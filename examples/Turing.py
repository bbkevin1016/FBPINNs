import matplotlib
# matplotlib.use('Agg')  # Comment out non-interactive backend
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer
from utils import plot_turing_patterns, plot_turing_evolution

class TuringSystem2D(Problem):
    """Solves the Turing reaction-diffusion system in 2D:
    ∂u/∂t = Du∇²u + f(u,v)
    ∂v/∂t = Dv∇²v + g(u,v)
    
    where:
    f(u,v) = r(a + u²/(v(1 + Ku²))) - μu
    g(u,v) = u² - νv
    
    with Neumann boundary conditions and localized initial perturbation
    """

    @staticmethod
    def init_params(Du=1.0, Dv=180.0, K=0.001, a=0.001, nu=5.5, mu=0.4, r=0.75):
        """Initialize parameters and compute steady state"""
        # Verify Turing instability conditions
        def check_turing_conditions(u_star, v_star, Du, Dv, K, a, nu, mu, r):
            # Compute Jacobian elements at steady state
            fu = 2*r*u_star/(v_star*(1 + K*u_star**2)) - \
                 2*r*K*u_star**3/(v_star*(1 + K*u_star**2)**2) - mu
            fv = -r*u_star**2/(v_star**2*(1 + K*u_star**2))
            gu = 2*u_star
            gv = -nu
            
            # Check conditions
            tr_J = fu + gv  # Trace of Jacobian
            det_J = fu*gv - fv*gu  # Determinant of Jacobian
            
            # Turing conditions
            homogeneous_stable = tr_J < 0 and det_J > 0
            cross_term = fu*gv - fv*gu > 0
            diffusion_term = Dv*fu + Du*gv > 0
            
            return {
                'homogeneous_stable': homogeneous_stable,
                'cross_term': cross_term,
                'diffusion_term': diffusion_term,
                'tr_J': tr_J,
                'det_J': det_J,
                'jacobian': {'fu': fu, 'fv': fv, 'gu': gu, 'gv': gv}
            }
        
        # Compute steady state
        def find_steady_state(r, a, mu, nu, K):
            # Solve cubic equation: K*u³ - (a*r*K/μ)*u² + u - r(ν+a)/μ = 0
            coeffs = [K, -a*r*K/mu, 1, -r*(nu+a)/mu]
            roots = np.roots(coeffs)
            # Find positive real root
            u_star = np.real(roots[np.logical_and(np.isreal(roots), roots > 0)][0])
            v_star = u_star**2/nu
            return u_star, v_star
        
        # Compute steady state
        u_star, v_star = find_steady_state(r, a, mu, nu, K)
        
        # Check Turing conditions
        turing_analysis = check_turing_conditions(u_star, v_star, Du, Dv, K, a, nu, mu, r)
        
        # Print analysis results
        print("\nTuring Analysis Results:")
        print(f"Steady state: u* = {u_star:.6f}, v* = {v_star:.6f}")
        print(f"Homogeneous stability: {turing_analysis['homogeneous_stable']}")
        print(f"Cross term positive: {turing_analysis['cross_term']}")
        print(f"Diffusion term positive: {turing_analysis['diffusion_term']}")
        print(f"Trace J = {turing_analysis['tr_J']:.6f}")
        print(f"Det J = {turing_analysis['det_J']:.6f}")
        
        static_params = {
            "dims": (2, 3),  # 2 outputs (u,v), 3 inputs (x,y,t)
            "Du": Du,        # Activator diffusion coefficient (normalized to 1)
            "Dv": Dv,        # Inhibitor diffusion coefficient
            "K": K,          # Saturation parameter
            "a": a,          # Basal production rate
            "nu": nu,        # Inhibitor removal rate
            "mu": mu,        # Activator removal rate
            "r": r,          # Reaction rate
            "u_star": u_star,  # Steady state activator
            "v_star": v_star,  # Steady state inhibitor
        }
        return static_params, {}

    @staticmethod
    def loss_fn(all_params, constraints):
        # Get parameters
        Du = all_params["static"]["problem"]["Du"]
        Dv = all_params["static"]["problem"]["Dv"]
        K = all_params["static"]["problem"]["K"]
        a = all_params["static"]["problem"]["a"]
        nu = all_params["static"]["problem"]["nu"]
        mu = all_params["static"]["problem"]["mu"]
        r = all_params["static"]["problem"]["r"]
        u_star = all_params["static"]["problem"]["u_star"]
        v_star = all_params["static"]["problem"]["v_star"]
        
        # Unpack constraints for PDE residuals
        x_batch_phys = constraints[0][0]
        derivs_phys = constraints[0][1:]
        
        # Get function values and derivatives for PDE
        u = derivs_phys[0]      # u value
        ux = derivs_phys[1]     # ∂²u/∂x²
        uy = derivs_phys[2]     # ∂²u/∂y²
        ut = derivs_phys[3]     # ∂u/∂t
        v = derivs_phys[4]      # v value
        vx = derivs_phys[5]     # ∂²v/∂x²
        vy = derivs_phys[6]     # ∂²v/∂y²
        vt = derivs_phys[7]     # ∂v/∂t
        
        # Get spatial and temporal coordinates for PDE
        x_phys = x_batch_phys[:, 0:1]
        y_phys = x_batch_phys[:, 1:2]
        t_phys = x_batch_phys[:, 2:3]
        
        # Compute time decay factor
        sd = 2.0  # Decay rate parameter
        e_phys = -0.5 * (x_phys**2 + y_phys**2 + t_phys**2) / (sd**2)
        decay_factor = jnp.exp(e_phys)
        
        # Compute Laplacians
        laplacian_u = ux + uy  # ∇²u
        laplacian_v = vx + vy  # ∇²v
        
        # Reaction terms with decay
        f = (r * (a + u**2/(v*(1 + K*u**2))) - mu*u) * decay_factor
        g = (u**2 - nu*v) * decay_factor

        # PDE residuals
        residual_u = ut - Du * laplacian_u - f
        residual_v = vt - Dv * laplacian_v - g
        pde_loss = jnp.mean(residual_u**2 + residual_v**2)
        
        # Unpack constraints for initial conditions
        x_batch_init = constraints[1][0]
        derivs_init = constraints[1][1:]
        
        # Get predicted values at initial conditions
        u_pred = derivs_init[0]  # Predicted u at t=0
        v_pred = derivs_init[1]  # Predicted v at t=0
        
        # Get coordinates for initial conditions
        x_init = x_batch_init[:, 0:1]
        y_init = x_batch_init[:, 1:2]
        
        # Compute reference initial conditions
        L1 = 5.0
        X_W = 10 * jnp.sqrt((x_init/L1)**2 + (y_init/L1)**2)
        
        def phi(s):
            return jnp.where(
                jnp.abs(s) <= 1.0,
                jnp.exp(1 - 1/(1 - s**2)),
                0.0
            )
        
        # Reference initial conditions
        perturbation = 0.1 * phi(X_W)
        u_ref = u_star * (1.0 + perturbation)  # Reference u at t=0
        v_ref = v_star * jnp.ones_like(x_init)  # Reference v at t=0
        
        # Initial condition loss
        init_loss = jnp.mean((u_pred - u_ref)**2 + (v_pred - v_ref)**2)
        
        # Total loss with weighted initial conditions
        total_loss = pde_loss + 1e6 * init_loss
             
        return total_loss

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        """Returns the steady state solution as reference"""
        u_star = all_params["static"]["problem"]["u_star"]
        v_star = all_params["static"]["problem"]["v_star"]
        
        return jnp.stack([
            u_star * jnp.ones(x_batch.shape[0]),
            v_star * jnp.ones(x_batch.shape[0])
        ], axis=1)

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Sample points for PDE constraints and initial conditions"""
        # Split random keys
        key1, key2 = jax.random.split(key)
        
        # Sample interior points for PDE
        x_batch_phys = domain.sample_interior(all_params, key1, sampler, batch_shapes[0])
        
        # Sample initial condition points (t=0)
        x_batch_init = domain.sample_interior(all_params, key2, sampler, batch_shapes[1])
        # Set time coordinate to 0 for initial conditions
        x_batch_init = x_batch_init.at[:,2].set(0.0)
        
        # Required derivatives for PDE residuals
        required_ujs_phys = [
            (0, ()),      # u value
            (0, (0,0)),  # ∂²u/∂x²
            (0, (1,1)),  # ∂²u/∂y²
            (0, (2,)),   # ∂u/∂t
            (1, ()),      # v value
            (1, (0,0)),  # ∂²v/∂x²
            (1, (1,1)),  # ∂²v/∂y²
            (1, (2,)),   # ∂v/∂t
        ]
        
        # Required derivatives for initial conditions
        required_ujs_init = [
            (0, ()),     # u value
            (1, ()),     # v value
        ]
        
        return [[x_batch_phys, required_ujs_phys], [x_batch_init, required_ujs_init]]

class TuringTrainer(PINNTrainer):
    def _test_step(self, all_params, i):
        # Get test predictions
        x_batch_test = self.c.domain.get_test_points(all_params, self.c.n_test)
        uv_test = self.network.apply(all_params, x_batch_test)
        u_test, v_test = uv_test[:, 0], uv_test[:, 1]
        
        # Create a new figure for this step
        plt.close('all')  # Close previous figures to free memory
        
        # Plot current state
        fig = plot_turing_patterns(x_batch_test, u_test, v_test, 
                                 t_idx=self.c.n_test[2]//2,  # middle time point
                                 n_test=self.c.n_test)
        
        # Add title with current epoch
        plt.suptitle(f'Training Progress - Epoch {i}')
        
        # Display the plot
        if self.c.show_figures:
            plt.pause(0.1)  # Add small pause to allow plot to update
            plt.show(block=False)  # Don't block execution
        
        return [("turing_pattern", fig)]

    def train(self):
        """Override train method to store the trained model and handle display"""
        plt.ion()  # Turn on interactive mode
        self.all_params = super().train()
        plt.ioff()  # Turn off interactive mode
        return self.all_params

# Domain setup
L = 10.0   # Reduced domain size
T = 20.0   # Reduced time range

subdomain_xs = [np.linspace(-L/2, L/2, 5),  # Reduced number of subdomains
                np.linspace(-L/2, L/2, 5), 
                np.linspace(0, T, 4)]
subdomain_ws = get_subdomain_ws(subdomain_xs, 1.9)

c = Constants(
    run="turing_pattern",
    domain=RectangularDomainND,
    domain_init_kwargs=dict(
        xmin=np.array([-L/2, -L/2, 0]),
        xmax=np.array([L/2, L/2, T]),
    ),
    problem=TuringSystem2D,
    problem_init_kwargs=dict(
        Du=1.0,     # Normalized activator diffusion
        Dv=180.0,   # Inhibitor diffusion (Dv >> Du required for Turing instability)
        K=0.001,    # Saturation parameter (K << 1)
        a=0.001,    # Basal production rate
        nu=5.5,     # Inhibitor removal rate
        mu=0.4,     # Activator removal rate
        r=0.75,     # Reaction rate
    ),
    decomposition=RectangularDecompositionND,
    decomposition_init_kwargs=dict(
        subdomain_xs=subdomain_xs,
        subdomain_ws=subdomain_ws,
        unnorm=(0., 1.),
    ),
    network=FCN,
    network_init_kwargs=dict(
        layer_sizes=[3, 64, 64, 32, 2],  # Reduced network size
    ),
    ns=((60,60,30), (60,60,1)),    # Reduced spatial and temporal resolution
    n_test=(80,80,20),             # Reduced test resolution
    n_steps=15000,                  # Reduced number of steps
    optimiser_kwargs=dict(learning_rate=1e-4),  # Keep stable learning rate
    summary_freq=100,               # Keep frequent updates
    test_freq=500,                 # Keep regular visualization
    model_save_freq=2500,
    show_figures=True,
    save_figures=True,
    clear_output=False,
)

# Train using PINN trainer
run = PINNTrainer(c)
run.train()

# After training, show final evolution plot
plt.close('all')

# Get test points and predictions
x_batch_test = run.c.domain.get_test_points(run.all_params, run.c.n_test)
uv_test = run.network.apply(run.all_params, x_batch_test)
u_test, v_test = uv_test[:, 0], uv_test[:, 1]

# Plot evolution
figs = plot_turing_evolution(x_batch_test, u_test, v_test, run.c.n_test)
plt.show()