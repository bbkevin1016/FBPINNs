import nbformat as nbf

nb = nbf.v4.new_notebook()

# First cell - imports
imports = '''import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from IPython.display import clear_output

from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.networks import FCN
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.trainers import FBPINNTrainer, PINNTrainer

%matplotlib inline'''

# Second cell - TuringSystem2D class
turing_system = '''class TuringSystem2D(Problem):
    """Solves the Turing reaction-diffusion system in 2D:
        ∂u/∂t = Du∇²u + f(u,v)
        ∂v/∂t = Dv∇²v + g(u,v)
        
        where f and g are the reaction terms from the Schnakenberg model:
        f(u,v) = a - u + u²v
        g(u,v) = b - u²v
        
        with periodic boundary conditions and random initial conditions
    """

    @staticmethod
    def init_params(Du=1.0, Dv=10.0, a=0.1, b=0.9):
        static_params = {
            "dims": (2, 3),  # 2 outputs (u,v), 3 inputs (x,y,t)
            "Du": Du,        # Diffusion coefficient for u
            "Dv": Dv,        # Diffusion coefficient for v
            "a": a,          # Reaction parameter
            "b": b,          # Reaction parameter
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
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
        
        # Required derivatives for initial conditions (no derivatives needed)
        required_ujs_init = [
            (0, ()),     # u value
            (1, ()),     # v value
        ]
        
        return [[x_batch_phys, required_ujs_phys], [x_batch_init, required_ujs_init]]

    @staticmethod
    def loss_fn(all_params, constraints):
        Du = all_params["static"]["problem"]["Du"]
        Dv = all_params["static"]["problem"]["Dv"]
        a = all_params["static"]["problem"]["a"]
        b = all_params["static"]["problem"]["b"]
        
        # Unpack constraints - each constraint is a list [x_batch, *ujs]
        x_batch_phys = constraints[0][0]
        x_batch_init = constraints[1][0]
        
        # Get derivatives from ujs
        derivs_phys = constraints[0][1:]  # Skip x_batch
        derivs_init = constraints[1][1:]  # Skip x_batch
        
        # Get function values and derivatives
        u = derivs_phys[0]      # u value
        ux = derivs_phys[1]     # ∂u/∂x
        uy = derivs_phys[2]     # ∂u/∂y
        ut = derivs_phys[3]     # ∂u/∂t
        v = derivs_phys[4]      # v value
        vx = derivs_phys[5]     # ∂v/∂x
        vy = derivs_phys[6]     # ∂v/∂y
        vt = derivs_phys[7]     # ∂v/∂t
        
        # Compute Laplacians using the provided derivatives
        laplacian_u = ux + uy  # ∂²u/∂x² + ∂²u/∂y²
        laplacian_v = vx + vy  # ∂²v/∂x² + ∂²v/∂y²
        
        # Reaction terms (Schnakenberg model)
        f = a - u + u**2 * v
        g = b - u**2 * v
        
        # PDE residuals
        residual_u = ut - Du * laplacian_u - f
        residual_v = vt - Dv * laplacian_v - g
        
        # Initial conditions
        u_init = derivs_init[0]  # u value at t=0
        v_init = derivs_init[1]  # v value at t=0
        
        # Random initial conditions centered around the homogeneous steady state
        key = jax.random.PRNGKey(0)
        u0 = a + b + jax.random.normal(key, u_init.shape) * 0.1
        v0 = b/((a + b)**2) + jax.random.normal(key, v_init.shape) * 0.1
        
        # Compute losses
        init_loss = jnp.mean((u_init - u0)**2 + (v_init - v0)**2)
        pde_loss = jnp.mean(residual_u**2 + residual_v**2)
        
        # Weight the losses
        total_loss = pde_loss + 10.0 * init_loss
        
        return total_loss

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        # For Turing patterns, there is no exact solution
        # Return homogeneous steady state as reference
        a = all_params["static"]["problem"]["a"]
        b = all_params["static"]["problem"]["b"]
        
        u_ss = a + b
        v_ss = b/((a + b)**2)
        
        # Reshape to match the expected output shape
        return jnp.stack([
            u_ss * jnp.ones(x_batch.shape[0]),
            v_ss * jnp.ones(x_batch.shape[0])
        ], axis=1)'''

# Third cell - Constants setup
setup = '''# Domain setup
Lx, Ly = 10.0, 10.0  # Smaller domain for better pattern formation
T = 5.0              # Shorter time range

subdomain_xs = [np.linspace(-Lx/2, Lx/2, 5), 
                np.linspace(-Ly/2, Ly/2, 5), 
                np.linspace(0, T, 5)]
subdomain_ws = get_subdomain_ws(subdomain_xs, 1.9)

c = Constants(
    run="turing_pattern",
    domain=RectangularDomainND,
    domain_init_kwargs=dict(
        xmin=np.array([-Lx/2, -Ly/2, 0]),
        xmax=np.array([Lx/2, Ly/2, T]),
    ),
    problem=TuringSystem2D,
    problem_init_kwargs=dict(
        Du=0.1,     # Smaller diffusion coefficient for u
        Dv=2.0,     # Larger diffusion coefficient ratio
        a=0.1,      # Reaction parameter a
        b=0.9,      # Reaction parameter b
    ),
    decomposition=RectangularDecompositionND,
    decomposition_init_kwargs=dict(
        subdomain_xs=subdomain_xs,
        subdomain_ws=subdomain_ws,
        unnorm=(0., 1.),
    ),
    network=FCN,
    network_init_kwargs=dict(
        layer_sizes=[3, 64, 64, 32, 2],  # Deeper network
    ),
    ns=((80,80,40), (80,80,1)),  # More points for better resolution
    n_test=(100,100,20),         # Test resolution
    n_steps=15000,               # More training steps
    optimiser_kwargs=dict(learning_rate=5e-4),  # Lower learning rate
    summary_freq=200,
    test_freq=200,
    model_save_freq=10000,
    show_figures=True,
    save_figures=False,
    clear_output=True,
)'''

# Fourth cell - Training
training = '''# Train using custom Turing trainer
run = PINNTrainer(c)
all_params = run.train()'''

# Add cells to notebook
nb['cells'] = [
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_code_cell(turing_system),
    nbf.v4.new_code_cell(setup),
    nbf.v4.new_code_cell(training)
]

# Write the notebook to a file
with open('examples/Turing_notebook.ipynb', 'w') as f:
    nbf.write(nb, f) 