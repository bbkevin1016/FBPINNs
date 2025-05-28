import numpy as np
import matplotlib.pyplot as plt

def plot_turing_patterns(x, y, u, v, t, title="Turing Patterns"):
    """Plot Turing patterns at a specific time point."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title} (t={t:.2f})")
    
    # Plot u concentration
    im1 = axes[0,0].pcolormesh(x, y, u, shading='auto')
    axes[0,0].set_title('u concentration')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot v concentration
    im2 = axes[0,1].pcolormesh(x, y, v, shading='auto')
    axes[0,1].set_title('v concentration')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Plot u-v pattern
    im3 = axes[1,0].pcolormesh(x, y, u-v, shading='auto')
    axes[1,0].set_title('Pattern (u-v)')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Plot total concentration
    im4 = axes[1,1].pcolormesh(x, y, u+v, shading='auto')
    axes[1,1].set_title('Total (u+v)')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    return fig

def create_grid(nx, ny, nt, xmin=-5, xmax=5, ymin=-5, ymax=5, tmin=0, tmax=5):
    """Create spatial and temporal grids."""
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    t = np.linspace(tmin, tmax, nt)
    
    X, Y = np.meshgrid(x, y)
    return x, y, t, X, Y

def plot_turing_evolution(x_batch_test, u_test, v_test, n_test, n_frames=5):
    """
    Plot the evolution of Turing patterns over time
    
    Args:
        x_batch_test: Test points coordinates
        u_test: First species concentration
        v_test: Second species concentration
        n_test: Tuple of (nx, ny, nt) test points
        n_frames: Number of time frames to plot
    """
    nx, ny, nt = n_test
    time_indices = np.linspace(0, nt-1, n_frames, dtype=int)
    
    figs = []
    for t_idx in time_indices:
        fig = plot_turing_patterns(x_batch_test[:,0], x_batch_test[:,1], u_test[t_idx], v_test[t_idx], t_idx/(nt-1),
                                 title=f"Turing Patterns Evolution (t={t_idx/(nt-1):.2f})")
        figs.append(fig)
    
    return figs 