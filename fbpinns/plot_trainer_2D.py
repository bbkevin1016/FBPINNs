"""
Defines plotting functions for 2D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import matplotlib.pyplot as plt

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy

def _plot_test_im(u_test, xlim0, ulim, n_test, it=None):
    # Handle different input shapes
    if len(n_test) == 3:  # 3D case (x, y, t)
        nx, ny, nt = n_test
        total_points = nx * ny
        
        # If the data is already in time-major format
        if len(u_test.shape) == 1 and u_test.size == total_points * nt:
            u_test = u_test.reshape(total_points, nt)
            if it is not None:
                u_test = u_test[:, it]
        elif len(u_test.shape) == 2:
            if it is not None:
                u_test = u_test[:, it]
        
        # Reshape to spatial dimensions
        u_test = u_test.reshape(nx, ny)
    else:  # 2D case (x, y)
        nx, ny = n_test[:2]  # Take only spatial dimensions
        u_test = u_test.reshape(nx, ny)
    
    # Create the plot
    plt.imshow(u_test.T, 
               origin="lower",
               extent=[xlim0[0], xlim0[1], xlim0[0], xlim0[1]],
               cmap="viridis",
               vmin=ulim[0],
               vmax=ulim[1])
    plt.colorbar()
    plt.gca().set_aspect("equal")

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    f = plt.figure(figsize=(8,10))

    # plot domain + x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solutions
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solution
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)