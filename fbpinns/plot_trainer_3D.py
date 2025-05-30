"""
Defines plotting functions for 3D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import matplotlib.pyplot as plt

from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy
from fbpinns.plot_trainer_2D import _plot_test_im

@_to_numpy
def plot_3D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    nt = n_test[-1]# slice across last dimension
    shape = (1+nt+1, 3)# nrows, ncols
    f = plt.figure(figsize=(8,8*shape[0]/3))

    # plot domain + x_batch
    for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
        plt.subplot2grid(shape,(0,iplot))
        plt.title(f"[{i}] Domain decomposition")
        plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
        decomposition.plot(all_params, active=active, create_fig=False, iaxes=[a,b])
        plt.xlim(xlim[0][a], xlim[1][a])
        plt.ylim(xlim[0][b], xlim[1][b])
        plt.gca().set_aspect("equal")

    # plot full solutions
    for it in range(nt):
        plt.subplot2grid(shape,(1+it,0))
        plt.title(f"[{i}] Full solution")
        _plot_test_im(u_test, xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,1))
        plt.title(f"[{i}] Ground truth")
        _plot_test_im(u_exact, xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,2))
        plt.title(f"[{i}] Difference")
        _plot_test_im(u_exact - u_test, xlim0, ulim, n_test, it=it)

    # plot raw hist
    plt.subplot2grid(shape,(1+nt,0))
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_3D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):
    # Get plot limits for both components
    xlim0 = [all_params["static"]["domain"]["xmin"][0], all_params["static"]["domain"]["xmax"][0]]
    xlim1 = [all_params["static"]["domain"]["xmin"][1], all_params["static"]["domain"]["xmax"][1]]
    
    # For multi-component systems, split into components if not already split
    n_components = all_params["static"]["problem"]["dims"][0]
    if n_components > 1:
        # Get solution limits for each component
        if len(u_test.shape) == 2:  # If data is already split into components
            u_tests = [u_test[:, i] for i in range(n_components)]
            u_exacts = [u_exact[:, i] for i in range(n_components)]
        else:  # If data needs to be split
            u_tests = [u_test[i::n_components] for i in range(n_components)]
            u_exacts = [u_exact[i::n_components] for i in range(n_components)]
        
        # Calculate limits for each component
        ulims = []
        for comp in range(n_components):
            u_min = min(u_exacts[comp].min(), u_tests[comp].min())
            u_max = max(u_exacts[comp].max(), u_tests[comp].max())
            ulims.append([u_min, u_max])
    else:
        u_tests = [u_test]
        u_exacts = [u_exact]
        u_min = min(u_exact.min(), u_test.min())
        u_max = max(u_exact.max(), u_test.max())
        ulims = [[u_min, u_max]]
    
    # Create figure
    nt = n_test[-1]  # Number of time steps
    n_rows = (nt + 1) // 2
    f = plt.figure(figsize=(12 * n_components, 4 * n_rows))
    
    # Plot each component
    for comp in range(n_components):
        for t in range(nt):
            plt.subplot(n_rows, 2 * n_components, t * n_components + comp + 1)
            _plot_test_im(u_tests[comp], xlim0, ulims[comp], n_test, it=t)
            plt.xlabel('x')
            plt.ylabel('y')
            component_name = 'u' if comp == 0 else 'v'
            plt.title(f'{component_name} at t={t}')

    plt.tight_layout()
    return [("test_solution", f)]