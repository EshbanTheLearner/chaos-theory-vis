import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

WIDTH, HEIGH, DPI = 1000, 750, 100

def lorenz_equations(t, X, sigma, beta, rho):
    u, v, w = X
    u_p = -sigma * (u - v)
    v_p = rho * u - v - u*w
    w_p = -beta * w + u*v
    return u_p, v_p, w_p

def solve_ODE(function, interval, initial_state, args):
    solution = solve_ivp(
        function,
        interval,
        initial_state,
        args,
        dense_output=True
    )
    return solution

def interpolating_to_grid(solution, interval, n_samples):
    t0, tmax = interval
    t = np.linspace(t0, tmax, n_samples)
    x, y, z = solution.sol(t)
    return x, y, z

def plot(width, height, dpi, n, coords, save=False):
    x, y, z = coords
    fig = plt.figure(facecolor="k", figsize=(width/dpi, height/dpi))
    ax = fig.gca(projection="3d")
    ax.set_facecolor("k")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    s = 10
    cmap = plt.cm.winter
    for i in range(0, n-s, s):
        ax.plot(
            x[i:i+s+1], 
            y[i:i+s+1], 
            z[1:i+s+1], 
            color=cmap(i/n), 
            alpha=0.4)
    ax.set_axis_off()
    if save:
        plt.savefig("lorenz.png", dpi=dpi)
    plt.show()

