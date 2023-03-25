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

