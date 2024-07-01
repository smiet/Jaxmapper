from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp

import math

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, basecase, no_modulo
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, apply_step, fixed_point_finder, fixed_point_trajectory, find_unique_fixed_points
from methods import mapping_vector, theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo

def plot_save_points(points, name='fig', colors='random'):
    """
    save a plot of the 'points' 
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    if colors== 'random':
        colors = onp.random.random((points.shape[0], 3))
    for i in range(points.shape[0]):
        ax.scatter(points[i,0, :], points[i,1,:], color=colors[i,:], s=0.1, marker='.')
    # give plot details
    plt.xlim([0,1])
    plt.ylim([-1.5,1.5])
    fig.savefig(name, bbox_inches='tight', dpi=300)

def plot_newtons_fractal_with_fixed_points():
    grid = grid_starting_points((-1,-1), (1,1), 300, 300)

    basecase_fixed_points = find_unique_fixed_points(basecase, no_modulo)
    unique_fixed_points = basecase_fixed_points(grid, step_NM)

    cmap = colormaps['gist_rainbow']
    colours = cmap(np.linspace(0, 1, unique_fixed_points.shape[0]))

    test = newton_fractal((-1, -1), (1,1), 10, 10, 
                        basecase, no_modulo, step_NM, niter=50,
                        test_grid=grid_starting_points((-1,1), (1,1), 10,10))


    plt.imshow(test, origin = 'lower', extent=(-1, 1, -1, 1))
    plt.scatter(unique_fixed_points[:, 0], unique_fixed_points[:, 1], facecolor=colours, marker='o', 
                edgecolor='black', linewidth = 2)

    plt.title('Newtons Fractal for roots of z^3 - 1')