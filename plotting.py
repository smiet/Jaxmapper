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
from methods import newton_fractal, apply_finder_to_grid

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
    grid = grid_starting_points((-1,-1), (1,1), 10, 10)

    basecase_fixed_points = find_unique_fixed_points(basecase, no_modulo)
    unique_fixed_points = basecase_fixed_points(grid, step_NM)

    # test = newton_fractal((-1, -1), (1,1), 10, 10, basecase, no_modulo, step_NM, niter=50, test_grid=grid_starting_points((-1,1), (1,1), 10,10))

    x_points=1000
    y_points=1000

    starts = grid_starting_points((-1,-1), (1,1), x_points=x_points, y_points=y_points)

    test = apply_finder_to_grid(basecase, no_modulo, step_NM, starts, x_points, y_points, unique_fixed_points, 15)

    colour_array = np.array([[114,229,239], [9,123,53], [42,226,130]])
    colour_list = (colour_array/255).tolist()

    test2 = assign_colours_to_grid(test, colour_array)

    plt.imshow(test2, origin = 'lower', extent=(-1, 1, -1, 1))
    plt.scatter(unique_fixed_points[:, 0], unique_fixed_points[:, 1], facecolors=colour_list, marker='o', 
                edgecolor='black', linewidth = 2)

    plt.title('Newtons Fractal for roots of z^3 - 1')

def assign_colours_to_grid(grid, colours):
    """
    Parameters:
    grid: MxN array
        array outputted from apply_finder_to_grid. 
        each element is an int corresponding to the fixed point nearest to that point after n iterations of the step.
    colours: Kx3 array
        array containing K RGB values, where K is the number of fixed points.
    """
    final = colours[grid]
    return final