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

def plot_newtons_fractal_with_fixed_points(map, modulo, step, **kwargs):
    grid = grid_starting_points((0,0), (1,1), 100, 100)

    map_fixed_points = find_unique_fixed_points(map, modulo)
    # use lambda to roll in the kwargs
    rolled_fixed_point_map = lambda xy, step: map_fixed_points(xy, step, **kwargs)

    unique_fixed_points = rolled_fixed_point_map(grid, step)

    expanded_fixed_points, colour_array = expand_fixed_points(unique_fixed_points, 0, 1, 0, 1)
    # test = newton_fractal((-1, -1), (1,1), 10, 10, basecase, no_modulo, step_NM, niter=50, test_grid=grid_starting_points((-1,1), (1,1), 10,10))

    x_points=1000
    y_points=1000

    starts = grid_starting_points((0,0), (1,1), x_points=x_points, y_points=y_points)

    test = apply_finder_to_grid(map, modulo, step, starts, x_points, y_points, expanded_fixed_points, 15, **kwargs)

    # colours = np.array([[114,229,239], [9,123,53], [42,226,130], [236,77,216], 
    #                     [157,187,230], [62,105,182], [149,200,88], [251,32,118],
    #                     [52,245,14], [225,50,25], [8,132,144], [218,164,249], 
    #                     [141,78,182], [250,209,57], [162,85,66], [233,191,152],
    #                     [111,125,67], [251,137,155], [231,134,7], [140,46,252]])

    colour_list = (colour_array/255).tolist()

    test2 = assign_colours_to_grid(test, colour_array)

    plt.imshow(test2, origin = 'lower', extent=(0, 1, 0, 1))
    plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
                edgecolor='black', linewidth = 2)

    # plt.title('Newtons Fractal for roots of z^3 - 1')

def plot_poincare_section(starts, niter, map, **kwargs):
    poincare = calculate_poincare_section(starts, niter, map, **kwargs)
    for i in range(poincare.shape[0]):
        plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                    color='gray', s=0.0001, marker ='.')


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

def expand_fixed_points(fixed_points, x_min, x_max, y_min, y_max):
    """
    Expands array of fixed points to include the modulo brothers.
    Generates array of colours where modulo brothers have the same colour.
    Outputs the array of expanded fixed points and the array of colours.

    Parameters:
    fixed_points: Kx2 array
        Array of fixed points from find_unique_fixed_points
    x_min: int
        lowest value of x-modulo
    x_max: int
        highest value of x-modulo
    y_min: int
        lowest value of y-modulo
    y_max: int
        highest value of y-modulo
    """
    colours = np.array([[114,229,239], [9,123,53], [42,226,130], [236,77,216], 
                        [157,187,230], [62,105,182], [149,200,88], [251,32,118],
                        [52,245,14], [225,50,25], [8,132,144], [218,164,249], 
                        [141,78,182], [250,209,57], [162,85,66], [233,191,152],
                        [111,125,67], [251,137,155], [231,134,7], [140,46,252]])
    
    colour_array = colours[0:fixed_points.shape[0], :]

    for i, point in enumerate(fixed_points):
        if point[0] == x_min:
            fixed_points = np.append(fixed_points, np.array([[x_max, point[1]]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)
        if point[0] == x_max:
            fixed_points = np.append(fixed_points, np.array([[x_min, point[1]]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)
        if point[1] == y_min:
            fixed_points = np.append(fixed_points, np.array([[point[0], y_max]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)
        if point[1] == y_max:
            fixed_points = np.append(fixed_points, np.array([[point[0], y_min]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)
        if point[0] == x_min and point[1] == y_min:
            fixed_points = np.append(fixed_points, np.array([[x_max, y_max]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)
        if point[0] == x_max and point[1] == y_max:
            fixed_points = np.append(fixed_points, np.array([[x_min, y_min]]), axis=0)
            colour_array = np.append(colour_array, np.array([colour_array[i]]), axis=0)

    return fixed_points, colour_array