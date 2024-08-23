from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp

from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, apply_step, fixed_point_finder, fixed_point_trajectory, find_unique_fixed_points
from methods import apply_finder_to_grid

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

def plot_poincare_section(starts, niter, map, **kwargs):
    """
    Produces plot of Poincare Section.

    Parameters:
    starts: Nx2 array
        line of starting points
    niter: int
        number of iterations of map
    map: function
        map which will be used 
    """
    poincare = calculate_poincare_section(starts, niter, map, **kwargs)
    for i in range(poincare.shape[0]):
        plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                    color='black', s=0.0001, marker ='.')

def plot_point_trajectories_to_fixed_points(starts, map, modulo, step, niter, **kwargs):
    """
    Produces plot of lines connecting the start points to their steps all the way to their final points (which ideally are the fixed points of the map).

    Parameters:
    starts: Nx2 array
        starting points to plot trajectories for. this function and plot are not ideal for a large number of starting points.
    map: function
        the map to investigate
    modulo: function
        the modulo linked to the map
    step: function
        the type of step used for the steps to the fixed points
    niter: int
        the number of iterations
    """
    steps = fixed_point_trajectory(starts, map, modulo, step, niter, **kwargs)

    cmap1 = colormaps['gist_heat']
    cmap2 = colormaps['PRGn']
    cmap3 = colormaps['PiYG']
    colors1 = cmap1(np.linspace(0.5, 1, steps.shape[2]))
    colors2 = cmap2(np.linspace(0.25, 0.5, steps.shape[2]))
    colors3 = cmap2(np.linspace(0.75, 0.5, steps.shape[2]))
    colors4 = cmap3(np.linspace(0.25, 0.5, steps.shape[2]))
    colors = [colors1, colors2, colors3, colors4]

    for j in range(steps.shape[0]): # for each fixed point
        for i in range(steps.shape[2]): # for each line segment
            plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                    color='blue',
                    ms=10, marker ='.', markerfacecolor=colors[j][i], markeredgecolor='blue')

def plot_fixed_points(grid, xy_start, xy_end, map, map_modulo, step, step_modulo, **kwargs):
    """
    Produces plot of fixed points of map.

    Parameters:
    grid: Nx2 array
        grid of starting points. doesn't need to be super high resolution.
    xy_start: tuple
        coordinates of lower-left corner of grid.
    xy_end: tuple
        coordinates of upper-right corner of grid.    
    map: function
        map which will be used 
    modulo: function
        the modulo linked to the map
    step: function
        the type of step used for the steps to the fixed points
    niter: int
        the number of iterations
    """
    map_fixed_points = find_unique_fixed_points(map, map_modulo)
    # use lambda to roll in the kwargs
    rolled_fixed_point_map = lambda xy, step: map_fixed_points(xy, step, step_modulo, **kwargs)
    unique_fixed_points = rolled_fixed_point_map(grid, step)
    expanded_fixed_points, colour_array = expand_fixed_points(unique_fixed_points, xy_start[0], xy_end[0], xy_start[1], xy_end[1])
    colour_list = (colour_array/255).tolist()

    plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
                edgecolor='black', linewidth = 2)

def plot_newtons_fractal(xy_start, xy_end, x_points, y_points, map, map_modulo, step, step_modulo, niter, **kwargs):
    """
    Produces plot of Newton's fractal of map.

    Parameters:
    xy_start: tuple
        coordinates of lower-left corner of grid.
    xy_end: tuple
        coordinates of upper-right corner of grid.    
    x_points: int
        number of columns in grid
    y_points: int
        number of rows in grid
    map: function
        map which will be used 
    modulo: function
        the modulo linked to the map
    step: function
        the type of step used for the steps to the fixed points
    niter: int
        the number of iterations
    """
    # initialise grid to find fixed points.
    grid = grid_starting_points(xy_start, xy_end, 100, 100)
    # initialise function to find fixed points.
    map_fixed_points = find_unique_fixed_points(map, map_modulo, 50)
    # use lambda to roll in the kwargs.
    rolled_fixed_point_map = lambda xy, step: map_fixed_points(xy, step, step_modulo, **kwargs)
    # use map_fixed_points and find fixed points.
    unique_fixed_points_array = rolled_fixed_point_map(grid, step)
    # use expand_fixed_points to generate expanded list of fixed points as well as corresponding colour array.
    expanded_fixed_points, colour_array = expand_fixed_points(unique_fixed_points_array, xy_start[0], xy_end[0], xy_start[1], xy_end[1])
    # initialise grid of starting points for newton's fractal.
    starts = grid_starting_points(xy_start, xy_end, x_points=x_points, y_points=y_points)
    # use newton's fractal function to get coordinate array with indices of fixed points as elements.
    fixed_point_index_grid = apply_finder_to_grid(map, map_modulo, step, step_modulo, starts, x_points, y_points, expanded_fixed_points, niter, **kwargs)
    # replace each index with rgb value.
    colour_grid = assign_colours_to_grid(fixed_point_index_grid, colour_array)
    # plot.
    plt.imshow(colour_grid, origin = 'lower', extent=(xy_start[0], xy_end[0], xy_start[1], xy_end[1]))

def assign_colours_to_grid(grid, colours):
    """
    Generates an MxNx3 array where each MxN coordinate has an RGB value attached to it.
    
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
    # colours = np.array([[114,229,239], [9,123,53], [42,226,130], [236,77,216], 
    #                     [157,187,230], [62,105,182], [149,200,88], [251,32,118],
    #                     [52,245,14], [225,50,25], [8,132,144], [218,164,249], 
    #                     [141,78,182], [250,209,57], [162,85,66], [233,191,152],
    #                     [111,125,67], [251,137,155], [231,134,7], [140,46,252]])
    
    colours = np.array([[236, 103, 62], [6, 149, 113], [121, 70, 129], [158, 102, 216],
                        [129, 255, 53], [214, 186, 169], [254, 192, 209], [39, 74, 138],
                        [88, 146, 90], [202, 220, 11], [17, 109, 238], [166, 12, 110],
                        [222, 136, 28], [10, 184, 72], [153, 18, 123], [178, 28, 11],
                        [110, 211, 62], [31, 244, 133], [58, 78, 236], [88, 116, 157],
                        [101, 6, 176], [52, 59, 178], [57, 5, 167], [146, 187, 18],
                        [44, 104, 41], [47, 105, 254], [58, 97, 53], [69, 156, 78],
                        [40, 55, 230], [169, 76, 236], [89, 58, 221], [209, 211, 7],
                        [184, 96, 21], [167, 111, 156], [123, 133, 204], [69, 193, 126],
                        [199, 195, 249], [234, 88, 48], [180, 177, 209], [152, 183, 206],
                        [114, 155, 195], [104, 249, 169], [232, 55, 34], [20, 152, 101],
                        [159, 107, 119], [181, 169, 28], [90, 152, 229], [22, 199, 248],
                        [149, 54, 185], [184, 40, 92],])

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