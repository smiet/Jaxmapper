from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
import math
from scipy.optimize import minimize
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

from maps import Nmap, sym_jac_func, no_modulo
from maps import standard_map_theta_modulo as no_p_mod

def calculate_poincare_section(starts, niter, map, modulo, **kwargs):
    """
    Calculate a Poincare section of a mapping with parameters a and b.
    Iterate niter times.
    Returns an array of shape (ijk) where i indexes the point in starts, j indexes the x/y value, 
    and k indexes the iteration number (0 is the original point before any mappings).    
    """
    # use lambda to "roll-in" the mapping kwargs and include modulo
    rolled_map = lambda xy: modulo(map(xy, **kwargs))
    # use vmap to create a function from all starts to all mapped points
    applymap = jit(vmap(rolled_map, in_axes=0))
    # initialize results array
    iterations = [starts, ]
    # calculate mapping of previous mappings niter times
    for _ in range(niter):
        iterations.append(applymap(iterations[-1]))
    # stack into a nice array for returning. 
    return np.stack(iterations, axis=-1)

def starts_nontwist(norbits_per_leg):
    """
    hacky function to create a bunch of starting points
    """
    x = np.linspace(0, 0.5, norbits_per_leg)
    y = np.linspace(-1, 1., norbits_per_leg)
    starts1 = np.stack([x,y], axis=1)
    x = 0.50* np.ones(norbits_per_leg)
    y = np.linspace(0., 1.0, norbits_per_leg)
    starts2 = np.stack([x,y], axis=1)
    x = np.zeros(norbits_per_leg)
    y = np.linspace(-1., 0, norbits_per_leg)
    starts3 = np.stack([x,y], axis=1)
    starts = np.append(starts1, starts2, axis=0)
    starts = np.append(starts, starts3, axis=0)
    return starts

def grid_starting_points(xy_start = tuple, xy_end = tuple, x_points = int, y_points = int):
    """
    Creates a grid of starting points determined by given points which are taken as corners of the grid.
    Returns an array of shape (ij) where i is the i-th point in the grid and j is the x/y-value.
    Length of i is x_points*y_points.
    """
    x_start = xy_start[0] + np.sqrt(2)*10e-6
    y_start = xy_start[1] + np.sqrt(2)*10e-6
    x_end = xy_end[0] + np.sqrt(2)*10e-6
    y_end = xy_end[1] + np.sqrt(2)*10e-6
    # create arrays for x and y points
    x = np.linspace(x_start, x_end, x_points)
    y = np.linspace(y_start, y_end, y_points)
    # create an array with shape (x_points*y_points,2)
    starts = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    return starts

def linear_starting_points(xy_start = tuple, xy_end = tuple, npoints = int):
    """
    Creates a bunch of starting points along a line determined by a given start and end point.
    Returns an array with shape (ij) where i is the i-th point in the line and j is the x/y-value.
    Length of i is npoints.
    """
    x_start, y_start = xy_start
    x_end, y_end = xy_end

    x_start += np.sqrt(2)*10e-6
    y_start += np.sqrt(2)*10e-6
    x_end += np.sqrt(2)*10e-6
    y_end += np.sqrt(2)*10e-6

    if x_start == x_end: # if the line is vertical
        # create y array with evenly spaced y_points
        y_array = np.linspace(y_start, y_end, num=npoints)
        # create a list containing the points
        starts = [np.array([x_start, y]) for y in y_array]
        # turn list into array and return
        return np.array(starts)
    else:
        # calculate gradient and y-intercept of line with given start and end point
        m = (y_end - y_start)/(x_end - x_start)
        c = y_start - m*x_start
        # create x array with evenly-spaced x-points
        x_array = np.linspace(x_start, x_end, num = npoints)
        # create y array with each point given by y = mx + c
        y_array = np.array([m*x + c for x in x_array])
        # create an array with shape (npoints, 2)
        starts = np.stack([x_array,y_array], axis=1)
        return starts

def apply_step(step, modulo):
    """
    Outputs a function which takes in an xy and returns modulo(xy + step(xy)).

    Parameters:
    step: function
        step function which is applied
    modulo: function
        modulo linked to the map 
    """
    def final(xy, **kwargs):
        return modulo(xy + step(xy, **kwargs))
    return jit(final)

def step_NM(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Newton step for xy towards a fixed point.
    
    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map. modulo is applied on map(xy) and then mod(map(xy))-xy is calculated
    method: string
        which method to use. choose between 'jax' or 'sympy'
    """
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        # use lambda to "roll-in" the mapping kwargs
        rolled_map = lambda xy: map(xy, **kwargs)
        M = jacfwd(rolled_map)(xy)
        A = M - np.eye(2)
        #diff = mapping_vector_modulo(rolled_map, modulo)
        diff = modded_mapping_vector(rolled_map, modulo)
        b = -diff(xy)
        delta = np.linalg.solve(A,b)
        return delta
    return jit(step)
    
def step_NM_sym(map):    
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Newton step for xy towards a fixed point.
    
    Parameters:
    map: sympy function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    method: string
        which method to use. choose between 'jax' or 'sympy'
    """
    def step(xy, xmod, ymod, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        xmod: divisor of modulo operation of x 
        ymod: divisior of modulo operation of y
        xmod and ymod are necessary for sympy as jacobian operation doesn't work on modulo operation in sympy.
        Put a large number for xmod/ymod if it doesn't need to be modded.
        """
        x, y = sym.symbols('x y')
        sym_expr = map(**kwargs)
        M = sym_jac_func(sym_expr)(xy)
        A = M - np.eye(2)
        f = sym.lambdify([x,y], sym_expr, 'numpy')(xy[0], xy[1])
        # the f above is a (ij) array where i is the x/y-value and j is a redudant axis of length 1.
        # so we flatten f.
        f = f.flatten()
        f = np.array([np.mod(f[0], xmod), np.mod(f[1], ymod)])
        b = xy-f
        delta = np.linalg.solve(A,b)
        ### NOTE: NO JIT APPLIED HERE AS IT DOESN'T WORK WITH SYMPY.
        return delta
    return step

def step_AGTNMx(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Topological Newton step for xy towards a fixed x-point.
    Step length is determined by magnitude of gradient of theta.

    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    """
    map_isotrope = isotrope(map, modulo)
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        delta = map_isotrope(xy, **kwargs)
        length = np.linalg.norm(delta)
        delta = (1/length)**2 * delta
        return np.nan_to_num(delta)
    return jit(step)

def step_AGTNMo(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Topological Newton step for xy towards a fixed o-point.
    Step length is determined by magnitude of gradient of theta.

    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    """
    map_isotrope = isotrope(map, modulo)
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        delta = map_isotrope(xy, **kwargs)
        length = np.linalg.norm(delta)
        delta = (1/length)**2 * delta
        return np.nan_to_num(-delta)
    return jit(step)

def step_OTNMx(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs an Optimised Topological Newton step for xy towards a fixed x-point.
    
    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    """
    map_isotrope = isotrope(map, modulo)
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        delta = map_isotrope(xy, **kwargs)
        unit_delta = delta / np.linalg.norm(delta)
        difference_vector = mapping_vector(map, no_p_mod)
        length = lambda xy: np.linalg.norm(difference_vector(xy, **kwargs))
        grad_length = grad(length)(xy)
        scaling = (length(xy))/abs(np.dot(grad_length, unit_delta))
        return np.nan_to_num(scaling * unit_delta)
    return jit(step)

def step_LGTNMx(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Topological Newton step for xy towards a fixed x-point.
    Step length is determined by distance between f(xy) and xy.

    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    """
    map_isotrope = isotrope(map, modulo)
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        delta = map_isotrope(xy, **kwargs)
        unit_delta = delta / np.linalg.norm(delta)

        length = lambda xy: modded_length(xy, no_p_mod(map(xy, **kwargs)), xmod=1, ymod=1)
        grad_length = grad(length)(xy)

        scaling = (length(xy))/abs(np.dot(grad_length, unit_delta))
        return np.nan_to_num(scaling * unit_delta)
    return jit(step)

def step_LGTNMo(map, modulo):
    """
    Outputs a function 'step' which takes in xy and **kwargs of map, and outputs a Topological Newton step for xy towards a fixed x-point.
    Step length is determined by distance between f(xy) and xy.
    
    Parameters:
    map: function
        map to calculate the Newton step of
    modulo: function
        modulo linked to the map
    """
    map_isotrope = isotrope(map, modulo)
    def step(xy, **kwargs):
        """
        Function which returns the step from xy towards the fixed point (to first order).
        """
        delta = map_isotrope(xy, **kwargs)
        unit_delta = delta / np.linalg.norm(delta)

        length = lambda xy: modded_length(xy, no_p_mod(map(xy, **kwargs)), xmod=1, ymod=1)
        grad_length = grad(length)(xy)

        scaling = (length(xy))/abs(np.dot(grad_length, unit_delta))
        return np.nan_to_num(-scaling * unit_delta)
    return jit(step)

def modulo_add(xmod, ymod):
    """
    Outputs a 9x2 array where the first axis indexes the step to the modulo brother and the second axis indexes the x/y-value.

    Parameters:
    xmod: int
        modulo for x coordinate (put large value if no modulo)
    ymod: int
        modulo for y coordinate (put large value if no modulo)
    """
    x_direction = np.array([-xmod,0,xmod])
    y_direction = np.array([-ymod,0,ymod])
    test2 = np.array(np.meshgrid(x_direction, y_direction))
    return test2.T.reshape(-1,2)

def modded_length(xy1, xy2, xmod, ymod):
    """
    Calculates the xmallest length between 2 points where the modulo brothers of the second point are considered.
    """
    modulo_steps = modulo_add(xmod, ymod)
    length_array = np.linalg.norm(xy1[None, :] - (xy2[None, :] + modulo_steps), axis=-1)
    return np.min(length_array)

def fixed_pt_traj_powell(map, modulo):
    """
    Returns a function which takes in a point and kwargs for map and outputs the fixed point trajectory.

    Parameters:
    map: function
        map to be used.
    modulo: function
        modulo to be used.
    """
    def traj(xy, **kwargs):
        """
        Returns an array of shape (ijk) where i is of length 1, j indexes the x/y value, 
        and k indexes the iteration number (0 is the original point before any steps).    

        Parameters:
        xy: 1D array of length 2
            point to find the trajectory of.
        """
        iterations =[xy]
        def write_to_iter(xk):
            iterations.append(xk)

        difference_vector = modded_mapping_vector(map, modulo)
        length = lambda xy: np.linalg.norm(difference_vector(xy, **kwargs))
        grad_length = jacfwd(length)
        result = minimize(fun=length, x0=xy, method='powell', jac=grad_length,
                          # NOTE: HARDCODED STANDARD MAP MODULO
                        #   bounds=((0,1), (0,1)),
                          callback = write_to_iter,
                          options={'direc': onp.eye(2)*0.1, 'ftol': 1e-11},
                          tol = 1e-11)
        iter_array = np.array(iterations)
        return np.swapaxes(iter_array, 0,1)[None,:,:], result
    return traj

def fixed_point_finder(map, map_modulo, step, step_modulo, Niter):
    """
    Takes as input a map and its modulo, a step function, and a number of iterations 'Niter'.
    Outputs a function which takes in a starting point and outputs the point 
    which is the n-th step away from the starting point after Niter steps.
    """
    step_for_map = step(map, step_modulo)
    apply_step_for_map = apply_step(step_for_map, map_modulo)
    Nstep = Nmap(apply_step_for_map, Niter)
    def final(xy, **kwargs):
        return Nstep(xy, **kwargs)
    
    return jit(final)

def fixed_point_trajectory(xy, map, map_modulo, step, step_modulo, niter, **kwargs):
    """
    Iterates niter times to move the starting points toward fixed points.
    Returns an array of shape (ijk) where i indexes the point in starts, j indexes the x/y value, 
    and k indexes the iteration number (0 is the original point before any steps).    

    Parameters:
    xy: Nx2 array
        array of starting points.
    map: function
        map which will be used.
    map_modulo: function
        modulo linked to map.
    step: funtion
        step which will be used.
    step_modulo: function
        modulo linked to step.
    niter: int
        number of iterations to run the fixed point algorithm.
    """

    step_for_map = step(map, step_modulo)
    apply_step_for_map = apply_step(step_for_map, map_modulo)
    # use lambda to "roll-in" the mapping kwargs
    rolled_apply_step = lambda xy: apply_step_for_map(xy, **kwargs)
    # jit rolled_delta
    applydelta = jit(vmap(rolled_apply_step, in_axes=0))
    # initialize results array
    iterations = [xy, ]
    # calculate mapping of previous mappings niter times
    for _ in range(niter):
        old_point = iterations[-1]
        new_point = applydelta(old_point)
        iterations.append(new_point)
    # stack into a nice array for returning. 
    return np.stack(iterations, axis=-1)

def fixed_point_steps(xy, map, map_modulo, step, step_modulo, niter, **kwargs):
    """
    Takes as input an array of starting points, a map, and a number of iterations.
    Also takes in a modulo function which is based on the modulo of the input map.
    Iterates niter points to move the starting points toward fixed points.
    Returns an array of steps of shape (ijk) where i indexes the point in starts, 
    j indexes the iteration number (0 is the original point before any steps),
    and k indexes the x/y value.    
    """

    step_for_map = step(map, step_modulo)
    rolled_step = lambda xy: step_for_map(xy, **kwargs)
    vmapped_step = vmap(rolled_step)

    apply_step_for_map = apply_step(step_for_map, map_modulo)
    # use lambda to "roll-in" the mapping kwargs
    rolled_apply_step = lambda xy: apply_step_for_map(xy, **kwargs)
    # jit rolled_delta
    applydelta = jit(vmap(rolled_apply_step, in_axes=0))
    # initialize results array
    iterations = [xy, ]
    steps = []
    # calculate mapping of previous mappings niter times
    for _ in range(niter):
        old_point = iterations[-1]
        new_point = applydelta(old_point)
        iterations.append(new_point)

        point_step = vmapped_step(old_point)
        steps.append(point_step)
    # stack into a nice array for returning. 
    return np.swapaxes(np.stack(steps, axis=-1), 1, 2)

def mapping_vector(map, modulo):
    """
    Returns a function which calculates the difference between point xy and the modulo of the mapping of xy.
    modulo(map(xy)) - xy
    """
    def diff(start, **kwargs):
        end = modulo(map(start, **kwargs))
        vec = end - start
        return vec
    return jit(diff)

def modded_mapping_vector(map, map_modulo):
    """
    Outputs a function which returns the vector between a point and the closest modulo brother of its map.

    Parameters:
    map: function
        map that is used
    map_modulo: function
        modulo linked to the map
    """
    def modded_vector(start, **kwargs):
        end = map_modulo(map(start, **kwargs))
        # NOTE: HARDCODED FOR STANDARD MAP
        modulo_steps = modulo_add(1, 1)
        modulo_brothers_of_end = end[None, :] + modulo_steps
        length_array = np.linalg.norm(start[None, :] - modulo_brothers_of_end, axis=-1)
        min_position = np.argmin(length_array)
        return modulo_brothers_of_end[min_position] - start
    return jit(modded_vector)

def theta(map, modulo):
    """
    Takes as input a map.
    Outputs a function which accepts xy and kwargs as input. This function calculates the angle between map(xy)-xy and the horizontal.
    """
    mapping_vector_fun = modded_mapping_vector(map, modulo)
    def final(xy, **kwargs):
        x, y = mapping_vector_fun(xy, **kwargs)
        return np.arctan2(y, x)
    return jit(final)

def isotrope(map, modulo):
    """
    Takes a mapping_vector function as input.
    Outputs a function which calculates the isotrope of xy given a certain map.
    Isotrope: Direction in which mapping vector doesn't change
    """
    map_theta = theta(map, modulo)
    def step(xy, **kwargs):
        rolled_theta = lambda xy: map_theta(xy, **kwargs)
        dtheta = grad(rolled_theta, argnums=0)
        final = np.array([[0,1],[-1,0]]) @ np.nan_to_num(dtheta(xy))
        return final
    return jit(step)

def test_isotrope(map):
    """
    Takes a mapping_vector function as input.
    Outputs a function which calculates the isotrope of xy given a certain map.
    Isotrope: Direction in which mapping vector doesn't change
    Different from main isotrope in that small_isotrope is used for testing whether the isotrope concept works.
    """
    map_theta = theta(map)
    def step_TNM(xy, **kwargs):
        rolled_theta = lambda xy: map_theta(xy, **kwargs)
        dtheta = grad(rolled_theta, argnums=0)
        isotrope =  np.array([[0,1],[-1,0]]) @ dtheta(xy)
        norm = np.linalg.norm(isotrope)
        small_isotrope = 10**(-7)/norm * isotrope
        return small_isotrope
    return step_TNM

def theta_comparison(map):
    """
    Takes a map as input.
    Outputs a function which takes in an xy and finds the step in the isotropic direction. 
    It calculates theta between xy and xy+step and returns the difference.
    """
    direction = test_isotrope(map)
    test_theta = theta(map)
    def final(xy, **kwargs):
        old_theta = test_theta(xy, **kwargs)
        step = direction(xy, **kwargs)
        new_theta = test_theta(xy+step, **kwargs)
        return new_theta - old_theta
    return final

def find_unique_fixed_points(map, map_modulo, Niter):
    """
    Returns a function which returns an array containing the fixed points of the map.
    Array is of shape (ij) where i indexes the fixed points and j indexes the x/y-value.
    """
    def final(grid, step, step_modulo, **kwargs):
        # initialise fixed point finder function. higher niter to ensure accuracy.
        map_fixed_point_finder = fixed_point_finder(map, map_modulo, step, step_modulo, Niter=Niter)
        # roll in kwargs for map_fixed_point_finder.
        rolled_fixed_point_finder = lambda xy: map_fixed_point_finder(xy, **kwargs)
        # vmap map_fixed_point_finder.
        vmapped_fixed_point_finder = vmap(rolled_fixed_point_finder, in_axes=0)
        # write destinations of points in grid into fixed_points array.
        fixed_points = vmapped_fixed_point_finder(grid)
        
        # for i in range(grid.shape[0]):
        #     final = map_fixed_point_finder(grid[i,:], **kwargs)
        #     # deal with nan points
        #     if math.isnan(final[0]) == False and math.isnan(final[1]) == False:
        #         fixed_points = fixed_points.at[i,:].set(final)
        
        # round fixed points to 5 decimal places.
        rounded_fixed_points = np.round(fixed_points, 10)
        # vmap modulo and apply on rounded fixed points.
        vmapped_modulo = vmap(map_modulo, in_axes=0)
        modded_fixed_points = vmapped_modulo(rounded_fixed_points)
        # np.unique to get array of unique fixed points.
        unique_fixed_points = np.unique(modded_fixed_points, axis=0)
        
        return unique_fixed_points
    # NOTE: CANT JIT, NOT SURE WHY
    return final

def newton_fractal(xy_start, xy_end, x_points, y_points, map, modulo, step, niter, test_grid=grid_starting_points((-1,-1), (1,1), 10, 10), **kwargs):
    # use find_unique_fixed_points to find the fixed points
    map_unique_fixed_points = find_unique_fixed_points(map, modulo)
    unique_fixed_points = map_unique_fixed_points(test_grid, step, **kwargs)
    
    # initialise color map based on number of points in unique_fixed_points.
    # from matplotlib import colormaps
    # cmap = colormaps['gist_rainbow']
    # colours = cmap(np.linspace(0, 1, unique_fixed_points.shape[0]))
    # colours = [np.array([114,229,239]), np.array([9,123,53]), np.array([42,226,130]), np.array([236,77,216]), 
    #            np.array([157,187,230]), np.array([62,105,182]), np.array([149,200,88]), np.array([251,32,118]),
    #            np.array([52,245,14]), np.array([225,50,25]), np.array([8,132,144]), np.array([218,164,249]), 
    #            np.array([141,78,182]), np.array([250,209,57]), np.array([162,85,66]), np.array([233,191,152]),
    #            np.array([111,125,67]), np.array([251,137,155]), np.array([231,134,7]), np.array([140,46,252])]
    
    # initialise fixed_point_finder to use niter argument.
    map_fixed_point_finder = vmap(fixed_point_finder(map, modulo, step, niter), in_axes=0)
    # initialise grid of starting points.
    start_grid = grid_starting_points(xy_start, xy_end, x_points, y_points)
    # apply map_fixed_point_finder on start_grid.
    end_grid = map_fixed_point_finder(start_grid)
    # initialise output array.
    output_grid = np.empty((y_points, x_points, 1))
    # calculate x_stpe and y_step for ease of use.
    x_min, y_min = xy_start
    x_max, y_max = xy_end
    x_step = (x_max - x_min)/(x_points-1)
    y_step = (y_max - y_min)/(y_points-1)
    from scipy import spatial
    # iterate over coordinate axes of colour_grid
    for index, point in enumerate(start_grid): 
        end = end_grid[index]
        # calculate indices of point.
        i = int((point[0] - x_min)/x_step)
        j = int((point[1] - y_min)/y_step)
        # set distance from end to closed fixed point to be 2 (arbitrary). iterate over unique_fixed_points and calculate distance.
        min_d=2
        # if distance is less than all distances encountered before, rewrite index value in output_grid to the closest fixed point.  
        for k in range(unique_fixed_points.shape[0]):
            distance = spatial.distance.euclidean(end, unique_fixed_points[k])
            if distance < min_d:
                min_d = distance
                output_grid = output_grid.at[j, i].set(k)
    # return output_grid.
    return output_grid

def apply_finder_to_grid(map, map_modulo, step, step_modulo, startpoints, x_points, y_points, fixedpoints, Niter, **kwargs):
    """
    apply the step function to a grid of starting points to find the fixed points.

    Parameters:
    step: function
        step function to be applied to the points in the grid startpoints
        (generated from a map and modulo using step_NM f.ex.)
    startpoints: Nx2 array
        grid of starting points
    fixedpoints: Kx2 array
        fixed points that you have pre-computed
    Niter: int
        number of iterations to run the step function
    """
    start_points = startpoints.reshape(y_points,x_points, 2)
    # initialise fixed point finder to use Niter argument.
    map_fixed_point_finder = fixed_point_finder(map, map_modulo, step, step_modulo, Niter)
    # use lambda to roll in the kwargs
    rolled_fixed_point_finder = lambda xy: map_fixed_point_finder(xy, **kwargs)
    # vmap the fixed_point_finder function. 
    vmap_fixed_point_finder = vmap(vmap(rolled_fixed_point_finder))
    # apply vmap_fixed_point_finder to the grid of starting points.
    end_points = vmap_fixed_point_finder(start_points)
    # calculate the distance to the fixed points
    distances = np.linalg.norm(end_points[:,:,None,:] - fixedpoints[None, None, :, :], axis=-1)
    # find the index of the closest fixed point
    closest = np.argmin(distances, axis=-1).T
    return closest