from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp
from scipy import optimize

import math

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, basecase, no_modulo
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, step_AGTNMx, apply_step, fixed_point_finder, fixed_point_trajectory, find_unique_fixed_points
from methods import modded_length
from methods import mapping_vector

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod


def dmin_against_N(points, map, map_modulo, step, step_modulo, max_N, fixed_points, **kwargs):
    """
    Outputs a Q x max_N array where the first axis indexes the point in points 
    and the second axis indexes the iteration value.

    Parameters:
    points: Mx2 array
        Points to perform function on. Ideally random.
    fixed_points: Kx2 array
        Array of fixed points. This should be the set of expanded fixed points (so including all modulo brothers).
    """
    # generate array of shape (ijk) using fixed_point_trajectory where i indexes the point in points, j indexes the x/y value, and k indexes the iteration number (0 is the original point before any steps).    
    iterations = fixed_point_trajectory(xy=points, 
                                   map=map, map_modulo=map_modulo,
                                   step=step, step_modulo=step_modulo,
                                   niter=max_N, 
                                   **kwargs)
    # swap axes to make j the last axis.
    iterations = np.swapaxes(iterations, 1, 2)
    # calculate distances from points to all fixed points in 1 step.
    distances = np.linalg.norm(iterations[:,:,None,:] - fixed_points[None,None,:,:], axis=-1)
    # generate MxN array where first axis indexes point in points, second axis indexes iteration number. each element represents the min distance to a fixed point.
    dmin = np.min(distances, axis=-1)
    return dmin

def map_difference_against_N(points, map, map_modulo, step, step_modulo, max_N, **kwargs):
    """
    Outputs a Q x max_N array where the first axis indexes the point in points 
    and the second axis indexes the iteration value.

    Parameters:
    points: Mx2 array
        Points to perform function on. Ideally random.
    fixed_points: Kx2 array
        Array of fixed points. This should be the set of expanded fixed points (so including all modulo brothers).
    """
    # generate array of shape (ijk) using fixed_point_trajectory where i indexes the point in points, j indexes the x/y value, and k indexes the iteration number (0 is the original point before any steps).    
    iterations = fixed_point_trajectory(xy=points, 
                                   map=map, map_modulo=map_modulo,
                                   step=step, step_modulo=step_modulo,
                                   niter=max_N, 
                                   **kwargs)
    # swap axes to make j the last axis.
    iterations = np.swapaxes(iterations, 1, 2)
    
    # NOTE: HARDCODED XMOD AND YMOD
    # consider the distance to all modulo brothers of its map and take the smallest one.
    rolled_modded_length = lambda xy: modded_length(xy, map_modulo(map(xy, **kwargs)), 1, 1)
    vmapped_modded_length = vmap(vmap(rolled_modded_length))
    # calculate distances from points to their maps in 1 step.
    distances = vmapped_modded_length(iterations)

    # map_mapping_vector = mapping_vector(map=map, modulo=map_modulo)
    # rolled_mapping_vector = lambda xy: map_mapping_vector(xy, **kwargs)
    # vmapped_mapping_vector = vmap(vmap(rolled_mapping_vector))
    # # calculate distances from points to their maps in 1 step.
    # distances = np.linalg.norm(vmapped_mapping_vector(iterations), axis=-1)
    
    return distances

def N_against_epsilon(iterations, epsilon_list):
    """
    Outputs an array of size MxK where first axis indexes the point 
    and the second axis indexes the epsilon value.
    Each element is the smallest N for dmin/map_difference to be smaller than epsilon for that epsilon and that point. 

    Parameters:
    iterations: MxN array
        array of distances from dmin_against_N or map_difference_against_N
    epsilon list: 1D array of length K
        array of epsilons. use np.logspace to generate.
    """    
    # Create a 3D array by expanding iterations to shape (M, N, 1) and repeating it K times along the third dimension
    expanded_iterations = np.expand_dims(iterations, axis=-1)  # Shape: (M, N, 1)
    expanded_iterations = np.repeat(expanded_iterations, len(epsilon_list), axis=-1)  # Shape: (M, N, K)

    # Compare each element of the 3D array with the corresponding epsilon value
    # This gives us a boolean array of shape (M, N, K)
    comparison = expanded_iterations < epsilon_list

    # Find the index of the first occurrence where the comparison is True along the N axis
    # np.argmax returns the index of the first occurrence of the maximum value (True here)
    # np.argmax will stop at the first True value, which is what we want
    final = np.argmax(comparison, axis=1)

    # If no True value is found for a specific epsilon, np.argmax will return 0
    # This is not what we want, so we need to handle such cases by setting them to N (as the smallest index not found)
    # Create a mask where we need to replace 0 with N
    mask = ~comparison.any(axis=1)
    final = final.at[mask].set(iterations.shape[1])
    return final

def order_of_conv(distances, epsilon):
    """
    Returns a 1D array of length M. each point is the ratio of distances for each point just before distance goes beow epsilon.

    Parameters:
    distances: MxN array
        array of distances from dmin_against_N or map_difference_against_N.
    epsilon: float
        error tolerance.
    """
    # generate boolean array where True in boolean_arr indicates that the corresponding element in distances is less than epsilon.
    boolean_arr = distances < epsilon
    # generate mask where element in mask is true if corresponding row in boolean_arr contains any True values.
    mask = boolean_arr.any(axis=1)
    # use mask to discard rows that do not have distances less than epsilon.
    distances = distances[mask, :]
    # regenerate boolean array.
    boolean_arr = distances < epsilon
    # generate index array. each element of index_array is the index in the corresponding row of distances where the distance drops below epsilon. 
    index_array = np.argmax(boolean_arr, axis=-1)

    # generate array of distances 1 step before the distance goes below epsilon.
    prev_point_array = distances[np.arange(distances.shape[0]), index_array-1]
    # generate array of distances 2 steps before the distance goes below epsilon.
    prev_prev_point_array = distances[np.arange(distances.shape[0]), index_array-2]

    return prev_point_array/prev_prev_point_array