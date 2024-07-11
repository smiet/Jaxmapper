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
from methods import theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

def dmin_against_N(points, map, map_modulo, step, step_modulo, max_N, fixed_points, **kwargs):
    """
    Outputs a Qx2 array where the first axis indexes the point on the metric against N scatter plot 
    and the second axis indexes the N/metric-value.

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