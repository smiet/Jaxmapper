from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

from maps import Nmap, standard_nontwist, sym_jac_func

def calculate_poincare_section(starts, niter, mapping=standard_nontwist, **kwargs):
    """
    Calculate a Poincare section of a mapping with parameters a and b.
    Iterate niter times.
    """
    # use lambda to "roll-in" the mapping kwargs
    rolled_map = lambda xy: mapping(xy, **kwargs)
    # use vmap to create a function from all starts to all mapped points
    applymap = jit(vmap(rolled_map, in_axes=0))
    # initialize results array
    iterations = [starts, ]
    # calculate mapping of previous mappings niter times
    for _ in range(niter):
        iterations.append(applymap(iterations[-1]))
    # stack into a nice array for returning. 
    # array has shape (ijk) where i indexes the point in starts,
    # j indexes the x/y value,
    # k indexes the iteration number (0 is the original point before any mappings)
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

def new_starts(xy_start = tuple, xy_end = tuple, x_points = int, y_points = int):
    """
    Creates a grid of starting points determined by given points which are taken as corners of the grid.
    """
    x_start = xy_start[0]
    y_start = xy_start[1]
    x_end = xy_end[0]
    y_end = xy_end[1]
    # create arrays for x and y points
    x = np.linspace(x_start, x_end, x_points)
    y = np.linspace(y_start, y_end, y_points)
    # create an array with shape (x_points*y_points,2)
    starts = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    return starts

def linear_starting_points(xy_start = tuple, xy_end = tuple, npoints = int):
    """
    Creates a bunch of starting points along a line determined by a given start and end point.
    """
    x_start, y_start = xy_start
    x_end, y_end = xy_end
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

def step_NM(xy, map, method='jax', **kwargs): #TODO rewrite as functional transformation (takes only xy and kwargs as input)
    """
    Takes as input a function map. 
    Then from fN calculates that matrix and takes a Newton step.
    Ensure that the method chosen is appropriate for the map (i.e. if method=sympy then map must be a Sympy map)!
    """
    if method=='jax':
        # use lambda to "roll-in" the mapping kwargs
        rolled_map = lambda xy: map(xy, **kwargs)
        M = jacfwd(rolled_map)(xy)
        A = M - np.eye(2)
        f = rolled_map(xy)
        b = xy-f
        delta = np.linalg.solve(A,b)
        return delta
    elif method=='sympy':
        x, y = sym.symbols('x y')
        sym_expr = map(**kwargs)
        M = sym_jac_func(sym_expr)(xy)
        A = M - np.eye(2)
        f = sym.lambdify([x,y], sym_expr, 'numpy')(xy[0], xy[1])
        # the f above is a (ij) array where i is the x/y-value and j is a redudant axis of length 1.
        # so we flatten f.
        f = f.flatten()
        b = xy-f
        delta = np.linalg.solve(A,b)
        return delta
    else:
        print("Invalid method!")

def mapping_vector(map): #TOCHECK
    """
    Returns a function which calculates the difference between point xy and the mapping of xy.
    """
    def diff(start, **kwargs):
        end = map(start, **kwargs)
        vec = np.subtract(end,start)
        return vec
    return jit(diff)

def isotrope(xy, mapping_vector_fun): #TODO
    """
    calculate the isotrope

    re-write so this monster is readable and understandable
    UNTESTED
    """
    return jacfwd(mapping_vector_fun)*(np.array([[0,1],[-1,0]]))
