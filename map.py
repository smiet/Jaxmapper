from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

@jit
def standard_nontwist(xx, a=0.51, b=0.31):
    """
    standard nontwist map.
    """
    y = xx[1] - b*np.sin(2*np.pi*xx[0])
    x = np.mod(xx[0] + a*(1 - y**2), 1)
    return np.array([x,y])


@jit
def tokamap(xx, K=0, w=0.666, w0=0.7, w1=0.3):
    """
    standard nontwist map.
    """
    psi_old = xx[1]
    theta_old = xx[0]
    P = psi_old - 1 - (K/(2*np.pi))*np.sin(2*np.pi*theta_old)
    a = (w-w0)/w
    c = 1 + ((w-w1)/(w-w0))**2
    psi = 0.5*(P + np.sqrt(P**2 + 4* psi_old))
    W = w*(1-a*(c*psi-1)**2)
    theta = np.mod(theta_old + W - (K/((2*np.pi)**2) * (1 + psi)**2)*np.cos(2*np.pi*theta_old), 1)
    #theta = np.mod(xx[0] + a*(1 - y**2), 1)
    return np.array([theta,psi])

@jit
def standard_map(xx, k = 0.1):
    """
    chirikov standard map
    """
    theta_old = xx[0]
    p_old = xx[1]
    # applying standard map on old coordinates to get new coordinates
    p = np.mod(p_old + (k/(2*np.pi))*np.sin(2*np.pi*theta_old), 1)
    theta = np.mod(theta_old + p + 0.5, 1)
    # return
    return np.array([theta, p])

def Nmap(map, N):
    def nmap(x, **kwargs):
        for _ in range(N):
            x = map(x, **kwargs)
        return x
    return jit(nmap)

def calculate_poincare_section(starts, niter, mapping=standard_nontwist, **kwargs):
    """
    Calculate a Poincare section of a mapping with parameters a and b.
    Iterate niter times.
    """
    # use lambda to "roll-in" the mapping kwargs
    fastmap = lambda xx: mapping(xx, **kwargs)
    # use vmap to create a function from all starts to all mapped points
    applymap = jit(vmap(fastmap, in_axes=0))
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

def step(Nmap,xx): #TODO
    """
    Takes as input a function Nmap (that is the map applied N times). 
    Then from fN calculates that matrix and takes a Newton step.
    """
    pass

def mapping_vector(starts, mapping, **kwargs): #TODO
    """
    returns the difference between point xx and the mapping of xx.
    """
    # use lambda to "roll-in" the mapping kwargs
    fastmap = lambda xx: mapping(xx, **kwargs)
    # use vmap to create a function from all starts to all mapped points
    applymap = jit(vmap(fastmap, in_axes=0))
    # calculate mapping of start points
    ends = applymap(starts)
    # calculate difference between starts and ends
    diff = np.subtract(ends, starts)
    # returns array with shape (ij) where i indexes the point in starts and j indexes the x/y value.
    # i is of the same length as 0th axis of starts.
    return diff

def isotrope(xx, mapping_vector_fun): #TODO
    """
    calculate the isotrope

    re-write so this monster is readable and understandable
    UNTESTED
    """
    return jacfwd(mapping_vector_fun)*(np.array([[0,1],[-1,0]]))



#remove script and put in different file
#starts = starts_nontwist(200)
#starts = new_starts(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
starts = np.array([0.2, 0.2])

#points = calculate_poincare_section(starts, 10000, mapping=standard_nontwist, b=0.2, a=0.53)
# points = calculate_poincare_section(starts, 100, mapping=standard_map, k = 0.5)

#plot_save_points(points, name='fig', colors='random')

# fig, ax = plt.subplots(figsize=(12, 6))
# for i in range(points.shape[0]):
#     ax.scatter(points[i,0, :], points[i,1,:], color=onp.random.random(3), s=10, marker ='.')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.tight_layout
# plt.show()

