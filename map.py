from jax import numpy as np
from jax import jit, grad, jacfwd, vmap
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

def calculate_poincare_section(starts, niter, mapping=standard_nontwist, **kwargs):
    """
    Calculate a Poincar\'e section of a mapping with parameters a and b.
    Iterate niter times.
    """
    fastmap = lambda xx: mapping(xx, **kwargs)
    applymap = jit(vmap(fastmap, in_axes=0))
    iterations = [starts, ]
    for _ in range(niter):
        iterations.append(applymap(iterations[-1]))
    return np.stack(iterations, axis=-1)

def plot_save_points(points, name='fig', colors='random'):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    if colors== 'random':
        colors = onp.random.random((points.shape[0], 3))
    for i in range(points.shape[0]):
        ax.scatter(points[i,0, :], points[i,1,:], color=colors[i,:], s=0.1, marker='.')

    plt.xlim([0,1])
    plt.ylim([-1.5,1.5])
    fig.savefig(name, bbox_inches='tight', dpi=300)

def starts_nontwist(norbits_per_leg):
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



starts = starts_nontwist(200)
points = calculate_poincare_section(starts, 10000, mapping=tokamap, K=0.01, w=1., w0 = 0.99, w1 = 0.9)


fig, ax = plt.subplots(figsize=(12, 6))
for i in range(points.shape[0]):
    ax.scatter(points[i,0, :], points[i,1,:], color=onp.random.random(3), s=0.1, marker ='.')
plt.xlim([0,1])
plt.ylim([-1.2,1.2])
plt.show()

a_scan = np.linspace(0.499, 0.52, 200)
colors = onp.random.random((points.shape[0], 3))
for num, a in enumerate(a_scan):
    points = calculate_poincare_section(a, 0.31, starts, 5000)
    plot_save_points(points, name='./dipolescan/{:04}scan_a_{}_b_0.31.png'.format(num, a), colors=colors)






    fastmap = lambda xx: standard_nontwist(xx, **kwargs)
    applymap = jit(vmap(fastmap, in_axes=0))
