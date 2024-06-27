from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section, step_NM
from methods import fixed_point_trajectory

from maps import standard_map_modulo as modulo

k=0.9

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=500)
initial_points = onp.random.random(2)

map3 = Nmap(standard_map, 3)

step = step_NM(map3)
points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = k)
steps = fixed_point_trajectory(initial_points, modulo=modulo, step=step, niter=1000, k=k)

fig, ax = plt.subplots(figsize=(12, 6))
#i = round(100*onp.random.rand())
colors = plt.cm.binary(np.linspace(0,1,points.shape[0]))
final_iter = points.shape[2]
for i in range(points.shape[0]):
    ax.scatter(points[i,0,:], points[i,1,:], 
               #color=onp.random.random(3), 
               color='gray',
               #color=colors[i],
               s=0.0001, marker ='.')

cmap1 = colormaps['gist_heat']
cmap2 = colormaps['PRGn']
cmap3 = colormaps['PiYG']
colors1 = cmap1(np.linspace(0.5, 1, steps.shape[1]))
colors2 = cmap2(np.linspace(0.25, 0.5, steps.shape[1]))
colors3 = cmap2(np.linspace(0.75, 0.5, steps.shape[1]))
colors4 = cmap3(np.linspace(0.25, 0.5, steps.shape[1]))
colors = [colors1, colors2, colors3, colors4]

for i in range(steps.shape[1]):
    ax.plot(steps[0,i:i+2], steps[1,i:i+2],
            color='black',
            ms=10, marker ='.', markerfacecolor=colors[0][i])

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'NM to find fixed points of 3-cycle. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()