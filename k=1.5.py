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

k=0.5

starts = linear_starting_points(xy_start=(0.5,0), xy_end=(0.5,1), npoints=500)
initial_points = linear_starting_points((0.35,0.75), (0.35, 0.85), npoints=4)

map2 = Nmap(standard_map, 2)

points = calculate_poincare_section(starts, 10000, mapping=standard_map, k = k)
steps = fixed_point_trajectory(initial_points, map = map2, modulo=modulo, step=step_NM, niter=50, k=k)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.binary(np.linspace(0,1,points.shape[0]))
final_iter = points.shape[2]
for i in range(points.shape[0]):
    ax.scatter(points[i,0,:], points[i,1,:], 
               color='gray', s=0.0001, marker ='.')

cmap1 = colormaps['gist_heat']
cmap2 = colormaps['PRGn']
cmap3 = colormaps['PiYG']
colors1 = cmap1(np.linspace(0.5, 1, steps.shape[2]))
colors2 = cmap2(np.linspace(0.25, 0.5, steps.shape[2]))
colors3 = cmap2(np.linspace(0.75, 0.5, steps.shape[2]))
colors4 = cmap3(np.linspace(0.25, 0.5, steps.shape[2]))
colors = [colors1, colors2, colors3, colors4]

for j in range(steps.shape[0]):
    for i in range(steps.shape[2]):
        ax.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color='black',
                ms=10, marker ='.', markerfacecolor=colors[j][i])

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'NM to find fixed points of 2-cycle. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()