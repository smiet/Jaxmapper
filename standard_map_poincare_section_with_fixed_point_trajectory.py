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
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM

k=0.5

###########################
## PLOT POINCARE SECTION ##
###########################
from methods import calculate_poincare_section
line = linear_starting_points(xy_start=(0,0), xy_end=(1,1), npoints=500)
poincare = calculate_poincare_section(starts=line, niter=10000, map=standard_map, modulo=modulo, k=k)
for i in range(poincare.shape[0]):
    plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                color='black', s=0.0001, marker ='.')
###########################
###########################


###################################
## PLOT FIXED POINT TRAJECTORIES ##
###################################
from methods import fixed_point_trajectory
starts = linear_starting_points((0.6,0.08), (0.9, 0.08), npoints=4)
steps = fixed_point_trajectory(xy=starts, 
                               map=standard_map, modulo=modulo,
                               step=step_NM, 
                               niter=50,
                               k=k)

cmap = colormaps['PiYG']
colors = cmap(np.linspace(0, 1, steps.shape[0]))

for j in range(steps.shape[0]): # for each fixed point
    for i in range(steps.shape[2]): # for each line segment
        plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color='blue',
                ms=10, marker ='.', markerfacecolor=colors[j], markeredgecolor='blue')
###################################
###################################

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'NM to find fixed points of 2-cycle. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()