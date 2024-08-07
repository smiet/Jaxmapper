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
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, no_modulo
from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod
from maps import original_standard_map as sm
from methods import grid_starting_points, linear_starting_points, step_NM, step_LGTNMx,step_LGTNMo

k=0.5

map2 = Nmap(standard_map, 2)

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


#########################################################################################################
##################################### PLOT FIXED POINT TRAJECTORIES #####################################
#########################################################################################################
from methods import fixed_point_trajectory
from groundtruth import starts50 as starts
starts = np.array([[1.45426592e-09, 4.99999996e-01]])
steps = fixed_point_trajectory(xy=starts, 
                               map=map2, map_modulo=modulo,
                               step=step_LGTNMx, step_modulo=modulo,
                               niter=80,
                               k=k)

cmap = colormaps['gist_rainbow']
colors = cmap(np.linspace(0, 1, steps.shape[0]))

for j in range(steps.shape[0]): # for each point
    for i in range(steps.shape[2]): # for each line segment
        plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color='blue',
                ms=5, marker ='o', markerfacecolor=colors[j], markeredgecolor=colors[j])
    plt.scatter(steps[j,0,0], steps[j,1,0], s=100, marker='.', color=colors[j])
    plt.scatter(steps[j,0,-1], steps[j,1,-1], s=100, marker='x', color='red')
#########################################################################################################
#########################################################################################################

plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title(f'LGTNMx to find fixed points of 2-Map. k={k}.')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout
plt.show()
# plt.savefig('sm2_ps_fptraj_TNM.pdf', bbox_inches='tight')