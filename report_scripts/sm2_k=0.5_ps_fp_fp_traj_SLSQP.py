from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import numpy as onp
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})


from tests import run_test
from maps import standard_map, Nmap, original_standard_map
from methods import grid_starting_points, linear_starting_points
from methods import step_NM, calculate_poincare_section
from plotting import expand_fixed_points

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

k=0.5

map2 = Nmap(standard_map, 2)


#################################################################################################################
############################################# PLOT POINCARE SECTION #############################################
#################################################################################################################
from methods import calculate_poincare_section
line = linear_starting_points(xy_start=(0,0), xy_end=(1,1), npoints=500)
poincare = calculate_poincare_section(starts=line, niter=10000, map=standard_map, modulo=modulo, k=k)
for i in range(poincare.shape[0]):
    plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                color='black', s=0.0001, marker ='.')
#################################################################################################################
#################################################################################################################


###################################################################################################################
################################################ PLOT FIXED POINTS ################################################
###################################################################################################################
# load fixed points from groundtruth.py.
from groundtruth import sm2_fixed_points
unique_fixed_points_array = sm2_fixed_points
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
colour_list = (colour_array/255).tolist()
plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
            edgecolor='black', s=70, linewidth = 2)
###################################################################################################################
###################################################################################################################

###################################################################################################################
########################################## PLOT FIXED POINT TRAJECTORIES ##########################################
###################################################################################################################
from methods import traj_slsqp, fixed_point_trajectory_scipy
from groundtruth import starts50
starts = starts50[0:10, :]
steps = fixed_point_trajectory_scipy(xy=starts, 
                                     map=map2, modulo=modulo, 
                                     traj=traj_slsqp, 
                                     k=k)

cmap = colormaps['gist_rainbow']
colors = cmap(np.linspace(0, 1, steps.shape[0]))

for j in range(steps.shape[0]): # for each fixed point
    for i in range(steps.shape[2]): # for each line segment
        plt.plot(steps[j, 0, i:i+2], steps[j, 1, i:i+2],
                color=colors[j],
                ms=5, marker ='o', markerfacecolor=colors[j], markeredgecolor=colors[j])
    plt.scatter(steps[j,0,0], steps[j,1,0], s=100, marker='.', color=colors[j])
    plt.scatter(steps[j,0,-1], steps[j,1,-1], s=100, marker='x', color='orangered')
###################################################################################################################
###################################################################################################################

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.xlim([0,1])
plt.ylim([0,1])
# plt.show()
plt.savefig('sm2_k=0.5_ps_fp_fp_traj_SLSQP.png', bbox_inches='tight', dpi=300)