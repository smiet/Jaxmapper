from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 15})

import numpy as onp

from tests import run_test
from maps import standard_map, Nmap, original_standard_map
from methods import grid_starting_points, linear_starting_points
from methods import step_NM, calculate_poincare_section

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

k=0.5

points = grid_starting_points((0,0), (1,1), 20, 20)
sm2 = Nmap(standard_map, 2)


#################################################################################################################
############################################# PLOT POINCARE SECTION #############################################
#################################################################################################################
from methods import calculate_poincare_section
line = linear_starting_points(xy_start=(0,0), xy_end=(1,1), npoints=100)
poincare = calculate_poincare_section(starts=line, niter=10000, map=standard_map, modulo=modulo, k=k)
for i in range(poincare.shape[0]):
    plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                color='lightgray', s=0.0001, marker ='.')
#################################################################################################################
#################################################################################################################

###################################################################################################################
################################################ PLOT FIXED POINTS ################################################
###################################################################################################################
# load fixed points from groundtruth.py.
from groundtruth import sm1_fixed_points
from plotting import expand_fixed_points
unique_fixed_points_array = sm1_fixed_points
expanded_fixed_points, colour_array = expand_fixed_points(fixed_points=unique_fixed_points_array,
                                                          x_min=0, x_max=1,
                                                          y_min=0, y_max=1)
colour_list = (colour_array/255).tolist()
plt.scatter(expanded_fixed_points[:, 0], expanded_fixed_points[:, 1], facecolors=colour_list, marker='o', 
            edgecolor='black', s=40, linewidth = 2)
###################################################################################################################
###################################################################################################################

from methods import isotrope
map_isotrope = isotrope(standard_map, modulo)
rolled_isotrope = lambda xy: map_isotrope(xy, k=k)
vmapped_isotrope = vmap(rolled_isotrope)

vmapped_modulo = vmap(modulo)

test = vmapped_isotrope(points)

test2 = np.linalg.norm(test, axis=-1)

test3 = -test/test2[:, None]

plt.quiver(points[:, 0], points[:, 1], test3[:, 0], test3[:, 1], facecolor='black')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.xlim([0,1])
plt.ylim([0,1])

# plt.show()
plt.savefig('sm1    _k=0.5_iso_field.png', bbox_inches='tight', dpi=300)