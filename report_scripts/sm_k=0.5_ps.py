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

from maps import standard_map_modulo as modulo

k=0.5

map6 = Nmap(standard_map, 6)


###########################
## PLOT POINCARE SECTION ##
###########################
line = linear_starting_points((0,0), (1,1), 500)
poincare = calculate_poincare_section(starts=line, niter=10000, map=standard_map, modulo=modulo, k=k)
for i in range(poincare.shape[0]):
    plt.scatter(poincare[i,0,:], poincare[i,1,:], 
                color='black', s=0.0001, marker ='.')
###########################
###########################

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.xlim([0,1])
plt.ylim([0,1])
# plt.show()
plt.savefig('sm1_k=0.5_ps.png', bbox_inches='tight', dpi=300)