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
from maps import original_standard_map, standard_map, Nmap, no_modulo
from maps import standard_map_theta_modulo as no_p_modulo
from maps import standard_map_modulo as modulo
from methods import grid_starting_points, linear_starting_points, step_NM, step_LGTNMx

from maps import standard_map_theta_modulo as no_p_mod


from convergence import map_difference_against_N, order_of_conv
k=0.5
sm5 = Nmap(standard_map, 5)
sm7 = Nmap(standard_map, 7)
sm11 = Nmap(standard_map, 11)
N=100

map1 = sm5
map2 = sm7
map3 = sm11

epsilon = 1E-12

xl = [1, 2, 3]
gap = 0.2

starts = np.load('starts1000.npy')

fig, ax = plt.subplots()

# colour for each method
dict_colours = {"NM": "lightskyblue",
                "LGTNMx": "lightgreen",
                "Powell": "mediumorchid",
                "SLSQP": "khaki"}

##################################
#### DATA FOR NETWON'S METHOD ####
##################################

# 2-map
NM_iterations = map_difference_against_N(points=starts,
                                      map=map1, map_modulo=modulo,
                                      step=step_NM, step_modulo=no_p_mod,
                                      max_N=N,
                                      k=k)
data_NM = order_of_conv(distances=NM_iterations, epsilon=epsilon)
NM_mean = np.mean(data_NM)
NM_std = np.std(data_NM)
ax.errorbar(xl[0] - gap, NM_mean, 
            yerr=NM_std, 
            ecolor=dict_colours["NM"], 
            fmt='o', markerfacecolor = dict_colours["NM"], markeredgewidth = 0)

# 3-map
NM_iterations = map_difference_against_N(points=starts,
                                      map=map2, map_modulo=modulo,
                                      step=step_NM, step_modulo=no_p_mod,
                                      max_N=N,
                                      k=k)
data_NM = order_of_conv(distances=NM_iterations, epsilon=epsilon)
NM_mean = np.mean(data_NM)
NM_std = np.std(data_NM)
ax.errorbar(xl[1] - gap, NM_mean, 
            yerr=NM_std, 
            ecolor=dict_colours["NM"], 
            fmt='o', markerfacecolor = dict_colours["NM"], markeredgewidth = 0)

#5-map
NM_iterations = map_difference_against_N(points=starts,
                                      map=map3, map_modulo=modulo,
                                      step=step_NM, step_modulo=no_p_mod,
                                      max_N=N,
                                      k=k)
data_NM = order_of_conv(distances=NM_iterations, epsilon=epsilon)
NM_mean = np.mean(data_NM)
NM_std = np.std(data_NM)
ax.errorbar(xl[2] - gap, NM_mean, 
            yerr=NM_std, 
            ecolor=dict_colours["NM"], 
            fmt='o', markerfacecolor = dict_colours["NM"], markeredgewidth = 0)
##################################
##################################


#########################
#### DATA FOR LGTNMx ####
#########################

# 2-map
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=map1, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[0] + gap, LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)

# 3-map
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=map2, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[1] + gap, LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)

# 5-map
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=map3, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[2] + gap, LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)
#########################
#########################


# generate legend
from matplotlib.patches import Patch
NM_patch = Patch(color=dict_colours["NM"], label="Newton's Method") 
LGTNMx_patch = Patch(color=dict_colours["LGTNMx"], label="LGTNMx")
# Powell_patch = Patch(color=dict_colours["Powell"], label="Powell's Method")
# SLSQP_patch = Patch(color=dict_colours["SLSQP"], label="SLSQP")   
ax.legend(handles=[NM_patch, LGTNMx_patch, 
                   #Powell_patch, 
                   #SLSQP_patch
                   ], loc='upper left')

ticks = ['5-map', '7-map', '11-map']

ax.set_xticks(xl, ticks)
ax.set_ylabel(r'$\gamma$')
# ax.set_ylim(-0.1, 0.1)
ax.set_yscale('log')

# plt.show()
plt.savefig('sm_k=0.5_ord_of_conv_alt_impt_higher.pdf', bbox_inches='tight', dpi=300)


print("NM:",  NM_mean, NM_std)
print("LGTNMx:",  LGTNMx_mean, LGTNMx_std)
# print("Powell:",  Powell_mean, Powell_std)
# print("SLSQP:",  SLSQP_mean, SLSQP_std)

# print(data_SLSQP)
