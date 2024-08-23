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
N=100
map = sm5

epsilon = 1E-14

fig, ax = plt.subplots()

# colour for each method
dict_colours = {"NM": "lightskyblue",
                "LGTNMx": "lightgreen",
                "Powell": "mediumorchid",
                "SLSQP": "khaki"}

##################################
#### DATA FOR NETWON'S METHOD ####
##################################
starts = np.load('starts1000.npy')
NM_iterations = map_difference_against_N(points=starts,
                                      map=map, map_modulo=modulo,
                                      step=step_NM, step_modulo=no_p_mod,
                                      max_N=N,
                                      k=k)

data_NM = order_of_conv(distances=NM_iterations, epsilon=epsilon)

NM_mean = np.mean(data_NM)
NM_std = np.std(data_NM)

ax.errorbar(1, NM_mean, 
            yerr=NM_std, 
            ecolor=dict_colours["NM"], 
            fmt='o', markerfacecolor = dict_colours["NM"], markeredgewidth = 0)
##################################
##################################


#########################
#### DATA FOR LGTNMx ####
#########################
starts = np.load('starts1000.npy')
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=map, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)

data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)

LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)

ax.errorbar(1.5, LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)
#########################
#########################


###################################
#### DATA FOR POWELLS'S METHOD ####
###################################
from methods import fixed_point_trajectory_scipy, traj_powell
from convergence import map_difference_against_N_scipy
starts = np.load('starts1000.npy')
starts = starts[300:500,:]
Powell_steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=map, modulo=modulo,
                                     traj=traj_powell,
                                     k=k)

Powell_iterations = map_difference_against_N_scipy(iterations=Powell_steps,
                                            map=map, map_modulo=modulo, 
                                            k=k)  

data_Powell = order_of_conv(distances=Powell_iterations, epsilon=epsilon)

Powell_mean = np.mean(data_Powell)
Powell_std = np.std(data_Powell)

ax.errorbar(2, Powell_mean, 
            yerr=Powell_std, 
            ecolor=dict_colours["Powell"], 
            fmt='o', markerfacecolor = dict_colours["Powell"], markeredgewidth = 0)
###################################
###################################


########################
#### DATA FOR SLSQP ####
########################
# from methods import fixed_point_trajectory_scipy, traj_slsqp
# from convergence import map_difference_against_N_scipy
# starts = np.load('starts1000.npy')
# starts = starts[300:500,:]
# SLSQP_steps = fixed_point_trajectory_scipy(xy=starts,
#                                      map=map, modulo=modulo,
#                                      traj=traj_slsqp,
#                                      k=k)

# SLSQP_iterations = map_difference_against_N_scipy(iterations=SLSQP_steps,
#                                             map=map, map_modulo=modulo, 
#                                             k=k)  

# data_SLSQP = order_of_conv(distances=SLSQP_iterations, epsilon=epsilon)
# data_SLSQP = data_SLSQP[np.isinf(data_SLSQP) == False]

# SLSQP_mean = np.mean(data_SLSQP)
# SLSQP_std = np.std(data_SLSQP)

# ax.errorbar(4, SLSQP_mean, 
#             yerr=SLSQP_std, 
#             ecolor=dict_colours["SLSQP"], 
#             fmt='o', markerfacecolor = dict_colours["SLSQP"], markeredgewidth = 0)
########################
########################


# generate legend
from matplotlib.patches import Patch
NM_patch = Patch(color=dict_colours["NM"], label="Newton's Method") 
LGTNMx_patch = Patch(color=dict_colours["LGTNMx"], label="LGTNMx")
Powell_patch = Patch(color=dict_colours["Powell"], label="Powell's Method")
# SLSQP_patch = Patch(color=dict_colours["SLSQP"], label="SLSQP")   
ax.legend(handles=[NM_patch, LGTNMx_patch, Powell_patch, 
                   #SLSQP_patch
                   ], loc='upper left')

ax.set_xticks([])
ax.set_ylabel(r'$\gamma$')

# plt.show()
plt.savefig('sm5_k=0.5_ord_of_conv.pdf', bbox_inches='tight', dpi=300)


print("NM:",  NM_mean, NM_std)
print("LGTNMx:",  LGTNMx_mean, LGTNMx_std)
print("Powell:",  Powell_mean, Powell_std)
# print("SLSQP:",  SLSQP_mean, SLSQP_std)

# print(data_SLSQP)
