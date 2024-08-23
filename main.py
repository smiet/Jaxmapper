from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as onp
from scipy import optimize

import math

from tests import run_test
from maps import standard_map, sym_standard_map, sym_jac_func, Nmap, basecase, no_modulo
from methods import grid_starting_points, linear_starting_points, calculate_poincare_section
from methods import step_NM, step_LGTNMx, apply_step, fixed_point_finder, fixed_point_trajectory, find_unique_fixed_points
from methods import theta, test_isotrope, isotrope, theta_comparison
from methods import newton_fractal

from maps import standard_map_modulo as modulo
from maps import standard_map_theta_modulo as no_p_mod

from convergence import map_difference_against_N, order_of_conv
k=0.5
sm2 = Nmap(standard_map, 2)
sm3 = Nmap(standard_map, 3)
sm5 = Nmap(standard_map, 5)
N=100

epsilon = 1E-14

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
                                      map=sm2, map_modulo=modulo,
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
                                      map=sm3, map_modulo=modulo,
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
                                      map=sm5, map_modulo=modulo,
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
                                      map=sm2, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[0], LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)

# 3-map
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=sm3, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[1], LGTNMx_mean, 
            yerr=LGTNMx_std, 
            ecolor=dict_colours["LGTNMx"], 
            fmt='o', markerfacecolor = dict_colours["LGTNMx"], markeredgewidth = 0)

# 5-map
LGTNMx_iterations = map_difference_against_N(points=starts,
                                      map=sm5, map_modulo=modulo,
                                      step=step_LGTNMx, step_modulo=modulo,
                                      max_N=N,
                                      k=k)
data_LGTNMx = order_of_conv(distances=LGTNMx_iterations, epsilon=epsilon)
LGTNMx_mean = np.mean(data_LGTNMx)
LGTNMx_std = np.std(data_LGTNMx)
ax.errorbar(xl[2], LGTNMx_mean, 
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
starts = starts[300:500,:]

# 2-map
Powell_steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=sm2, modulo=modulo,
                                     traj=traj_powell,
                                     k=k)
Powell_iterations = map_difference_against_N_scipy(iterations=Powell_steps,
                                            map=sm2, map_modulo=modulo, 
                                            k=k)  
data_Powell = order_of_conv(distances=Powell_iterations, epsilon=epsilon)
Powell_mean = np.mean(data_Powell)
Powell_std = np.std(data_Powell)
ax.errorbar(xl[0] + gap, Powell_mean, 
            yerr=Powell_std, 
            ecolor=dict_colours["Powell"], 
            fmt='o', markerfacecolor = dict_colours["Powell"], markeredgewidth = 0)

# 3-map
Powell_steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=sm3, modulo=modulo,
                                     traj=traj_powell,
                                     k=k)
Powell_iterations = map_difference_against_N_scipy(iterations=Powell_steps,
                                            map=sm3, map_modulo=modulo, 
                                            k=k)  
data_Powell = order_of_conv(distances=Powell_iterations, epsilon=epsilon)
Powell_mean = np.mean(data_Powell)
Powell_std = np.std(data_Powell)
ax.errorbar(xl[1] + gap, Powell_mean, 
            yerr=Powell_std, 
            ecolor=dict_colours["Powell"], 
            fmt='o', markerfacecolor = dict_colours["Powell"], markeredgewidth = 0)

# 5-map
Powell_steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=sm5, modulo=modulo,
                                     traj=traj_powell,
                                     k=k)
Powell_iterations = map_difference_against_N_scipy(iterations=Powell_steps,
                                            map=sm5, map_modulo=modulo, 
                                            k=k)  
data_Powell = order_of_conv(distances=Powell_iterations, epsilon=epsilon)
Powell_mean = np.mean(data_Powell)
Powell_std = np.std(data_Powell)
ax.errorbar(xl[2] + gap, Powell_mean, 
            yerr=Powell_std, 
            ecolor=dict_colours["Powell"], 
            fmt='o', markerfacecolor = dict_colours["Powell"], markeredgewidth = 0)
###################################
###################################


# generate legend
from matplotlib.patches import Patch
NM_patch = Patch(color=dict_colours["NM"], label="Newton's Method") 
LGTNMx_patch = Patch(color=dict_colours["LGTNMx"], label="LGTNMx")
Powell_patch = Patch(color=dict_colours["Powell"], label="Powell's Method")
# SLSQP_patch = Patch(color=dict_colours["SLSQP"], label="SLSQP")   
ax.legend(handles=[NM_patch, LGTNMx_patch, Powell_patch, 
                   #SLSQP_patch
                   ], loc='upper left')

ticks = ['2-map', '3-map', '5-map']

ax.set_xticks(xl, ticks)
ax.set_ylabel(r'$\gamma$')

plt.show()
# plt.savefig('sm2_k=0.5_ord_of_conv.pdf', bbox_inches='tight', dpi=300)


print("NM:",  NM_mean, NM_std)
print("LGTNMx:",  LGTNMx_mean, LGTNMx_std)
print("Powell:",  Powell_mean, Powell_std)
# print("SLSQP:",  SLSQP_mean, SLSQP_std)

# print(data_SLSQP)




# from methods import fixed_point_trajectory_scipy
# test4 = starts[300:301,:]
# test5 = fixed_point_trajectory_scipy(test4, sm2, modulo, traj_slsqp, k=k)
# print(test5)

#TODO: use colours for the fixed points in a neater way
#TODO: compare different number of iterations
#TODO: check if the points all make it to the fixed points

#TODO: deliverables 
#TODO: TODO analytic expressions from sympy
#TODO: TODO numerical outputs from sympy_jac and jacfwd for 2, 3, 4maps
#TODO: TODO numerical outputs from step
#DONE: split up map.py
#DONE: make sympy part of step work properly and do comparison
#DONE: convert step into a functional transformation
#DONE: make a folder called tests with a few simple tests (sanity checks. eg f(x + dx) = f(x) + Mdx, make dx random)
#DONE: switch to imshow
#DONE: add simple case (z^3 - 1) for baseline
#DONE: plot fixed points on the same diagram
