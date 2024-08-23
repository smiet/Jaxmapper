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


from convergence import map_difference_against_N, N_against_epsilon

k=0.5
sm2 = Nmap(standard_map, 2)
N=100
map = sm2

e_list = np.logspace(-14, -1, 14)

##################################
#### DATA FOR NETWON'S METHOD ####
##################################
starts = np.load('starts1000.npy')
NM_iterations = map_difference_against_N(points=starts,
                                      map=map, map_modulo=modulo,
                                      step=step_NM, step_modulo=no_p_mod,
                                      max_N=N,
                                      k=k)

data_NM = N_against_epsilon(iterations=NM_iterations, epsilon_list=e_list)
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

data_LGTNMx = N_against_epsilon(iterations=LGTNMx_iterations, epsilon_list=e_list)
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

data_Powell = N_against_epsilon(iterations=Powell_iterations, epsilon_list=e_list)  
###################################
###################################


########################
#### DATA FOR SLSQP ####
########################
from methods import fixed_point_trajectory_scipy, traj_slsqp
from convergence import map_difference_against_N_scipy
starts = np.load('starts1000.npy')
starts = starts[300:500,:]
SLSQP_steps = fixed_point_trajectory_scipy(xy=starts,
                                     map=map, modulo=modulo,
                                     traj=traj_slsqp,
                                     k=k)

SLSQP_iterations = map_difference_against_N_scipy(iterations=SLSQP_steps,
                                            map=map, map_modulo=modulo, 
                                            k=k)  

data_SLSQP = N_against_epsilon(iterations=SLSQP_iterations, epsilon_list=e_list)  
########################
########################

# colour for each method
dict_colours = {"NM": "lightskyblue",
                "LGTNMx": "lightgreen",
                "Powell": "mediumorchid",
                "SLSQP": "khaki"}

# initialise plot
fig, ax = plt.subplots(figsize = (12,6))

# generate box plots
ax.boxplot(data_NM, positions = np.array(np.arange(len(e_list)))*7.0-1.5, widths = 0.6,
           patch_artist = True, 
           boxprops = dict(facecolor = dict_colours["NM"]),
           flierprops = dict(marker = 'o', markerfacecolor = dict_colours["NM"], markersize = 2),
           medianprops = dict(linestyle = '-', linewidth = 2, color = 'firebrick'))

ax.boxplot(data_LGTNMx, positions = np.array(np.arange(len(e_list)))*7.0-0.5, widths = 0.6,
           patch_artist = True, 
           boxprops = dict(facecolor = dict_colours["LGTNMx"]),
           flierprops = dict(marker = 'o', markerfacecolor = dict_colours["LGTNMx"], markersize = 2),
           medianprops = dict(linestyle = '-', linewidth = 2, color = 'firebrick'))

ax.boxplot(data_Powell, positions = np.array(np.arange(len(e_list)))*7.0+0.5, widths = 0.6,
           patch_artist = True, 
           boxprops = dict(facecolor = dict_colours["Powell"]),
           flierprops = dict(marker = 'o', markerfacecolor = dict_colours["Powell"], markersize = 2),
           medianprops = dict(linestyle = '-', linewidth = 2, color = 'firebrick'))

ax.boxplot(data_SLSQP, positions = np.array(np.arange(len(e_list)))*7.0+1.5, widths = 0.6,
           patch_artist = True, 
           boxprops = dict(facecolor = dict_colours["SLSQP"]),
           flierprops = dict(marker = 'o', markerfacecolor = dict_colours["SLSQP"], markersize = 2),
           medianprops = dict(linestyle = '-', linewidth = 2, color = 'firebrick'))

# generate legend
from matplotlib.patches import Patch
NM_patch = Patch(color=dict_colours["NM"], label="Newton's Method") 
LGTNMx_patch = Patch(color=dict_colours["LGTNMx"], label="LGTNMx")
Powell_patch = Patch(color=dict_colours["Powell"], label="Powell's Method")
SLSQP_patch = Patch(color=dict_colours["SLSQP"], label="SLSQP")   
ax.legend(handles=[NM_patch, LGTNMx_patch, Powell_patch, SLSQP_patch])

# set ticks 
ticks = np.int64(np.round(np.log10(e_list), 0))
ax.set_xticks(np.arange(0, len(e_list) * 7, 7), ticks)

# ax.set_yscale('symlog')
ax.set_ylim(bottom=-0.1)
ax.set_ylim(top=20)

ax.set_xlabel(r'$\log_{10}(\epsilon)$')
ax.set_ylabel(r'$N$')

plt.show()
# plt.savefig('sm2_k=0.5_N_against_ep_map_diff.pdf', bbox_inches='tight', dpi=300)