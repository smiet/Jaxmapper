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

from plotting import plot_newtons_fractal, plot_fixed_points
from maps import standard_map_theta_modulo as no_p_mod

from methods import fixed_point_trajectory, theta
from plotting import expand_fixed_points

k=0.5
N=2

sm2 = Nmap(standard_map, 2)

starts = np.array([[1.45426592e-09, 4.99999996e-01]])
steps = fixed_point_trajectory(xy=starts, 
                               map=sm2, map_modulo=modulo,
                               step=step_LGTNMx, step_modulo=modulo,
                               niter=80,
                               k=k)
test = np.swapaxes(steps, 1,2)

rolled_sm2 = lambda xy: modulo(sm2(xy, k=k))

vmapped_sm2 = vmap(vmap(rolled_sm2))

test2 = vmapped_sm2(test)

# print(test2.shape)

# # plt.scatter(test[:,:,0], test[:,:, 1], color = 'r')
# # plt.scatter(test2[:, :, 0], test2[:, :, 1])

# for index in range(80):
#     plt.plot([test[:, index, 0], test2[:, index, 0]],[test[:, index, 1], test2[:, index, 1]], marker = 'o')

# plt.show()

x_0 = 2.30258482e-01; y_0=9.60516963e-01
x_0 = 0; y_0 = 0.5
npoints = 100; len=0.03
points = np.array([[x_0+len*np.sin(theta), y_0 + len*np.cos(theta)] for theta in np.linspace(0, np.pi*2, npoints)])

from methods import isotrope
map_isotrope = isotrope(standard_map, modulo)
rolled_isotrope = lambda xy: map_isotrope(xy, k=k)
vmapped_isotrope = vmap(rolled_isotrope)

vmapped_modulo = vmap(modulo)

test = vmapped_isotrope(points)

test2 = np.linalg.norm(test, axis=-1)

test3 = test/test2[:, None]

plt.quiver(points[:, 0], points[:, 1], test3[:, 0], test3[:, 1])

plt.xlim([-0.04, 0.04])
plt.ylim([0.46, 0.54])

plt.show()


####### TESTING WHY THE MAP DIFFERENCE PLOT HAS SOME POINTS THAT GO UP

# k=0.5
# start = np.array([[0.48747675, 0.04207196]])
# start = np.array([[1.45426592e-09, 4.99999996e-01]])
# start = np.array([[1.5426592e-09, 4.99999996e-01]])
# # start = np.array([[0.5, 0.5]])

# sm2 = Nmap(standard_map, 2)

# test = fixed_point_trajectory(start, sm2, modulo, step_LGTNMx, modulo, 40, k=k)
# test = np.swapaxes(test, 1, 2) 
# # print(test)

# from methods import fixed_point_steps
# test2 = fixed_point_steps(start, sm2, modulo, step_LGTNMx, modulo, 20, k=k)
# # print(test2)

# test3 = test[0, -2, :]
# test4 = np.array([0., 0.5])
# # print(test3-test4)

# step_for_map = step_LGTNMx(sm2, modulo)
# rolled_step = lambda xy: step_for_map(xy, k=k)
# test4 = rolled_step(test3)
# # print(test4)

# map_theta = theta(sm2, modulo)
# rolled_theta = lambda xy: map_theta(xy, k=k)
# dtheta = grad(rolled_theta, argnums=0)
# final = np.array([[0,1],[-1,0]]) @ np.nan_to_num(dtheta(test3))
# # print(dtheta(test3))

# xy = np.array([1.5426592e-09, 4.99999996e-01])
# xy = np.array([1.36107700e-02, 4.65976648e-01])
# xy = np.array([2.36107700e-02, 4.65976648e-01])

# map_isotrope = isotrope(sm2, modulo)
# delta = map_isotrope(xy, k=k)
# print(delta)
# unit_delta = delta / np.linalg.norm(delta)

# from methods import modded_length
# length = lambda xy: modded_length(xy, no_p_mod(sm2(xy, k=k)), xmod=1, ymod=1)
# grad_length = grad(length)(test3)

# scaling = (length(test3))/abs(np.dot(grad_length, unit_delta))
# print(np.nan_to_num(scaling * unit_delta))

#######


####### TESTING DMIN PLOT

# k=0.5
# points = np.array([[0.47585491, 0.23475008]])

# from groundtruth import sm2_fixed_points

# fixed_points = expand_fixed_points(sm2_fixed_points, 0, 1, 0, 1)[0]

# iterations = fixed_point_trajectory(xy=points, 
#                                    map=sm2, map_modulo=modulo,
#                                    step=step_LGTNMx, step_modulo=modulo,
#                                    niter=20, 
#                                    k=k)
# iterations = np.swapaxes(iterations, 1, 2)
# distances = np.linalg.norm(iterations[:,:,None,:] - fixed_points[None,None,:,:], axis=-1)
# # print(distances[-1, -1])

#######