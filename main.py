from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

#starts = starts_nontwist(200)
#starts = new_starts(xy_start=(0,0), xy_end=(1,1), x_points=10, y_points=10)
starts = np.array([0.1, 0.1])

#delta = step_NM(standard_map, starts)

#points = calculate_poincare_section(starts, 10000, mapping=standard_nontwist, b=0.2, a=0.53)
# points = calculate_poincare_section(starts, 100, mapping=standard_map, k = 0.5)

#plot_save_points(points, name='fig', colors='random')

# fig, ax = plt.subplots(figsize=(12, 6))
# for i in range(points.shape[0]):
#     ax.scatter(points[i,0, :], points[i,1,:], color=onp.random.random(3), s=10, marker ='.')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.tight_layout
# plt.show()