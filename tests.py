from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

def run_test():
    test_step_NM()
    test_first_order_jacobian()
    print('All tests passed')


def test_step_NM():
    """
    Tests step_NM from methods.py with the chirikov standard map.
    Specifically tests whether step_NM with sympy method = step_NM with jax method
    """
    from methods import step_NM
    from maps import standard_map, sym_standard_map
    
    starts = onp.random.rand(2)
    k = onp.random.rand()
    
    jax_delta = step_NM(standard_map, method='jax')
    jax_step = jax_delta(starts, k=k)
    jax_step = np.round(jax_step, 12)

    sym_delta = step_NM(sym_standard_map, method='sympy')
    sym_step = sym_delta(starts, xmod=1, ymod=1, k=k)
    sym_step = np.round(sym_step, 12)

    assert np.array_equal(jax_step, sym_step) == True, f'Something wrong with step_NM (jax and sym not giving same values), {starts} {k}'

def test_first_order_jacobian():
    """
    Tests whether f(xy+delta) = f(xy) + Mdelta to first order
    """
    from maps import standard_map
    xy = onp.random.rand(2)
    k = onp.random.rand()
    delta = 10**(-7)*onp.random.rand(2)
    fxy = standard_map(xy, k=k)
    rolled_map = lambda xy: standard_map(xy, k=k)
    M = jacfwd(rolled_map)(xy)

    first_order = fxy + np.matmul(M, delta)
    diff = fxy - first_order
    assert np.sum(np.absolute(diff)) < 10**(-6), f'Something wrong with first-order-approximation of standard map, {xy}, {delta}, {k}'

