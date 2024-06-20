from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

@jit
def standard_nontwist(xx, a=0.51, b=0.31):
    """
    standard nontwist map.
    """
    y = xx[1] - b*np.sin(2*np.pi*xx[0])
    x = np.mod(xx[0] + a*(1 - y**2), 1)
    return np.array([x,y])

@jit
def tokamap(xx, K=0, w=0.666, w0=0.7, w1=0.3):
    """
    standard nontwist map.
    """
    psi_old = xx[1]
    theta_old = xx[0]
    P = psi_old - 1 - (K/(2*np.pi))*np.sin(2*np.pi*theta_old)
    a = (w-w0)/w
    c = 1 + ((w-w1)/(w-w0))**2
    psi = 0.5*(P + np.sqrt(P**2 + 4* psi_old))
    W = w*(1-a*(c*psi-1)**2)
    theta = np.mod(theta_old + W - (K/((2*np.pi)**2) * (1 + psi)**2)*np.cos(2*np.pi*theta_old), 1)
    #theta = np.mod(xx[0] + a*(1 - y**2), 1)
    return np.array([theta,psi])

@jit
def standard_map(xx, k = 0.1):
    """
    chirikov standard map
    """
    theta_old = xx[0]
    p_old = xx[1]
    # applying standard map on old coordinates to get new coordinates
    p = np.mod(p_old + (k/(2*np.pi))*np.sin(2*np.pi*theta_old), 1)
    theta = np.mod(theta_old + p + 0.5, 1)
    # return
    return np.array([theta, p])

def Nmap(map, N):
    def nmap(x, **kwargs):
        for _ in range(N):
            x = map(x, **kwargs)
        return x
    return jit(nmap)

def symbolic_nmap(func, N):
    """
    Takes a Sympy expression as input and outputs a Sympy expression which is the N-map.
    IMPT: func must only have 2 Sympy symbols - x and y. 
    """
    x, y, z = sym.symbols('x y z')
    iter = func
    for _ in range(N-1):
        old_x = iter[0]
        old_y = iter[1]
        iter = iter.subs(y, z)
        iter = iter.subs(x, old_x)
        iter = iter.subs(z, old_y)
    return iter

def sym_jac_func(func): #TODO num_jac takes 2 separate inputs. find a way to change to 1 input which is an array of 2 points
    """
    Takes a Sympy expression as input and outputs a Python function which evaluates the Jacobian of func.
    IMPT: func must only have 2 Sympy symbols - x and y.
    The output function has 2 arguments. 1st is x-value and 2nd is y-value. 
    """
    x, y = sym.symbols('x y')
    X=sym.Matrix([x,y])
    sym_jac = func.jacobian(X)
    num_jac = sym.lambdify([x,y], sym_jac)
    return num_jac 



