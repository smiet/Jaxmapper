from jax import numpy as np
from jax import jit, grad, jacfwd, vmap, jacobian
from jax.tree_util import Partial
from jax import config
import sympy as sym
from sympy import Subs
config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
import numpy as onp

@jit
def standard_nontwist(xy, a=0.51, b=0.31):
    """
    standard nontwist map.
    """
    y = xy[1] - b*np.sin(2*np.pi*xy[0])
    x = np.mod(xy[0] + a*(1 - y**2), 1)
    return np.array([x,y])

@jit
def tokamap(xy, K=0, w=0.666, w0=0.7, w1=0.3):
    """
    standard nontwist map.
    """
    psi_old = xy[1]
    theta_old = xy[0]
    P = psi_old - 1 - (K/(2*np.pi))*np.sin(2*np.pi*theta_old)
    a = (w-w0)/w
    c = 1 + ((w-w1)/(w-w0))**2
    psi = 0.5*(P + np.sqrt(P**2 + 4* psi_old))
    W = w*(1-a*(c*psi-1)**2)
    theta = np.mod(theta_old + W - (K/((2*np.pi)**2) * (1 + psi)**2)*np.cos(2*np.pi*theta_old), 1)
    #theta = np.mod(xy[0] + a*(1 - y**2), 1)
    return np.array([theta,psi])

@jit
def basecase(xy):
    """
    f(z) = z^3 - 1 (want to find the roots of this)
    g(z) = z^3 - 1 + z (finding the fixed points of this is equivalent to finding the roots of f)
    """
    x = xy[0]
    y = xy[1]
    x_new = x**3 -3*x*y**2 + x - 1
    y_new = 3*y*x**2 - y**3 + y 
    return np.array([x_new, y_new])

@jit
def no_modulo(xy):
    """
    Returns argument.
    Required as some of the maps don't have modulo.
    """
    return xy

@jit
def standard_map(xy, k):
    """
    Definition of the Chirikov Standard Map. Takes in an xy coordinate and a k-value.
    x and p coordinates are normalised and modulo 1.
    0.5 is added to the theta coordinate in order to shift the plot up by 0.5.
    """
    theta_old = xy[0]
    p_old = xy[1]
    # applying standard map on old coordinates to get new coordinates
    p = np.mod(p_old + (k/(2*np.pi))*np.sin(2*np.pi*theta_old), 1)
    theta = np.mod(theta_old + p + 0.5, 1)
    # return
    return np.array([theta, p])

@jit
def standard_map_modulo(xy):
    """
    Returns modulo of arguments based on Standard Map definition.
    """
    theta = xy[0]
    p=xy[1]

    theta = np.mod(theta, 1)
    p = np.mod(p, 1)
    return np.array([theta, p])

def sym_standard_map(k):
    """
    Returns a Sympy expression for the Chirikov standard map.
    Since it is a Sympy expression, it does not take any inputs.
    """
    x, y = sym.symbols('x y')
    x_new = x + y + (k/(2*np.pi))*sym.sin(2*np.pi*x) + 0.5
    y_new = y + (k/(2*np.pi))*sym.sin(2*np.pi*x)
    F = sym.Matrix([x_new, y_new])
    return F

def Nmap(map, N=int):
    """
    Takes in a function and an integer N. Outputs a function which is the map applied N times.
    """
    def nmap(x, **kwargs):
        for _ in range(N):
            x = map(x, **kwargs)
        return x
    return jit(nmap)

def sym_nmap(expr, N):
    """
    Takes a Sympy expression as input and outputs a Sympy expression which is the N-map.
    IMPT: func must only have 2 Sympy symbols - x and y. 
    """
    x, y, z = sym.symbols('x y z')
    iter = expr
    for _ in range(N-1):
        old_x = iter[0]
        old_y = iter[1]
        iter = iter.subs(y, z)
        iter = iter.subs(x, old_x)
        iter = iter.subs(z, old_y)
    return iter

def sym_jac_func(expr):
    """
    Takes a Sympy expression as input and outputs a Python function which evaluates the Jacobian of func.
    IMPT: func must only have 2 Sympy symbols - x and y.
    The output function takes in a 1D array of length 2 (x/y value).
    """
    x, y = sym.symbols('x y')
    X=sym.Matrix([x,y])
    sym_jac = expr.jacobian(X)
    first_num_jac = sym.lambdify([x,y], sym_jac)
    def num_jac(xy):
        x = xy[0]
        y = xy[1]
        return first_num_jac(x, y)
    return num_jac 


