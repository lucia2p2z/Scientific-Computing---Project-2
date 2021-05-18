"""
Lucia Filippozzi 
Student ID: 223447
Course of Scientific Computing - Project 2
"""

import numpy as np
import numpy.linalg as la 
import matplotlib.pyplot as plt

"""
AUXILIARY FUNCTIONS
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""

""" 
Implement y =  A_h u_h, where u_h is N+1 dimensional. 
"""
def Au_op_1d(u):
    n = len(u) - 1
    h = 2/n
    Au = np.zeros_like(u)
    Au[0] = u[0]            # left b.c.
    Au[-1] = u[-1]          # right b.c.
    for i in range(1,n):
        ci = np.abs(-1 + i*h)**2
        Au[i] = ((2+h**2 *ci) * u[i] - u[i-1] - u[i+1])/(h**2)
    return Au

"""
Implementation of the restriction operator:
I_h^2h : Omega_h —> Omega_2h
[v2h]_i = 1/4 ([vh]_2i-1 + 2[vh]_2i + [vh]_2i-1)   for i =1... N/2 -1
"""
def restriction_op_1d(v):
    N = len(v) - 1
    v2 = np.zeros(N//2+1)
    v2[0], v2[-1] = v[0], v[-1]
    for i in range(1, N//2):
        v2[i] = (v[2*i-1] + 2*v[2*i] + v[2*i+1])/4
    return v2

"""
Implementation of the prolongation operator in 1d:
I_2h^h : Omega_2h —> Omega_h
[vh]_2i+1 = 1/2([v2h]_i + [v2h]_i+1)   for i =0... N/2 -1
[vh]_2 = [v2h]_i                       for i =0... N/2
"""
def prolongation_op_1d(v2):
    N = 2*(len(v2)-1)
    v = np.zeros(N+1)
    #v[0], v[-1] = v2[0], v2[-1]
    for i in range(N//2):  # i=0...N/2 -1 
        v[2*i] = v2[i]
        v[2*i+1] = (v2[i] + v2[i+1])/2
    v[-1] = v2[-1]
    return v


"""
Implementation of the restriction operator in 2d:
I_h^2h : Omega_h —> Omega_2h
"""
def restriction_op_2d(v):
    N = len(v) - 1
    v2 = np.zeros([N//2+1, N//2+1])
    #v2[0:], v2[-1:] = v[0:], v[-1:]
    #v2[:0], v2[:-1] = v[:0], v[:-1]
    for i in range(1, N//2):
        for j in range(1, N//2):
            v2[i,j] = ( v[2*i-1, 2*j-1] + v[2*i-1, 2*j+1] + v[2*i+1,2*j-1] + v[2*i+1, 2*j+1] +
            2*(v[2*i, 2*j-1] + v[2*i, 2*j+1] + v[2*i-1, 2*j] + v[2*i+1, 2*j]) +
            4*v[2*i, 2*j] )/ 16
    return v2

"""
Implementation of the prolongation operator in 2d:
I_2h^h : Omega_2h —> Omega_h
"""
def prolongation_op_2d(v2):
    N = 2*(len(v2)-1)
    v = np.zeros([N+1, N+1])
    for i in range(N//2):                       # i = 0...N/2 -1 
        #v[2*i+1, 2*N//2] = (v2[i,N//2] + v2[i+1,N//2])/2
        for j in range(N//2):                   # j = 0...N/2 -1 
            v[2*i, 2*j] = v2[i,j]
            v[2*i+1, 2*j] = (v2[i,j] + v2[i+1,j])/2
            v[2*i, 2*j+1] = (v2[i,j] + v2[i,j+1])/2
            v[2*i+1, 2*j+1] = (v2[i,j] + v2[i+1,j] + v2[i,j+1] + v2[i+1,j+1])/4
    #v[N//2:], v[:N//2] = v2[N//2:], v2[:N//2]
    #for j in range(N//2):  
    #    v[2*N//2, 2*j+1] = (v2[N//2,j] + v2[N//2,j+1])/2
    return v

""" 
Implement y =  A_h u_h, where u_h is N+1 x N+1 dimensional. 
"""
def Au_op_2d(u, c):
    
    n = len(u) - 1
    h= 2/n
    Au = np.zeros_like(u)
    Au[0,:] = u[0,:]            # bottom b.c.
    Au[-1,:] = u[-1,:]          # top b.c.
    Au[:,0] = u[:,0]            # left b.c.
    Au[:,-1] = u[:,-1]          # right b.c.
    for i in range(1,n):
        for j in range(1,n):
            Au[i,j] = ((4 + c[i,j]* h**2)* u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1]) / (h**2)
    return Au


"""
CODE FOR 1 DIMENSIONAL CASE
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""
def jacobi_step_1d(uh, fh, omega):
    """  Implementation of one  step  of  the weighted Jacobi method with weight ω for the linear system A_h u_h = f_h. 

    INPUTS:
    - uh: the initial guess u(k)_h
    - fh: the right hand side of the linear system
    - omega: the weight ω.  
    
    OUTPUTS: 
    - new iterate u(k+1)_h (in place of u_h)
    - pseudo-residual |u(k+1)h−u(k)h|∞ , which acts as the return value of the function.
    """
    
    n = len(uh) - 1
    h = 2/n
    v = uh.copy()
    
    # one step of weighted jacobi:
    for i in range(1,n):
        ci = np.abs(-1 + i*h)**2
        v[i] = (h**2 * fh[i] + uh[i-1] + uh[i+1]) / (2 + h**2 * ci)
        v[i] = (1-omega)*uh[i] + omega*v[i]
    pseudo_res = la.norm(v- uh, np.inf)
    uh[:] = v
    return pseudo_res


def v_cycle_step_1d(uh, fh, omega):
    n = len(uh) - 1
    if n == 2:                                              # case for the coarsest grid Omega_1
        uh[1] = fh[1]/2
        return 0
    else:
        jacobi_step_1d(uh,fh,omega)                         # pre - smoothing 
        f2h = restriction_op_1d(fh - Au_op_1d(uh))
        u2h = np.zeros_like(f2h)
        v_cycle_step_1d(u2h, f2h, omega)
        uh += prolongation_op_1d(u2h)
        pseudo_res = jacobi_step_1d(uh,fh,omega)            # post - smoothing 
    return pseudo_res


def full_mg_1d(uh, fh, omega):
    n = len(uh) - 1
    h = 2/n
    if n == 2:                                              # case for the coarsest grid Omega_1
        c1 = np.abs(-1 + 1*h)**2
        uh[1] = fh[1]/(2+c1*h)                              # exact solve: uh = Ah^{-1}fh = 
        return 0
    else:
        v = uh.copy()
        f2h = restriction_op_1d(fh)
        u2h = np.zeros_like(f2h)
        full_mg_1d(u2h, f2h, omega)                         # recursive call on the other grid
        v = prolongation_op_1d(u2h)
        r = v_cycle_step_1d(v, fh, omega)
        uh[:] = v
        return r                  


"""
CODE FOR 2 DIMENSIONAL CASE
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""
def jacobi_step_2d(uh, fh, omega):
    n = len(uh) - 1
    h = 2/n

    v = uh.copy()
    x = np.linspace(-1, 1, n+1)  
    y = np.linspace(-1, 1, n+1) 
    x,y = np.meshgrid(x,y) 
    c = x**2 + y**2                             # c(X) = c(x,y) = || (x,y) ||^2

    for i in range(1,n):                        # one step of weighted jacobi:
        for j in range(1,n):
            v[i,j] = (h**2 * fh[i,j] + uh[i-1,j] + uh[i+1,j] + uh[i,j-1] + uh[i,j+1]) / (4 + h**2 * c[i,j])
    v[:] = (1-omega)*uh[:] + omega*v[:]
    pseudo_res = la.norm((v- uh).flat, np.inf)
    uh[:] = v
    return pseudo_res

def v_cycle_step_2d(uh, fh, omega):
    
    n = len(uh) - 1
    h = 2/n
    x = np.linspace(-1, 1, n+1)  
    y = np.linspace(-1, 1, n+1) 
    x,y = np.meshgrid(x,y) 
    c = x**2 + y**2                                         # c(X) = c(x,y) = || (x,y) ||^2
    
    if n == 2:                                              # case for the coarsest grid Omega_1
        uh[1,1] = fh[1,1]*(h**2/(4+c[1,1]))
        return 0
    else:
        jacobi_step_2d(uh,fh,omega)                         # pre - smoothing 
        f2h = restriction_op_2d(fh - Au_op_2d(uh, c))
        u2h = np.zeros_like(f2h)
        v_cycle_step_2d(u2h, f2h, omega)
        uh += prolongation_op_2d(u2h)
        pseudo_res = jacobi_step_2d(uh,fh,omega)            # post - smoothing 
    return pseudo_res

def full_mg_2d(uh, fh, omega):
    n = len(uh) - 1
    h = 2/n
    x = np.linspace(-1, 1, n+1)  
    y = np.linspace(-1, 1, n+1) 
    x,y = np.meshgrid(x,y) 
    c = x**2 + y**2                                         # c(X) = c(x,y) = || (x,y) ||^2
    if n == 2:                                              # case for the coarsest grid Omega_1
        uh[1,1] = fh[1,1]*(h**2/(4+c[1,1]))
        return 0
    else:
        v = uh.copy()
        f2h = restriction_op_2d(fh)
        u2h = np.zeros_like(f2h)
        full_mg_2d(u2h, f2h, omega)                         # recursive call on the other grid
        v = prolongation_op_2d(u2h)
        r = v_cycle_step_2d(v, fh, omega)
        uh[:,:] = v
        return r   

