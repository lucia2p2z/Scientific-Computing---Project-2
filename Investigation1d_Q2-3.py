
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time

from project2 import jacobi_step_1d #, v_cycle_step_1d, full_mg_1d

""" 
Function f of the given linear system 
"""
def f_1d(x): 
    return (x - 1/2)**4

""" 
Perform a weighted Jacobi in 1 dim until convergence is reached 
or until we excess the max number of iterations 
"""
def weighted_jacobi_1d(uh, fh, omega, tol=1e-8, kmax=100):
    for k in range(kmax):
        if jacobi_step_1d(uh, fh, omega) < tol: return k + 1
    return -1

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
Implementation of the prolongation operator:
I_h^2h : Omega_2h —> Omega_h
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
Perform 1 step of the 2 grid correction scheme for the weighted Jacobi in 1 dim.
Steps:
- 1 step of weighted_jacobi to find u_h(k) approx. solution of the lin. system A_h u_h = f_h 
- compute the residuals
- restriction of residuals
- Solve the residual equations with 10 steps of weighted Jacobi 
- Coarse grid correction to find the new u_h(k+1) (new initial guess for the following step)
- 1 step of weighted_jacobi to find u_h(k) of the new lin. system A_h u_h = f_h 
"""
def two_grid_correction_step_1d(uh, fh, omega, tol=1e-8, kmax=100):
    v = uh.copy()
    res = 0 
    jacobi_step_1d(uh, fh, omega)                               # pre - smoothing 
    rh = fh - Au_op_1d(uh)                                         # compute r_h = f_h - A_h * u_h(K)
    r2h = restriction_op_1d(rh)                              # restriction of residuals
    e2h = np.zeros_like(r2h)
    for i in range(10):                                         # ten steps of weighted Jacobi on the coarse grid
        jacobi_step_1d(e2h, r2h, omega)
    uh += prolongation_op_1d(e2h) 
    jacobi_step_1d(uh, fh, omega)                               # post - smoothing 
    res = la.norm(uh-v, np.inf)
    return res

"""
Perform a 2 grid correction for the weighted Jacobi in 1 dim 
until convergence is reached we excess the max number of iterations.
"""
def two_grid_correction_jacobi_1d(uh, fh, omega, tol=1e-8, kmax=100):
    for k in range(kmax):
        if two_grid_correction_step_1d(uh, fh, omega) < tol: return k + 1
    return -1

"""
Q2: 
Investigate the number of iterations needed for the weighted Jacobi method with ω = 2/3
and the effect on the CPU time, as h decreases.
"""
def investigation_weighted_jacobi_1d():
    omega, kmax = 2/3, 10000000                                         # weight and max number of iterations
    N = [2**3, 2**4, 2**5, 2**6 , 2**7]                               # different values for n
    print("Weighted Jacobi —— Investigation on the number of iterations and the effect on CPU time")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = False                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                  # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        plt.plot(x, uh, color = "blue", label = "u_h^(0)")

        tic = time()
        k = weighted_jacobi_1d(uh, fh, omega, 1e-8, kmax)            # Re-do jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        if graphics: 
            plt.plot(x, uh, color = "red", label = f"u_h^({kmax})")   # plot the obtained solution from jacobi
            plt.legend()
            plt.title(f"Meshgrid size: h = 2/{n} = {h}")
            plt.show()
        print(f"Needed {k} Jacobi steps. CPU time: {el:.4f}s.")

"""
Q3: 
Implement the two-grid correction scheme, where instead of solving A2h e2h = r2h, 
you perform ten weighted Jacobi steps on the coarse grid. 
Investigate the improvement on the number of iterations and on the CPU time, 
compared to the standard Jacobi iterations you have used above.
"""
def investigation_2grid_correction_jacobi_1d():
    omega, kmax = 2/3, 10000000                                         # weight and max number of iterations
    N = [2**3, 2**4, 2**5, 2**6, 2**7]                               # different values for n
    print("\nTwo grid correction scheme —— Investigation on the number of iterations and the effect on CPU time")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = False                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                  # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        plt.plot(x, uh, color = "blue", label = "u_h^(0)")

        tic = time()
        k = two_grid_correction_jacobi_1d(uh, fh, omega, 1e-8, kmax)            # Re-do jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        
        if graphics: 
            plt.plot(x, uh, color = "red", label = f"u_h^({kmax})")   # plot the obtained solution from jacobi
            plt.legend()
            plt.title(f"Meshgrid size: h = 2/{n} = {h}")
            plt.show()
        print(f"Needed {k} two-grid steps. CPU time: {el:.4f}s.")

if __name__ == "__main__":
    investigation_weighted_jacobi_1d()
    investigation_2grid_correction_jacobi_1d()
