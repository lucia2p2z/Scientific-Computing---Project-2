
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time
import math

from project2 import jacobi_step_1d, prolongation_op_1d, restriction_op_1d, Au_op_1d

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
Function to plot different functions on the same image. 
Shows the current plot and wait a little bit to show the next plot with a different colour
"""
def plot_solution(x, u, n, kmax):
    """ Plot solution at time t and wait a bit to create animation effect. """
    if n== 0:
        plt.plot(x, u, label = f"u_h(0)")                   # plot the initial guess u_h(0)
    else: 
        plt.plot(x, u, label = f"h=2/(2^{math.log2(n)})")   # plot the final approximated solution u_h^{kmax} where the meshgrid size is h = 2/n
    plt.legend()
    plt.title(f"Graphics of the solution u_h for different values of h")
    plt.pause(0.6)                                          # wait a bit before showing next plot



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
    jacobi_step_1d(uh, fh, omega)                           # pre - smoothing 
    rh = fh - Au_op_1d(uh)                                  # compute r_h = f_h - A_h * u_h(K)
    r2h = restriction_op_1d(rh)                             # restriction of residuals
    e2h = np.zeros_like(r2h)
    for i in range(10):                                     # ten steps of weighted Jacobi on the coarse grid
        jacobi_step_1d(e2h, r2h, omega)
    uh += prolongation_op_1d(e2h) 
    smax = jacobi_step_1d(uh, fh, omega)                    # post - smoothing
    return smax                                             # returns the pseudo residuals as Jacobi method

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
    count_plot_u0 = 0                                               # counter to show the plot of the initial guess u_0
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    N = [2**3, 2**4, 2**5, 2**6 , 2**7]                             # different values for n
    
    print("Weighted Jacobi —— Investigation on the number of iterations and the effect on CPU time")
    
    for n in N:
        print(f"**** h={2/n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if graphics and count_plot_u0 == 0:
            plot_solution(x, uh, 0, kmax)
            count_plot_u0 += 1


        tic = time()
        k = weighted_jacobi_1d(uh, fh, omega, 1e-8, kmax)            # Jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic

        if graphics: 
            plot_solution(x, uh, n, kmax) 
            count_plot_u0 += 1

        if graphics and count_plot_u0 == len(N)+1:
            plt.savefig('Jacobi_uh.png')

        print(f"Needed {k} Jacobi steps. CPU time: {el:.4f}s.\n")

"""
Q3: 
Implement the two-grid correction scheme, where instead of solving A2h e2h = r2h, 
you perform ten weighted Jacobi steps on the coarse grid. 
Investigate the improvement on the number of iterations and on the CPU time, 
compared to the standard Jacobi iterations you have used above.
"""
def investigation_2grid_correction_jacobi_1d():
    count_plot_u0 = 0                                                   # counter to show the plot of the initial guess u_0
    omega, kmax = 2/3, 10000000                                         # weight and max number of iterations
    N = [2**3, 2**4, 2**5, 2**6, 2**7]                                  # different values for n
    
    print("\nTwo grid correction scheme —— Investigation on the number of iterations and the effect on CPU time")
    
    for n in N:
        print(f"**** h={2/n} ****")
        h = 2/n                                                         # amplitute of the intervals: as n increases, h decreases
        graphics = True                                                 # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                     # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)

        if graphics and count_plot_u0 == 0:
            plot_solution(x, uh, 0, kmax)
            count_plot_u0 += 1

        tic = time()
        k = two_grid_correction_jacobi_1d(uh, fh, omega, 1e-8, kmax)    # Jacobi 2GridCorrection method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        
        if graphics: 
            plot_solution(x, uh, n, kmax) 
            count_plot_u0 += 1

        if graphics and count_plot_u0 == len(N)+1:
            plt.savefig('Jacobi2GridCorrect_uh.png')

        print(f"Needed {k} two-grid steps. CPU time: {el:.4f}s.\n")

if __name__ == "__main__":
    investigation_weighted_jacobi_1d()
    investigation_2grid_correction_jacobi_1d()
