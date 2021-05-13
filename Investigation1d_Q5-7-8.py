from operator import index
import numpy as np
import math 
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time

from project2 import v_cycle_step_1d, full_mg_1d

""" 
Function f of the given linear system 
"""
def f_1d(x): 
    return (x - 1/2)**4

""" 
Implement y =  A_h u_h, where u_h is N+1 dimensional. 
"""
def Au_op(u):
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
Function to plot different functions on the same image. 
Shows the current plot and wait a little bit to show the next plot with a different colour
"""
def plot_solution(x, u, n, kmax): #, t):
    """ Plot solution at time t and wait a bit to create animation effect. """
    plt.plot(x, u, label = f"h=2/(2^{math.log2(n)})")   # plot the final approximated solution u_h^{kmax} where the meshgrid size is h = 2/n
    plt.legend()
    plt.title(f"Graphics of u_h^{kmax} for different values of h")
    plt.pause(0.6)                                      # wait a bit before showing next plot


""" 
Perform a complete V-cycle moethod in 1 dim until convergence is reached 
or until we excess the max number of iterations 
"""
def v_cycle_1d(uh, fh, omega, tol=1e-8, kmax=100):
    for k in range(kmax):
        if v_cycle_step_1d(uh, fh, omega) < tol: return k + 1
    return -1

"""
Q5: 
Investigate the number of iterations needed for the V-cycle method with ω = 2/3
and the effect on the CPU time, as h decreases.
"""
def investigation_vstep_1d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(3,21)]                                  # different values for n: N = 2^l for l = 2,3...,20
    print("V steps —— Investigation on the number of iterations and the effect on CPU time")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                      # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                  # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if count_plot_u0 ==0:
            plot_solution(x, uh, n, kmax)
            count_plot_u0 += 1

        tic = time()
        k = v_cycle_1d(uh, fh, omega, 1e-8, kmax)                        # Re-do jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        if graphics: 
            plot_solution(x, uh, n, kmax) 
        print(f"Needed {k} V-cycle steps. CPU time: {el:.4f}s. \n")

"""
Q7: 
Investigate the number of iterations needed for a full multigrid step with ω = 2/3
and the effect on the CPU time, as h decreases.
"""
def investigation_full_mg_1d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(3,15)]                                  # different values for n: N = 2^l for l = 2,3...,20
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                      # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                  # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if count_plot_u0 ==0:
            plot_solution(x, uh, n, kmax)
            count_plot_u0 += 1

        pseudo_res = full_mg_1d(uh, fh, omega)                        
        res = la.norm(fh - Au_op(uh), np.inf)

        if graphics: 
            plot_solution(x, uh, n, kmax) 
            
        print(f"Pseudo residuals |s_h^(k+1)| = {pseudo_res:6.4e}. \nResiduals |r_h^(k+1)| = {res:6.4e}. \n")


"""
Q8: 
Investigate the number of additional V-cycles needed, and the effect on the total CPU time,
by using the method full_mg_1d() to provide an initial guess for your V-cycle multigrid
method.
Your sequence of grid sizes should include as least N = 2^l, l = 3, . . . , 14.
"""
def investigation_full_mg_addV_cycle_1d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(3,15)]                                  # different values for n: N = 2^l for l = 2,3...,20
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                      # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                  # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if count_plot_u0 ==0:
            plot_solution(x, uh, n, kmax)
            count_plot_u0 += 1

        tic = time()
        full_mg_1d(uh, fh, omega) 
        k = v_cycle_1d(uh, fh, omega, tol=1e-8, kmax=100)     
        el = time() - tic                 

        if graphics: 
            plot_solution(x, uh, n, kmax) 
            
        print(f"Needed {k} additional V-cycles needed. CPU time: {el:.4f}s. \n") 

"""
Q9
Plot an approximation to the solution u over Closure(Omega) and provide an accurate estimate of max u(x) for x in Omega
"""
def plot_solution_and_max():
    graphics = True
    omega, kmax = 2/3, 10000 
    n = 2**16
    h = 2/n    
    x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
    fh = f_1d(x)
    fh[0] = 0
    fh[-1] = 0 
    uh = np.zeros_like(x)
    full_mg_1d(uh, fh, omega) 
    v_cycle_1d(uh, fh, omega, tol=1e-8, kmax=100)    
    if graphics:
        plt.plot(x, uh, color = "red", label = f"u_h^({kmax})")     # plot the final approximated solution u_h^{kmax} where the meshgrid size is h = 2/n
        plt.legend()
        plt.title(f"Approximation to the solution u over [-1,1]:")

    approx_max = max(uh)
    index_max = np.argmax(uh)
    x_max = x[index_max]
    print(f"Estimate of max_[-1,1] u(x) = {approx_max}. Xmax = {x_max}")

    err = fh - Au_op(uh)
    err_x  = err[index_max]
    print(f"Error = {err_x}")
    
    if graphics:
        plt.scatter(x_max, approx_max)
        plt.show()





if __name__ == "__main__":
    plot_solution_and_max()
