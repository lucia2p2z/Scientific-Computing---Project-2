from operator import index
import numpy as np
import math 
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time

from project2 import v_cycle_step_1d, full_mg_1d, Au_op_1d

""" 
Function f of the given linear system 
"""
def f_1d(x): 
    return (x - 1/2)**4

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

def plot_residuals(n,res): 
    """ Plot residuals at time t and wait a bit to create animation effect. """
    #if n== 0:
    #    plt.plot(x, u, label = f"u_h(0)")                   # plot the initial guess u_h(0)
    #else: 
    l = math.log2(n)
    plt.plot(l, res, label = f"h=2/(2^{l})")   # plot the final approximated solution u_h^{kmax} where the meshgrid size is h = 2/n
    plt.legend()
    plt.title(f"Graphics of the residuals for different values of h")
    plt.pause(0.6)  

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
Your sequence of grid sizes should include as least N = 2^l, l = 3, . . . , 14.
"""
def investigation_vstep_1d():
    count_plot_u0 = 0                                               # counter to show the plot of the initial guess u_0
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    N = [2**i for i in range(3,16)]                                 # different values for n: N = 2^l for l = 2,3...,15
    
    print("V steps —— Investigation on the number of iterations and the effect on CPU time")
    
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if count_plot_u0 ==0:
            plot_solution(x, uh, 0, kmax)
            count_plot_u0 += 1

        tic = time()
        k = v_cycle_1d(uh, fh, omega, 1e-8, kmax)                   # V cycle until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        
        if graphics: 
            plot_solution(x, uh, n, kmax) 
            count_plot_u0 +=1

        if graphics and count_plot_u0 == len(N)+1:
            plt.savefig('Vstep_uh.png')

        print(f"Needed {k} V-cycle steps. CPU time: {el:.4f}s. \n")

"""
Q7: 
Investigate the effect on the CPU time, the residuals and the pseudo residuals
for a full multigrid step with ω = 2/3.
Your sequence of grid sizes should include as least N = 2^l, l = 3, . . . , 14.

"""
def investigation_full_mg_1d():
    count_plot_u0 = 0                                               # counter to show the plot of the initial guess u_0
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    N = [2**i for i in range(3,16)]                                 # different values for n: N = 2^l for l = 2,3...,15
    
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = False                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if graphics and count_plot_u0 == 0:
            plot_solution(x, uh, 0, kmax)
            count_plot_u0 += 1

        tic = time()
        pseudo_res = full_mg_1d(uh, fh, omega)                      # full multigrid method: returns pseudo residuals
        el = time() - tic                        
        res = la.norm(fh - Au_op_1d(uh), np.inf)                    # compute the "real" residuals

        if graphics: 
            plot_solution(x, uh, n, kmax) 
            count_plot_u0 +=1

        if graphics and count_plot_u0 == len(N)+1:
            plt.savefig('FullMultigrid_uh.png')

        print(f"CPU time: {el:6.3e} s.\nPseudo residuals |s_h^(k+1)| = {pseudo_res:6.3e}. \nResiduals        |r_h^(k+1)| = {res:6.3e}. \n")


"""
Q8: 
Investigate the number of additional V-cycles needed, and the effect on the total CPU time,
by using the method full_mg_1d() to provide an initial guess for your V-cycle multigrid
method.
Your sequence of grid sizes should include as least N = 2^l, l = 3, . . . , 14.
"""
def investigation_full_mg_addV_cycle_1d():
    count_plot_u0 = 0                                               # counter to show the plot of the initial guess u_0
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    N = [2**i for i in range(3,16)]                                 # different values for n: N = 2^l for l = 2,3...,15
    
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        fh = f_1d(x)
        fh[0] = 0
        fh[-1] = 0
        uh = np.zeros_like(x)
        
        if count_plot_u0 ==0:
            plot_solution(x, uh, n, kmax)
            count_plot_u0 += 1

        tic = time()
        full_mg_1d(uh, fh, omega)                                   # full multigrid method: returns pseudo residuals   
        k = v_cycle_1d(uh, fh, omega, tol=1e-8, kmax=100)           # v-cycles added until convergence
        el = time() - tic                 

        if graphics: 
            plot_solution(x, uh, n, kmax) 
            count_plot_u0 +=1

        if graphics and count_plot_u0 == len(N)+1:
            plt.savefig('FullMultigrid+Vstep_uh.png')
            
        print(f"Needed {k} additional V-cycles needed. CPU time: {el:.4f}s. \n") 

"""
Q9
Plot an approximation to the solution u over Closure(Omega) and provide an accurate estimate of max u(x) for x in Omega
"""
def plot_solution_and_max():

    graphics = True                                                 # boolean to show the plots
    omega, kmax = 2/3, 1000 
    n = 2**16
    h = 2/n                                                         # amplitute of the intervals: as n increases, h decreases
    x = np.linspace(-1, 1, n+1)                                     # uniform partitioning of [-1,1]: grid for x in 1d
    
    fh = f_1d(x)
    fh[0] = 0
    fh[-1] = 0 
    uh = np.zeros_like(x)
    
    full_mg_1d(uh, fh, omega)                                       # Find the approximated solution
    v_cycle_1d(uh, fh, omega, tol=1e-8, kmax=100)    
    
    if graphics:
        plt.plot(x, uh, color = "red", label = f"u_h")              # Plot the final approximated solution u_h^{kmax} where the meshgrid size is h = 2/n
        plt.title(f"Approximation to the solution u over [-1,1]:")

    approx_max = max(uh)                                            # Compute the approximated max u_h
    index_max = np.argmax(uh)
    x_max = x[index_max]
    print(f"Estimate of max_[-1,1] u(x) = {approx_max}. Xmax = {x_max}")

    err = fh - Au_op_1d(uh)                                         # Compute the error
    err_x  = err[index_max]
    print(f"Error = {err_x}")
    
    if graphics:                                                    # Plot the solutions with the maximum 
        plt.scatter(x_max, approx_max, color = "blue", label = f"(x_max, max)")
        plt.legend()
        plt.savefig('Plot_max_uh.png')
        plt.show()





if __name__ == "__main__":
    #investigation_vstep_1d()
    #investigation_full_mg_1d()
    #investigation_full_mg_addV_cycle_1d()
    plot_solution_and_max()
