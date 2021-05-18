import numpy as np
import math 
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time

from project2 import v_cycle_step_2d, full_mg_2d, Au_op_2d

""" 
Function f of the given linear system 
"""
def f_2d(x,y): 
    return x * math.e**(2*y) 

"""
Function to plot different functions on the same image. 
Shows the current plot and wait a little bit to show the next plot with a different colour
"""
def plot_solution_2d(ax, x, y, u, n, kmax): 
    h= 2/n
    ax.plot_surface(x, y, u)  #, label = f"h=2/(2^{math.log2(n)})")
    ax.set_title(f"Meshgrid size: h = 2/(2^{math.log2(n)}) = {h:4.4e}")
    plt.pause(1.2)                       # wait a bit before showing next plot
    ax.cla()


""" 
Perform a complete V-cycle method in 2 dim until convergence is reached 
or until we excess the max number of iterations 
"""
def v_cycle_2d(uh, fh, omega, tol=1e-8, kmax=100):
    for k in range(kmax):
        if v_cycle_step_2d(uh, fh, omega) < tol: return k + 1
    return -1

"""
Q5: 
Investigate the number of iterations needed for the V-cycle method with ω = 2/3
and the effect on the CPU time, as h decreases.
"""
def investigation_vstep_2d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(2,10)]                                  # different values for n: N = 2^l for l = 2,3...,9
    print("V steps —— Investigation on the number of iterations and the effect on CPU time")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                      # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        y = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for y in 1d
        x,y = np.meshgrid(x,y)
        fh = f_2d(x,y)
        uh = np.zeros_like(fh)

        if graphics and count_plot_u0 ==0:
            fig = plt.figure()                                      # set up a figure for plotting
            ax = fig.add_subplot(111, projection="3d")
            plot_solution_2d(ax, x, y, uh, n, kmax)
            count_plot_u0 += 1

        tic = time()
        k = v_cycle_2d(uh, fh, omega, 1e-8, kmax)                        # Re-do jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        if graphics: 
            plot_solution_2d(ax, x, y, uh, n, kmax) 
        print(f"Needed {k} V-cycle steps. CPU time: {el:.4f}s. \n")

"""
Q7: 
Investigate the effect on the CPU time for a full multigrid step with ω = 2/3
and the pseudo and non residuals, as h decreases.
"""
def investigation_full_mg_2d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(2,10)]                                  # different values for n: N = 2^l for l = 2,3...,10
    
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                      # amplitute of the intervals: as n increases, h decreases
        graphics = False                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        y = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for y in 1d
        x,y = np.meshgrid(x,y)
        c = x**2 + y**2
        fh = f_2d(x,y)
        uh = np.zeros_like(fh)

        if graphics and count_plot_u0 ==0:
            fig = plt.figure()                                      # set up a figure for plotting
            ax = fig.add_subplot(111, projection="3d")               
            plot_solution_2d(ax, x, y, uh, n, kmax)
            count_plot_u0 += 1

        tic = time()
        pseudo_res = full_mg_2d(uh, fh, omega)   
        el = time() - tic                     
        
        rmax =0                                                     # compute the residuals
        rh = fh - Au_op_2d(uh, c)
        for i in range(1,n):
            for j in range(1,n):
                rmax = max(rmax, abs(rh[i,j]))

        if graphics: 
            plot_solution_2d(ax, x, y, uh, n, kmax) 
            
        print(f"CPU time: {el:.4f} s. \nPseudo residuals |s_h^(k+1)| = {pseudo_res:.4f}. \nResiduals        |r_h^(k+1)| = {rmax:.4f}. \n")

"""
Q8: 
Investigate the number of additional V-cycles needed, and the effect on the total CPU time,
by using the method full_mg_2d() to provide an initial guess for your V-cycle multigrid
method.
Your sequence of grid sizes should include as least N = 2^l, l = 3, . . . , 8.
"""
def investigation_full_mg_addV_cycle_2d():
    omega, kmax = 2/3, 10000                                      # weight and max number of iterations
    count_plot_u0 = 0
    N = [2**i for i in range(2,10)]                                  # different values for n: N = 2^l for l = 2,3...,9
    print("Full multigrid step —— Investigation on the size of the final pseudo residuals S_h and of the real ones r_h")
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        y = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for y in 1d
        x,y = np.meshgrid(x,y)
        c = x**2 + y**2
        fh = f_2d(x,y)
        uh = np.zeros_like(fh)

        if graphics and count_plot_u0 ==0: 
            fig = plt.figure()                                      # set up a figure for plotting
            ax = fig.add_subplot(111, projection="3d")
        
        k = 0 
        tol = 10e-8
        tic = time()
        a = full_mg_2d(uh, fh, omega) 
        if a > tol:
            k = v_cycle_2d(uh, fh, omega, tol=1e-8, kmax=100)     
        el = time() - tic                 

        if graphics: 
            plot_solution_2d(ax, x, y, uh, n, kmax)
            count_plot_u0 += 1
            
        print(f"Needed {k} additional V-cycles needed. CPU time: {el:.4f}s. \n") 

"""
Q9
Plot an approximation to the solution u over Closure(Omega) and provide an accurate estimate of max u(x) for x in Omega
"""
def plot_solution_and_max():
    omega, kmax = 2/3, 10000 
    n = 2**8
    h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
    graphics = True                                             # boolean to show the plots

    x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
    y = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for y in 1d
    x,y = np.meshgrid(x,y)
    c = x**2 + y**2
    fh = f_2d(x,y)
    uh = np.zeros_like(fh)
    
    full_mg_2d(uh, fh, omega) 
    v_cycle_2d(uh, fh, omega, tol=1e-8, kmax=100)     

    approx_max = np.max(uh)
    index_max = np.argmax(uh.reshape(uh.shape[0],uh.shape[1],-1),axis=2)
    
    max_index_x = np.where(uh==approx_max)[0][0]
    max_index_y = np.where(uh==approx_max)[1][0]
    x_max = x[max_index_x, max_index_y]
    y_max = y[max_index_x, max_index_y]

    print(x_max)
    print(y_max)
    print(f"Estimate of max_[-1,1]x[-1,1] u(x) = {approx_max}. \n(Xmax, Ymax) = ({x_max}, {y_max})")

    err = fh - Au_op_2d(uh,c)
    err_x  = err[max_index_x, max_index_y]
    print(f"Error = {err_x}")
    
    if graphics:
        fig = plt.figure()                                      # set up a figure for plotting
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, uh)#, label = f"u_h")  
        ax.set_title(f"Approximation to the solution u over [-1,1]x[-1,1]:\n\n")
        ax.scatter(x_max, y_max, approx_max, color = "red", label = f"(x_max, y_max, max)")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("u_h")
        plt.savefig('Plot_max_uh_2d.png')
        plt.show()




if __name__ == "__main__":
    #investigation_vstep_2d()
    #investigation_full_mg_2d()
    #investigation_full_mg_addV_cycle_2d()
    plot_solution_and_max()
    