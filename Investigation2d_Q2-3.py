import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D 
from time import time

from project2 import jacobi_step_2d, prolongation_op_2d, restriction_op_2d, Au_op_2d

""" 
Function f of the given linear system 
"""
def f_2d(x,y): 
    return x * math.e**(2*y) 

"""
Function to plot different functions on the same image. 
Shows the current plot and wait a little bit to show the next plot with a different colour
"""
def plot_solution(ax, x, y, u, n, kmax): 
    h= 2/n
    ax.plot_surface(x, y, u) 
    ax.set_title(f"Meshgrid size: h = 2/(2^{math.log2(n)}) = {h:4.4e}")
    plt.pause(1.2)                       # wait a bit before showing next plot
    ax.cla()


""" 
Perform a weighted Jacobi in 1 dim until convergence is reached 
or until we excess the max number of iterations 
"""
def weighted_jacobi_2d(uh, fh, omega, tol=1e-8, kmax=100):
    for k in range(kmax):
        if jacobi_step_2d(uh, fh, omega) < tol: return k + 1
    return -1


"""
Perform 1 step of the 2 grid correction scheme for the weighted Jacobi in 2 dim.
"""
def two_grid_correction_step_2d(uh, fh, omega, c, tol=1e-8, kmax=100):
    v = uh.copy()
    res = 0 
    jacobi_step_2d(uh, fh, omega)                               # pre - smoothing 
    rh = fh - Au_op_2d(uh, c)                                   # compute r_h = f_h - A_h * u_h(K)
    r2h = restriction_op_2d(rh)                                 # restriction of residuals
    e2h = np.zeros_like(r2h)
    for i in range(10):                                         # ten steps of weighted Jacobi on the coarse grid
        jacobi_step_2d(e2h, r2h, omega)
    uh += prolongation_op_2d(e2h) 
    smax = jacobi_step_2d(uh, fh, omega)                        # post - smoothing 
    return smax                                                 # returns the pseudo residuals as Jacobi method

"""
Perform a 2 grid correction for the weighted Jacobi in 2 dim 
until convergence is reached we excess the max number of iterations.
"""
def two_grid_correction_jacobi_2d(uh, fh, omega, c, tol=1e-8, kmax=100):
    for k in range(kmax):
        if two_grid_correction_step_2d(uh, fh, omega, c) < tol: return k + 1
    return -1

"""
Q2: 
Investigate the number of iterations needed for the weighted Jacobi method with ω = 2/3
and the effect on the CPU time, as h decreases.
"""
def investigation_weighted_jacobi_2d():
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    counter_plot = 0                                                # counter to save some plots as png
    N = [2**2, 2**3, 2**4, 2**5, 2**6]                              # different values for n
    
    print("Weighted Jacobi —— Investigation on the number of iterations and the effect on CPU time")
    
    for n in N:
        print(f"**** N={n} ****")
        h = 2/n                                                     # amplitute of the intervals: as n increases, h decreases
        graphics = True                                             # boolean to show the plots

        x = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for x in 1d
        y = np.linspace(-1, 1, n+1)                                 # uniform partitioning of [-1,1]: grid for y in 1d
        x,y = np.meshgrid(x,y)
        
        fh = f_2d(x,y)
        uh = np.zeros_like(fh)

        if graphics: 
            fig = plt.figure()                                      # set up a figure for plotting
            ax = fig.add_subplot(111, projection="3d")

        tic = time()
        k = weighted_jacobi_2d(uh, fh, omega, 1e-8, kmax)           # Jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        if graphics: 
            plot_solution(ax, x, y, uh, n, kmax)                    # Plot the obtained solution from jacobi
            counter_plot +=1
        if graphics: plt.savefig("jacobi_2d.png")

        print(f"Needed {k} Jacobi steps. CPU time: {el:.4f}s.")

"""
Q3: 
Implement the two-grid correction scheme, where instead of solving A2h e2h = r2h, 
you perform ten weighted Jacobi steps on the coarse grid. 
Investigate the improvement on the number of iterations and on the CPU time, 
compared to the standard Jacobi iterations you have used above.
"""
def investigation_2grid_correction_jacobi_2d():
    count_plot_u0 = 0
    omega, kmax = 2/3, 10000000                                     # weight and max number of iterations
    N = [2**2, 2**3, 2**4, 2**5, 2**6]                              # different values for n
    
    print("\nTwo grid correction scheme —— Investigation on the number of iterations and the effect on CPU time")
    
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
            count_plot_u0 +=1

        tic = time()
        k = two_grid_correction_jacobi_2d(uh, fh, omega, c, 1e-8, kmax) # Jacobi method until convergence (or until we reach the max number of iterations) 
        el = time()-tic
        
        if graphics: 
            plot_solution(ax, x, y, uh, n, kmax)                        # plot the obtained solution from jacobi

        print(f"Needed {k} two-grid steps. CPU time: {el:.4f}s.")

if __name__ == "__main__":
    investigation_weighted_jacobi_2d()
    investigation_2grid_correction_jacobi_2d()
