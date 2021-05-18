""" Test the functions from project2.py. """
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def unittest2_1d():
    """ Here we test all the 1d routines. """
    from project2 import jacobi_step_1d, v_cycle_step_1d, full_mg_1d

    def f_1d(x): return (x - 1/2)**4

    def weighted_jacobi_1d(uh, fh, omega=1, tol=1e-8, kmax=100):
        for k in range(kmax):
            if jacobi_step_1d(uh, fh, omega) < tol: return k + 1
        return -1

    def mg_v_cycle_1d(uh, fh, omega=1, tol=1e-8, kmax=100):
        for k in range(kmax):
            if v_cycle_step_1d(uh, fh, omega) < tol: return k + 1
        return -1

    def w_i(i, x): return np.sin(i * (x + 1) / 2 * np.pi)

    omega, kmax, n, graphics = 2/3, 50, 4096, False
    #h = 2/n
    x = np.linspace(-1, 1, n+1)
    fh = np.zeros_like(x)
    uh = 0.1 * w_i(1, x) + 0.5 * w_i(n//2, x) + w_i(n-3, x)
    if graphics: 
        plt.plot(x, uh, color = "blue", label = "u_h^0")

    weighted_jacobi_1d(uh, fh, omega, 0, kmax)

    if graphics: 
        plt.plot(x, uh, color = "red", label = f"u_h^{kmax}")
        plt.legend()
        plt.show()
    error = la.norm(uh, np.inf)
    print(f"The error after {kmax} Jacobi steps is {error:6.4e}.")

    uh = w_i(n-1, x)
    k = weighted_jacobi_1d(uh, fh, omega, 1e-6, kmax)
    error = la.norm(uh, np.inf)
    print(f"Needed {k} Jacobi steps to reduce the error to {error:6.4e}.")

    uh = 0.1 * w_i(1, x) + 0.5 * w_i(n//2, x) + w_i(n-3, x)
    k = mg_v_cycle_1d(uh, fh, omega, 1e-8, kmax)
    error = la.norm(uh, np.inf)
    print(f"Needed {k} V-cycle steps to reduce the error to {error:6.4e}.")

    uh[:], fh[:] = 0, f_1d(x)
    fh[0], fh[-1] = 0, 0
    k = mg_v_cycle_1d(uh, fh, omega, 1e-8, kmax)
    if graphics: 
        plt.plot(x, uh, color = "blue")
        plt.show()
    print(f"Needed {k} V-cycle steps to converge.")

    uh[:] = 0
    smax = full_mg_1d(uh, fh, omega)
    if graphics: 
        plt.plot(x, uh, color = "blue")
        plt.show()
    print(f"The pseudo-residual after a full MG step is {smax:6.4e}.") 

def unittest2_2d():
    """ Here we test all the 2d routines. """
    from project2 import jacobi_step_2d, v_cycle_step_2d, full_mg_2d
    from mpl_toolkits.mplot3d import Axes3D

    def f_2d(x, y): return x * np.exp(2*y)

    def weighted_jacobi_2d(uh, fh, omega=1, tol=1e-8, kmax=100):
        for k in range(kmax):
            if jacobi_step_2d(uh, fh, omega) < tol: return k + 1
        return -1

    def mg_v_cycle_2d(uh, fh, omega=1, tol=1e-8, kmax=100):
        for k in range(kmax):
            if v_cycle_step_2d(uh, fh, omega) < tol: return k + 1
        return -1

    def w_ij(i, j, x, y): 
        return np.sin(i * (x + 1)/2 * np.pi) * np.sin(j * (y + 1)/2 * np.pi)

    omega, kmax, n, graphics = 2/3, 50, 256, False
    h = 2/n
    x = np.linspace(-1, 1, n+1)
    y = np.linspace(-1, 1, n+1)
    x, y = np.meshgrid(x, y)
    fh = np.zeros_like(x)
    uh = (0.1 * w_ij(1, 1, x, y) + 0.5 * w_ij(n//2, n//2, x, y) 
          + w_ij(n-3, n-3, x, y))

    if graphics: 
        fig = plt.figure()
        ax = fig.add_subplot(211, projection="3d")
        ax.set_title("u_h^0")
        ax.plot_surface(x, y, uh)
    weighted_jacobi_2d(uh, fh, omega, 0, kmax)
    if graphics: 
        ax = fig.add_subplot(212, projection="3d")
        ax.set_title(f"u_h^{kmax}")
        ax.plot_surface(x, y, uh)
        plt.show()
    error = la.norm(uh.flat, np.inf)
    print(f"The error after {kmax} Jacobi steps is {error:6.4e}.")

    uh = w_ij(n-1, n-1, x, y)
    k = weighted_jacobi_2d(uh, fh, omega, 1e-6, kmax)
    error = la.norm(uh.flat, np.inf)
    print(f"Needed {k} Jacobi steps to reduce the error to {error:6.4e}.")
    uh = (0.1 * w_ij(1, 1, x, y) + 0.5 * w_ij(n//2, n//2, x, y) 
          + w_ij(n-3, n-3, x, y))
    k = mg_v_cycle_2d(uh, fh, omega, 1e-8, kmax)
    error = la.norm(uh.flat, np.inf)
    print(f"Needed {k} V-cycle steps to reduce the error to {error:6.4e}.")

    uh[:], fh[:] = 0, f_2d(x,y)
    fh[0,:], fh[-1,:], fh[:,0], fh[:,-1] = 0, 0, 0, 0
    k = mg_v_cycle_2d(uh, fh, omega, 1e-8, kmax)
    if graphics: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, uh)
        plt.show()
    print(f"Needed {k} V-cycle steps to converge.")

    uh[:] = 0
    smax = full_mg_2d(uh, fh, omega)
    if graphics: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, uh)
        plt.show()
    print(f"The pseudo-residual after a full MG step is {smax:6.4e}.")

print("****** 1d tests ********")
unittest2_1d()
print("****** 2d tests ********")
unittest2_2d()
