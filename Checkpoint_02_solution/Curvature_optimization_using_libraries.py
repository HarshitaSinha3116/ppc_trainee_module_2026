import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import approx_fprime  

def main(args=None):
    eta = 0.001
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 3, 2, 3], dtype=float)

    tck, u = splprep([x, y], s=0, per=False)
    u_new = np.linspace(0, 1, 100)
    x_smooth, y_smooth = splev(u_new, tck)

    def computeCurvature(y_opt):        
        y_full = np.concatenate([[y[0]], y_opt, [y[-1]]])
        tck_, _ = splprep([x, y_full], s=0, per=False)
        dx, dy   = splev(u_new, tck_, der=1)
        d2x, d2y = splev(u_new, tck_, der=2)
        kappa_sq = (dx * d2y - dy * d2x)**2 / (dx**2 + dy**2)**3
        ds = np.sqrt(dx**2 + dy**2)
        return np.trapz(kappa_sq * ds, u_new)

    y_opt = y[1:-1].copy()
   
    eps = 0.00001
    gradient = approx_fprime(y_opt, computeCurvature, eps)
    
    y_opt = y_opt - eta * gradient
    y_opt = np.concatenate([[y[0]], y_opt, [y[-1]]])
   
    tck_opt, _ = splprep([x, y_opt], s=0, per=False)
    x_opt_smooth, y_opt_smooth = splev(u_new, tck_opt)

    plt.figure(figsize=(6, 6))
    plt.plot(x_smooth, y_smooth, label='Cubic Spline', linewidth=2)
    plt.plot(x_opt_smooth, y_opt_smooth, label="Optimized Path", linewidth = 2)
    plt.scatter(x, y, color='red', label='Waypoints')
    plt.scatter(x, y_opt, color='blue')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Before vs After curvature optimization")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()
