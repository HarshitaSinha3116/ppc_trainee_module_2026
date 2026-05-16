import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import approx_fprime  
import os

def main(args=None):
    eta = 0.01
    csv_path = os.path.expanduser('/home/harshita/Downloads/loop_track_waypoints.csv')
    
    dataPoints = pd.read_csv(csv_path)
    x = dataPoints['X'].values
    y = dataPoints['Y'].values

    if (x[0] != x[-1]) or (y[0] != y[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    tck, u = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, 100)
    x_smooth, y_smooth = splev(u_new, tck)
    
    def computeCurvature(y_opt):
      tck_, _ = splprep([x, y_opt], s=0, per=True)
      dx, dy = splev(u_new, tck_, der=1)
      d2x, d2y = splev(u_new, tck_, der=2)

      kappa = np.abs(dx*d2y - dy*d2x) / (dx**2 + dy**2)**1.5
      J = np.sum(kappa**2)
      return J
    
    y_opt = y.copy()#considerng non-endpoints
    eps = 0.00001

    gradient = approx_fprime(y_opt, computeCurvature, eps)

    y_opt -= eta*gradient
    y_opt[0] = y[0]
    y_opt[-1] = y[-1]

    tck_opt, u_opt = splprep([x, y_opt], s=0, per = True)
    x_opt_smooth, y_opt_smooth = splev(u_new, tck_opt)

    plt.figure(figsize=(6, 6))
    plt.plot(x_smooth, y_smooth, label='Cubic Spline', linewidth=2)
    plt.plot(x_opt_smooth, y_opt_smooth, label = "Optimised Curvature", linewidth = 2)
    plt.scatter(x, y, color='red', label='Waypoints')
    plt.scatter(x, y_opt, color = 'blue', label = 'Optimised_Waypoints')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Before vs After curvature optimization")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()



