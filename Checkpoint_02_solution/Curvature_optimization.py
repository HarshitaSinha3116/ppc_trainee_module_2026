import matplotlib.pyplot as plt
import numpy as np

def main(args=None):
    eta = 0.01
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 3, 2, 3], dtype = float)
    A = np.zeros((12, 12))
    B = np.zeros(12)
    eqn = 0
    for i in range(3):
        A[eqn, 4*i] = 1
        B[eqn] = y[i]
        eqn += 1
        dx = x[i+1] - x[i]
        A[eqn,4*i] = 1
        A[eqn, 4*i+1] = dx
        A[eqn, 4*i+2] = dx**2
        A[eqn, 4*i+3] = dx**3
        B[eqn] = y[i+1]
        eqn += 1
    for i in range(1,3):
        dx = x[i] - x[i-1]

        A[eqn, 4*(i-1)+1] = 1
        A[eqn, 4*(i-1)+2] = 2*dx
        A[eqn, 4*(i-1)+3] = 3*dx**2
        A[eqn,4*i+1] = -1
        eqn += 1
    for i in range(1,3):
        dx = x[i] - x[i-1]
        A[eqn, 4*(i-1)+2] = 2
        A[eqn, 4*(i-1)+3] = 6*dx
        A[eqn, 4*i+2] = -2
        eqn += 1
    A[eqn, 2] = 2
    eqn += 1
    dx = x[3] - x[2]
    A[eqn, 10] = 2
    A[eqn, 11] = dx*6
    eqn += 1
    coeffs = np.linalg.solve(A, B)

    def computePoly(i, xi):
        a = coeffs[4*i]
        b = coeffs[4*i+1]
        c = coeffs[4*i+2]
        d = coeffs[4*i+3]
        return a + b*(xi-x[i]) + c*(xi-x[i])**2 + d*(xi-x[i])**3

    X, Y = [], []
    for i in range(3):
        xi_vals = np.linspace(x[i], x[i+1], 50)
        yi_vals = computePoly(i, xi_vals)
        X.extend(xi_vals)
        Y.extend(yi_vals)
 
    def computeCurvature(y_input):
      
      B_temp = np.zeros(12)

      eqn = 0

      for i in range(3):

        B_temp[eqn] = y_input[i]
        eqn += 1

        B_temp[eqn] = y_input[i+1]
        eqn += 1

      B_temp[eqn] = 0
      eqn += 1
      B_temp[eqn] = 0

      coeffs_temp = np.linalg.solve(A, B_temp)

      total_curvature = 0

      for i in range(3):

        xi_vals = np.linspace(x[i], x[i+1], 50)

        for xi in xi_vals:

            b = coeffs_temp[4*i+1]
            c = coeffs_temp[4*i+2]
            d = coeffs_temp[4*i+3]

            yd = b + 2*c*(xi - x[i]) + 3*d*(xi - x[i])**2
            y2d = 2*c + 6*d*(xi - x[i])

            kappa = abs(y2d)/(1 + yd**2)**1.5

            total_curvature += kappa**2

      return total_curvature
      
    J = computeCurvature(y)
    
    def computeGradient(index):
      eps = 0.000001
      J1 = computeCurvature(y)
      y_temp = y.copy()
      y_temp[index] += eps
      J2 = computeCurvature(y_temp)
      
      print(J1)
      print(J2)
      print(J2 - J1)

      gradient = (J2 - J1)/eps
      return gradient
 
    grad1 = computeGradient(1)
    grad2 = computeGradient(2)
    
    print(grad1)
    print(" ")
    print(grad2)
    print("\n")
   
    y_opt = y.copy()

    y_opt[1] -= eta*grad1
    y_opt[2] -= eta*grad2

    B_opt = np.zeros(12)

    eqn = 0
    for i in range(3):

      B_opt[eqn] = y_opt[i]
      eqn += 1

      B_opt[eqn] = y_opt[i+1]
      eqn += 1

    coeffs_opt = np.linalg.solve(A, B_opt)

    X_opt = []
    Y_opt = []

    for i in range(3):

       xi_vals = np.linspace(x[i], x[i+1], 50)

       a = coeffs_opt[4*i]
       b = coeffs_opt[4*i+1]
       c = coeffs_opt[4*i+2]
       d = coeffs_opt[4*i+3]

       yi_vals = (
        a
        + b*(xi_vals-x[i])
        + c*(xi_vals-x[i])**2
        + d*(xi_vals-x[i])**3
)
       X_opt.extend(xi_vals)
       Y_opt.extend(yi_vals)

    plt.plot(X, Y, label="Original Spline")
    plt.plot(X_opt, Y_opt, label="Optimized Spline")
    plt.scatter(x, y, color='blue')
    plt.scatter(x, y_opt, color='red')
    plt.legend()
    plt.show()

