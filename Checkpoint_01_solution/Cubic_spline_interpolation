import matplotlib.pyplot as plt
import numpy as np
import math

x = np.array([1, 2, 3, 4])
y = np.array([1, 3, 2, 3])

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
  eqn+=1

for i in range(1,3):
  dx = x[i] - x[i-1]

  A[eqn, 4*(i-1)+1] = 1
  A[eqn, 4*(i-1)+2] = 2*dx
  A[eqn, 4*(i-1)+3] = 3*dx**2

  A[eqn,4*i+1] = -1
  eqn+=1

for i in range(1,3):
    dx = x[i] - x[i-1]

    A[eqn, 4*(i-1)+2] = 2
    A[eqn, 4*(i-1)+3] = 6*dx

    A[eqn, 4*i+2] = -2
    
    eqn+=1

A[eqn, 2] = 2
eqn+=1

dx = x[3] - x[2]
A[eqn, 10] = 2
A[eqn, 11] = dx*6
eqn+=1

coeffs = np.linalg.solve(A,B)

def computePoly(i, xi):
  a = coeffs[4*i]
  b = coeffs[4*i+1]
  c = coeffs[4*i+2]
  d = coeffs[4*i+3]

  return a + b*(xi-x[i]) + c*(xi-x[i])**2 + d*(xi-x[i])**3

X,Y =[],[]

for i in range(3):
  xi_vals = np.linspace(x[i], x[i+1], 50)
  yi_vals = computePoly(i, xi_vals)
  X.extend(xi_vals)
  Y.extend(yi_vals)

plt.plot(X,Y,label="Cubic Spline Interpolation")
plt.scatter(x, y, color = 'blue')
plt.legend()
plt.show()
