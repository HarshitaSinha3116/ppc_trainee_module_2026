import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from utilities import PIDController, pure_pursuit_steering

waypoints = np.load('waypoints.npy')
blue_cones = np.load('blue_cones.npy')
yellow_cones = np.load('yellow_cones.npy')
# This is the map which was obtained from optimising the waypoints

class Vehicle:
    def __init__(self, x=0, y=0, yaw=0 , v=0.0, L=2.5):  # Initial position, yaw, velocity, and wheelbase

        self.x = x              #  IMP ! : Change the initial pose of the car to determine the starting position
        self.y = y  
        self.yaw = yaw
        self.v = v
        self.L = L  # Wheelbase

    def update(self, throttle, delta, dt=0.1):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(delta) * dt
        self.v += throttle * dt
        self.v = max(0.0, self.v)  # No reverse
    

    pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5)
    def compute_control(self, waypoints, target_idx, k_dd=0.5, ld_min=2.0):
      
    # Limit steering to realistic bounds
      max_steer = np.radians(30)

    # Use the imported Pure Pursuit steering function
      steer, target_idx = pure_pursuit_steering(
        self.x, self.y, self.yaw, self.v,
        waypoints, L=self.L, k_dd=k_dd, ld_min=ld_min, max_steer=max_steer
    )

    # Throttle control using PID
      target_v = 10.0  # Target velocity - retune gains if you change this
      speed_error = target_v - self.v
      dt = 0.1
      throttle = self.pid.update(speed_error, dt)

      return throttle, steer, target_idx


# Set the starting pose to the first waypoint, heading toward the second
start_yaw = np.arctan2(waypoints[1,1] - waypoints[0,1], waypoints[1,0] - waypoints[0,0])
vehicle = Vehicle(x=waypoints[0,0], y=waypoints[0,1], yaw=start_yaw)
history = {'x': [], 'y': []}
target_idx = 0

fig, ax = plt.subplots(figsize=(8, 8))
track_line, = ax.plot(waypoints[:,0], waypoints[:,1], 'k--', label='Track')
ax.scatter(blue_cones[:,0], blue_cones[:,1], c='blue', s=30, label='Blue Cones')
ax.scatter(yellow_cones[:,0], yellow_cones[:,1], c='gold', s=30, label='Yellow Cones')
car_dot, = ax.plot([], [], 'ro', markersize=8, label='Car')
path_line, = ax.plot([], [], 'r-', linewidth=1, alpha=0.6, label='Driven Path')
ax.set_aspect('equal')
ax.legend()
ax.grid(True)
ax.set_title('PID Throttle + Pure Pursuit Steering Animation')

def init():
    car_dot.set_data([], [])
    path_line.set_data([], [])
    return car_dot, path_line

def animate(i):
    global target_idx
    throttle, steer, target_idx = vehicle.compute_control(waypoints, target_idx)
    vehicle.update(throttle, steer)

    history['x'].append(vehicle.x)
    history['y'].append(vehicle.y)

    car_dot.set_data([vehicle.x], [vehicle.y])
    path_line.set_data(history['x'], history['y'])
    return car_dot, path_line

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=50, blit=True)
plt.show()  # works in a plain script
plt.close()  # Prevent double display in some notebooks

from IPython.display import HTML
HTML(ani.to_jshtml())


plt.figure(figsize=(8, 8))
plt.plot(waypoints[:,0], waypoints[:,1], 'k--', label="Track")
plt.scatter(blue_cones[:,0], blue_cones[:,1], c='blue', s=30, label='Blue Cones')
plt.scatter(yellow_cones[:,0], yellow_cones[:,1], c='gold', s=30, label='Yellow Cones')
plt.axis("equal")
plt.title("Track with Cones")
plt.legend()
plt.grid()
plt.show()
