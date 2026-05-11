import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class Drone:
    def __init__(self):
        self.height = 0.0
        self.velocity = 0.0

    def update(self, thrust, dt):
        gravity = -9.81
        mass = 1.0
        acceleration = (thrust / mass) + gravity
        self.velocity += acceleration * dt
        self.height += self.velocity * dt
        return self.height

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.previous_error = 0
        # you might need to add more variables here... hint: for Integral controller

    def update(self, error, dt):
        P_term = self.Kp * error

        self.integral_error += error * dt
        I_term = self.Ki * self.integral_error

        derivative_error = (error - self.previous_error) / dt
        D_term = self.Kd * derivative_error

        self.previous_error = error

        thrust = P_term + I_term + D_term
        return thrust

Kp = 5.0   
Ki = 10.0   
Kd = 6.0   


target_height = 10.0    # target height of drone
sim_time = 10    # increase this if you want the animation to be longer
dt = 0.05
steps = int(sim_time / dt)

drone = Drone()
pid = PIDController(Kp, Ki, Kd)

heights = []
times = []

for i in range(steps):
    t = i * dt

    error =  target_height - drone.height

    thrust = pid.update(error, dt)
    h = drone.update(thrust, dt)
    times.append(t)
    heights.append(h)

fig, ax = plt.subplots(figsize=(4, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(0, 15)   # increase the 2nd argument of ylim if you wanna see further above the setpoint
ax.set_xlabel("Drone")
ax.set_ylabel("Height (m)")
ax.set_title("Drone Height Stabilization (PID Control)")

drone_body, = ax.plot([], [], 'bo', markersize=15)
target_line = ax.axhline(y=target_height, color='r', linestyle='--', label='Target Height')
ax.legend()

def init():
    drone_body.set_data([], [])
    return drone_body,

def update(frame):
    x = 0
    y = heights[frame]
    drone_body.set_data([x], [y])
    return drone_body,

ani = animation.FuncAnimation(
    fig, update, frames=len(heights), init_func=init,
    interval=dt*800, blit=True
)

plt.close(fig)
HTML(ani.to_jshtml())
