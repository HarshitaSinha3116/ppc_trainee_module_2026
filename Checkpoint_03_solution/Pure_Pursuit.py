import numpy as np

def pure_pursuit_steering(x, y, yaw, v, waypoints, L=2.5, k_dd=0.5, ld_min=2.0, max_steer=np.radians(30)):
  ld = k_dd * v + ld_min

  rear_wheel = np.array([x, y])

  distances = np.linalg.norm(waypoints - rear_wheel, axis=1)
  closest_idx = np.argmin(distances)

  def find_lookahead_point_idx(x, y, waypoints, current_idx, ld):
    for i in range(current_idx, len(waypoints)):
      distance_from_car_to_waypoint = np.linalg.norm(waypoints[i] - np.array([x,y]))
      if distance_from_car_to_waypoint >=ld:
        return i
    return len(waypoints) - 1 

  lookahead_idx = find_lookahead_point_idx(x, y, waypoints, closest_idx, ld)
  target_x = waypoints[lookahead_idx][0]
  target_y = waypoints[lookahead_idx][1]


  def compute_alpha(x, y, yaw, target_x, target_y):
    alpha = np.arctan2(target_y - y, target_x - x) - yaw
    return alpha

  alpha = compute_alpha(x, y, yaw, target_x, target_y)

  def compute_steering_angle(L,ld, alpha, max_steer):
    steer = np.arctan2(2 * L * np.sin(alpha), ld)
    steer = np.clip(steer, -max_steer, max_steer) 
    return steer

  steer = compute_steering_angle(L, ld, alpha, max_steer)
  return steer, lookahead_idx 
