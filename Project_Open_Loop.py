import numpy as np
from Project_MPC import rect_edge_point, get_B, rotmat, angle_diff, get_B_k

T = 300 # Total Steps
dt = 0.05
# Body Parameters
B_WIDTH = 0.1 # m
B_HEIGHT = 0.05
b_mu_ground = 0.35
b_mu_pusher = 0.3
b_mass = 0.8
g = 9.81
# Maximum force before slipping
f_max = b_mu_ground*b_mass*g 
# Maximum moment before slipping
m_max = b_mu_ground*b_mass*g*B_WIDTH*B_HEIGHT*8
# Ellipsoid defining the limit surface
B_L = np.diag([2/f_max**2, 2/f_max**2, 2/m_max**2]) 
print(f"Identified maximum force of {f_max:.3f}, maximum moment of {m_max:.3f}")

# Mode Scheduling
n_buckets = 8

def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def controlopenloop():
    # State: [bx, by, theta, phi1, phi2]  
    x_init = np.array([0.4, 0.3, 0, 21*np.pi/12, 15*np.pi/12]).reshape(-1, 1)
    nx = len(x_init)
    # Input: [f_n1, f_n2, f_t1, f_t2] 
    nu = 4

    # Generate reference trajectory (circular path)
    t_ref = np.linspace(0, 2*np.pi, T+1)

    # Circular Reference
    x_ref_values = np.vstack([
        0.3 + 0.1 * np.cos(t_ref),  # px reference (circle center 0.3,0.3 with radius 0.1)
        0.3 + 0.1 * np.sin(t_ref),  # py reference
        # np.full(T+1,0), # Zero theta reference
        np.arctan2(0.1 * np.sin(t_ref), 0.1 * np.cos(t_ref)),  # theta reference (tangent to circle)
        np.full(T+1,0), np.full(T+1,0)  # pusher angles (irrelevant)
    ])
    
    # Linear Reference
    # x_ref_values = np.vstack([
    #     0.4 + 0.03 * t_ref,  # px reference (Line starting at 0.4, 0.3)
    #     0.3 + 0.03 * t_ref,  # py reference
    #     np.full(T+1,0), # Zero theta reference
    #     # np.arctan2(-0.1 * np.sin(t_ref), 0.1 * np.cos(t_ref)),  # theta reference spinning
    #     np.full(T+1,0), np.full(T+1,0)  # pusher angles (irrelevant)
    # ])

    # Initialize trajectories
    x_trajectory = [x_init]
    u_trajectory = []

    # Precompute pusher positions and directions from initial conditions (since pusher location doesn't change)
    px, py, normal, tangential = rect_edge_point(B_WIDTH, B_HEIGHT, x_init[3:].flatten())
    # Precompute B matrix
    b = get_B(px, py, normal, tangential)

    # MPC Loop
    for t in range(T):
        x0 = x_trajectory[-1].flatten() # Get latest state
        B_k = get_B_k(x0, b, B_L)
        u_opt = np.array([0.1,0.2,0.3,0.4]).reshape(-1,1)
        x_new = x_trajectory[-1] + dt * (B_k @ u_opt)
        x_trajectory.append(x_new)
        u_trajectory.append(u_opt)

    return x_trajectory, u_trajectory, x_ref_values

if __name__ == "__main__":
    from Project_Visualize import playback
    x_traj, u_traj, x_ref = controlopenloop()
    playback(x_traj, u_traj, x_ref, B_WIDTH, B_HEIGHT, dt)
