import numpy as np
import cvxpy as cp
import time

# MPC Parameters
T = 300 # Total Steps
N = 30 # Prediction Horizon
dt = 0.05 # Seconds
lookahead_skip = 3 # Amount by which to decrease the resolution of MPC prediction. A value of 1 means normal MPC
# At an N-to-skip ratio of 10:1, with a dt of 0.05, processing usually keeps up with real time.

# Weights
Q = np.diag([30, 30, 1]) # Penalizing bx, by, theta
Q_N = 200*Q # Making a powerful terminal cost
R = 0.2*np.diag([1, 1, 1, 1]) # Small penalty on all controls

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


def rect_edge_point(width, height, phi):
    phi = np.asarray(phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    dx = np.where(cos_phi != 0, np.abs(width / 2 / cos_phi), np.inf)
    dy = np.where(sin_phi != 0, np.abs(height / 2 / sin_phi), np.inf)
    t = np.minimum(dx, dy)
    x = t * cos_phi
    y = t * sin_phi
    normal_x = np.where(dx < dy, np.copysign(1, -cos_phi), 0)
    normal_y = np.where(dx >= dy, np.copysign(1, -sin_phi), 0)
    tangential_x = np.where(dx < dy, 0, np.copysign(1, sin_phi))
    tangential_y = np.where(dx >= dy, 0, np.copysign(1, -cos_phi))
    normal = np.stack((normal_x, normal_y), axis=-1)
    tangential = np.stack((tangential_x, tangential_y), axis=-1)
    return x, y, normal, tangential

def get_B(x, y, normal, tangential):
    x = np.asarray(x)
    y = np.asarray(y)
    C = x.shape[0]
    J = np.zeros((C, 2, 3))
    J[:, 0, 0] = 1
    J[:, 1, 1] = 1
    J[:, 0, 2] = -y
    J[:, 1, 2] = x
    N = np.einsum('cij,cj->ci', J.transpose(0, 2, 1), normal)
    T = np.einsum('cij,cj->ci', J.transpose(0, 2, 1), tangential)
    B = np.hstack((N.T, T.T))
    return B

def rotmat(t):
    ct = np.cos(t)
    st = np.sin(t)
    return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def get_B_k(x, b, B_L):
    r = rotmat(x[2])
    rl = r @ B_L
    rlb = rl @ b
    return rlb

def cost_function(x, x_ref, u, u_ref):
    x_diff = x_ref - x
    u_diff = u_ref - u
    return x_diff.T @ Q @ x_diff + u_diff.T @ R @ u_diff

def controlclosedloop(pusher_plan, x_ref_values):
    # State: [bx, by, theta]  
    x_init = np.array([0.4, 0.3, 0]).reshape(-1, 1)
    nx = len(x_init)
    # Input: [f_n1, f_n2, f_t1, f_t2] 
    nu = 4

    # Variables
    x = cp.Variable((nx, N+1))  # State trajectory
    u = cp.Variable((nu, N))    # Control trajectory
    x_ref = cp.Parameter((nx, N+1))  # Reference trajectory
    x0 = cp.Parameter(nx)  # Initial state
    
    u_ref = np.vstack([
        0, 0, 0, 0  # force reference
    ])

    # Initialize trajectories (State outputs are augmented x; internal dynamics don't directly consider phi)
    x_trajectory = [x_init]
    y_trajectory = [list(x_init.flatten()) + [pusher_plan[0]["phi1"], pusher_plan[0]["phi2"]]]
    u_trajectory = []
    cost_trajectory = []

    # MPC Loop
    start_time = time.time()
    for t in range(T):
        if T-t-1 < N:
            horizon = max(1,T-t-1) # Truncate horizon if approaching the end
            x = cp.Variable((nx, horizon+1))
            u = cp.Variable((nu, horizon))
            x_ref = cp.Parameter((nx, horizon+1))
        else:
            horizon = N
        x0.value = x_trajectory[-1].flatten() # Get latest state
        theta_now = x0.value[2]
        theta_ref_horizon = x_ref_values[2, t:t+horizon+1]
        theta_error = angle_diff(theta_ref_horizon, theta_now)
        x_ref.value = np.vstack([
            x_ref_values[0, t:t+horizon+1], x_ref_values[1, t:t+horizon+1],
            theta_now + theta_error # Angle Wrapping Countermeasure
        ])

        constraints = [x[:, 0] == x0]
        cost = 0

        # Horizon Loop
        for k in range(0, horizon - lookahead_skip + 1, lookahead_skip):
            global_time = t + k

            if global_time >= len(pusher_plan):
                global_time = len(pusher_plan) - 1  # Safety bound

            plan = pusher_plan[global_time]
            active1 = plan["active1"]
            active2 = plan["active2"]
            phi1 = plan["phi1"]
            phi2 = plan["phi2"]

            if not active1:
                constraints += [
                    u[0, k] == 0,  # No normal from pusher 1
                    u[2, k] == 0   # No tangential from pusher 1
                ]
            if not active2:
                constraints += [
                    u[1, k] == 0,  # No normal from pusher 2
                    u[3, k] == 0   # No tangential from pusher 2
                ]
            
            # Pusher positions and directions from phi values
            px, py, normal, tangential = rect_edge_point(B_WIDTH, B_HEIGHT, [phi1, phi2])
            # B matrix
            b = get_B(px, py, normal, tangential)
            B_k = get_B_k(x0.value, b, B_L)
            
            # Dynamics Update
            constraints.append(x[:, k+lookahead_skip] == x[:, k] + dt * (B_k @ u[:, k]) * lookahead_skip)

            # Input constraints
            constraints.append(cp.norm(u[:, k], 'inf') <= f_max)  # Force limits
            constraints.append(u[:2, k] >= 0)  # Normal Force limits
            constraints.append(cp.abs(u[2,k]) <= b_mu_pusher * u[0,k]) # Force Cone
            constraints.append(cp.abs(u[3,k]) <= b_mu_pusher * u[1,k]) # Force Cone

            # Cost function
            cost += cp.quad_form(x[:, k] - x_ref[:, k], Q) + cp.quad_form(u[:, k] - u_ref.flatten(), R)

        # Add terminal cost
        cost += cp.quad_form(x[:, horizon] - x_ref[:, horizon], Q_N)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # prob.solve(solver=cp.ECOS) # fails less often
        prob.solve()

        # Apply first control input and update system
        if u[:, 0].value is not None:
            u_opt = u[:, 0].value.reshape(-1, 1)
        else:
            u_opt = np.array([0,0,0,0]).reshape(-1,1)
        x_new = x_trajectory[-1] + dt * (B_k @ u_opt)
        y_new = list(x_new.flatten()) + [pusher_plan[t]["phi1"], pusher_plan[t]["phi2"]]
        x_trajectory.append(x_new)
        y_trajectory.append(y_new)
        u_trajectory.append(u_opt)
        cost_trajectory.append(cost_function(x_new, x_ref.value[:,:1], u_opt, u_ref).flatten())
        if np.mod(t,1//dt) == 0:
            print(f"Completed Trajectory at t = {t} / {T} ({t*dt:.2f} sec in sim, {time.time() - start_time:.2f} sec real processing time)")

    return y_trajectory, u_trajectory, x_ref_values, cost_trajectory