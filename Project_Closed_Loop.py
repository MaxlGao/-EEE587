import numpy as np
import cvxpy as cp
from Project_MPC import rect_edge_point, get_B, rotmat, angle_diff, get_B_k
import time
import itertools
from multiprocessing import Pool
import concurrent.futures

# MPC Parameters
T = 300 # Total Steps
N = 30 # Prediction Horizon
dt = 0.05 # Seconds
lookahead_skip = 3 # Amount by which to decrease the resolution of MPC prediction. A value of 1 means normal MPC
# At an N-to-skip ratio of 10:1, with a dt of 0.05, processing keeps up with real time.

# Weights
Q = np.diag([30, 30, 1, 0, 0]) # Penalizing bx, by, theta (not contact point locations)
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

# Mode Scheduling
n_buckets = 8
bucket_size = T // n_buckets

def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def controlclosedloop(pusher2_active_plan):
    # State: [bx, by, theta, phi1, phi2]  
    x_init = np.array([0.4, 0.3, 0, 21*np.pi/12, 15*np.pi/12]).reshape(-1, 1)
    nx = len(x_init)
    # Input: [f_n1, f_n2, f_t1, f_t2] 
    nu = 4

    # Variables
    x = cp.Variable((nx, N+1))  # State trajectory
    u = cp.Variable((nu, N))    # Control trajectory
    x_ref = cp.Parameter((nx, N+1))  # Reference trajectory
    x0 = cp.Parameter(nx)  # Initial state
    
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

    u_ref_values = np.vstack([
        0, 0, 0, 0  # force reference
    ])

    # Initialize trajectories
    x_trajectory = [x_init]
    u_trajectory = []

    # Precompute pusher positions and directions from initial conditions (since pusher location doesn't change)
    px, py, normal, tangential = rect_edge_point(B_WIDTH, B_HEIGHT, x_init[3:].flatten())
    # Precompute B matrix
    b = get_B(px, py, normal, tangential)

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
            x_ref_values[0, t:t+horizon+1],
            x_ref_values[1, t:t+horizon+1],
            theta_now + theta_error, # Angle Wrapping Countermeasure
            x_ref_values[3, t:t+horizon+1],
            x_ref_values[4, t:t+horizon+1]
        ])
        B_k = get_B_k(x0.value, b, B_L)

        constraints = [x[:, 0] == x0]
        cost = 0

        # Horizon Loop
        for k in range(0, horizon - lookahead_skip + 1, lookahead_skip):
            global_time = t + k

            # Get current mode bucket
            bucket_idx = min(global_time // bucket_size, n_buckets - 1)
            pusher2_active = pusher2_active_plan[bucket_idx]

            # Shut off second pusher if inactive in this bucket
            if not pusher2_active:
                constraints += [
                    u[1, k] == 0,  # No normal from pusher 2
                    u[3, k] == 0   # No tangential from pusher 2
                ]
            
            # Dynamics Update
            constraints.append(x[:, k+lookahead_skip] == x[:, k] + dt * (B_k @ u[:, k]) * lookahead_skip)

            # Input constraints
            constraints.append(cp.norm(u[:, k], 'inf') <= f_max)  # Force limits
            constraints.append(u[:2, k] >= 0)  # Normal Force limits
            constraints.append(cp.abs(u[2,k]) <= b_mu_pusher * u[0,k]) # Force Cone
            constraints.append(cp.abs(u[3,k]) <= b_mu_pusher * u[1,k]) # Force Cone

            # Cost function
            cost += cp.quad_form(x[:, k] - x_ref[:, k], Q) + cp.quad_form(u[:, k] - u_ref_values.flatten(), R)

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
        x_trajectory.append(x_new)
        u_trajectory.append(u_opt)
        # if np.mod(t,1//dt) == 0:
        #     print(f"Completed Trajectory at t = {t} / {T} ({t*dt:.2f} sec in sim, {time.time() - start_time:.2f} sec real processing time)")
        #     print(f"Horizon Cost: {prob.value:.3f}")
        #     u_opt = u_opt.flatten()
        #     print(f"Pusher 1: {u_opt[0]:.2f} norm, {u_opt[2]: .2f} tang.")
        #     print(f"Pusher 2: {u_opt[1]:.2f} norm, {u_opt[3]: .2f} tang.") 

    return x_trajectory, u_trajectory, x_ref_values

def generate_all_mode_schedules(n_buckets):
    return list(itertools.product([False, True], repeat=n_buckets))

def run_experiment(pusher2_active_plan, label=None):
    # Run controller
    x_traj, u_traj, x_ref = controlclosedloop(pusher2_active_plan)
    # Compute total cost or final error (your metric of interest)
    pos_error = sum(np.linalg.norm(x[:2] - x_ref[:2, i].reshape(-1,1)) for i, x in enumerate(x_traj)) # distance integral from reference
    ang_error = sum(np.abs(angle_diff(x[2], x_ref[2, i])) for i, x in enumerate(x_traj)) # rotation integral from reference
    ang_error = ang_error[0]

    # Compute total force usage per pusher (sum of force vector magnitudes)
    force_usage = np.zeros(2)
    for u in u_traj:
        u = u.flatten()
        f1 = np.array([u[0], u[2]])  # [f_n1, f_t1]
        f2 = np.array([u[1], u[3]])  # [f_n2, f_t2]
        force_usage[0] += np.linalg.norm(f1)
        force_usage[1] += np.linalg.norm(f2)

    # Normalize over time
    pos_error *= dt
    ang_error *= dt
    force_usage *= dt

    # Save results
    result = {
        "label": label,
        "pusher2_active_plan": pusher2_active_plan,
        "x_traj": np.array(x_traj),
        "u_traj": np.array(u_traj),
        "x_ref": x_ref,
        "position_error": pos_error,
        "angular_error": ang_error,
        "force_usage": force_usage
    }

    return result


def run_all_experiments():
    all_plans = generate_all_mode_schedules(n_buckets)
    results = []
    for i, plan in enumerate(all_plans):
        label = f"plan_{i:04d}"
        print(f"\n>>> Running experiment {label} with schedule {plan}")
        start_time = time.time()
        result = run_experiment(plan, label)
        print(f"Done; took {time.time() - start_time:.2d} seconds")
        results.append(result)
        np.savez(f"results/{label}.npz", **result)

    # experiments = {
    #     "All On":        [True]*10,
    #     "2 of 3 On":        [True, True, False, True, True, False, True, True, False, True],
    #     "1 of 2 On":        [True, False, True, False, True, False, True, False, True, False],
    #     "1 of 3 On":        [True, False, False, True, False, False, True, False, False, True],
    #     "All Off":       [False]*10,
    # }

    # results = []
    # for label, plan in experiments.items():
    #     print(f"\n>>> Running experiment: {label}")
    #     result = run_experiment(plan, label)
    #     results.append(result)
    #     np.savez(f"results/{label.replace(' ', '_')}.npz", **result)

    return results

def single_run(args):
    idx, plan = args
    label = f"plan_{idx:04d}"
    print(f"Running {label}")
    result = run_experiment(plan, label)
    np.savez(f"results/{label}.npz", **result)
    return result

def run_all_experiments_parallel():
    all_plans = generate_all_mode_schedules(n_buckets)

    results = []
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(single_run, enumerate(all_plans))):
            results += result
            if np.mod(i, 50) == 49:
                print(f"Evaluated {i+1:3d} experiments of {len(all_plans)}. Running Time: {time.time() - start_time:.2f}")

    # with Pool() as pool:
    #     results = pool.map(single_run, enumerate(all_plans))

    return results

def summarize_results(results, sort_by="position_error"):
    print(f"\n===== Sorted by {sort_by} =====")
    results_sorted = sorted(results, key=lambda r: r[sort_by])
    for res in results_sorted:
        print(f"{res['label']:<10} | Pos Err: {res['position_error']:.4f} | "
              f"Ang Err: {res['angular_error']:.4f} | "
              f"Force 1/2: {res['force_usage'][0]:.2f}, {res['force_usage'][1]:.2f}")


def load_and_play(label):
    from Project_Visualize import playback
    data = np.load(f"results/{label.replace(' ', '_')}.npz", allow_pickle=True)
    playback(data['x_traj'], data['u_traj'], data['x_ref'], B_WIDTH, B_HEIGHT, dt)


if __name__ == "__main__":
    results = run_all_experiments()
    summarize_results(results)
    load_and_play("All Off")
