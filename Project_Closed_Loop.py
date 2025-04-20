import numpy as np
from Project_MPC import T, angle_diff, controlclosedloop
import time
import itertools
from concurrent.futures import ProcessPoolExecutor

# Mode Scheduling
n_buckets = 8
bucket_size = T // n_buckets

# Generate reference trajectory (circular path)
t_ref = np.linspace(0, 2*np.pi, T+1)
x_ref_values = np.vstack([
    0.3 + 0.1 * np.cos(t_ref),  # px reference (circle center 0.3,0.3 with radius 0.1)
    0.3 + 0.1 * np.sin(t_ref),  # py reference
    np.arctan2(0.1 * np.sin(t_ref), 0.1 * np.cos(t_ref))  # theta reference (tangent to circle)
])

def generate_all_buckets(n_buckets):
    # Returns list of lists of true/falses. Each corresponds to one schedule
    return list(itertools.product([False, True], repeat=n_buckets))

def make_plan_from_buckets(bucket_schedule, bucket_size, T, phi1=15*np.pi/12, phi2=21*np.pi/12):
    """
    Creates a full-length pusher plan compatible with controlclosedloop.

    Args:
        bucket_schedule (list of bool): Whether pusher 2 is active per bucket.
        bucket_size (int): How many time steps each bucket spans.
        T (int): Total number of time steps.
        phi1 (float): Fixed angle for pusher 1 (radians).
        phi2 (float): Fixed angle for pusher 2 (radians).
    
    Returns:
        List of dicts for each timestep with keys: 'active1', 'phi1', 'active2', 'phi2'
    """
    plan = []
    for t in range(T):
        bucket_idx = min(t // bucket_size, len(bucket_schedule) - 1)
        p2_active = bucket_schedule[bucket_idx]
        plan.append({
            "active1": True,
            "phi1": phi1,
            "active2": p2_active,
            "phi2": phi2
        })
    return plan

def run_experiment(pusher_plan, label=None):
    # Run controller
    y_traj, u_traj, x_ref, cost_traj = controlclosedloop(pusher_plan, x_ref_values)
    
    pos_error = sum(np.linalg.norm(y[:2] - x_ref[:2, i].reshape(-1,1)) for i, y in enumerate(y_traj)) # distance integral from reference
    ang_error = sum(np.abs(angle_diff(y[2], x_ref[2, i])) for i, y in enumerate(y_traj)) # rotation integral from reference
    total_cost = sum(cost_traj)

    # Compute total force usage per pusher (sum of force vector magnitudes)
    force_usage = np.zeros(2)
    for u in u_traj:
        u = u.flatten()
        f1 = np.array([u[0], u[2]])  # [f_n1, f_t1]
        f2 = np.array([u[1], u[3]])  # [f_n2, f_t2]
        force_usage[0] += np.linalg.norm(f1)
        force_usage[1] += np.linalg.norm(f2)

    # Average over all steps
    pos_error /= T
    ang_error /= T
    force_usage /= T
    total_cost /= T

    # Save results
    result = {
        "label": label,
        "pusher_plan": pusher_plan,
        "y_traj": np.array(y_traj), # [x, y, theta, phi1, phi2]
        "u_traj": np.array(u_traj),
        "x_ref": x_ref,
        "cost_traj": cost_traj,
        "position_error": pos_error,
        "angular_error": ang_error,
        "force_usage": force_usage,
        "total_cost": total_cost
    }

    return result

def single_run(args, results_dir="results/experiment_0"):
    idx, plan = args
    label = f"plan_{idx:04d}"
    print(f"\nRunning {label}")
    start_time = time.time()
    result = run_experiment(plan, label)
    print(f"Done with {label}; took {time.time() - start_time:.2f} seconds")
    np.savez(f"{results_dir}/{label}.npz", **result)

def run_all_experiments_buckets(start_idx=0, parallel=False, max_workers=12):
    all_plans = generate_all_buckets(n_buckets)
    jobs = []
    for i, bucket_plan in enumerate(all_plans[start_idx:], start=start_idx):
        plan = make_plan_from_buckets(bucket_plan, bucket_size, T)
        jobs.append((i, plan))
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(single_run, jobs)
    else:
        for args in jobs:
            single_run(args)

if __name__ == "__main__":
    run_all_experiments_buckets(parallel=True)
