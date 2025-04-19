import numpy as np
import os
import matplotlib.pyplot as plt
from Project_MPC import B_WIDTH, B_HEIGHT, dt, Q, R

# This script works only with an older version of result data

def load_all_results(results_dir):
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(results_dir, fname), allow_pickle=True)
            pos_error = data["position_error"].item()
            ang_error = data["angular_error"].item()
            force_error = np.linalg.norm(data["force_usage"])
            total_error = Q[0, 0] * pos_error**2 + Q[2,2] * ang_error**2 + R[0, 0] * force_error**2
            results.append({
                "label": data["label"].item(),
                "pusher2_active_plan": data["pusher2_active_plan"],
                "position_error": data["position_error"].item(),
                "angular_error": data["angular_error"].item(),
                "force_usage": data["force_usage"],
                "total_cost": total_error
            })
    return results

def summarize_results(results, top_n=10):
    results_sorted = sorted(results, key=lambda r: r["position_error"])
    print("\n=== Top Experiments by Position Error ===")
    for r in results_sorted[:top_n]:
        print(f"{r['label']:<10} | Pos Err: {r['position_error']:.4f} | "
              f"Ang Err: {r['angular_error']:.4f} | "
              f"Force: {r['force_usage'][0]:.2f}, {r['force_usage'][1]:.2f}")

def plot_error_distribution(results):
    pos_errors = [r["position_error"] for r in results]
    ang_errors = [r["angular_error"] for r in results]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(pos_errors, bins=30, alpha=0.7)
    plt.title("Position Error Distribution")
    plt.xlabel("m-s")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(ang_errors, bins=30, alpha=0.7, color='orange')
    plt.title("Angular Error Distribution")
    plt.xlabel("rad-s")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_force_scatter(results):
    f1 = [r["force_usage"][0] for r in results]
    f2 = [r["force_usage"][1] for r in results]
    pos_errors = [r["position_error"] for r in results]

    plt.scatter(f1, f2, c=pos_errors, cmap='viridis', s=40)
    plt.colorbar(label="Position Error (m-s)")
    plt.xlabel("Force Usage: Pusher 1 (N-s)")
    plt.ylabel("Force Usage: Pusher 2 (N-s)")
    plt.title("Force Usage vs. Position Error")
    plt.grid(True)
    plt.show()

def plot_error_vs_active_time(results, bucket_count=8):
    # Count active time per plan (as % of total buckets)
    active_stats = {}
    for r in results:
        plan = r["label"] if isinstance(r["label"], list) else r["pusher2_active_plan"]
        if not isinstance(plan, (list, np.ndarray)): continue  # skip if label isn't a plan
        active_count = sum(plan)
        pct_active = (active_count / bucket_count) * 100
        if pct_active not in active_stats:
            active_stats[pct_active] = []
        active_stats[pct_active].append(r["total_cost"])

    # Prepare data
    active_pcts = sorted(active_stats.keys())
    avg_errors = [np.mean(active_stats[p]) for p in active_pcts]
    min_errors = [np.min(active_stats[p]) for p in active_pcts]
    max_errors = [np.max(active_stats[p]) for p in active_pcts]

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(active_pcts, avg_errors, marker='o', label="Average Cost")
    plt.plot(active_pcts, min_errors, marker='s', label="Best Cost")
    plt.plot(active_pcts, max_errors, marker='x', label="Worst Cost")
    plt.xlabel("Pusher 2 Active Time (%)")
    plt.ylabel("Total Cost J (Averaged over T)")
    plt.title("Total Cost vs. Pusher 2 Active Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_best_schedules(results, bucket_count=8):
    # Group by active % and track best (lowest position error) plan per group
    best_plans = {}
    for r in results:
        plan = r["label"] if isinstance(r["label"], list) else r["pusher2_active_plan"]
        if not isinstance(plan, (list, np.ndarray)): continue
        active_count = sum(plan)
        pct_active = (active_count / bucket_count) * 100
        if pct_active not in best_plans or r["total_cost"] < best_plans[pct_active]["total_cost"]:
            best_plans[pct_active] = {"plan": plan, "total_cost": r["total_cost"], "label": r["label"]}

    # Sort by increasing activity
    sorted_pcts = sorted(best_plans.keys())
    plans = [best_plans[p]["plan"] for p in sorted_pcts]
    for percent in best_plans:
        bp = best_plans[percent]
        print(f"Best plan for percent {percent} is {bp['label']}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(plans) * 0.5))
    for row_idx, plan in enumerate(plans):
        for col_idx, is_active in enumerate(plan):
            color = "black" if is_active else "white"
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, facecolor=color, edgecolor="gray")
            ax.add_patch(rect)

    ax.set_xlim(0, bucket_count)
    ax.set_ylim(0, len(plans))
    ax.set_xticks([i + 0.5 for i in range(bucket_count)])
    ax.set_xticklabels([i + 1 for i in range(bucket_count)])
    ax.set_yticks([i + 0.5 for i in range(len(plans))])
    ax.set_yticklabels([f"{int(p)}%" for p in sorted_pcts])
    ax.set_xlabel("Schedule Bucket")
    ax.set_ylabel("Pusher 2 Active Time")
    ax.set_title("Best Plan per Activity Level")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_worst_schedules(results, bucket_count=8):
    # Group by active % and track best (lowest position error) plan per group
    worst_plans = {}
    for r in results:
        plan = r["label"] if isinstance(r["label"], list) else r["pusher2_active_plan"]
        if not isinstance(plan, (list, np.ndarray)): continue
        active_count = sum(plan)
        pct_active = (active_count / bucket_count) * 100
        if pct_active not in worst_plans or r["total_cost"] > worst_plans[pct_active]["total_cost"]:
            worst_plans[pct_active] = {"plan": plan, "total_cost": r["total_cost"], "label": r["label"]}

    # Sort by increasing activity
    sorted_pcts = sorted(worst_plans.keys())
    plans = [worst_plans[p]["plan"] for p in sorted_pcts]
    for percent in worst_plans:
        wp = worst_plans[percent]
        print(f"Worst plan for percent {percent} is {wp['label']}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(plans) * 0.5))
    for row_idx, plan in enumerate(plans):
        for col_idx, is_active in enumerate(plan):
            color = "black" if is_active else "white"
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, facecolor=color, edgecolor="gray")
            ax.add_patch(rect)

    ax.set_xlim(0, bucket_count)
    ax.set_ylim(0, len(plans))
    ax.set_xticks([i + 0.5 for i in range(bucket_count)])
    ax.set_xticklabels([i + 1 for i in range(bucket_count)])
    ax.set_yticks([i + 0.5 for i in range(len(plans))])
    ax.set_yticklabels([f"{int(p)}%" for p in sorted_pcts])
    ax.set_xlabel("Schedule Bucket")
    ax.set_ylabel("Pusher 2 Active Time")
    ax.set_title("Worst Plan per Activity Level")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def load_and_play(label, speed, results_dir):
    from Project_Visualize import playback
    data = np.load(f"{results_dir}/{label.replace(' ', '_')}.npz", allow_pickle=True)
    playback(data['x_traj'], data['u_traj'], data['x_ref'], B_WIDTH, B_HEIGHT, dt, speed)

if __name__ == "__main__":
    results_dir = "results/batch_bucket_schedules_0"
    results = load_all_results(results_dir)
    # summarize_results(results)
    # plot_error_distribution(results)
    # plot_force_scatter(results)
    # plot_error_vs_active_time(results)
    plot_best_schedules(results)
    plot_worst_schedules(results)


    # load_and_play("plan_0000", 2, results_dir)
    # Best plan for percent 0.0 is plan_0000
    # Worst plan for percent 0.0 is plan_0000

    # load_and_play("plan_0016", 2, results_dir)
    # Best plan for percent 12.5 is plan_0016
    # Worst plan for percent 12.5 is plan_0001

    # load_and_play("plan_0036", 2, results_dir)
    # load_and_play("plan_0003", 2, results_dir)
    # Best plan for percent 25.0 is plan_0036
    # Worst plan for percent 25.0 is plan_0003

    # Best plan for percent 37.5 is plan_0084
    # Worst plan for percent 37.5 is plan_0131

    # load_and_play("plan_0085", 2, results_dir)
    # Best plan for percent 50.0 is plan_0085
    # Worst plan for percent 50.0 is plan_0195

    load_and_play("plan_0241", 2, results_dir)
    # Best plan for percent 62.5 is plan_0087
    # Worst plan for percent 62.5 is plan_0241

    # Best plan for percent 75.0 is plan_0095
    # Worst plan for percent 75.0 is plan_0249

    # Best plan for percent 87.5 is plan_0127
    # Worst plan for percent 87.5 is plan_0254

    # Best plan for percent 100.0 is plan_0255
    # Worst plan for percent 100.0 is plan_0255