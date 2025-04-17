import numpy as np
import os
import matplotlib.pyplot as plt

def load_all_results(results_dir="results"):
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(results_dir, fname), allow_pickle=True)
            results.append({
                "label": data["label"].item(),
                "pusher2_active_plan": data["pusher2_active_plan"],
                "position_error": data["position_error"].item(),
                "angular_error": data["angular_error"].item(),
                "force_usage": data["force_usage"]
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
        active_stats[pct_active].append(r["position_error"])

    # Prepare data
    active_pcts = sorted(active_stats.keys())
    avg_errors = [np.mean(active_stats[p]) for p in active_pcts]
    min_errors = [np.min(active_stats[p]) for p in active_pcts]

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(active_pcts, avg_errors, marker='o', label="Average Position Error")
    plt.plot(active_pcts, min_errors, marker='s', label="Best Position Error")
    plt.xlabel("Pusher 2 Active Time (%)")
    plt.ylabel("Position Error (m-s)")
    plt.title("Position Error vs. Pusher 2 Active Time")
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
        if pct_active not in best_plans or r["position_error"] < best_plans[pct_active]["position_error"]:
            best_plans[pct_active] = {"plan": plan, "position_error": r["position_error"]}

    # Sort by increasing activity
    sorted_pcts = sorted(best_plans.keys())
    plans = [best_plans[p]["plan"] for p in sorted_pcts]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(plans) * 0.5))
    for row_idx, plan in enumerate(plans):
        for col_idx, is_active in enumerate(plan):
            color = "black" if is_active else "white"
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, facecolor=color, edgecolor="gray")
            ax.add_patch(rect)

    ax.set_xlim(0, bucket_count)
    ax.set_ylim(0, len(plans))
    ax.set_xticks(range(bucket_count))
    ax.set_yticks([i + 0.5 for i in range(len(plans))])
    ax.set_yticklabels([f"{int(p)}%" for p in sorted_pcts])
    ax.set_xlabel("Schedule Bucket")
    ax.set_ylabel("Pusher 2 Active Time")
    ax.set_title("Best Plan per Activity Level")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = load_all_results()
    summarize_results(results)
    plot_error_distribution(results)
    plot_force_scatter(results)
    plot_error_vs_active_time(results)
    plot_best_schedules(results)
