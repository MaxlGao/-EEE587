import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Project_MPC import B_WIDTH, B_HEIGHT, dt

def load_all_results(results_dir):
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(results_dir, fname), allow_pickle=True)
            # pos_error = data["position_error"].item()
            # ang_error = data["angular_error"].item()
            # force_error = np.linalg.norm(data["force_usage"])
            results.append({
                "label": data["label"].item(),
                "pusher_plan": data["pusher_plan"],
                "y_traj": data["y_traj"],
                "u_traj": data["u_traj"],
                "x_ref": data["x_ref"],
                "cost_traj": data["cost_traj"],
                "position_error": data["position_error"].item(),
                "angular_error": data["angular_error"].item(),
                "force_usage": data["force_usage"],
                "total_cost": data["total_cost"]
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

def plot_cost_trajs(results, buckets=8):
    total_steps = len(results[0]["pusher_plan"])

    # Compute % active and organize by that
    grouped = {}
    for r in results:
        plan = r["pusher_plan"]
        active_count = sum(p['active2'] for p in plan)
        pct_active = round((active_count / len(plan)) * buckets) * 100 / buckets
        if pct_active not in grouped:
            grouped[pct_active] = []
        grouped[pct_active].append(r)

    sorted_keys = sorted(grouped.keys())
    # color_map = cm.get_cmap('viridis', len(sorted_keys))
    # color_dict = {k: color_map(i) for i, k in enumerate(sorted_keys)}

    # plt.figure(figsize=(8,5))
    # for pct_active, group in grouped.items():
    #     color = color_dict[pct_active]
    #     for r in group:
    #         plt.plot(range(total_steps), r["cost_traj"], color=color, alpha=0.6)
    
    # # Custom legend
    # legend_handles = [
    #     plt.Line2D([0], [0], color=color_dict[k], label=f"{k:.1f}% active")
    #     for k in sorted_keys
    # ]
    # plt.legend(handles=legend_handles, title="Pusher 2 Active Time")

    # plt.xlabel("Time Steps")
    # plt.ylabel("Cost incurred per time step")
    # plt.title("Cost Trajectories Grouped by Pusher 2 Activation %")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # Subplotting method
    n_cols = int(np.ceil(np.sqrt(buckets+1)))
    n_rows = int(np.ceil((buckets+1) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()  # For easy indexing
    # Plot each group in its subplot
    for i, pct_active in enumerate(sorted_keys):
        ax = axs[i]
        group = grouped[pct_active]
        for r in group:
            ax.plot(range(total_steps), r["cost_traj"], alpha=0.6)
        ax.set_title(f"{pct_active:.1f}% Active")
        ax.grid(True)
        ax.set_xticks(np.arange(0, total_steps+1, total_steps // buckets))
        if i % n_cols == 0:
            ax.set_ylabel("Cost per step")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("Time Steps")

    # Hide any unused subplots
    for j in range(buckets+1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Cost Trajectories by Pusher 2 Active Percentage", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()

def plot_trajectories_subplots(results, buckets=8):
    # Group by rounded pusher 2 activation %
    grouped = {}
    for r in results:
        plan = r["pusher_plan"]
        active_count = sum(p['active2'] for p in plan)
        pct_active = round((active_count / len(plan)) * buckets) * 100 / buckets
        if pct_active not in grouped:
            grouped[pct_active] = []
        grouped[pct_active].append(r)

    sorted_keys = sorted(grouped.keys())
    num_plots = len(sorted_keys)
    
    # Determine subplot grid layout
    n_cols = int(np.ceil(np.sqrt(num_plots)))
    n_rows = int(np.ceil(num_plots / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, pct_active in enumerate(sorted_keys):
        ax = axs[i]
        for r in grouped[pct_active]:
            traj = np.array(r["y_traj"])  # shape (T, 5): x, y, theta, phi1, phi2
            x_ref = np.array(r["x_ref"])  # shape (T, 3): x, y, theta

            ax.plot(x_ref[:,0], x_ref[:,1], '--', color='gray', alpha=0.7, label='Ref' if 'Ref' not in ax.get_legend_handles_labels()[1] else "")
            ax.plot(traj[:,0], traj[:,1], alpha=0.8)

        ax.set_title(f"{pct_active:.1f}% Active")
        ax.set_aspect('equal')
        ax.grid(True)
        if i % n_cols == 0:
            ax.set_ylabel("Y Position")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("X Position")

    # Hide any unused subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Object Trajectories by Pusher 2 Activation %", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_error_vs_active_time(results, buckets=8):
    # Count active time per plan (as % of total buckets)
    active_stats = {}
    for r in results:
        plan = r["label"] if isinstance(r["label"], list) else r["pusher_plan"]
        if not isinstance(plan, (list, np.ndarray)): continue  # skip if label isn't a plan
        active_count = sum([p['active2'] for p in plan])
        pct_active = round((active_count / len(plan)) * buckets) * 100 / buckets
        if pct_active not in active_stats:
            active_stats[pct_active] = []
        active_stats[pct_active].append(r["total_cost"])
        # cost = r["total_cost"]
        # print(f"Plan {r['label']}, which is {pct_active}% pusher 2, has cost {cost}")

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
    plt.xticks(np.arange(0, 101, 100 / buckets))
    plt.xlabel("Pusher 2 Active Time (%)")
    plt.ylabel("Total Cost J (Averaged over T)")
    plt.title("Total Cost vs. Pusher 2 Active Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_best_worst_schedules(results, best=True, buckets=8):
    # Group by active % and track best (lowest position error) plan per group
    best_plans = {}
    worst_plans = {}
    for r in results:
        plan = r["label"] if isinstance(r["label"], list) else r["pusher_plan"]
        if not isinstance(plan, (list, np.ndarray)): continue
        active_count = sum([p['active2'] for p in plan])
        pct_active = round((active_count / len(plan)) * buckets) * 100 / buckets
        if pct_active not in best_plans or r["total_cost"] < best_plans[pct_active]["total_cost"]:
            best_plans[pct_active] = {"plan": plan, "total_cost": r["total_cost"], "label": r["label"]}
        if pct_active not in worst_plans or r["total_cost"] > worst_plans[pct_active]["total_cost"]:
            worst_plans[pct_active] = {"plan": plan, "total_cost": r["total_cost"], "label": r["label"]}

    # Sort by increasing activity
    if best:
        sorted_best_pcts = sorted(best_plans.keys())
        plans = [best_plans[p]["plan"] for p in sorted_best_pcts]
        for percent in best_plans:
            bp = best_plans[percent]
            print(f"Best plan for percent {percent} is {bp['label']}")
    else:
        sorted_worst_pcts = sorted(worst_plans.keys())
        plans = [worst_plans[p]["plan"] for p in sorted_worst_pcts]
        for percent in worst_plans:
            wp = worst_plans[percent]
            print(f"Worst plan for percent {percent} is {wp['label']}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(plans) * 0.5))
    for row_idx, plan in enumerate(plans):
        for col_idx, p in enumerate(plan):
            is_active = p['active2']
            color = "black" if is_active else "white"
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, facecolor=color)
            ax.add_patch(rect)

    ax.set_xlim(0, len(plan))
    ax.set_ylim(0, len(plans))
    # ax.set_xticks([i + 0.5 for i in range(bucket_count)])
    # ax.set_xticklabels([i + 1 for i in range(bucket_count)])
    ax.set_yticks([i + 0.5 for i in range(len(plans))])
    if best:
        ax.set_yticklabels([f"{int(p)}%" for p in sorted_best_pcts])
    else:
        ax.set_yticklabels([f"{int(p)}%" for p in sorted_worst_pcts])
    ax.set_xlabel("Schedule Bucket")
    ax.set_ylabel("Pusher 2 Active Time")
    if best:
        ax.set_title("Best Plan per Activity Level")
    else:
        ax.set_title("Worst Plan per Activity Level")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def load_and_play(label, speed, results_dir):
    from Project_Visualize import playback
    data = np.load(f"{results_dir}/{label.replace(' ', '_')}.npz", allow_pickle=True)
    playback(data['x_traj'], data['u_traj'], data['x_ref'], B_WIDTH, B_HEIGHT, dt, speed)

if __name__ == "__main__":
    results_dir = "results/batch_bucket_schedules_1"
    results = load_all_results(results_dir)
    # summarize_results(results)
    # plot_error_distribution(results)
    # plot_force_scatter(results)
    # plot_cost_trajs(results)
    plot_trajectories_subplots(results)
    # plot_error_vs_active_time(results)
    # plot_best_worst_schedules(results, best=True)
    # plot_best_worst_schedules(results, best=False)


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

    # load_and_play("plan_0241", 2, results_dir)
    # Best plan for percent 62.5 is plan_0087
    # Worst plan for percent 62.5 is plan_0241

    # Best plan for percent 75.0 is plan_0095
    # Worst plan for percent 75.0 is plan_0249

    # Best plan for percent 87.5 is plan_0127
    # Worst plan for percent 87.5 is plan_0254

    # Best plan for percent 100.0 is plan_0255
    # Worst plan for percent 100.0 is plan_0255