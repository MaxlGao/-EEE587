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

def summarize_results(results, top_n=50):
    results_sorted = sorted(results, key=lambda r: r["total_cost"])
    print("\n=== Top Experiments by Total Cost ===")
    for r in results_sorted[:top_n]:
        print(f"{r['label']:<10} | cost: {r['total_cost'][0]:.4f}")

def plot_force_scatter(results):
    f1 = [r["force_usage"][0] for r in results]
    f2 = [r["force_usage"][1] for r in results]
    total_cost = [r["total_cost"] for r in results]

    plt.scatter(f1, f2, c=total_cost, cmap='viridis', s=40)
    plt.colorbar(label="Total Cost")
    plt.xlabel("Pusher 1 Average Force")
    plt.ylabel("Pusher 2 Average Force")
    plt.title("Pusher 1 Force vs. Pusher 2 Force vs. Cost")
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
    # Subplotting method
    n_cols = int(np.ceil(np.sqrt(buckets+1)))
    n_rows = int(np.ceil((buckets+1) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()  # For easy indexing
    # Plot each group in its subplot
    for i, pct_active in enumerate(sorted_keys):
        ax = axs[i]
        group = grouped[pct_active]
        # Find best (lowest cost) run
        best_run = min(group, key=lambda r: r["total_cost"])
        for r in group:
            is_best = (r == best_run)
            alpha = 0.9 if is_best else 0.2
            lw = 2.5 if is_best else 1.0

            ax.plot(range(total_steps), r["cost_traj"], alpha=alpha, linewidth=lw)
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

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, pct_active in enumerate(sorted_keys):
        ax = axs[i]
        group = grouped[pct_active]

        # Find best (lowest cost) run
        best_run = min(group, key=lambda r: r["total_cost"])
        for r in group:
            traj = np.array(r["y_traj"])
            x_ref = np.array(r["x_ref"])
            x_ref = x_ref.T
            is_best = (r == best_run)

            alpha = 0.9 if is_best else 0.2
            lw = 2.5 if is_best else 1.0
            label = "Ref Traj" if is_best and 'Ref Traj' not in ax.get_legend_handles_labels()[1] else None

            # Reference trajectory (dashed, light)
            if is_best:
                ax.plot(x_ref[:,0], x_ref[:,1], '--', color='gray', alpha=0.7, label=label)

            # Actual trajectory
            ax.plot(traj[:,0], traj[:,1], alpha=alpha, linewidth=lw)

        ax.set_title(f"{pct_active:.1f}% Active")
        ax.set_aspect('equal')
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
    ax.set_yticks([i + 0.5 for i in range(len(plans))])
    if best:
        ax.set_yticklabels([f"{int(p)}%" for p in sorted_best_pcts])
    else:
        ax.set_yticklabels([f"{int(p)}%" for p in sorted_worst_pcts])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Percent active in Plan")
    if best:
        ax.set_title("Best Plan per Percent Active (Black = Pusher 2 active)")
    else:
        ax.set_title("Worst Plan per Percent Active (Black = Pusher 2 active)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def load_and_play(label, speed, results_dir):
    from Project_Visualize import playback
    data = np.load(f"{results_dir}/{label.replace(' ', '_')}.npz", allow_pickle=True)
    playback(data['y_traj'], data['u_traj'], data['x_ref'], B_WIDTH, B_HEIGHT, dt, speed)

if __name__ == "__main__":
    results_dir = "results/experiment_2"
    results = load_all_results(results_dir)
    # summarize_results(results)
    # plot_force_scatter(results)
    # plot_cost_trajs(results)
    # plot_trajectories_subplots(results)
    # plot_error_vs_active_time(results)
    # plot_best_worst_schedules(results, best=True)
    # plot_best_worst_schedules(results, best=False)

    load_and_play("plan_0000", 3, results_dir)
