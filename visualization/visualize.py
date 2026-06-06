"""
Visualization module for PPO robot navigation experiments.
Generates PNG figures from evaluation results (JSON) and training logs.
Saves all figures to results/figures/.
"""
import os
import json
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Global plot styling applied to all figures.
# Removing top and right spines keeps the charts clean without a full box frame.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# Helpers

def _load_eval_json(path):
    """
    Load a single evaluation result JSON file produced by evaluate().

    Parameters
    path: str
        Absolute path to the JSON file.

    Returns
    dict
        Parsed evaluation result dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_eval_npz(log_key):
    """
    Load the EvalCallback checkpoint file (evaluations.npz) from logs/.

    Stable-Baselines3 EvalCallback saves one npz file per training run under
    logs/<log_key>/evaluations.npz. The file contains:
      - timesteps: 1-D array of training steps at which eval was run
      - results: 2-D array (n_evals, n_eval_episodes) with per-episode rewards
      - ep_lengths: 2-D array (n_evals, n_eval_episodes) with per-episode lengths

    Parameters
    log_key: str
        Subfolder name inside LOGS_DIR.
        This matches the log_path passed to EvalCallback in callbacks.py.

    Returns
    tuple of (np.ndarray, np.ndarray, np.ndarray or None)
        (timesteps, mean_rewards, mean_ep_lengths).
        mean_rewards and mean_ep_lengths are averaged across eval episodes.
        Returns None if the file does not exist.
    """
    path = os.path.join(LOGS_DIR, log_key, "evaluations.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    if "ep_lengths" in data:
        mean_lengths = data["ep_lengths"].mean(axis=1) 
    else:
        mean_lengths = None
    return timesteps, mean_rewards, mean_lengths


def _load_tb_scalar(log_path, tag):
    """
    Load a scalar time series from a TensorBoard event file.

    Used for reading rollout/ep_rew_mean from PPO_* log folders, which are
    not saved as npz (only EvalCallback folders have npz files).
    Requires tensorboard to be installed; returns (None, None) silently if not.

    Parameters
    log_path: str
        Path to log files.
    tag: str
        TensorBoard scalar tag.

    Returns
    tuple of (np.ndarray or None, np.ndarray or None)
        (steps, values). Returns (None, None) if tensorboard is not installed or the folder/tag does not exist.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None
    if os.path.isabs(log_path):
        path = log_path  
    else:
        path = os.path.join(LOGS_DIR, log_path)
    if not os.path.isdir(path):
        return None, None
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()
    try:
        events = event_accumulator.Scalars(tag)
        steps = []
        values = []
        for e in events:
            steps.append(e.step)
            values.append(e.value)
        steps = np.array(steps)
        values = np.array(values)
        return steps, values
    except Exception:
        return None, None


def _smooth(values, window=7):
    """
    Apply a simple moving average to reduce noise in training curves.

    Parameters
    values: array-like
        Raw scalar values (e.g. per-iteration rewards).
    window: int
        Number of points to average over. Larger values produce a smoother curve but lag behind sudden changes. Default 7 works well for ~180 data points over 3M timesteps.

    Returns
    np.ndarray
        Smoothed array of the same length as values.
    """
    values = np.array(values)
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _save(figure, filename):
    """
    Save a matplotlib figure to FIGURES_DIR and close it.

    Parameters
    fig: matplotlib.figure.Figure
        The figure to save.
    filename: str
        Output filename. Saved inside FIGURES_DIR.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {path}")


def _bar_chart(ax, labels, values, colors, ylabel, title, fmt="{:.1f}%"):
    """
    Draw a bar chart on the given axes with value labels above each bar.

    Parameters
    ax: matplotlib.axes.Axes
        Axes to draw on.
    labels: list of str
        X-axis category labels (one per bar).
    values: list of float
        Bar heights.
    colors: list of str
        Matplotlib named color string for each bar.
        Must be the same length as values.
    ylabel: str
        Y-axis label.
    title: str
        Axes title.
    fmt: str
        Format string for the value annotation above each bar.
        Default "{:.1f}%" is for percentages; use "{:.1f}" for raw numbers.
    """
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, pad=10)
    if values:
        ax.set_ylim(0, max(values) * 1.25)
    else:
        ax.set_ylim(0,1)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            fmt.format(val),
            ha="center", va="bottom", fontsize=10,
        )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))


# Evaluation metric figures
def plot_e1_metrics():
    """
    E1 — bar charts of success rate and average episode length vs obstacle count.
    """
    configs = [
        ("N=3", "e1/eval_obs3_spd1.0.json"),
        ("N=6", "e1/eval_obs6_spd1.0.json"),
        ("N=10", "e1/eval_obs10_spd1.0.json"),
    ]
    labels, success, average_steps = [], [], []
    for label, rel_path in configs:
        path = os.path.join(RESULTS_DIR, rel_path)
        if not os.path.exists(path):
            continue
        d = _load_eval_json(path)
        labels.append(label)
        success.append(d["success_rate"] * 100)
        average_steps.append(d["avg_episode_length"])

    if not labels:
        print("E1 eval JSON files not found, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("E1 — effect of obstacle density", fontsize=13, y=1.01)

    _bar_chart(ax1, labels, success, ["mediumpurple", "coral", "darkgray"], "Success rate (%)", "Success rate")
    _bar_chart(ax2, labels, average_steps, ["mediumpurple", "coral", "darkgray"], "Steps", "Average episode length", fmt="{:.1f}")

    fig.tight_layout()
    _save(fig, "e1_metrics.png")


def plot_e2_metrics():
    """
    E2 — bar charts of success rate and average episode length vs obstacle speed.
    """
    configs = [
        ("spd=0.5", "e2/eval_obs6_spd0.5.json"),
        ("spd=1.0", "e2/eval_obs6_spd1.0.json"),
        ("spd=1.5", "e2/eval_obs6_spd1.5.json"),
    ]
    labels, success, average_steps = [], [], []
    for label, rel_path in configs:
        path = os.path.join(RESULTS_DIR, rel_path)
        if not os.path.exists(path):
            continue
        d = _load_eval_json(path)
        labels.append(label)
        success.append(d["success_rate"] * 100)
        average_steps.append(d["avg_episode_length"])

    if not labels:
        print("E2 eval JSON files not found, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("E2 — effect of obstacle speed", fontsize=13, y=1.01)

    _bar_chart(ax1, labels, success, ["steelblue", "mediumpurple", "mediumseagreen"], "Success rate (%)", "Success rate")
    _bar_chart(ax2, labels, average_steps, ["steelblue", "mediumpurple", "mediumseagreen"], "Steps", "Average episode length", fmt="{:.1f}")

    fig.tight_layout()
    _save(fig, "e2_metrics.png")


def plot_e3_metrics():
    """
    E3 — bar charts of success rate and average episode length for all unseen configs.

    Bar color encodes performance relative to the training-config baseline (8%):
      mediumseagreen = success rate >= 10% (above baseline)
      darkgray  = success rate 6-9% (near baseline)
      coral = success rate < 6% (below baseline)
    """
    pattern = os.path.join(RESULTS_DIR, "e3", "eval_obs*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("E3 eval JSON files not found, skipping.")
        return

    labels, success, average_steps = [], [], []
    for path in files:
        d = _load_eval_json(path)
        base  = os.path.basename(path).replace(".json", "").replace("eval_", "")
        parts = base.replace("obs", "N=").replace("_spd", "\nspd=")
        labels.append(parts)
        success.append(d["success_rate"] * 100)
        average_steps.append(d["avg_episode_length"])

    # Color each bar by whether it beats, matches, or falls below the baseline.
    colors = []
    for s in success:
        if s >= 10:
            colors.append("mediumseagreen")
        elif s < 6:
            colors.append("coral")  
        else:
            colors.append("darkgray")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("E3 — generalization to unseen configurations", fontsize=13, y=1.01)

    _bar_chart(ax1, labels, success, colors, "Success rate (%)", "Success rate")
    _bar_chart(ax2, labels, average_steps, ["mediumpurple"] * len(labels), "Steps", "Average episode length", fmt="{:.1f}")

    ax1.tick_params(axis="x", labelsize=9)
    ax2.tick_params(axis="x", labelsize=9)

    # Dashed reference line showing the training-configuration success rate (N=6, spd=1.0).
    ax1.axhline(y=8.0, color="coral", linestyle="--", linewidth=1, alpha=0.6, label="Training config baseline (8%)")
    ax1.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "e3_metrics.png")


def plot_e4_metrics():
    """
    E4 — bar charts comparing success rate, average episode length and average collisions between the model trained with reward shaping and the one trained without.
    """
    configs = [
        ("With shaping", "e4/eval_obs6_spd1.0_shaping.json"),
        ("Without shaping", "e4/eval_obs6_spd1.0_no_shaping.json"),
    ]
    labels, success, average_steps, average_col = [], [], [], []
    for label, rel_path in configs:
        path = os.path.join(RESULTS_DIR, rel_path)
        if not os.path.exists(path):
            continue
        d = _load_eval_json(path)
        labels.append(label)
        success.append(d["success_rate"] * 100)
        average_steps.append(d["avg_episode_length"])
        average_col.append(d["avg_collision_count"])

    if not labels:
        print("E4 eval JSON files not found, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle("E4 — reward shaping vs no reward shaping", fontsize=13, y=1.01)

    _bar_chart(axes[0], labels, success, ["mediumpurple", "coral"], "Success rate (%)", "Success rate")
    _bar_chart(axes[1], labels, average_steps, ["mediumpurple", "coral"], "Steps", "Average episode length", fmt="{:.1f}")
    _bar_chart(axes[2], labels, average_col, ["mediumpurple", "coral"], "Collisions / episode", "Average collisions", fmt="{:.3f}")

    fig.tight_layout()
    _save(fig, "e4_metrics.png")


# Training curve figures
def plot_e1_training_curves():
    """
    E1 — EvalCallback reward curves over training for N=0, 3, 6, 10.

    Each point is the mean reward over 20 evaluation episodes, measured every 80 000 timesteps by EvalCallback during training.
    Line styles differ so the chart is readable in greyscale print.
    """
    # Each tuple: (legend label, log subfolder key, named color, line style).
    series = [
        ("N=0 (phase 1)", "eval_obs0_spd1.0", "mediumpurple", "-"),
        ("N=3", "eval_obs3_spd1.0", "mediumseagreen",  "--"),
        ("N=6", "eval_obs6_spd1.0", "coral", "-."),
        ("N=10", "eval_obs10_spd1.0", "darkgray", ":"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("E1 — training reward curves (EvalCallback)", pad=10)

    plotted = False
    for label, log_key, color, ls in series:
        result = _load_eval_npz(log_key)
        if result is None:
            continue
        timesteps, mean_rewards, _ = result
        ax.plot(timesteps / 1e6, mean_rewards, color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.9)
        plotted = True

    if not plotted:
        print("E1 npz files not found, skipping training curves.")
        plt.close(fig)
        return

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward (20-episode eval)")
    ax.legend(fontsize=10)
    ax.set_ylim(-11, -2)
    fig.tight_layout()
    _save(fig, "e1_training_curves.png")


def plot_e2_training_curves():
    """
    E2 — EvalCallback reward curves over training for spd=0.5, 1.0, 1.5.
    """
    series = [
        ("spd=0.5", "eval_obs6_spd0.5", "steelblue", "-"),
        ("spd=1.0", "eval_obs6_spd1.0", "mediumpurple", "--"),
        ("spd=1.5", "eval_obs6_spd1.5", "mediumseagreen", "-."),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("E2 — training reward curves (EvalCallback)", pad=10)

    plotted = False
    for label, log_key, color, ls in series:
        result = _load_eval_npz(log_key)
        if result is None:
            continue
        timesteps, mean_rewards, _ = result
        ax.plot(timesteps / 1e6, mean_rewards, color=color, linestyle=ls,
                linewidth=1.8, label=label, alpha=0.9)
        plotted = True

    if not plotted:
        print("E2 npz files not found, skipping training curves.")
        plt.close(fig)
        return

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward (20-episode eval)")
    ax.legend(fontsize=10)
    ax.set_ylim(-11, -2)
    fig.tight_layout()
    _save(fig, "e2_training_curves.png")


def plot_e4_training_curves():
    """
    E4 — rollout/ep_rew_mean from TensorBoard for shaping vs no shaping.

    The raw signal is noisy so a moving average (_smooth) is plotted on top.
    Requires tensorboard to be installed; skips silently if not available.
    """
    # Each tuple: (legend label, TensorBoard log folder, named color, line style).
    series = [
        ("With shaping", "models/ppo_robot_nav_obs6_spd1.0_shaping_meta.json",    "mediumpurple", "-"),
        ("Without shaping", "models/ppo_robot_nav_obs6_spd1.0_no_shaping_meta.json", "coral",        "--"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("E4 — training reward curves (rollout)", pad=10)

    plotted = False
    for label, meta_path, color, ls in series:
        full_meta = os.path.join(os.path.dirname(__file__), "..", meta_path)
        if not os.path.exists(full_meta):
            continue
        with open(full_meta) as f:
            folder = json.load(f)["tb_log_folder"]
        steps, values = _load_tb_scalar(folder, "rollout/ep_rew_mean")
        if steps is None:
            continue
        smoothed = _smooth(values, window=9)
        # Raw signal at low opacity as background context.
        ax.plot(steps / 1e6, values, color=color, linewidth=0.5, alpha=0.25)
        # Smoothed signal as the main readable line.
        ax.plot(steps / 1e6, smoothed, color=color, linestyle=ls, linewidth=2, label=label)
        plotted = True

    if not plotted:
        print("E4 TensorBoard logs not found or tensorboard not installed, skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward (rollout)")
    ax.legend(fontsize=10)
    ax.set_ylim(-11, -2)
    fig.tight_layout()
    _save(fig, "e4_training_curves.png")


def plot_all_eval_curves():
    """
    Overview figure — EvalCallback reward curves for all available npz logs
    in a single plot, for cross-experiment comparison.

    Colors and line styles are assigned automatically by index so all series remain visually distinct without requiring manual configuration.
    """
    pattern = os.path.join(LOGS_DIR, "eval_*", "evaluations.npz")
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        print("No npz eval files found, skipping overview curves.")
        return

    named_colors = ["mediumpurple", "mediumseagreen", "coral", "steelblue", "goldenrod", "darkgray", "mediumvioletred"]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("All experiments — EvalCallback reward curves", pad=10)

    for i, npz_path in enumerate(npz_files):
        key = os.path.basename(os.path.dirname(npz_path))
        label = key.replace("eval_", "").replace("_", " ")
        data = np.load(npz_path)
        timesteps = data["timesteps"]
        mean_rewards = data["results"].mean(axis=1)
        ax.plot(
            timesteps / 1e6, mean_rewards,
            color=named_colors[i % len(named_colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.8, label=label, alpha=0.9
        )

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward (20-episode eval)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-11, -2)
    fig.tight_layout()
    _save(fig, "all_eval_curves.png")


# Entry point
def run_visualization():
    """
    Generate all evaluation metric plots and training curve plots.
    Saves PNG files to results/figures/.
    """
    print("Generating visualizations...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Evaluation metrics:")
    plot_e1_metrics()
    plot_e2_metrics()
    plot_e3_metrics()
    plot_e4_metrics()

    print("Training curves:")
    plot_e1_training_curves()
    plot_e2_training_curves()
    plot_e4_training_curves()
    plot_all_eval_curves()

    print(f"Done. Figures saved to: {os.path.abspath(FIGURES_DIR)}")