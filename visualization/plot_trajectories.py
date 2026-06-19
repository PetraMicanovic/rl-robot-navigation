"""
Visualization of agent trajectories in the 2D robot navigation environment.

Each episode dict passed to plot_multiple_trajectories has the form:
    {
        "trajectory": list of np.ndarray (2,) - agent_position snapshots,
        "goal_pos": np.ndarray (2,) - target_position at episode start,
        "static_obstacles": list of (cx, cy, hw, hh) - same as env.static_obstacles,
        "outcome": str - "success" | "collision" | "timeout",
    }
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


# Colour palette 
_OUTCOME_COLORS = {
    "success": "#2ecc71", # green
    "collision": "#e74c3c", # red
    "timeout": "#e67e22", # orange
}
_AGENT_COLOR = "#2980b9" # blue
_GOAL_COLOR = "#f1c40f" # yellow-gold
_OBSTACLE_COLOR = "#7f8c8d" # grey
_BG_COLOR = "#f8f8f8"


# Helpers 
def _draw_arena(ax, world_size):
    """
    Draw world boundary and background.

    Parameters
    ax: matplotlib.axes.Axes
        Axes object to draw onto.
    world_size: float
        Side length of the (square) world; env.WORLD_SIZE.
    """
    ax.set_facecolor(_BG_COLOR)
    border = patches.Rectangle((0, 0), world_size, world_size, linewidth=2, edgecolor="black", facecolor="none", zorder=1)
    ax.add_patch(border)
    ax.set_xlim(-0.2, world_size + 0.2)
    ax.set_ylim(-0.2, world_size + 0.2)
    ax.set_aspect("equal")


def _draw_static_obstacles(ax, static_obstacles):
    """
    Draw static rectangular obstacles (cx, cy, half_w, half_h).

    Parameters
    ax: matplotlib.axes.Axes
        Axes object to draw onto.
    static_obstacles: list of (cx, cy, half_w, half_h)
        Same list as env.static_obstacles. Each tuple gives the center coordinates and half-width/half-height of one
        rectangular obstacle.
    """
    for (cx, cy, hw, hh) in static_obstacles:
        rect = patches.Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, color=_OBSTACLE_COLOR, zorder=2)
        ax.add_patch(rect)


def _draw_goal(ax, goal_pos, target_radius=0.3, alpha=1.0):
    """
    Draw the goal marker as a filled circle.

    Parameters
    ax: matplotlib.axes.Axes
        Axes object to draw onto.
    goal_pos: array-like (2,)
        (x, y) coordinates of the goal/target position.
    target_radius: float
        Visual radius of the goal marker (use env.TARGET_RADIUS).
    alpha: float
        Opacity of the goal marker, in [0, 1].
    """
    ax.add_patch(patches.Circle(goal_pos, target_radius, color=_GOAL_COLOR, zorder=3, alpha=alpha))

def _draw_trajectory(ax, trajectory, color, alpha=1.0, linewidth=1.5):
    """
    Plot a single trajectory line with start/end markers.

    Parameters
    ax: matplotlib.axes.Axes
        Axes object to draw onto.
    trajectory: list of np.ndarray (2,)
        Sequence of agent positions recorded at every step.
    color: str
        Line/end-marker color (typically from _OUTCOME_COLORS).
    alpha: float
        Opacity of the trajectory line and markers, in [0, 1].
    linewidth: float
        Width of the trajectory line.
    """
    if len(trajectory) < 2:
        return
    xs = []
    ys = []
    for p in trajectory:
        xs.append(p[0])
        ys.append(p[1])
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=4)
    # start marker
    ax.plot(xs[0], ys[0], "s", color=_AGENT_COLOR, markersize=6, zorder=5, alpha=alpha)
    # end marker
    ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, zorder=5, alpha=alpha)


# Public API 
def plot_trajectory(trajectory, goal_pos, static_obstacles, world_size, outcome="success", target_radius=0.3, title="Agent Trajectory", 
                    save_path=None,):
    """
    Plot a single episode trajectory.

    Parameters
    trajectory: list of np.ndarray (2,)
        Sequence of agent positions recorded at every step.
    goal_pos: array-like (2,)
        Target position for this episode.
    static_obstacles: list of (cx, cy, half_w, half_h)
        Same list as env.static_obstacles.
    world_size: float
        env.WORLD_SIZE.
    outcome: str
        "success" | "collision" | "timeout"
    target_radius: float
        Visual radius of the goal marker (use env.TARGET_RADIUS).
    title: str
        Plot title.
    save_path: str or None
        If given, saves the figure to this path; otherwise calls plt.show().
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=11)

    _draw_arena(ax, world_size)
    _draw_static_obstacles(ax, static_obstacles)
    _draw_goal(ax, goal_pos, target_radius)

    color = _OUTCOME_COLORS.get(outcome, "blue")
    _draw_trajectory(ax, trajectory, color)

    # legend
    legend_elements = [
        Line2D([0], [0], color=_AGENT_COLOR, marker="s", linestyle="none", markersize=7, label="Start"),
        patches.Patch(facecolor=_GOAL_COLOR, label="Goal"),
        patches.Patch(facecolor=_OBSTACLE_COLOR, label="Static obstacle"),
        Line2D([0], [0], color=color, linewidth=2, label=f"Trajectory ({outcome})"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_multiple_trajectories(episodes, world_size, target_radius=0.3, title="Agent trajectories", max_to_plot=50, alpha=0.35,
                               save_path=None,):
    """
    Overlay multiple episode trajectories on a single map.

    Parameters
    episodes: list of dict
        Each dict must have keys:
            "trajectory" - list of np.ndarray (2,)
            "goal_pos" - np.ndarray (2,)
            "static_obstacles" - list of (cx, cy, hw, hh)
            "outcome" - "success" | "collision" | "timeout"
        All episodes are assumed to share the same static_obstacles layout.
    world_size: float
        env.WORLD_SIZE.
    target_radius: float
        env.TARGET_RADIUS.
    title: str
        Plot title.
    max_to_plot: int
        Cap on number of episodes drawn (avoids clutter for 200-episode eval).
    alpha: float
        Transparency for individual trajectory lines.
    save_path: str or None
        Save path or None to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title, fontsize=12)

    if episodes:
        static_obstacles = episodes[0]["static_obstacles"]  
    else:
        static_obstacles = []
    _draw_arena(ax, world_size)
    _draw_static_obstacles(ax, static_obstacles)

    counts = {"success": 0, "collision": 0, "timeout": 0}

    for ep in episodes[:max_to_plot]:
        outcome = ep.get("outcome", "timeout")
        color = _OUTCOME_COLORS.get(outcome, "blue")
        counts[outcome] = counts.get(outcome, 0) + 1

        _draw_goal(ax, ep["goal_pos"], target_radius, alpha=0.3)
        _draw_trajectory(ax, ep["trajectory"], color, alpha=alpha, linewidth=1.0)

    # legend with outcome counts
    legend_elements = [
        Line2D([0], [0], color=_OUTCOME_COLORS["success"], linewidth=2, label=f'Success ({counts["success"]})'),
        Line2D([0], [0], color=_OUTCOME_COLORS["collision"], linewidth=2, label=f'Collision ({counts["collision"]})'),
        Line2D([0], [0], color=_OUTCOME_COLORS["timeout"], linewidth=2, label=f'Timeout ({counts["timeout"]})'),
        patches.Patch(facecolor=_GOAL_COLOR, alpha=0.5, label="Goal"),
        patches.Patch(facecolor=_OBSTACLE_COLOR, label="Static obstacle"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
