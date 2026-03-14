"""
MASOS - Create animated GIF of trained model simulation.

Generates a visual GIF showing agents searching for targets in the grid world.

Usage:
    python scripts/create_simulation_gif.py --algorithm maac --task 1 --seed 42
    python scripts/create_simulation_gif.py --algorithm ac --task 1 --seed 42
    python scripts/create_simulation_gif.py --algorithm maac --task 1 --seed 42 --compare
"""
import sys
import os
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from configs.default import TrainingConfig, TASK_CONFIGS
from envs.grid_world import GridWorldEnv
from algorithms.ac_trainer import ACTrainer
from algorithms.maac_trainer import MAACTrainer
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint, find_latest_checkpoint


def render_frame(env, ax, title="", step_rewards=None):
    """Render a single frame of the environment to a matplotlib axis."""
    ax.clear()
    gs = env.grid_size

    # Background: explored vs unexplored
    display = np.zeros((gs, gs, 3), dtype=np.float32)
    display[:, :] = [0.92, 0.92, 0.92]  # Unexplored = light gray
    explored = env.grid_explored
    display[explored] = [1.0, 1.0, 1.0]  # Explored = white

    # Obstacles = dark gray
    for obs in env.obstacles:
        for r, c in obs.cells:
            display[r, c] = [0.25, 0.25, 0.25]

    # Found targets = light green (already found)
    for target in env.targets:
        if target.found:
            r, c = target.position
            display[r, c] = [0.7, 1.0, 0.7]

    # Unfound targets = bright green
    for target in env.targets:
        if not target.found:
            r, c = target.position
            display[r, c] = [0.0, 0.85, 0.0]

    ax.imshow(display, origin="upper", interpolation="nearest")

    # Draw agent trails (last 5 positions) - faded
    AGENT_COLORS = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#42d4f4", "#f032e6",
        "#9A6324", "#808000", "#469990", "#000075",
        "#aaffc3", "#dcbeff", "#800000", "#ffd8b1",
    ]

    # Draw agents as colored circles
    for agent in env.agents:
        if not agent.alive:
            continue
        r, c = agent.position
        color = AGENT_COLORS[agent.id % len(AGENT_COLORS)]

        # Agent body
        circle = plt.Circle((c, r), 0.45, color=color, zorder=5, alpha=0.9)
        ax.add_patch(circle)

        # Agent border
        border = plt.Circle((c, r), 0.45, fill=False, edgecolor='white',
                           linewidth=1.5, zorder=6)
        ax.add_patch(border)

        # Agent ID
        ax.text(c, r, str(agent.id), ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=7)

        # Detection radius indicator (faint circle)
        detect = plt.Circle((c, r), env.config.obs_radius, fill=False,
                           edgecolor=color, linewidth=0.3, alpha=0.3,
                           linestyle='--', zorder=3)
        ax.add_patch(detect)

    # Title with metrics
    coverage = np.sum(explored) / (gs * gs) * 100
    remaining = sum(1 for t in env.targets if not t.found)
    info_str = (f"{title}  |  Step: {env.step_count}/{env.max_steps}  |  "
                f"Found: {env.total_found}/{env.n_targets}  |  "
                f"Remaining: {remaining}  |  "
                f"Coverage: {coverage:.1f}%")
    ax.set_title(info_str, fontsize=9, fontweight='bold')

    ax.set_xlim(-0.5, gs - 0.5)
    ax.set_ylim(gs - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=[0.92, 0.92, 0.92], edgecolor='gray',
                      label="Unexplored"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Explored"),
        mpatches.Patch(facecolor=[0.25, 0.25, 0.25], label="Obstacle"),
        mpatches.Patch(facecolor=[0.0, 0.85, 0.0], label="Target"),
        mpatches.Patch(facecolor=[0.7, 1.0, 0.7], label="Found"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6,
             framealpha=0.8)


def run_episode_and_save_frames(trainer, env, frame_dir, title_prefix,
                                 save_every=5, seed=0):
    """Run one episode and save frames at regular intervals."""
    os.makedirs(frame_dir, exist_ok=True)

    obs_dict = env.reset(seed=seed)
    dones_dict = {"__all__": False}
    frame_count = 0
    total_reward = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Save initial frame
    render_frame(env, ax, title=title_prefix)
    fig.savefig(os.path.join(frame_dir, f"frame_{frame_count:05d}.png"),
                dpi=100, bbox_inches="tight", facecolor='white')
    frame_count += 1

    step = 0
    while not dones_dict["__all__"]:
        # Select actions
        result = trainer.select_actions(obs_dict)
        if isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
        else:
            actions = result

        next_obs_dict, rewards_dict, dones_dict, info = env.step(actions)
        total_reward += sum(rewards_dict.values()) / env.n_agents
        obs_dict = next_obs_dict
        step += 1

        # Save frame every N steps
        if step % save_every == 0 or dones_dict["__all__"]:
            render_frame(env, ax, title=title_prefix)
            fig.savefig(os.path.join(frame_dir, f"frame_{frame_count:05d}.png"),
                        dpi=100, bbox_inches="tight", facecolor='white')
            frame_count += 1

    plt.close(fig)

    return {
        "total_reward": total_reward,
        "targets_found": info["total_found"],
        "coverage": info["coverage"],
        "steps": info["step"],
        "frames_saved": frame_count,
    }


def create_gif_from_frames(frame_dir, output_path, duration=100):
    """Create a GIF from saved PNG frames using PIL."""
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] PIL/Pillow not installed. Install with: pip install Pillow")
        print("        Frames are saved in:", frame_dir)
        return False

    # Get sorted frame files
    frame_files = sorted([
        os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
        if f.endswith('.png')
    ])

    if not frame_files:
        print("[ERROR] No frames found in", frame_dir)
        return False

    print(f"  Creating GIF from {len(frame_files)} frames...")

    # Load frames
    frames = []
    for f in frame_files:
        img = Image.open(f)
        frames.append(img.copy())
        img.close()

    # Add pause at the end (repeat last frame)
    for _ in range(15):
        frames.append(frames[-1].copy())

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # ms per frame
        loop=0,  # infinite loop
        optimize=True,
    )

    print(f"  GIF saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="MASOS Simulation GIF Creator")
    parser.add_argument("--algorithm", type=str, default="maac",
                        choices=["ac", "maac"])
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episode_seed", type=int, default=123,
                        help="Seed for the evaluation episode")
    parser.add_argument("--save_every", type=int, default=3,
                        help="Save a frame every N steps (lower = smoother GIF)")
    parser.add_argument("--gif_speed", type=int, default=80,
                        help="Milliseconds per frame in GIF (lower = faster)")
    parser.add_argument("--compare", action="store_true",
                        help="Generate GIFs for both MAAC and AC side by side")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint path (optional)")
    return parser.parse_args()


def run_single(args, algorithm):
    """Run simulation for a single algorithm and generate GIF."""
    print(f"\n{'='*60}")
    print(f"  Generating simulation: {algorithm.upper()} - Task {args.task}")
    print(f"{'='*60}")

    # Config
    config = TrainingConfig()
    if args.task in TASK_CONFIGS:
        config.update(TASK_CONFIGS[args.task])
    config.device = args.device
    config.seed = args.seed
    set_seed(args.seed)

    # Environment
    env = GridWorldEnv(config)

    # Create trainer
    if algorithm == "ac":
        trainer = ACTrainer(config)
    elif algorithm == "maac":
        trainer = MAACTrainer(config)

    # Load checkpoint
    if args.checkpoint and algorithm == args.algorithm:
        ckpt = load_checkpoint(args.checkpoint, device=config.device)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(project_root, "results", "checkpoints")
        prefix = f"{algorithm}_task{args.task}_seed{args.seed}"
        latest = find_latest_checkpoint(ckpt_dir, prefix)
        if latest:
            ckpt = load_checkpoint(latest, device=config.device)
        else:
            print(f"[ERROR] No checkpoint found for {algorithm}")
            return None

    trainer.load_models(ckpt["models"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")

    # Output paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame_dir = os.path.join(project_root, "results", "simulation_frames",
                            f"{algorithm}_task{args.task}")
    gif_path = os.path.join(project_root, "results",
                           f"simulation_{algorithm}_task{args.task}_seed{args.seed}.gif")

    # Clean previous frames
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)

    # Run episode and save frames
    print(f"  Running episode (save_every={args.save_every} steps)...")
    results = run_episode_and_save_frames(
        trainer, env, frame_dir,
        title_prefix=f"{algorithm.upper()} (epoch {ckpt['epoch']})",
        save_every=args.save_every,
        seed=args.episode_seed,
    )

    print(f"\n  Episode Results:")
    print(f"    Targets Found: {results['targets_found']}/{env.n_targets}")
    print(f"    Coverage:      {results['coverage']*100:.1f}%")
    print(f"    Total Steps:   {results['steps']}")
    print(f"    Total Reward:  {results['total_reward']:.1f}")
    print(f"    Frames Saved:  {results['frames_saved']}")

    # Create GIF
    print(f"\n  Creating GIF (speed={args.gif_speed}ms per frame)...")
    success = create_gif_from_frames(frame_dir, gif_path, duration=args.gif_speed)

    if success:
        print(f"\n  [OK] GIF ready: {gif_path}")

    return results


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        # Generate both MAAC and AC GIFs
        print("\n" + "="*60)
        print("  COMPARISON MODE: Generating MAAC and AC simulations")
        print("="*60)

        results_maac = run_single(args, "maac")
        results_ac = run_single(args, "ac")

        if results_maac and results_ac:
            print(f"\n\n{'='*60}")
            print(f"  COMPARISON RESULTS")
            print(f"{'='*60}")
            print(f"  {'Metric':<20} {'MAAC':>10} {'AC':>10} {'Diff':>10}")
            print(f"  {'-'*50}")
            print(f"  {'Targets Found':<20} {results_maac['targets_found']:>10} "
                  f"{results_ac['targets_found']:>10} "
                  f"{results_maac['targets_found'] - results_ac['targets_found']:>+10}")
            print(f"  {'Coverage %':<20} {results_maac['coverage']*100:>9.1f}% "
                  f"{results_ac['coverage']*100:>9.1f}% "
                  f"{(results_maac['coverage'] - results_ac['coverage'])*100:>+9.1f}%")
            print(f"  {'Total Steps':<20} {results_maac['steps']:>10} "
                  f"{results_ac['steps']:>10}")
            print(f"  {'Reward':<20} {results_maac['total_reward']:>10.1f} "
                  f"{results_ac['total_reward']:>10.1f} "
                  f"{results_maac['total_reward'] - results_ac['total_reward']:>+10.1f}")
            print(f"{'='*60}\n")
    else:
        run_single(args, args.algorithm)
