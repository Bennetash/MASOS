"""
MASOS - Evaluate trained models.

Usage:
    python scripts/evaluate.py --algorithm maac --task 1 --checkpoint path/to/model.pt
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from configs.default import TrainingConfig, TASK_CONFIGS
from envs.grid_world import GridWorldEnv
from envs.renderer import GridWorldRenderer
from algorithms.ac_trainer import ACTrainer
from algorithms.maac_trainer import MAACTrainer
from algorithms.maddpg_trainer import MADDPGTrainer
from utils.seed import set_seed
from utils.checkpoint import load_checkpoint, find_latest_checkpoint
from utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="MASOS Evaluation")
    parser.add_argument("--algorithm", type=str, default="maac",
                        choices=["ac", "maac", "maddpg"])
    parser.add_argument("--task", type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="Directory to save rendered frames")
    return parser.parse_args()


def evaluate(trainer, env, n_episodes, render=False, renderer=None,
             save_dir=None):
    """
    Run evaluation episodes.

    Args:
        trainer: Trained agent
        env: GridWorldEnv
        n_episodes: Number of episodes to evaluate
        render: Whether to render
        renderer: GridWorldRenderer instance
        save_dir: Directory for saving frames

    Returns:
        MetricsTracker with evaluation results
    """
    metrics = MetricsTracker(window_size=n_episodes)

    for ep in range(n_episodes):
        obs_dict = env.reset(seed=ep * 1000)
        dones_dict = {"__all__": False}
        episode_reward = 0.0

        while not dones_dict["__all__"]:
            # Select actions (deterministic for evaluation)
            if hasattr(trainer, 'select_actions'):
                result = trainer.select_actions(obs_dict)
                if isinstance(result, dict) and "actions" in result:
                    actions = result["actions"]
                else:
                    actions = result
            else:
                actions = {i: 4 for i in range(env.n_agents)}  # stay

            # For MADDPG, use deterministic
            if hasattr(trainer, 'epsilon'):
                old_eps = trainer.epsilon
                trainer.epsilon = 0.0
                result = trainer.select_actions(obs_dict, explore=False)
                actions = result["actions"]
                trainer.epsilon = old_eps

            next_obs_dict, rewards_dict, dones_dict, info = env.step(actions)
            episode_reward += sum(rewards_dict.values()) / env.n_agents
            obs_dict = next_obs_dict

            if render and renderer is not None:
                title = f"Ep {ep+1}"
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir,
                                       f"ep{ep}_step{info['step']:04d}.png")
                    renderer.render(env, title=title, save_path=path)
                else:
                    renderer.render(env, title=title)

        metrics.add_episode(
            total_reward=episode_reward,
            episode_length=info["step"],
            targets_found=info["total_found"],
            coverage=info["coverage"],
        )
        print(f"  Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"Found={info['total_found']}, "
              f"Coverage={info['coverage']:.3f}")

    return metrics


if __name__ == "__main__":
    args = parse_args()

    # Config
    config = TrainingConfig()
    if args.task in TASK_CONFIGS:
        config.update(TASK_CONFIGS[args.task])
    config.device = args.device
    config.seed = args.seed
    set_seed(config.seed)

    # Environment
    env = GridWorldEnv(config)

    # Create trainer
    if args.algorithm == "ac":
        trainer = ACTrainer(config)
    elif args.algorithm == "maac":
        trainer = MAACTrainer(config)
    elif args.algorithm == "maddpg":
        trainer = MADDPGTrainer(config)

    # Load checkpoint
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, device=config.device)
        trainer.load_models(ckpt["models"])
        print(f"[Eval] Loaded checkpoint from epoch {ckpt['epoch']}")
    else:
        # Try finding latest
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(project_root, "results", "checkpoints")
        prefix = f"{args.algorithm}_task{args.task}_seed{args.seed}"
        latest = find_latest_checkpoint(ckpt_dir, prefix)
        if latest:
            ckpt = load_checkpoint(latest, device=config.device)
            trainer.load_models(ckpt["models"])
            print(f"[Eval] Loaded latest checkpoint from epoch {ckpt['epoch']}")
        else:
            print("[Eval] WARNING: No checkpoint found, using random policy")

    # Renderer
    renderer = None
    if args.render:
        renderer = GridWorldRenderer(grid_size=config.grid_size)

    # Evaluate
    print(f"\n[Eval] Algorithm: {args.algorithm.upper()}, Task: {args.task}")
    print(f"[Eval] Running {args.n_episodes} evaluation episodes...\n")

    metrics = evaluate(
        trainer, env, args.n_episodes,
        render=args.render, renderer=renderer,
        save_dir=args.save_frames,
    )

    stats = metrics.get_stats()
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.algorithm.upper()}, Task {args.task}):")
    print(f"  Avg Reward:   {stats['avg_reward']:.2f}")
    print(f"  Avg Found:    {stats['avg_found']:.1f}")
    print(f"  Max Found:    {stats['max_found']:.0f}")
    print(f"  Avg Coverage: {stats['avg_coverage']:.3f}")
    print(f"  Avg Length:   {stats['avg_length']:.0f}")
    print(f"{'='*50}")

    if renderer:
        renderer.close()
