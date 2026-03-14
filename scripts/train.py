"""
MASOS - Main training entry point.

Usage:
    python scripts/train.py --algorithm maac --task 1 --seed 42 --device cuda
    python scripts/train.py --algorithm ac --task 1 --seed 42
    python scripts/train.py --algorithm maddpg --task 1 --seed 42
"""
import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from configs.default import TrainingConfig, TASK_CONFIGS
from envs.grid_world import GridWorldEnv
from algorithms.ac_trainer import ACTrainer
from algorithms.maac_trainer import MAACTrainer
from algorithms.maddpg_trainer import MADDPGTrainer
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="MASOS Training")
    parser.add_argument("--algorithm", type=str, default="maac",
                        choices=["ac", "maac", "maddpg"],
                        help="Algorithm to use")
    parser.add_argument("--task", type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6],
                        help="Task configuration (1-6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Override log directory")
    parser.add_argument("--no_layernorm", action="store_true",
                        help="Disable LayerNorm in MAAC (for ablation)")
    return parser.parse_args()


def create_trainer(algorithm, config, **kwargs):
    """Create the appropriate trainer based on algorithm name."""
    if algorithm == "ac":
        return ACTrainer(config)
    elif algorithm == "maac":
        use_ln = kwargs.get("use_layer_norm", True)
        return MAACTrainer(config, use_layer_norm=use_ln)
    elif algorithm == "maddpg":
        return MADDPGTrainer(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_on_policy(trainer, env, config, logger, metrics, n_epochs):
    """Training loop for on-policy methods (AC, MAAC)."""
    print(f"\n[Train] Starting on-policy training for {n_epochs} epochs...")

    for epoch in range(1, n_epochs + 1):
        obs_dict = env.reset()
        episode_reward = 0.0
        dones_dict = {"__all__": False}

        while not dones_dict["__all__"]:
            # Build alive mask from env
            alive_mask = {a.id: a.alive for a in env.agents}

            # Select actions
            result = trainer.select_actions(obs_dict, alive_mask=alive_mask)
            actions = result["actions"]
            log_probs = result["log_probs"]
            values = result["values"]

            # Step environment
            next_obs_dict, rewards_dict, dones_dict, info = env.step(actions)

            # Store transition
            trainer.store_transition(
                obs_dict, actions, rewards_dict, values, log_probs, dones_dict
            )

            # Accumulate reward
            episode_reward += sum(rewards_dict.values()) / config.n_agents
            obs_dict = next_obs_dict

        # Update networks
        losses = trainer.update(next_obs_dict=obs_dict)

        # Track metrics
        metrics.add_episode(
            total_reward=episode_reward,
            episode_length=info["step"],
            targets_found=info["total_found"],
            coverage=info["coverage"],
        )

        # Logging
        if epoch % config.log_interval == 0:
            stats = metrics.get_stats()
            log_data = {**losses, **stats}
            logger.log_scalars(log_data, step=epoch)

            print(f"  Epoch {epoch}/{n_epochs} | "
                  f"Reward: {stats['avg_reward']:.1f} | "
                  f"Found: {stats['avg_found']:.1f} | "
                  f"Coverage: {stats['avg_coverage']:.3f} | "
                  f"ActLoss: {losses.get('actor_loss', 0):.4f} | "
                  f"CritLoss: {losses.get('critic_loss', 0):.4f}")

        # Save checkpoint
        if epoch % config.save_interval == 0:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results", "checkpoints"
            )
            save_checkpoint(
                models=trainer.get_models(),
                epoch=epoch,
                save_dir=save_dir,
                prefix=f"{args.algorithm}_task{args.task}_seed{args.seed}",
            )


def train_off_policy(trainer, env, config, logger, metrics, n_epochs):
    """Training loop for off-policy methods (MADDPG)."""
    print(f"\n[Train] Starting off-policy training for {n_epochs} epochs...")

    global_step = 0

    for epoch in range(1, n_epochs + 1):
        obs_dict = env.reset()
        episode_reward = 0.0
        dones_dict = {"__all__": False}

        while not dones_dict["__all__"]:
            # Select actions with exploration
            result = trainer.select_actions(obs_dict, explore=True)
            actions = result["actions"]

            # Step environment
            next_obs_dict, rewards_dict, dones_dict, info = env.step(actions)

            # Store transition in replay buffer
            trainer.store_transition(obs_dict, actions, rewards_dict,
                                   next_obs_dict, dones_dict)

            # Update networks
            losses = trainer.update()

            episode_reward += sum(rewards_dict.values()) / config.n_agents
            obs_dict = next_obs_dict
            global_step += 1

        # Track metrics
        metrics.add_episode(
            total_reward=episode_reward,
            episode_length=info["step"],
            targets_found=info["total_found"],
            coverage=info["coverage"],
        )

        # Logging
        if epoch % config.log_interval == 0:
            stats = metrics.get_stats()
            log_data = {**losses, **stats} if losses else stats
            logger.log_scalars(log_data, step=epoch)

            print(f"  Epoch {epoch}/{n_epochs} | "
                  f"Reward: {stats['avg_reward']:.1f} | "
                  f"Found: {stats['avg_found']:.1f} | "
                  f"Coverage: {stats['avg_coverage']:.3f} | "
                  f"Epsilon: {trainer.epsilon:.3f}")

        # Save checkpoint
        if epoch % config.save_interval == 0:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results", "checkpoints"
            )
            save_checkpoint(
                models=trainer.get_models(),
                epoch=epoch,
                save_dir=save_dir,
                prefix=f"{args.algorithm}_task{args.task}_seed{args.seed}",
            )


if __name__ == "__main__":
    args = parse_args()

    # Setup config
    config = TrainingConfig()
    if args.task in TASK_CONFIGS:
        config.update(TASK_CONFIGS[args.task])

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        config.device = "cpu"
    else:
        config.device = args.device

    if args.n_epochs is not None:
        config.n_epochs = args.n_epochs

    config.seed = args.seed

    # Set seed
    set_seed(config.seed)

    # Create environment
    env = GridWorldEnv(config)

    # Create trainer
    trainer = create_trainer(
        args.algorithm, config,
        use_layer_norm=not args.no_layernorm,
    )

    # Setup logging
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(project_root, "results", "logs")
    experiment_name = f"{args.algorithm}_task{args.task}_seed{args.seed}"
    logger = Logger(log_dir=log_dir, experiment_name=experiment_name)
    metrics = MetricsTracker(window_size=100)

    print("=" * 60)
    print(f"MASOS Training")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print(f"  Task: {args.task} ({TASK_CONFIGS[args.task]})")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Agents: {config.n_agents}, Targets: {config.n_targets}")
    if args.algorithm == "maac":
        print(f"  LayerNorm: {not args.no_layernorm}")
    print("=" * 60)

    start_time = time.time()

    # Train
    if args.algorithm in ["ac", "maac"]:
        train_on_policy(trainer, env, config, logger, metrics, config.n_epochs)
    else:
        train_off_policy(trainer, env, config, logger, metrics, config.n_epochs)

    elapsed = time.time() - start_time
    print(f"\n[Done] Training completed in {elapsed:.1f}s")
    print(f"  Final stats: {metrics.get_stats()}")

    logger.close()
