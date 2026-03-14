"""
MASOS - Batch runner for all experiments.
Runs all 6 tasks with all 3 algorithms across multiple seeds.

Usage:
    python scripts/run_all_tasks.py --seeds 42 123 456 --device cuda
"""
import sys
import os
import subprocess
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="MASOS Run All Tasks")
    parser.add_argument("--algorithms", nargs="+",
                        default=["ac", "maac", "maddpg"],
                        help="Algorithms to run")
    parser.add_argument("--tasks", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6],
                        help="Task numbers to run")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(scripts_dir, "train.py")

    total = len(args.algorithms) * len(args.tasks) * len(args.seeds)
    print(f"[RunAll] Total experiments: {total}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print()

    count = 0
    start = time.time()

    for algo in args.algorithms:
        for task in args.tasks:
            for seed in args.seeds:
                count += 1
                cmd = [
                    sys.executable, train_script,
                    "--algorithm", algo,
                    "--task", str(task),
                    "--seed", str(seed),
                    "--device", args.device,
                ]
                if args.n_epochs is not None:
                    cmd.extend(["--n_epochs", str(args.n_epochs)])

                print(f"[{count}/{total}] {algo} task={task} seed={seed}")

                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                else:
                    try:
                        result = subprocess.run(
                            cmd, check=True,
                            capture_output=False,
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"  ERROR: {e}")
                    except KeyboardInterrupt:
                        print("\n[RunAll] Interrupted by user")
                        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n[RunAll] Completed {count} experiments in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
