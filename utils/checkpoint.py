"""
MASOS - Model checkpoint save/load utilities.
"""
import os
import torch
from typing import Dict, Optional


def save_checkpoint(
    models: Dict[str, torch.nn.Module],
    epoch: int,
    save_dir: str,
    prefix: str = "checkpoint",
    extra: Optional[Dict] = None,
):
    """
    Save model checkpoints.

    Args:
        models: Dictionary of model_name -> model
        epoch: Current epoch number
        save_dir: Directory to save checkpoints
        prefix: Filename prefix
        extra: Extra data to save (e.g., optimizer states)
    """
    os.makedirs(save_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "models": {name: model.state_dict() for name, model in models.items()},
    }
    if extra is not None:
        state.update(extra)

    path = os.path.join(save_dir, f"{prefix}_epoch_{epoch}.pt")
    torch.save(state, path)
    print(f"[Checkpoint] Saved: {path}")

    # Also save as latest
    latest_path = os.path.join(save_dir, f"{prefix}_latest.pt")
    torch.save(state, latest_path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
) -> Dict:
    """
    Load a checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary with 'epoch', 'models', etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    print(f"[Checkpoint] Loaded: {path} (epoch {state.get('epoch', '?')})")
    return state


def find_latest_checkpoint(save_dir: str, prefix: str = "checkpoint") -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        save_dir: Directory to search
        prefix: Filename prefix

    Returns:
        Path to latest checkpoint, or None if not found
    """
    latest_path = os.path.join(save_dir, f"{prefix}_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    return None
