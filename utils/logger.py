"""
MASOS - TensorBoard + CSV logging utility.
"""
import os
import csv
import time
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class Logger:
    """
    Combined TensorBoard and CSV logger.

    Args:
        log_dir: Directory for log files
        experiment_name: Name prefix for the run
        use_tensorboard: Whether to use TensorBoard
    """

    def __init__(self, log_dir: str, experiment_name: str = "run",
                 use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and HAS_TB:
            tb_dir = os.path.join(log_dir, "tb", experiment_name)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        # CSV
        self.csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self.csv_file = None
        self.csv_writer = None
        self.csv_fields = None

        self.start_time = time.time()

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, data: Dict[str, float], step: int):
        """Log multiple scalar values."""
        for tag, value in data.items():
            self.log_scalar(tag, value, step)

        # Also write to CSV
        self._write_csv(data, step)

    def _write_csv(self, data: Dict[str, float], step: int):
        """Write a row to the CSV file."""
        row = {"step": step, "time": time.time() - self.start_time}
        row.update(data)

        if self.csv_file is None:
            self.csv_fields = list(row.keys())
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=self.csv_fields,
                extrasaction="ignore"
            )
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """Close all log files."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
