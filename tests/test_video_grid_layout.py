import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(TRAIN_DIR))

from sample import create_video_grid  # noqa: E402


def test_video_grid_stacks_predictions_below_ground_truth():
    predicted = torch.zeros(2, 3, 3, 4, 4)
    ground_truth = torch.ones(2, 3, 3, 4, 4)

    image = create_video_grid(predicted, ground_truth, num_samples=2)

    assert image.height > image.width
