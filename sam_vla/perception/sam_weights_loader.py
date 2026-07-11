"""Loads the custom SimpleSAM2Seg checkpoint."""

import os

import torch

# ADJUST HERE: import path for the SimpleSAM2Seg architecture class (4-class head
# on a SAM2.1 Hiera-L backbone). Point this at wherever that class actually lives.
from sam2_custom_head import SimpleSAM2Seg

SAM_CHECKPOINT_PATH = "sam/sam/sam_v2_dataset/training_output/checkpoints/best_model.pth"


def load_sam_model(checkpoint_path: str = SAM_CHECKPOINT_PATH, device: str = "cuda") -> object:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Custom SimpleSAM2Seg weights required, none found at '{checkpoint_path}'. "
        )

    model = SimpleSAM2Seg()

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
