"""Offline physical objectives in the same world frame used by evaluation.

These are dense-joint diagnostics/training losses, not SMPL FK. No ground-truth
realignment is applied. The previous reference is the batch's committed frame.
"""

import torch


def weighted_representation_mse(prediction, target):
    weights = prediction.new_ones(243, dtype=torch.float32)
    weights[198:207] = 8
    return ((prediction.float() - target.float()).square() * weights).sum(-1).mean() / weights.sum()


def dense_position_errors(codec, prediction, previous_reference, target_joints):
    if prediction.ndim != 3 or prediction.shape[1:] != (1, 243):
        raise ValueError("Expected one [B,1,243] current prediction.")
    if previous_reference.shape != (len(prediction), 4, 4) or target_joints.shape != (len(prediction), 22, 3):
        raise ValueError("Expected matching previous references and world target joints.")
    # Decode outside autocast: 6D Gram-Schmidt and geometry need float32.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        body = codec.decode_current(prediction[:, 0].float(), previous_reference.float())
        return body.joints[..., :3, 3] - target_joints.float()


def physical_objective(codec, prediction, batch, *, geometry_weight=1.0, scale_m=0.1):
    if geometry_weight < 0 or scale_m <= 0:
        raise ValueError("Invalid physical-loss weight or scale.")
    representation = weighted_representation_mse(prediction, batch["target"])
    geometry = representation.new_zeros(())
    if geometry_weight:
        error = dense_position_errors(codec, prediction, batch["previous_reference"], batch["target_joints"])
        geometry = error.square().sum(-1).mean() / (scale_m**2)
    return representation + geometry_weight * geometry


@torch.no_grad()
def dense_position_mm(codec, prediction, batch):
    return float(
        dense_position_errors(codec, prediction, batch["previous_reference"], batch["target_joints"])
        .norm(dim=-1)
        .mean()
        * 1000
    )
