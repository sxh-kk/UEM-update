"""OFFLINE training cache of reachable predicted body histories.

Only train takes are accepted. This is never imported by online rollout.
"""

from pathlib import Path

import torch


class PredictedHistoryFrames:
    def __init__(self, path, *, train_takes, reference_mode, stats_sha256=None):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
        self.identity = payload["identity"]
        if self.identity.get("scope") != "training_only_predicted_histories":
            raise ValueError("Expected a training-only predicted-history cache.")
        if self.identity.get("reference_mode") != reference_mode:
            raise ValueError("Replay and training must use the same reference convention.")
        if stats_sha256 is not None and self.identity.get("stats_sha256") != stats_sha256:
            raise ValueError("Replay normalization statistics do not match training.")
        if not set(payload["take_names"]).issubset(set(train_takes)):
            raise ValueError("Replay contains takes outside the training split.")
        self.tensors = payload["tensors"]
        self.record_ids, self.time_indices = payload["record_ids"], payload["time_indices"]
        self.count = len(self.record_ids)
        if not self.count or len(payload["take_names"]) != self.count or len(self.time_indices) != self.count:
            raise ValueError("Replay metadata lengths do not match.")
        for key, value in self.tensors.items():
            if value.shape[0] != self.count or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid/nonfinite cached field {key}.")

    def __len__(self):
        return self.count

    def batch(self, indices, device):
        return {key: value[indices].to(device) for key, value in self.tensors.items()}


def training_batch(teacher, replay, count, generator, device, probability=0.5):
    batch = teacher.batch(torch.randint(len(teacher), (count,), generator=generator), device)
    if replay is None:
        return batch
    if not 0 <= probability <= 1:
        raise ValueError("Replay probability must be in [0,1].")
    predicted = replay.batch(torch.randint(len(replay), (count,), generator=generator), device)
    if set(batch) != set(predicted):
        raise ValueError("Replay and teacher batches have different fields.")
    choose = (torch.rand(count, generator=generator) < probability).to(device)
    return {
        key: torch.where(choose.reshape(count, *([1] * (value.ndim - 1))), predicted[key], value)
        for key, value in batch.items()
    }
