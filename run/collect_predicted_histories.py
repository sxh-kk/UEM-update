"""Collect TRAINING-only predicted histories with frozen P/G and planar codec.

GT is used for the first 20 TRAINING frames and for offline targets. Subsequent
histories contain only committed predictions. No dev/holdout take is collected.
Both source policies feed one pooled cache so source training can share H/mu.
"""

import argparse
import json
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from egorecover.annotations import body_states_from_supervision
from egorecover.calibration import estimate_bootstrap_floor
from egorecover.codec import BodyState, MotionCodec
from egorecover.data import open_dataset
from egorecover.history import HistoryBuffer
from egorecover.history_flow import HistoryFlow
from egorecover.prior import HistoryPrior
from model.history_uniegomotion import HistoryUniEgoMotion
from run.engineering_pilot import sha256


def slice_state(state, index):
    return BodyState(*(getattr(state, name)[index] for name in ("joints", "reference", "auxiliary")))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["clean", "freeze_3s", "drift_0p03mps"])
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    experiment = json.loads((args.experiment / "report.json").read_text())
    dataset, handoff = open_dataset()
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False)).to(device)
    prior = HistoryPrior().to(device)
    prior.load_state_dict(
        torch.load(args.experiment / "prior.pt", map_location=device, weights_only=True)["state_dict"]
    )
    prior.freeze()
    cache = {
        key: []
        for key in (
            "history_motion",
            "base_mu",
            "velocity_mu",
            "target",
            "traj",
            "img_embs",
            "img_available",
            "traj_available",
            "previous_reference",
            "target_joints",
        )
    }
    metadata = {key: [] for key in ("record_ids", "time_indices", "take_names", "generator_modes")}
    summaries = []
    for mode in ("gaussian", "history"):
        checkpoint = torch.load(args.experiment / f"g_{mode}.pt", map_location=device, weights_only=True)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=mode, sigma=checkpoint["sigma"])
        for variant_index, variant in enumerate(args.variants):
            records = [
                r
                for r in dataset.records
                if r["base_take_name"] in experiment["splits"]["train"] and r["variant_name"] == variant
            ]
            if not records or any(r["num_frames"] != 200 or r["bootstrap_frames"] != 20 for r in records):
                raise ValueError("Expected 200-frame train episodes with 20-frame bootstrap.")
            observed, truth, buffers, floors = [], [], [], []
            for record in records:
                packet = {
                    k: v.to(device)
                    for k, v in dataset.observations(record["variant_id"], as_of=199, history_frames=200).items()
                }
                clean = {
                    k: v.to(device)
                    for k, v in dataset.observations(record["clean_variant_id"], as_of=199, history_frames=200).items()
                }
                floor = estimate_bootstrap_floor(packet, codec, 20)
                supervision = dataset.supervision(record["variant_id"])
                states = body_states_from_supervision(
                    supervision,
                    clean["aria_traj_obs"],
                    floor_height=floor,
                    contact_floor_height=float(supervision["floor_height"]),
                )
                prefix = slice_state(states, slice(0, 20))
                codes = codec.encode_history(
                    BodyState(*(getattr(prefix, k)[None] for k in ("joints", "reference", "auxiliary"))),
                    states.reference[0:1],
                )[0]
                buffers.append(HistoryBuffer.from_bootstrap(codec, codes, states.reference[0]))
                observed.append(packet)
                truth.append(states)
                floors.append(floor)
            errors = []
            print(f"Collect {mode}/{variant}: {len(records)} training episodes", flush=True)
            for t in range(20, 200):
                conditions = [
                    buffer.conditions({k: v[t : t + 1] for k, v in packet.items()}, floor_height=floor)
                    for buffer, packet, floor in zip(buffers, observed, floors)
                ]
                y = {key: torch.cat([condition[key] for condition in conditions]) for key in conditions[0]}
                previous = torch.stack([buffer.previous_reference for buffer in buffers])
                target_state = BodyState(
                    *(
                        torch.stack([getattr(state, key)[t] for state in truth])
                        for key in ("joints", "reference", "auxiliary")
                    )
                )
                row = {
                    "history_motion": y["history_motion"],
                    "base_mu": y["prior_mu"],
                    "velocity_mu": torch.stack(
                        [codec.constant_velocity_prior(buffer.states[-2], buffer.states[-1]) for buffer in buffers]
                    )[:, None],
                    "target": codec.encode_current(target_state, previous)[:, None],
                    "previous_reference": previous,
                    "target_joints": target_state.joints[..., :3, 3],
                    "traj": y["traj"],
                    "img_embs": y["img_embs"],
                    "img_available": ~y["img_mask"],
                    "traj_available": ~y["traj_mask"],
                }
                y["prior_mu"] = prior(y["history_motion"], y["history_valid"], y["prior_mu"])
                epsilon = torch.randn(
                    len(records),
                    1,
                    243,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(args.seed + 1000 * variant_index + t),
                )
                prediction = flow.sample(model, y, epsilon=epsilon)[:, 0]
                states = [buffer.commit(pred, t) for buffer, pred in zip(buffers, prediction)]
                error = (
                    torch.stack([state.joints[..., :3, 3] for state in states]) - row["target_joints"]
                ).double().norm(dim=-1).mean(-1) * 1000
                errors.extend(error.cpu().tolist())
                for key, value in row.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Nonfinite training cache {mode}/{variant}/{t}/{key}")
                    cache[key].append(value.cpu())
                metadata["record_ids"].extend(r["variant_id"] for r in records)
                metadata["take_names"].extend(r["base_take_name"] for r in records)
                metadata["time_indices"].extend([t] * len(records))
                metadata["generator_modes"].extend([mode] * len(records))
            summaries.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "episodes": len(records),
                    "mean_dense22_mm": sum(errors) / len(errors),
                    "max_dense22_mm": max(errors),
                }
            )
            print(summaries[-1], flush=True)
        del model, checkpoint
    identity = {
        "scope": "training_only_predicted_histories",
        "reference_mode": codec.reference_mode,
        "bootstrap": "GT first 20 frames on TRAINING takes only; no resets afterwards",
        "policy": "a11 frozen P/G; pooled gaussian/history policies",
        "generator_experiment": str(args.experiment),
        "g_sha256": {mode: sha256(args.experiment / f"g_{mode}.pt") for mode in ("gaussian", "history")},
        "p_sha256": sha256(args.experiment / "prior.pt"),
        "stats_sha256": sha256(stats_path),
        "train_takes": experiment["splits"]["train"],
        "variants": args.variants,
        "seed": args.seed,
        "sigma": experiment["sigma"],
        "nfe": 10,
    }
    output = args.output / "frames.pt"
    torch.save({"identity": identity, "tensors": {k: torch.cat(v) for k, v in cache.items()}, **metadata}, output)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "frames": len(metadata["record_ids"]),
                "episodes": summaries,
                "cache_sha256": sha256(output),
                "completed": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
