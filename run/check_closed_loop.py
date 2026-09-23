"""Predicted-history stability check with an explicitly identified initializer.

Without pretrained E7, --allow-random-initializer is required and all outputs
are labelled as an untrained-startup diagnostic, never method performance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config.defaults import get_cfg_defaults
from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.prior import HistoryPrior
from egorecover.rollout import run_episode
from egorecover.history_flow import HistoryFlow
from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import UniEgoMotion
from module.ema import apply_ema_weights_from_checkpoint
from mydiffusion.flow_matching import FlowMatching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--allow-random-initializer", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--reference-mode", choices=("planar", "legacy_se3"), default="planar")
    parser.add_argument("--take-index", type=int, default=0)
    parser.add_argument("--source-modes", nargs="+", choices=("gaussian", "history"), default=["gaussian", "history"])
    parser.add_argument("--variants", nargs="+", default=["clean", "freeze_3s", "drift_0p03mps"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    if args.bootstrap_checkpoint is None and not args.allow_random_initializer:
        parser.error("Supply E7 weights or explicitly permit an untrained-startup diagnostic.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    experiment = json.loads((args.experiment / "report.json").read_text())
    dataset, _ = open_dataset()
    # Development takes only; keep the engineering holdout untouched.
    take = experiment["splits"]["dev"][args.take_index]
    records = [
        record
        for record in dataset.records
        if record["base_take_name"] == take and record["variant_name"] in args.variants
    ]
    stats = torch.load(
        dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt", map_location="cpu", weights_only=False
    )
    codec = MotionCodec(stats, reference_mode=args.reference_mode).to(device)
    prior = HistoryPrior().to(device)
    prior.load_state_dict(
        torch.load(args.experiment / "prior.pt", map_location=device, weights_only=True)["state_dict"]
    )
    prior.freeze()
    torch.manual_seed(args.seed)
    initializer = UniEgoMotion(get_cfg_defaults()).to(device).eval()
    if args.bootstrap_checkpoint:
        checkpoint = torch.load(args.bootstrap_checkpoint, map_location=device, weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        initializer.load_state_dict(state, strict=True)
        if args.bootstrap_weight_source == "ema" and not apply_ema_weights_from_checkpoint(initializer, checkpoint):
            raise ValueError("Requested bootstrap EMA format is unavailable.")
    initializer.requires_grad_(False)
    report = {
        "scope": "closed_loop_stability_only",
        "reference_mode": codec.reference_mode,
        "completion_semantics": "execution and finite metrics only; inspect magnitude separately",
        "initialization": str(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else "random E7; not pretrained",
        "bootstrap_is_trained": args.bootstrap_checkpoint is not None,
        "history_source": "model_predictions_only; no resets",
        "take": take,
        "metric": "dense22_world_position_error_mm; not SMPL FK",
        "results": [],
    }
    for mode in args.source_modes:
        checkpoint = torch.load(args.experiment / f"g_{mode}.pt", map_location=device, weights_only=True)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=mode, sigma=checkpoint["sigma"])
        for record in records:
            last_requested = [-1]

            def observations(**kwargs):
                last_requested[0] = kwargs["as_of"]
                return dataset.observations(record["variant_id"], **kwargs)

            print(f"Rolling out {mode}/{record['variant_name']} with predicted history", flush=True)
            try:
                result = run_episode(
                    model,
                    flow,
                    initializer,
                    FlowMatching(),
                    codec,
                    observations,
                    num_frames=min(args.frames, record["num_frames"]),
                    prior=prior,
                    seed=args.seed,
                )
                torch.save(result, args.output / f"{mode}_{record['variant_name']}.pt")
                # Supervision and fault windows are read ONLY after inference.
                target = torch.from_numpy(dataset.supervision(record["variant_id"])["kp3d"][:, :22])
                indices = result["frame_indices"]
                error = (result["dense_world_joints"] - target[indices]).norm(dim=-1).mean(-1) * 1000
                onset = record["fault_onset"]
                sections = {
                    "pre_fault": [i for i, t in enumerate(indices) if t < onset],
                    "fault": [i for i, t in enumerate(indices) if onset <= t < onset + 30],
                    "recovery": [i for i, t in enumerate(indices) if t >= onset + 30],
                }
                row = {
                    "source_mode": mode,
                    "variant": record["variant_name"],
                    "finite": bool(torch.isfinite(error).all()),
                    "mean_dense22_mm": float(error.mean()),
                    "last_frame_dense22_mm": float(error[-1]),
                    "first_frame_dense22_mm": float(error[0]),
                    "peak_dense22_mm": float(error.max()),
                    "phase_dense22_mm": {
                        name: float(error[items].mean()) if items else None for name, items in sections.items()
                    },
                    "latency_median_ms": float(np.median(result["latency_ms"])),
                    "latency_p95_ms": float(np.percentile(result["latency_ms"], 95)),
                    "floor_estimate_m": result["floor_estimate_m"],
                    "frames": len(indices),
                    "bootstrap_dense22_mm": float(
                        (result["bootstrap_world_joints"] - target[: result["bootstrap_frames"]]).norm(dim=-1).mean()
                        * 1000
                    ),
                }
            except (ValueError, RuntimeError) as error:
                row = {
                    "source_mode": mode,
                    "variant": record["variant_name"],
                    "completed": False,
                    "error": str(error),
                    "last_requested_frame": last_requested[0],
                }
            report["results"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row, indent=2), flush=True)
        del model, checkpoint
    report["completed"] = all(row.get("finite", False) for row in report["results"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
