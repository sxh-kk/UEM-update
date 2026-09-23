"""Small real-data learning diagnostic, explicitly using teacher-forced history.

Not a reproduction or final benchmark: official-val engineering takes, no
SMPL FK, and no claim that teacher-forced gains survive closed-loop rollout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from config.defaults import get_cfg_defaults
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.engineering import TeacherForcedFrames, conditions_from_batch, split_takes
from egorecover.history_flow import HistoryFlow
from egorecover.losses import physical_objective, dense_position_mm, weighted_representation_mse
from egorecover.prior import HistoryPrior
from egorecover.replay import PredictedHistoryFrames, training_batch
from egorecover.utility import generate_utility_labels
from model.history_uniegomotion import HistoryUniEgoMotion


def weighted_mse(prediction, target):
    return weighted_representation_mse(prediction, target)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--prior-steps", type=int, default=200)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--contact-floor-source", choices=("annotation", "startup_estimate"), default="annotation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--reference-mode", choices=("planar", "legacy_se3"), default="planar")
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--prior-selection", choices=("dense", "representation"), default="dense")
    parser.add_argument("--flow-selection", choices=("dense", "final"), default="dense")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--replay-probability", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve previous experiments.")
    if args.prior_steps < 1 or args.flow_steps < 1 or args.batch_size < 1:
        parser.error("Step counts and batch size must be positive.")
    if args.geometry_weight < 0 or args.eval_every < 1:
        parser.error("Geometry weight must be nonnegative and eval interval positive.")
    if not 0 <= args.replay_probability <= 1:
        parser.error("Replay probability must be in [0,1].")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset, handoff = open_dataset(signal=args.signal)
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    codec = MotionCodec(stats, reference_mode=args.reference_mode)
    geometry_codec = MotionCodec(stats, reference_mode=args.reference_mode).to(device)
    splits = split_takes(dataset, args.seed)
    replay = (
        PredictedHistoryFrames(
            args.history_cache,
            train_takes=splits["train"],
            reference_mode=codec.reference_mode,
            stats_sha256=sha256(stats_path),
        )
        if args.history_cache
        else None
    )
    report = {
        "scope": "teacher_forced_engineering_only",
        "base_split": "official_val",
        "history_source": "GT for development diagnostic; not online evaluation",
        "floor_source": "startup_head_z_median_minus_training_mean_head_height",
        "contact_floor_source": args.contact_floor_source,
        "metric": "dense22_position_error_mm (not SMPL22 FK MPJPE)",
        "initialization": str(args.checkpoint) if args.checkpoint else "random; no pretrained E7",
        "seed": args.seed,
        "splits": splits,
        "sigma": args.sigma,
        "nfe": 10,
        "stats_sha256": sha256(stats_path),
        "dataset_handoff": handoff,
        "prior_steps": args.prior_steps,
        "flow_steps": args.flow_steps,
        "batch_size": args.batch_size,
        "reference_mode": args.reference_mode,
        "geometry_weight": args.geometry_weight,
        "geometry_scale_m": 0.1,
        "prior_selection": args.prior_selection,
        "flow_selection": args.flow_selection,
        "predicted_history_cache": (
            {
                "path": str(args.history_cache),
                "sha256": sha256(args.history_cache),
                "identity": replay.identity,
                "frames": len(replay),
                "sample_probability": args.replay_probability,
            }
            if replay
            else None
        ),
        "sources": {},
    }
    if replay:
        report["scope"] = "mixed_history_engineering_only"
        report["history_source"] = (
            "training mixture of GT and frozen-model predicted histories; GT-bootstrap only on train takes; dev diagnostic remains GT-history"
        )
    report["evaluation_history_source"] = "GT for the single-step dev diagnostic; evaluate closed loop separately"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Preparing take-disjoint teacher-forced engineering batches", flush=True)
    prior_train = TeacherForcedFrames(
        dataset, codec, splits["train"], clean_only=True, contact_floor_source=args.contact_floor_source
    )
    prior_dev = TeacherForcedFrames(
        dataset, codec, splits["dev"], clean_only=True, contact_floor_source=args.contact_floor_source
    )
    report["floor_diagnostics"] = prior_train.floor_diagnostics | prior_dev.floor_diagnostics
    prior = HistoryPrior().to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=0.01)
    random = torch.Generator().manual_seed(args.seed)
    dev = prior_dev.batch(torch.arange(len(prior_dev)), device)
    hv = torch.ones(dev["history_motion"].shape[:2], device=device, dtype=torch.bool)
    baseline = float(weighted_mse(dev["base_mu"], dev["target"]))
    baseline_dense = dense_position_mm(geometry_codec, dev["base_mu"], dev)
    best = baseline_dense if args.prior_selection == "dense" else baseline
    best_step, best_state, selected_representation = 0, cpu_state(prior), baseline
    selected_dense = baseline_dense
    curve = []
    for step in range(1, args.prior_steps + 1):
        prior.train()
        batch = training_batch(prior_train, replay, args.batch_size, random, device, args.replay_probability)
        valid = torch.ones(batch["history_motion"].shape[:2], device=device, dtype=torch.bool)
        prediction = prior(batch["history_motion"], valid, batch["base_mu"])
        loss = physical_objective(geometry_codec, prediction, batch, geometry_weight=args.geometry_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == args.prior_steps:
            prior.eval()
            with torch.no_grad():
                dev_prediction = prior(dev["history_motion"], hv, dev["base_mu"])
                value = float(weighted_mse(dev_prediction, dev["target"]))
                dev_dense = dense_position_mm(geometry_codec, dev_prediction, dev)
            score = dev_dense if args.prior_selection == "dense" else value
            if score < best:
                best, best_step, best_state = score, step, cpu_state(prior)
                selected_representation, selected_dense = value, dev_dense
            curve.append(
                {"step": step, "train_loss": float(loss.detach()), "dev_loss": value, "dev_dense22_mm": dev_dense}
            )
            print(
                f"P {step}/{args.prior_steps}: train={float(loss.detach()):.6f} dev_mse={value:.6f} dev_mm={dev_dense:.3f}",
                flush=True,
            )
    prior.load_state_dict(best_state, strict=True)
    prior.freeze()
    torch.save(
        {
            "state_dict": best_state,
            "kind": "history_prior",
            "scope": report["scope"],
            "selected_step": best_step,
            "reference_mode": codec.reference_mode,
            "selection": args.prior_selection,
        },
        args.output / "prior.pt",
    )
    report["prior"] = {
        "baseline_dev_loss": baseline,
        "selected_dev_loss": selected_representation,
        "baseline_dev_dense22_mm": baseline_dense,
        "selected_dev_dense22_mm": selected_dense,
        "selected_step": best_step,
        "curve": curve,
    }
    del optimizer, best_state, prior_train, prior_dev, dev
    train = TeacherForcedFrames(dataset, codec, splits["train"], contact_floor_source=args.contact_floor_source)
    development = TeacherForcedFrames(dataset, codec, splits["dev"], contact_floor_source=args.contact_floor_source)
    # Holdout takes remain unused for all model selection and this diagnostic.
    indices = torch.linspace(0, len(development) - 1, 64).long()
    evaluation = development.batch(indices, device)
    y_eval = conditions_from_batch(evaluation, prior=prior)
    codec = codec.to(device)
    for source_mode in ("gaussian", "history"):
        torch.manual_seed(args.seed)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device)
        if args.checkpoint:
            load_e7_weights(model, args.checkpoint, weight_source=args.weight_source)
        flow = HistoryFlow(source_mode=source_mode, sigma=args.sigma)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8.5e-5, weight_decay=0.01)
        random = torch.Generator().manual_seed(args.seed + 1)
        noise = torch.Generator(device=device).manual_seed(args.seed + 2)
        fixed_epsilon = torch.randn(evaluation["target"].shape, device=device, generator=noise)
        model.eval()
        with torch.no_grad():
            before = float(
                flow.training_losses(model, evaluation["target"], y_eval, epsilon=fixed_epsilon, t=0.5)["loss"].mean()
            )
        curve = []
        best_flow_mm, selected_flow_step, best_flow_state = float("inf"), 0, None
        for step in range(1, args.flow_steps + 1):
            model.train()
            batch = training_batch(train, replay, args.batch_size, random, device, args.replay_probability)
            action = torch.randint(4, (args.batch_size,), generator=random).to(device)
            y = conditions_from_batch(batch, prior=prior, action=action)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = flow.training_losses(
                    model, batch["target"], y, generator=noise, return_diagnostics=args.geometry_weight > 0
                )
            loss = (
                physical_objective(codec, result["model_output"], batch, geometry_weight=args.geometry_weight)
                if args.geometry_weight
                else result["loss"].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step == args.flow_steps:
                curve.append({"step": step, "train_loss": float(loss.detach())})
                print(f"G/{source_mode} {step}/{args.flow_steps}: train={float(loss.detach()):.6f}", flush=True)
            if args.flow_selection == "dense" and (step % args.eval_every == 0 or step == args.flow_steps):
                model.eval()
                with torch.no_grad():
                    value = dense_position_mm(codec, flow.sample(model, y_eval, epsilon=fixed_epsilon), evaluation)
                if value < best_flow_mm:
                    best_flow_mm, selected_flow_step, best_flow_state = value, step, cpu_state(model)
                curve.append({"step": step, "train_loss": float(loss.detach()), "dev_sample_dense22_mm": value})
                print(f"G/{source_mode} dev sample {step}: {value:.3f} mm", flush=True)
        if best_flow_state is not None:
            model.load_state_dict(best_flow_state, strict=True)
            del best_flow_state
        else:
            selected_flow_step = args.flow_steps
        model.eval()
        with torch.no_grad():
            after = float(
                flow.training_losses(model, evaluation["target"], y_eval, epsilon=fixed_epsilon, t=0.5)["loss"].mean()
            )
        checkpoint_path = args.output / f"g_{source_mode}.pt"
        torch.save(
            {
                "state_dict": cpu_state(model),
                "source_mode": source_mode,
                "sigma": args.sigma,
                "scope": report["scope"],
                "history_length": 20,
                "steps": args.flow_steps,
                "selected_step": selected_flow_step,
                "reference_mode": codec.reference_mode,
                "geometry_weight": args.geometry_weight,
            },
            checkpoint_path,
        )

        def dense_error(prediction):
            body = codec.decode_current(prediction[:, 0], evaluation["previous_reference"])
            return (body.joints[..., :3, 3] - evaluation["target_joints"]).norm(dim=-1).mean(-1) * 1000

        labels = generate_utility_labels(
            model, flow, y_eval, fixed_epsilon, dense_error, checkpoint_id=sha256(checkpoint_path)
        )
        labels["identity"].update(
            metric=report["metric"],
            scope=report["scope"],
            prior_sha256=sha256(args.output / "prior.pt"),
            stats_sha256=report["stats_sha256"],
            reference_mode=codec.reference_mode,
        )
        torch.save(
            {
                **labels,
                "record_ids": [development.record_ids[i] for i in indices],
                "time_indices": [development.time_indices[i] for i in indices],
            },
            args.output / f"utility_diagnostic_{source_mode}.pt",
        )
        errors = labels["errors"]
        report["sources"][source_mode] = {
            "dev_loss_before": before,
            "dev_loss_after": after,
            "selected_step": selected_flow_step,
            "curve": curve,
            "dense22_mm_per_action": dict(zip(("a11", "a10", "a01", "a00"), errors.mean(0).tolist())),
            "dense22_oracle_gap_mm": float((errors[:, 0] - errors.min(1).values).mean()),
            "oracle_action_counts": torch.bincount(errors.argmin(1), minlength=4).tolist(),
            "checkpoint_sha256": labels["identity"]["checkpoint_id"],
        }
        print(json.dumps(report["sources"][source_mode], indent=2), flush=True)
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, optimizer, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["completed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved engineering diagnostic to {args.output}; not a closed-loop/FK result.", flush=True)


if __name__ == "__main__":
    main()
