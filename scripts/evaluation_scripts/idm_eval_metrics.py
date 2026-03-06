#!/usr/bin/env python3
"""
Compute scalar action metrics (MSE, MAE, correlation) between IDM inference and real
action parquet data. Use this to compare checkpoints: run dump_idm_actions with different
checkpoints, then run this script on each output; lower MSE/MAE = better.

Avoids frame-length issues by optionally resampling both trajectories to the same
number of steps per episode (normalized time), so you get a single number per checkpoint.

Example (compare two checkpoints):
  # Checkpoint 10k
  python IDM_dump/dump_idm_actions.py --checkpoint .../checkpoint-10000 --dataset ... --output_dir .../lerobot_10k
  python scripts/idm_eval_metrics.py --inference_data_dir .../lerobot_10k/data --real_data_dir .../real/data --max_episodes 10

  # Checkpoint 15k
  python IDM_dump/dump_idm_actions.py --checkpoint .../checkpoint-15000 --dataset ... --output_dir .../lerobot_15k
  python scripts/idm_eval_metrics.py --inference_data_dir .../lerobot_15k/data --real_data_dir .../real/data --max_episodes 10

  # Compare the printed "Overall action MSE" to see if more steps helped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_episodes_parquet(
    data_root: Path,
    max_episodes: int,
    action_key: str = "action",
) -> tuple[np.ndarray, list[int]]:
    """Load action arrays and per-episode lengths from parquet files."""
    parquet_paths = sorted(data_root.glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {data_root}")
    paths = parquet_paths[:max_episodes]
    actions_list = []
    lengths = []
    for p in tqdm(paths, desc="Loading episodes"):
        df = pd.read_parquet(p)
        if action_key in df.columns:
            a = np.stack([np.asarray(x, dtype=np.float32) for x in df[action_key]])
            actions_list.append(a)
            lengths.append(len(df))
    actions = np.concatenate(actions_list, axis=0) if actions_list else np.zeros((0, 0))
    return actions, lengths


def resample_to_length(arr: np.ndarray, new_len: int) -> np.ndarray:
    """Resample trajectory to new_len steps along axis 0 (time)."""
    T, D = arr.shape
    if T == new_len:
        return arr
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_len)
    out = np.zeros((new_len, D), dtype=arr.dtype)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, arr[:, d])
    return out


def compute_metrics(
    inf_actions: np.ndarray,
    real_actions: np.ndarray,
    len_inf: list[int],
    len_real: list[int],
    normalize_time: bool = True,
    target_steps: int | None = None,
) -> dict:
    """
    Compute per-episode and overall MSE, MAE, and correlation.
    If normalize_time=True, resample both to the same length per episode (min of the two,
    or target_steps if set) so frame-length mismatch doesn't dominate.
    """
    n_episodes = min(len(len_inf), len(len_real))
    action_dim = min(inf_actions.shape[1], real_actions.shape[1])
    if action_dim == 0:
        return {}

    mse_per_ep, mae_per_ep = [], []
    all_inf, all_real = [], []

    for ep in range(n_episodes):
        start_inf = sum(len_inf[:ep])
        end_inf = start_inf + len_inf[ep]
        start_real = sum(len_real[:ep])
        end_real = start_real + len_real[ep]
        a_inf = inf_actions[start_inf:end_inf, :action_dim]
        a_real = real_actions[start_real:end_real, :action_dim]
        T_inf, T_real = a_inf.shape[0], a_real.shape[0]

        if normalize_time:
            T = target_steps if target_steps is not None else min(T_inf, T_real)
            T = max(1, T)
            a_inf = resample_to_length(a_inf, T)
            a_real = resample_to_length(a_real, T)
        else:
            # Align by resampling real to inference length (same as compare_inference_vs_real)
            if T_real != T_inf:
                x = np.linspace(0, T_inf - 1, T_real)
                a_real_resampled = np.zeros((T_inf, action_dim), dtype=a_real.dtype)
                for d in range(action_dim):
                    a_real_resampled[:, d] = np.interp(np.arange(T_inf), x, a_real[:, d])
                a_real = a_real_resampled

        all_inf.append(a_inf)
        all_real.append(a_real)
        diff = a_inf - a_real
        mse_per_ep.append(float(np.mean(diff ** 2)))
        mae_per_ep.append(float(np.mean(np.abs(diff))))

    all_inf = np.concatenate(all_inf, axis=0)
    all_real = np.concatenate(all_real, axis=0)

    # Correlation per dimension (over all episodes)
    corr_per_dim = []
    for d in range(action_dim):
        c = np.corrcoef(all_inf[:, d], all_real[:, d])[0, 1]
        corr_per_dim.append(float(c) if np.isfinite(c) else 0.0)

    overall_mse = float(np.mean((all_inf - all_real) ** 2))
    overall_mae = float(np.mean(np.abs(all_inf - all_real)))

    return {
        "n_episodes": n_episodes,
        "action_dim": action_dim,
        "normalize_time": normalize_time,
        "target_steps": target_steps,
        "mse_per_episode": mse_per_ep,
        "mae_per_episode": mae_per_ep,
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "correlation_per_dim": corr_per_dim,
        "mean_correlation": float(np.mean(corr_per_dim)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute action MSE/MAE/correlation between IDM inference and real parquet (for checkpoint comparison)."
    )
    parser.add_argument("--inference_data_dir", type=Path, required=True, help="Path to inference dataset data/ (parquet)")
    parser.add_argument("--real_data_dir", type=Path, required=True, help="Path to real reference dataset data/ (parquet)")
    parser.add_argument("--max_episodes", type=int, default=10, help="Max episodes to load (default: 10)")
    parser.add_argument(
        "--normalize_time",
        action="store_true",
        default=True,
        help="Resample both to same length per episode to reduce frame-length bias (default: True)",
    )
    parser.add_argument(
        "--no_normalize_time",
        action="store_false",
        dest="normalize_time",
        help="Do not resample; align real to inference length (like compare_inference_vs_real plots)",
    )
    parser.add_argument(
        "--target_steps",
        type=int,
        default=None,
        help="When normalize_time=True, resample to this many steps per episode (default: min(inf_len, real_len))",
    )
    parser.add_argument("--output_json", type=Path, default=None, help="Save metrics to this JSON file")
    args = parser.parse_args()

    for d in (args.inference_data_dir, args.real_data_dir):
        if not d.resolve().is_dir():
            raise SystemExit(f"Not a directory: {d}")

    print("Loading inference actions...")
    inf_actions, len_inf = load_episodes_parquet(args.inference_data_dir, args.max_episodes)
    print("Loading real actions...")
    real_actions, len_real = load_episodes_parquet(args.real_data_dir, args.max_episodes)

    print(f"Inference: {inf_actions.shape[0]} frames, {inf_actions.shape[1]} action dims, lengths: {len_inf}")
    print(f"Real:      {real_actions.shape[0]} frames, {real_actions.shape[1]} action dims, lengths: {len_real}")

    metrics = compute_metrics(
        inf_actions,
        real_actions,
        len_inf,
        len_real,
        normalize_time=args.normalize_time,
        target_steps=args.target_steps,
    )

    print("\n--- IDM vs real action metrics ---")
    print(f"Episodes: {metrics['n_episodes']}, action dims: {metrics['action_dim']}")
    print(f"Normalize time: {metrics['normalize_time']}, target_steps: {metrics['target_steps']}")
    print(f"Overall action MSE:  {metrics['overall_mse']:.6f}")
    print(f"Overall action MAE:  {metrics['overall_mae']:.6f}")
    print(f"Mean correlation (per dim): {metrics['mean_correlation']:.4f}")
    print(f"MSE per episode: {[f'{x:.6f}' for x in metrics['mse_per_episode']]}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output_json}")

    print("\nUse the same --inference_data_dir/--real_data_dir with different checkpoint outputs to compare checkpoints.")


if __name__ == "__main__":
    main()
