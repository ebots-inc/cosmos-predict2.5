#!/usr/bin/env python3
"""
Compare inference vs real data using only the first N frames of each real episode
(no resampling). Real episodes are truncated to match inference episode length
(e.g. first 93 frames), so comparison is 1:1 frame-aligned.

Use this when inference episodes have a fixed length (e.g. 93 frames) and you
want to compare against the same number of initial frames from real demos
without any temporal resampling.

Example:
  python scripts/compare_inference_vs_real_first_n_frames.py \\
    --inference_data_dir /path/to/batch_inference_test_long_pickup_93_fi_20_lerobot/data \\
    --real_data_dir /path/to/pickUp_jointStates_cartStates_stage_v4Config_0pct_224res/data \\
    --real_max_frames 93 \\
    --max_episodes 10 \\
    --output_dir ./eval_plots_first_n
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_episodes_parquet(
    data_root: Path,
    max_episodes: int,
    state_key: str = "observation.state",
    action_key: str = "action",
    max_frames_per_episode: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load state and action arrays from parquet files (first max_episodes episodes).

    If max_frames_per_episode is set, each episode is truncated to the first
    max_frames_per_episode frames (no resampling). Shorter episodes are kept as-is.
    """
    parquet_paths = sorted(data_root.glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {data_root}")
    paths = parquet_paths[:max_episodes]
    states_list = []
    actions_list = []
    lengths = []
    for p in tqdm(paths, desc="Loading episodes"):
        df = pd.read_parquet(p)
        if max_frames_per_episode is not None and len(df) > max_frames_per_episode:
            df = df.head(max_frames_per_episode)
        if state_key in df.columns:
            s = np.stack([np.asarray(x, dtype=np.float32) for x in df[state_key]])
            states_list.append(s)
        if action_key in df.columns:
            a = np.stack([np.asarray(x, dtype=np.float32) for x in df[action_key]])
            actions_list.append(a)
        lengths.append(len(df))
    states = np.concatenate(states_list, axis=0) if states_list else np.zeros((0, 0))
    actions = np.concatenate(actions_list, axis=0) if actions_list else np.zeros((0, 0))
    return states, actions, lengths


def plot_distribution_comparison(
    inference_states: np.ndarray,
    inference_actions: np.ndarray,
    real_states: np.ndarray,
    real_actions: np.ndarray,
    out_dir: Path,
    inference_label: str = "inference",
    real_label: str = "real",
    max_dims_to_plot: int = 20,
) -> None:
    """Plot per-dimension distribution (mean ± std, min–max) for state and action."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _summary(arr: np.ndarray, name: str) -> dict:
        if arr.size == 0:
            return {}
        return {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
        }

    def _plot_one(
        inf_arr: np.ndarray,
        real_arr: np.ndarray,
        title: str,
        filename: str,
        inf_label: str,
        real_label: str,
    ) -> None:
        nd_inf = inf_arr.shape[1] if inf_arr.ndim > 1 and inf_arr.size else 0
        nd_real = real_arr.shape[1] if real_arr.ndim > 1 and real_arr.size else 0
        if nd_inf == 0 and nd_real == 0:
            return
        n_plot = min(nd_inf, nd_real, max_dims_to_plot) if (nd_inf and nd_real) else min(max(nd_inf, nd_real), max_dims_to_plot)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax_mean, ax_range = axes

        x = np.arange(n_plot)
        width = 0.35
        if nd_inf >= n_plot and inf_arr.size:
            s_inf = _summary(inf_arr, "inf")
            ax_mean.bar(x - width / 2, s_inf["mean"][:n_plot], width, yerr=s_inf["std"][:n_plot], label=inf_label, alpha=0.8)
            ax_range.bar(x - width / 2, s_inf["max"][:n_plot] - s_inf["min"][:n_plot], width, bottom=s_inf["min"][:n_plot], label=inf_label, alpha=0.8)
        if nd_real >= n_plot and real_arr.size:
            s_real = _summary(real_arr, "real")
            ax_mean.bar(x + width / 2, s_real["mean"][:n_plot], width, yerr=s_real["std"][:n_plot], label=real_label, alpha=0.8)
            ax_range.bar(x + width / 2, s_real["max"][:n_plot] - s_real["min"][:n_plot], width, bottom=s_real["min"][:n_plot], label=real_label, alpha=0.8)

        ax_mean.set_ylabel("Mean ± Std")
        ax_mean.set_title(f"{title} (first {n_plot} dims)")
        ax_mean.legend()
        ax_mean.grid(True, alpha=0.3)
        ax_range.set_ylabel("Min–Max range")
        ax_range.set_xlabel("Dimension index")
        ax_range.legend()
        ax_range.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    _plot_one(
        inference_states, real_states,
        "State distribution comparison",
        "state_distribution.png",
        inference_label, real_label,
    )
    _plot_one(
        inference_actions, real_actions,
        "Action distribution comparison",
        "action_distribution.png",
        inference_label, real_label,
    )


def plot_action_timeseries_only(
    inf_actions: np.ndarray,
    real_actions: np.ndarray,
    episode_lengths_inf: list[int],
    episode_lengths_real: list[int],
    out_dir: Path,
    n_dims: int = 6,
    episode_idx: int = 0,
    filename: str | None = None,
    frame_aligned: bool = True,
) -> None:
    """Plot action dimensions over time for one episode (inference vs real).

    When frame_aligned is True and both episodes have the same length, plot
    1:1 frame alignment (no resampling). Otherwise fall back to linspace resampling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dims = min(n_dims, inf_actions.shape[1], real_actions.shape[1])
    if n_dims == 0:
        return
    start_inf = sum(episode_lengths_inf[:episode_idx])
    end_inf = start_inf + episode_lengths_inf[episode_idx]
    start_real = sum(episode_lengths_real[:episode_idx])
    end_real = start_real + episode_lengths_real[episode_idx]
    a_inf = inf_actions[start_inf:end_inf, :n_dims]
    a_real = real_actions[start_real:end_real, :n_dims]
    T_inf, T_real = a_inf.shape[0], a_real.shape[0]
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims))
    if n_dims == 1:
        axes = [axes]
    for i in range(n_dims):
        axes[i].plot(a_inf[:, i], label="inference", color="C0")
        if frame_aligned and T_inf == T_real:
            axes[i].plot(a_real[:, i], label="real", color="C1", alpha=0.8)
        else:
            axes[i].plot(np.linspace(0, T_inf - 1, T_real), a_real[:, i], label="real", color="C1", alpha=0.8)
        axes[i].set_ylabel(f"Action dim {i}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"Action comparison (episode {episode_idx}) — first-N frames, no resampling")
    plt.tight_layout()
    out_name = filename if filename is not None else "timeseries_action_only.png"
    plt.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series_sample(
    inf_states: np.ndarray,
    inf_actions: np.ndarray,
    real_states: np.ndarray,
    real_actions: np.ndarray,
    episode_lengths_inf: list[int],
    episode_lengths_real: list[int],
    out_dir: Path,
    n_dims: int = 6,
    episode_idx: int = 0,
    filename: str | None = None,
    frame_aligned: bool = True,
) -> None:
    """Plot state/action dimensions over time for one episode (inference vs real).

    When frame_aligned is True and both episodes have the same length, plot
    1:1 frame alignment (no resampling).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dims = min(n_dims, inf_states.shape[1] if inf_states.size else 0, real_states.shape[1] if real_states.size else 0)
    if n_dims == 0:
        return
    start_inf = sum(episode_lengths_inf[:episode_idx])
    end_inf = start_inf + episode_lengths_inf[episode_idx]
    s_inf = inf_states[start_inf:end_inf, :n_dims]
    a_inf = inf_actions[start_inf:end_inf, :n_dims]
    start_real = sum(episode_lengths_real[:episode_idx])
    end_real = start_real + episode_lengths_real[episode_idx]
    s_real = real_states[start_real:end_real, :n_dims]
    a_real = real_actions[start_real:end_real, :n_dims]
    T_inf, T_real = s_inf.shape[0], s_real.shape[0]
    same_len = frame_aligned and (T_inf == T_real)
    fig, axes = plt.subplots(n_dims, 2, figsize=(14, 2 * n_dims))
    for i in range(n_dims):
        axes[i, 0].plot(s_inf[:, i], label="inference", color="C0")
        if same_len:
            axes[i, 0].plot(s_real[:, i], label="real", color="C1", alpha=0.8)
        else:
            axes[i, 0].plot(np.linspace(0, T_inf - 1, T_real), s_real[:, i], label="real", color="C1", alpha=0.8)
        axes[i, 0].set_ylabel(f"State dim {i}")
        axes[i, 0].legend(loc="upper right", fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(a_inf[:, i], label="inference", color="C0")
        if same_len:
            axes[i, 1].plot(a_real[:, i], label="real", color="C1", alpha=0.8)
        else:
            axes[i, 1].plot(np.linspace(0, T_inf - 1, T_real), a_real[:, i], label="real", color="C1", alpha=0.8)
        axes[i, 1].set_ylabel(f"Action dim {i}")
        axes[i, 1].legend(loc="upper right", fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
    axes[0, 0].set_title(f"State (episode {episode_idx}) — first-N frames")
    axes[0, 1].set_title(f"Action (episode {episode_idx}) — first-N frames")
    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    plt.tight_layout()
    out_name = filename if filename is not None else "timeseries_episode_sample.png"
    plt.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_histograms_global(
    inference_states: np.ndarray,
    inference_actions: np.ndarray,
    real_states: np.ndarray,
    real_actions: np.ndarray,
    out_dir: Path,
    state_dims_for_hist: int | None = 17,
) -> None:
    """Global histograms for quick sanity check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nd_inf_s = inference_states.shape[1] if inference_states.ndim > 1 and inference_states.size else 0
    nd_real_s = real_states.shape[1] if real_states.ndim > 1 and real_states.size else 0
    if state_dims_for_hist is not None and (nd_inf_s > state_dims_for_hist or nd_real_s > state_dims_for_hist):
        n_state = min(state_dims_for_hist, nd_inf_s, nd_real_s) if (nd_inf_s and nd_real_s) else min(state_dims_for_hist, max(nd_inf_s, nd_real_s))
        inf_state_hist = inference_states[:, :n_state].flatten()
        real_state_hist = real_states[:, :n_state].flatten()
    else:
        inf_state_hist = inference_states.flatten()
        real_state_hist = real_states.flatten()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for arr, label, ax in [
        (inf_state_hist, "Inference state", axes[0, 0]),
        (real_state_hist, "Real state", axes[0, 1]),
        (inference_actions.flatten(), "Inference action", axes[1, 0]),
        (real_actions.flatten(), "Real action", axes[1, 1]),
    ]:
        if arr.size:
            ax.hist(arr, bins=80, alpha=0.7, density=True, label=label)
        ax.set_title(label)
        ax.set_xlabel("Value (each state/action coordinate)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "histograms_global.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference vs real (first N frames of real only, no resampling)"
    )
    parser.add_argument(
        "--inference_data_dir",
        type=Path,
        required=True,
        help="Path to inference dataset data/ folder (contains chunk-XXX/*.parquet)",
    )
    parser.add_argument(
        "--real_data_dir",
        type=Path,
        required=True,
        help="Path to real dataset data/ folder",
    )
    parser.add_argument(
        "--real_max_frames",
        type=int,
        default=93,
        help="Use only the first N frames of each real episode to match inference (default: 93). No resampling.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=10,
        help="Max episodes to load from each dataset (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./eval_plots_first_n"),
        help="Directory to save plots (default: ./eval_plots_first_n)",
    )
    parser.add_argument(
        "--inference_label",
        type=str,
        default="inference",
        help="Label for inference in plots",
    )
    parser.add_argument(
        "--real_label",
        type=str,
        default="real",
        help="Label for real data in plots",
    )
    parser.add_argument(
        "--max_dims",
        type=int,
        default=20,
        help="Max dimensions to show in bar plots (default: 20)",
    )
    parser.add_argument(
        "--timeseries_episode",
        type=int,
        default=0,
        help="Episode index for time-series sample plot (default: 0)",
    )
    parser.add_argument(
        "--timeseries_dims",
        type=int,
        default=17,
        help="Number of action/state dims in time-series plot (default: 17)",
    )
    parser.add_argument(
        "--timeseries_all_episodes",
        action="store_true",
        help="Save one time-series plot per episode",
    )
    parser.add_argument(
        "--state_dims_histogram",
        type=int,
        default=17,
        help="First N state dims for global histograms when dims differ (default: 17). Set 0 to use all.",
    )
    args = parser.parse_args()
    inference_root = args.inference_data_dir.resolve()
    real_root = args.real_data_dir.resolve()
    for d in (inference_root, real_root):
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    print("Loading inference episodes (full length)...")
    inf_states, inf_actions, len_inf = load_episodes_parquet(
        inference_root, args.max_episodes
    )
    print(f"Loading real episodes (first {args.real_max_frames} frames only, no resampling)...")
    real_states, real_actions, len_real = load_episodes_parquet(
        real_root, args.max_episodes, max_frames_per_episode=args.real_max_frames
    )
    print(f"Inference: {inf_states.shape[0]} frames, state dim {inf_states.shape[1]}, action dim {inf_actions.shape[1]}")
    print(f"Real:      {real_states.shape[0]} frames (first {args.real_max_frames} per episode), state dim {real_states.shape[1]}, action dim {real_actions.shape[1]}")
    print(f"Inference lengths: {len_inf}")
    print(f"Real lengths (after truncation): {len_real}")
    if len_inf != len_real or (len_inf and any(a != b for a, b in zip(len_inf, len_real))):
        print("WARNING: Episode length mismatch — time-series will use resampling for those episodes.")
    else:
        print("Episode lengths match — time-series use 1:1 frame alignment (no resampling).")
    print(f"Distribution/histogram: aggregated over all {args.max_episodes} episodes.")
    if args.timeseries_all_episodes:
        print(f"Time-series: one per episode (timeseries_*_ep01.png .. ep{args.max_episodes:02d}.png).")
    else:
        print(f"Time-series: episode index {args.timeseries_episode} only.")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plot_distribution_comparison(
        inf_states, inf_actions, real_states, real_actions,
        out, args.inference_label, args.real_label, args.max_dims,
    )
    state_dims_hist = args.state_dims_histogram if args.state_dims_histogram else None
    plot_histograms_global(inf_states, inf_actions, real_states, real_actions, out, state_dims_for_hist=state_dims_hist)

    n_ts_dims = min(args.timeseries_dims, inf_actions.shape[1], real_actions.shape[1]) if (inf_actions.shape[1] and real_actions.shape[1]) else args.timeseries_dims
    num_episodes = min(len(len_inf), len(len_real))
    episode_indices = range(num_episodes) if args.timeseries_all_episodes else [args.timeseries_episode]
    frame_aligned = len_inf == len_real and (not len_inf or all(a == b for a, b in zip(len_inf, len_real)))

    if inf_states.shape[1] == real_states.shape[1] and inf_actions.shape[1] == real_actions.shape[1]:
        for ep_idx in episode_indices:
            if ep_idx >= num_episodes:
                continue
            fname = f"timeseries_episode_sample_ep{ep_idx + 1:02d}.png" if args.timeseries_all_episodes else None
            plot_time_series_sample(
                inf_states, inf_actions, real_states, real_actions,
                len_inf, len_real, out, n_dims=min(n_ts_dims, inf_states.shape[1], real_states.shape[1]),
                episode_idx=ep_idx, filename=fname, frame_aligned=frame_aligned,
            )
    elif inf_actions.shape[1] == real_actions.shape[1] and inf_actions.shape[1] > 0:
        for ep_idx in episode_indices:
            if ep_idx >= num_episodes:
                continue
            fname = f"timeseries_action_only_ep{ep_idx + 1:02d}.png" if args.timeseries_all_episodes else None
            plot_action_timeseries_only(
                inf_actions, real_actions, len_inf, len_real, out,
                n_dims=n_ts_dims, episode_idx=ep_idx, filename=fname, frame_aligned=frame_aligned,
            )
    else:
        print("Skipping time-series overlay (state/action dims differ).")
    print(f"Plots saved to {out}")


if __name__ == "__main__":
    main()
