# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Export LeRobot (ebots) dataset to flat layout for Cosmos-Predict2.5 VideoDataset.
# Works on raw LeRobot with 3 cameras: tiles them into one video per episode, then
# writes flat videos/ + metas/ (no dependency on cosmos-predict2).
#
# Input (LeRobot at input_dir, e.g. $HOME/set_1):
#   meta/episodes.jsonl     # episode_index, tasks[], length
#   videos/chunk-000/observation.images.cam_high/episode_000000.mp4
#   videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
#   videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
#
# Output (flat for Predict2.5 VideoDataset):
#   output_dir/videos/episode_000000.mp4   # one tiled video per episode (2x2: high|left, right|black)
#   output_dir/metas/episode_000000.txt    # one line = task description
#
# Usage:
#   python -m scripts.export_lerobot_to_flat --input_dir $HOME/set_1 --output_dir $HOME/set_1_flat

import argparse
import json
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
from decord import VideoReader, cpu


# Ebots camera keys (same as cosmos-predict2 tile_lerobot_videos)
VIDEO_KEY_HIGH = "observation.images.cam_high"
VIDEO_KEY_LEFT = "observation.images.cam_left_wrist"
VIDEO_KEY_RIGHT = "observation.images.cam_right_wrist"


def load_episode_tasks(meta_dir: Path) -> dict[int, str]:
    """Load episode_index -> task description from meta/episodes.jsonl."""
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        return {}
    tasks = {}
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = obj.get("episode_index")
            task_list = obj.get("tasks") or []
            if ep_idx is not None and task_list:
                tasks[ep_idx] = task_list[0].strip()
    return tasks


def find_chunk_dir(videos_root: Path) -> Path | None:
    """Return first videos/chunk-XXX path."""
    if not videos_root.is_dir():
        return None
    for d in sorted(videos_root.iterdir()):
        if d.is_dir() and d.name.startswith("chunk-"):
            return d
    return None


def get_episode_paths(chunk_dir: Path, episode_index: int) -> tuple[Path | None, Path | None, Path | None]:
    """Return (high_path, left_path, right_path) for episode. None if missing."""
    base = f"episode_{episode_index:06d}.mp4"
    high = chunk_dir / VIDEO_KEY_HIGH / base
    left = chunk_dir / VIDEO_KEY_LEFT / base
    right = chunk_dir / VIDEO_KEY_RIGHT / base
    return (
        high if high.is_file() else None,
        left if left.is_file() else None,
        right if right.is_file() else None,
    )


def tile_three_views_to_mp4(
    high_path: Path,
    left_path: Path,
    right_path: Path,
    out_path: Path,
) -> None:
    """Tile 3 views into 2x2 layout (high|left, right|black) and write one mp4. Same layout as cosmos-predict2 tile_lerobot_videos."""
    vr_h = VideoReader(str(high_path), ctx=cpu(0), num_threads=1)
    vr_l = VideoReader(str(left_path), ctx=cpu(0), num_threads=1)
    vr_r = VideoReader(str(right_path), ctx=cpu(0), num_threads=1)

    n = min(len(vr_h), len(vr_l), len(vr_r))
    if n <= 0:
        raise ValueError(f"Empty video among: {high_path}, {left_path}, {right_path}")

    fps = float(vr_h.get_avg_fps() or 0.0)
    if fps <= 0:
        fps = 15.0

    f0h = vr_h[0].asnumpy()
    if f0h.ndim != 3 or f0h.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {f0h.shape} from {high_path}")
    h, w = f0h.shape[:2]
    f0l = vr_l[0].asnumpy()
    f0r = vr_r[0].asnumpy()
    if f0l.shape[:2] != (h, w) or f0r.shape[:2] != (h, w):
        raise ValueError(f"View resolutions differ: high={f0h.shape}, left={f0l.shape}, right={f0r.shape}")

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_tiled_", suffix=".mp4", dir=str(out_path.parent))
    os.close(fd)
    writer = imageio.get_writer(tmp_path, fps=fps)
    try:
        for i in range(n):
            fh = vr_h[i].asnumpy()
            fl = vr_l[i].asnumpy()
            fr = vr_r[i].asnumpy()
            top = np.concatenate([fh, fl], axis=1)
            bot = np.concatenate([fr, blank], axis=1)
            tiled = np.concatenate([top, bot], axis=0)
            writer.append_data(tiled)
        writer.close()
        os.replace(tmp_path, str(out_path))
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LeRobot (ebots) to flat videos/ + metas/ for Cosmos-Predict2.5 VideoDataset (tiles 3 cams per episode)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path.home() / "set_1"),
        help="LeRobot dataset root (meta/, videos/chunk-*/observation.images.cam_*)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "set_1_flat"),
        help="Output directory; creates output_dir/videos and output_dir/metas",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    meta_dir = input_dir / "meta"
    videos_root = input_dir / "videos"
    episode_tasks = load_episode_tasks(meta_dir)
    if not episode_tasks:
        print("Warning: no entries in meta/episodes.jsonl; metas will use placeholder.")

    chunk_dir = find_chunk_dir(videos_root)
    if chunk_dir is None:
        raise SystemExit(f"No videos/chunk-* found under {videos_root}")

    videos_out = output_dir / "videos"
    metas_out = output_dir / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    exported = 0
    for ep_idx in sorted(episode_tasks.keys()):
        high_p, left_p, right_p = get_episode_paths(chunk_dir, ep_idx)
        if high_p is None or left_p is None or right_p is None:
            print(f"Skip episode {ep_idx}: missing view(s)")
            continue

        base = f"episode_{ep_idx:06d}"
        dest_mp4 = videos_out / f"{base}.mp4"
        if dest_mp4.exists() and not args.overwrite:
            exported += 1
            # still write meta
        else:
            try:
                tile_three_views_to_mp4(high_p, left_p, right_p, dest_mp4)
                exported += 1
            except Exception as e:
                print(f"Skip episode {ep_idx}: {e}")
                continue

        task = episode_tasks.get(ep_idx, "")
        with open(metas_out / f"{base}.txt", "w") as f:
            f.write(task if task else "(no task description)\n")

    print(f"Exported {exported} episodes to {output_dir}")
    print(f"  videos: {videos_out}")
    print(f"  metas:  {metas_out}")


if __name__ == "__main__":
    main()
