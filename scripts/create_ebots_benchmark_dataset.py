#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Create an eBots "pickup" benchmark folder from the flat export: one image + one .txt
# per episode, so you can run prepare_batch_input_json and then batch inference.
#
# Prerequisites: Run scripts/export_lerobot_to_flat.py first to get flat_dir with
#   flat_dir/videos/episode_*.mp4 and flat_dir/metas/episode_*.txt
#
# Example:
#   python -m scripts.create_ebots_benchmark_dataset \
#     --flat_dir /path/to/set_1_flat \
#     --output_dir dataset_benchmark_inference/ebots_pickup \
#     --frame_index 0
#   python -m scripts.prepare_batch_input_json \
#     --dataset_path dataset_benchmark_inference/ebots_pickup \
#     --output_path assets/sample_ebots/batch_input_image2world.json \
#     --format cosmos_predictv2p5

import argparse
import os
import sys
import tempfile

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def _expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _atomic_save_png(frame: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_frame_", suffix=".png", dir=os.path.dirname(out_path) or ".")
    os.close(fd)
    try:
        Image.fromarray(frame).save(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Build eBots benchmark folder (images + .txt prompts) from flat export."
    )
    p.add_argument(
        "--flat_dir",
        type=str,
        required=True,
        help="Path to flat export (contains videos/ and metas/ with episode_*.mp4 and episode_*.txt)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder for <episode_id>.png and <episode_id>.txt (e.g. dataset_benchmark_inference/ebots_pickup)",
    )
    p.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index to extract from each video (default: 0)",
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Max number of episodes to process (default: all)",
    )
    return p.parse_args()


def extract_frame(video_path: str, frame_index: int) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n = len(vr)
    if n <= frame_index:
        raise ValueError(f"Video has only {n} frames, cannot read index={frame_index}: {video_path}")
    frame = vr[frame_index].asnumpy()
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 (or HxWx4) uint8 frame, got shape={frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    return frame


def main():
    args = parse_args()
    flat_dir = _expand_path(args.flat_dir)
    output_dir = _expand_path(args.output_dir)
    videos_dir = os.path.join(flat_dir, "videos")
    metas_dir = os.path.join(flat_dir, "metas")

    if not os.path.isdir(videos_dir):
        print(f"Error: videos dir not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(metas_dir):
        print(f"Error: metas dir not found: {metas_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    videos = sorted([f for f in os.listdir(videos_dir) if f.endswith(".mp4")])
    if args.max_episodes is not None:
        videos = videos[: args.max_episodes]

    for i, vid_name in enumerate(videos):
        stem = os.path.splitext(vid_name)[0]
        video_path = os.path.join(videos_dir, vid_name)
        meta_path = os.path.join(metas_dir, stem + ".txt")
        out_png = os.path.join(output_dir, stem + ".png")
        out_txt = os.path.join(output_dir, stem + ".txt")

        if not os.path.exists(meta_path):
            print(f"Skip (no meta): {stem}", flush=True)
            continue

        try:
            frame = extract_frame(video_path, args.frame_index)
            _atomic_save_png(frame, out_png)
        except Exception as e:
            print(f"Skip (video error): {stem} - {e}", flush=True)
            continue

        with open(meta_path) as f:
            prompt = f.read().strip()
        with open(out_txt, "w") as f:
            f.write(prompt)

        print(f"[{i+1}/{len(videos)}] {stem}", flush=True)

    print(f"Done. Created {output_dir} with {len(videos)} image+txt pairs.", flush=True)


if __name__ == "__main__":
    main()
