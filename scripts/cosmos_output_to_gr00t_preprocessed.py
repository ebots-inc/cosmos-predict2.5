# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Converts Cosmos-Predict2.5 batch inference output (episode_*.mp4 + episode_*.json)
# to the format expected by GR00T-Dreams preprocess_video.py: videos/ and labels/
# with numeric indices and prompts taken from the .json files (not from filenames).
#
# Usage:
#   python -m scripts.cosmos_output_to_gr00t_preprocessed \
#     --source_dir /path/to/batch_output_folder \
#     --output_dir /path/to/preprocessed \
#     [--subdir_name NAME]
#
# Then run GR00T-Dreams preprocess_video and raw_to_lerobot on output_dir.

import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cosmos-Predict2.5 inference output to GR00T-Dreams preprocessed format (videos/ + labels/ with prompts from .json)."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing episode_*.mp4 and episode_*.json from Predict2.5 batch inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. Creates output_dir/videos/ and output_dir/labels/ (no extra subdir).",
    )
    parser.add_argument(
        "--subdir_name",
        type=str,
        default=None,
        help="If set, creates output_dir/subdir_name/videos/ and labels/ instead of output_dir/videos/ and labels/. Omit for flat layout (recommended for GR00T-Dreams).",
    )
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    out_base = Path(args.output_dir).resolve()
    if args.subdir_name:
        out_processed = out_base / args.subdir_name
    else:
        out_processed = out_base
    videos_dir = out_processed / "videos"
    labels_dir = out_processed / "labels"
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(source.glob("*.mp4"))
    if not mp4_files:
        raise SystemExit(f"No .mp4 files found in {source}")

    for idx, mp4_path in enumerate(mp4_files, start=1):
        stem = mp4_path.stem
        json_path = source / f"{stem}.json"
        prompt = ""
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                prompt = (data.get("prompt") or "").strip()
            except Exception as e:
                print(f"Warning: could not read prompt from {json_path}: {e}")
        if not prompt:
            print(f"Warning: no prompt for {mp4_path.name}, using placeholder.")
            prompt = stem.replace("_", " ")

        label_path = labels_dir / f"{idx}.txt"
        with open(label_path, "w") as f:
            f.write(prompt)

        dest_mp4 = videos_dir / f"{idx}.mp4"
        shutil.copy2(mp4_path, dest_mp4)
        print(f"  {mp4_path.name} -> {idx}.mp4, prompt: {prompt[:60]}...")

    print(f"Done. Wrote {len(mp4_files)} videos and labels to {out_processed}")


if __name__ == "__main__":
    main()
