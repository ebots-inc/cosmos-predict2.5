# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare batch input for inference.

Cosmos-Predict2 format (default): single JSON with input_video, prompt, output_video per item.
Cosmos-Predict2.5 format: .jsonl file, one line per sample, for use with examples/inference.py -i batch.jsonl.
"""

import argparse
import glob
import json
import os


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)

# DEFAULT_NEGATIVE_PROMPT = ("extra arms, using the right arm, switching arms, multiple wires, missing wire, wire disappears, wire duplicates, wire changes color, wire teleports, grasp happens without contact, gripper closes in empty space, repeated grasp attempts, retry motions, oscillation, jittery end-effector, jerky motion, sudden jumps, unnatural bending of the arm, broken kinematics, camera moves, zoom, camera shake, viewpoint change, lighting flicker, exposure pumping, motion blur, heavy grain, compression artifacts, low frame rate, choppy motion.")

# DEFAULT_NEGATIVE_PROMPT = ("slow motion, hesitation, lingering, prolonged alignment, hovering, failing to grasp, near-contact without grasping, weak grasp, slipping, delayed grasp, incomplete pickup, repeated grasp attempts, re-grasp, open-close cycles, tapping, oscillation, jitter, jerky motion, sudden jumps, broken kinematics, wire disappears, wire teleports, multiple wires, extra arms, using the right arm, camera movement, viewpoint change, lighting flicker")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare batch input JSON for inference.",
        epilog=(
            "dataset_path: folder of input images (.jpg/.png) and matching .txt prompts (same basename, e.g. frame_01.png + frame_01.txt). "
            "output_path: where to write the batch JSON (.json) or .jsonl. "
            "For cosmos_predictv2p5, generated videos are saved to the directory you pass with -o when running examples/inference.py (--save_path is ignored)."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        required=True,
        help="One or more folders with images (.jpg/.png) and .txt prompts (same basename). If multiple, use same number of --output_path.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="[Cosmos-Predict2 only] Directory for output videos. Ignored for --format cosmos_predictv2p5 (use -o in examples/inference.py instead)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        nargs="+",
        required=True,
        help="One or more output paths for the batch file(s). If multiple, must match --dataset_path count (pairwise).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["cosmos_predict2", "cosmos_predictv2p5"],
        default="cosmos_predict2",
        help="Output format: cosmos_predict2 (single JSON) or cosmos_predictv2p5 (.jsonl for examples/inference.py -i)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="448,448",
        help="Resolution H,W for cosmos_predictv2p5 (default: 448,448)",
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=93,
        help="Frames per video for cosmos_predictv2p5 (default: 93)",
    )
    parser.add_argument(
        "--enable_autoregressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable autoregressive generation for longer videos (default: True, matches ebots_image2world_long.json)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=93,
        help="Chunk size for autoregressive mode (default: 93)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1,
        help="Chunk overlap for autoregressive mode (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_paths = [os.path.abspath(d) for d in args.dataset_path]
    output_paths = list(args.output_path)

    if len(dataset_paths) != len(output_paths):
        raise SystemExit(
            f"--dataset_path and --output_path must have the same number of arguments "
            f"(got {len(dataset_paths)} vs {len(output_paths)})."
        )

    for dataset_path, output_path in zip(dataset_paths, output_paths):
        _process_one(dataset_path, output_path, args)


def _process_one(dataset_path: str, output_path: str, args) -> None:
    save_path = args.save_path

    input_files = sorted(
        glob.glob(os.path.join(dataset_path, "*.jpg")) + glob.glob(os.path.join(dataset_path, "*.png"))
    )
    if not input_files:
        print(f"No .jpg/.png found in {dataset_path}", flush=True)
        return

    if args.format == "cosmos_predictv2p5":
        if not output_path.endswith(".jsonl"):
            output_path = output_path.replace(".json", ".jsonl") if output_path.endswith(".json") else output_path.rstrip("/") + ".jsonl"
        count = 0
        with open(output_path, "w") as f:
            for input_file in input_files:
                prompt_file = input_file.replace(".jpg", ".txt").replace(".png", ".txt")
                if not os.path.exists(prompt_file):
                    prompt_file = input_file.replace(".jpg", "..txt").replace(".png", "..txt")
                if not os.path.exists(prompt_file):
                    print(f"Skip (no prompt): {input_file}", flush=True)
                    continue
                name = os.path.splitext(os.path.basename(input_file))[0]
                sample = {
                    "inference_type": "image2world",
                    "name": name,
                    "prompt": open(prompt_file).read().strip(),
                    "input_path": os.path.abspath(input_file),
                    "resolution": args.resolution,
                    "num_output_frames": args.num_output_frames,
                    "guidance": 7,
                    "seed": 0,
                    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                    "enable_autoregressive": args.enable_autoregressive,
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                }
                f.write(json.dumps(sample) + "\n")
                count += 1
        print(f"Saved {count} items to {output_path} (Cosmos-Predict2.5 .jsonl)", flush=True)
        return

    # Cosmos-Predict2 format
    if not args.save_path:
        raise SystemExit("--save_path is required for --format cosmos_predict2 (output directory for videos).")
    save_path = args.save_path
    output_json = []
    for input_file in input_files:
        print(input_file, flush=True)
        prompt_file = input_file.replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(prompt_file):
            prompt_file = input_file.replace(".jpg", "..txt").replace(".png", "..txt")

        output_json.append(
            {
                "input_video": input_file,
                "prompt": open(prompt_file).read(),
                "output_video": os.path.join(
                    save_path, os.path.basename(input_file.replace(".jpg", ".mp4").replace(".png", ".mp4"))
                ),
            }
        )
    print(f"Saved {len(output_json)} items to {output_path}", flush=True)
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    main()
