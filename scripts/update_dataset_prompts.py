#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Update all .txt prompt files in one or more dataset folders with a new prompt.
# Use this to try different prompts for batch inference without regenerating images.
#
# Example:
#   # Single folder
#   python -m scripts.update_dataset_prompts \
#     --dirs assets/sample_ebots/dataset_benchmark_inference/ebots_pickup/fi_0 \
#     --prompt "The robot uses the left arm to reach into the bin, grasp the white wire, lift it out of the bin, and hold it clearly in the gripper."
#
#   # Multiple folders (fi_0 and fi_93)
#   python -m scripts.update_dataset_prompts \
#     --dirs assets/sample_ebots/dataset_benchmark_inference/ebots_pickup/fi_0 \
#            assets/sample_ebots/dataset_benchmark_inference/ebots_pickup/fi_93 \
#     --prompt "Your new prompt here."
#
#   # Prompt from a file (e.g. for long prompts)
#   python -m scripts.update_dataset_prompts \
#     --dirs dataset_benchmark_inference/ebots_pickup/fi_0 \
#     --prompt_file my_prompt.txt
#
#   # Dry run: show which files would be updated
#   python -m scripts.update_dataset_prompts --dirs fi_0 --prompt "Test" --dry_run

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Update all .txt prompt files in dataset folder(s) with a new prompt.",
        epilog="Provide exactly one of --prompt or --prompt_file.",
    )
    p.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing .txt files to overwrite (e.g. fi_0 fi_93)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The new prompt text (one line; use --prompt_file for multi-line)",
    )
    g.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a file whose content is the new prompt",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which files would be updated, do not write",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.prompt_file:
        path = os.path.expanduser(args.prompt_file)
        if not os.path.isfile(path):
            raise SystemExit(f"Prompt file not found: {path}")
        prompt = open(path).read().strip()
    else:
        prompt = args.prompt.strip()

    if not prompt:
        raise SystemExit("Prompt is empty.")

    updated = 0
    for dir_path in args.dirs:
        d = os.path.expanduser(dir_path)
        if not os.path.isdir(d):
            print(f"Skip (not a directory): {d}", flush=True)
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".txt"):
                continue
            fpath = os.path.join(d, name)
            if args.dry_run:
                print(f"Would update: {fpath}", flush=True)
            else:
                with open(fpath, "w") as f:
                    f.write(prompt)
                print(f"Updated: {fpath}", flush=True)
            updated += 1

    print(f"\nTotal: {updated} .txt file(s) {'would be ' if args.dry_run else ''}updated.", flush=True)


if __name__ == "__main__":
    main()
