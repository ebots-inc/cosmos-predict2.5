# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Update the "prompt" field in Cosmos-Predict2.5 inference output JSON files
# (e.g. episode_000000.json) under one or more directories. Other keys are preserved.
#
# Usage:
#   python -m scripts.update_inference_json_prompts \
#     --dirs /path/to/batch_output_folder \
#     --prompt "Your new prompt text here."
#
#   python -m scripts.update_inference_json_prompts \
#     --dirs /path/to/folder1 /path/to/folder2 \
#     --prompt_file /path/to/prompt.txt
#
#   python -m scripts.update_inference_json_prompts --dirs /path/to/folder --prompt "..." --dry_run

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Update the 'prompt' field in inference output .json files (e.g. episode_*.json)."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing .json files to update.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="New prompt text. Ignored if --prompt_file is set.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Path to a file whose content is the new prompt (overrides --prompt).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be updated without writing files.",
    )
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            new_prompt = f.read().strip()
    elif args.prompt:
        new_prompt = args.prompt.strip()
    else:
        raise SystemExit("Provide either --prompt or --prompt_file.")

    if not new_prompt:
        raise SystemExit("Prompt text is empty.")

    updated = 0
    for dir_path in args.dirs:
        d = Path(dir_path)
        if not d.is_dir():
            print(f"Skip (not a directory): {d}")
            continue
        for jpath in sorted(d.glob("*.json")):
            try:
                with open(jpath) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skip (read error) {jpath}: {e}")
                continue
            if "prompt" not in data:
                print(f"Skip (no 'prompt' key) {jpath}")
                continue
            old = data.get("prompt", "")
            if old == new_prompt:
                print(f"Unchanged {jpath}")
                continue
            if args.dry_run:
                print(f"Would update {jpath}: prompt len {len(old)} -> {len(new_prompt)}")
                updated += 1
                continue
            data["prompt"] = new_prompt
            with open(jpath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated {jpath}")
            updated += 1

    print(f"Total: {updated} file(s) updated." + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
