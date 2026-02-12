# FILE: run_pipeline.py
#
# DESCRIPTION:
#     Run the end-to-end GPT labeling pipeline: OCR -> raw GPT labels -> canonical
#     fields -> canonical relabeling. Optionally upload and build a model.

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GPT labeling pipeline.")
    parser.add_argument("--upload", action="store_true", help="Upload training data after relabeling.")
    parser.add_argument("--build", action="store_true", help="Build model after upload.")
    return parser.parse_args()


def run_step(title: str, command: List[str]) -> None:
    print(f"\n== {title} ==")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    steps = [
        ("Generate OCR", [sys.executable, str(script_dir / "generate_ocr_data.py")]),
        ("Generate raw GPT labels", [sys.executable, str(script_dir / "label_ocr_with_gpt.py")]),
        ("Normalize canonical fields", [sys.executable, str(script_dir / "normalize_fields_with_gpt.py")]),
        ("Relabel with canonical fields", [sys.executable, str(script_dir / "relabel_with_canonical_fields.py")]),
    ]

    if args.upload:
        steps.append(("Upload training data", [sys.executable, str(script_dir / "upload_training_data.py")]))
    if args.build:
        steps.append(("Build custom model", [sys.executable, str(script_dir / "build_custom_model.py"), "--build"]))

    for title, command in steps:
        run_step(title, command)


if __name__ == "__main__":
    main()
