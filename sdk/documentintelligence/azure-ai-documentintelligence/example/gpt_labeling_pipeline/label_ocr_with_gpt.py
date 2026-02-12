# FILE: label_ocr_with_gpt.py
#
# DESCRIPTION:
#     Use Azure OpenAI to propose raw key-value labels from OCR content.
#
# USAGE:
#     python label_ocr_with_gpt.py
#
#     Optional arguments:
#       --source   Directory containing *.ocr.json files.
#       --output   Output directory for raw label files.
#       --endpoint Azure OpenAI endpoint (https://<resource>.openai.azure.com).
#       --model    Azure OpenAI deployment name (defaults to gpt-5.2).

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from pipeline_utils import create_openai_client, parse_json_from_response, resolve_azure_openai_endpoint


def parse_args(default_source: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw labels from OCR using Azure OpenAI.")
    parser.add_argument("--source", default=str(default_source), help="Directory with *.ocr.json files.")
    parser.add_argument("--output", default=str(default_output), help="Directory for raw labels.")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint.")
    parser.add_argument("--model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"), help="Deployment.")
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"), help="Azure OpenAI API key.")
    return parser.parse_args()


def build_prompt(content: str) -> str:
    return (
        "Extract key-value labels from the OCR content. "
        "Return JSON only in this format: "
        "{\"labels\": [{\"field\": \"snake_case\", \"value\": \"exact text\"}]}. "
        "Field names should be concise and consistent. "
        "Values must be exact substrings from the OCR content. "
        "Omit fields that are missing.\n\n"
        "OCR content:\n"
        f"{content}"
    )


def normalize_raw_labels(payload: Dict) -> List[Dict]:
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if labels is None:
        labels = payload

    normalized = []
    if isinstance(labels, dict):
        for field_name, value in labels.items():
            if field_name and value is not None:
                normalized.append({"field": str(field_name), "value": str(value)})
        return normalized

    if isinstance(labels, list):
        for entry in labels:
            if not isinstance(entry, dict):
                continue
            field_name = entry.get("field") or entry.get("name")
            value_text = entry.get("value")
            if field_name and value_text is not None:
                normalized.append({"field": str(field_name), "value": str(value_text)})
    return normalized


def filter_labels_by_content(labels: List[Dict], content: str) -> List[Dict]:
    filtered = []
    for label in labels:
        value_text = label.get("value")
        if value_text and value_text in content:
            filtered.append(label)
        else:
            field_name = label.get("field")
            print(f"Value not found in OCR content, skipping '{field_name}'.")
    return filtered


def save_raw_labels(output_dir: Path, file_name: str, labels: List[Dict]) -> Path:
    payload = {"document": file_name, "labels": labels}
    output_path = output_dir / f"{file_name}.labels.raw.json"
    output_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return output_path


def main() -> None:
    load_dotenv()
    default_source = Path(__file__).resolve().parent / "output" / "ocr"
    default_output = Path(__file__).resolve().parent / "output" / "raw_labels"
    args = parse_args(default_source, default_output)

    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = resolve_azure_openai_endpoint(args.endpoint)
    client = create_openai_client(endpoint, args.api_key)

    ocr_files = sorted(source_dir.glob("*.ocr.json"))
    if not ocr_files:
        raise FileNotFoundError(f"No *.ocr.json files found in {source_dir}")

    for ocr_path in ocr_files:
        file_name = ocr_path.name[: -len(".ocr.json")]
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        content = payload.get("content", "")
        if not content:
            print(f"OCR content missing for {file_name}")
            continue
        prompt = build_prompt(content)
        response = client.chat.completions.create(
            model=args.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        message = response.choices[0].message.content
        parsed = parse_json_from_response(message)
        if parsed is None:
            print(f"Unable to parse labels for {file_name}")
            continue
        normalized = normalize_raw_labels(parsed)
        normalized = filter_labels_by_content(normalized, content)
        output_path = save_raw_labels(output_dir, file_name, normalized)
        print(f"Wrote raw labels: {output_path}")


if __name__ == "__main__":
    main()
