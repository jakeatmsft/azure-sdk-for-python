# FILE: normalize_fields_with_gpt.py
#
# DESCRIPTION:
#     Use Azure OpenAI to normalize raw per-document labels into a canonical
#     field list.
#
# USAGE:
#     python normalize_fields_with_gpt.py
#
#     Optional arguments:
#       --source   Directory containing *.labels.raw.json files.
#       --output   Output JSON file for canonical fields.
#       --endpoint Azure OpenAI endpoint (https://<resource>.openai.azure.com).
#       --model    Azure OpenAI deployment name (defaults to gpt-5.2).

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from pipeline_utils import (
    coerce_fields_to_strings,
    create_openai_client,
    normalize_fields,
    parse_json_from_response,
    resolve_azure_openai_endpoint,
)


FIELDS_SCHEMA = "http://www.azure.com/schema/formrecognizer/fields.json"


def parse_args(default_source: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize GPT labels into canonical fields.")
    parser.add_argument("--source", default=str(default_source), help="Directory with raw label files.")
    parser.add_argument("--output", default=str(default_output), help="Output JSON file for fields.")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint.")
    parser.add_argument("--model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"), help="Deployment.")
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"), help="Azure OpenAI API key.")
    return parser.parse_args()


def build_prompt(field_samples: List[Dict]) -> str:
    return (
        "Normalize the field names into a single canonical set. "
        "Return JSON only with this shape: {\"fields\": [{\"name\": \"snake_case\", \"type\": \"string\"}]}. "
        "Use snake_case for names, merge synonyms, and include only the core fields. "
        "Pick the most suitable type for each field (string, number, integer, date, time, boolean).\n\n"
        "Field candidates with example values:\n"
        f"{json.dumps(field_samples, indent=2)}"
    )


def load_raw_labels(source_dir: Path) -> Tuple[List[Dict], List[str]]:
    label_files = sorted(source_dir.glob("*.labels.raw.json"))
    if not label_files:
        raise FileNotFoundError(f"No *.labels.raw.json files found in {source_dir}")
    raw_payloads = []
    for label_path in label_files:
        raw_payloads.append(json.loads(label_path.read_text(encoding="utf-8")))
    return raw_payloads, [path.name for path in label_files]


def summarize_fields(raw_payloads: List[Dict]) -> List[Dict]:
    field_values: Dict[str, List[str]] = defaultdict(list)
    for payload in raw_payloads:
        labels = payload.get("labels", [])
        for label in labels:
            if not isinstance(label, dict):
                continue
            field_name = label.get("field")
            value_text = label.get("value")
            if not field_name:
                continue
            if value_text:
                field_values[field_name].append(str(value_text))
            else:
                field_values[field_name].append("")

    field_samples = []
    for field_name, values in sorted(field_values.items()):
        examples = [value for value in values if value][:3]
        field_samples.append({"field": field_name, "examples": examples})
    return field_samples


def normalize_response(payload: Dict) -> List[Dict]:
    fields = payload.get("fields") if isinstance(payload, dict) else None
    if fields is None:
        fields = payload
    if not fields:
        return []
    normalized = normalize_fields(fields)
    normalized = coerce_fields_to_strings(normalized)
    seen = set()
    unique_fields = []
    for field in normalized:
        name = field.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        unique_fields.append(field)
    return unique_fields


def build_fields_payload(fields: List[Dict]) -> Dict:
    return {
        "$schema": FIELDS_SCHEMA,
        "fields": [
            {
                "fieldKey": field.get("name"),
                "fieldType": "string",
                "fieldFormat": "not-specified",
                "groupIndex": 0,
            }
            for field in fields
            if field.get("name")
        ],
        "definitions": {},
    }


def main() -> None:
    load_dotenv()
    default_source = Path(__file__).resolve().parent / "output" / "raw_labels"
    default_output = Path(__file__).resolve().parent / "output" / "canonical_fields.json"
    args = parse_args(default_source, default_output)

    source_dir = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_payloads, source_files = load_raw_labels(source_dir)
    field_samples = summarize_fields(raw_payloads)
    if not field_samples:
        raise ValueError("No labels found in raw label files.")

    endpoint = resolve_azure_openai_endpoint(args.endpoint)
    client = create_openai_client(endpoint, args.api_key)
    prompt = build_prompt(field_samples)
    response = client.chat.completions.create(
        model=args.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    parsed = parse_json_from_response(response.choices[0].message.content)
    if parsed is None:
        raise ValueError("Unable to parse canonical field response.")

    fields = normalize_response(parsed)
    if not fields:
        raise ValueError("No canonical fields returned by the model.")

    output_payload = {
        "description": "Canonical fields generated from GPT raw labels.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "fields": fields,
    }
    output_path.write_text(json.dumps(output_payload, separators=(",", ":"), ensure_ascii=False))
    fields_payload = build_fields_payload(fields)
    fields_output_path = output_path.parent / "fields.json"
    fields_output_path.write_text(
        json.dumps(fields_payload, separators=(",", ":"), ensure_ascii=False)
    )
    print(f"Wrote canonical fields: {output_path}")


if __name__ == "__main__":
    main()
