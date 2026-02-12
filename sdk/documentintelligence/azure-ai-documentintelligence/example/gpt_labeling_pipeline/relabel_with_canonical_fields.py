# FILE: relabel_with_canonical_fields.py
#
# DESCRIPTION:
#     Use canonical fields with Azure OpenAI to generate Document Intelligence
#     label artifacts and assemble training data.
#
# USAGE:
#     python relabel_with_canonical_fields.py
#
#     Optional arguments:
#       --docs     Directory with source documents.
#       --ocr      Directory with *.ocr.json files.
#       --fields   Canonical fields JSON file.
#       --output   Output directory for training artifacts.
#       --endpoint Azure OpenAI endpoint (https://<resource>.openai.azure.com).
#       --model    Azure OpenAI deployment name (defaults to gpt-5.2).

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from pipeline_utils import (
    build_label_entry,
    build_word_index,
    coerce_fields_to_strings,
    create_openai_client,
    find_value_match,
    normalize_fields,
    parse_json_from_response,
    resolve_azure_openai_endpoint,
)


FIELDS_SCHEMA = "http://www.azure.com/schema/formrecognizer/fields.json"


def parse_args(default_docs: Path, default_ocr: Path, default_fields: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel documents with canonical fields.")
    parser.add_argument("--docs", default=str(default_docs), help="Directory containing source documents.")
    parser.add_argument("--ocr", default=str(default_ocr), help="Directory containing *.ocr.json files.")
    parser.add_argument("--fields", default=str(default_fields), help="Canonical fields JSON file.")
    parser.add_argument("--output", default=str(default_output), help="Output directory for training artifacts.")
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"), help="Azure OpenAI endpoint.")
    parser.add_argument("--model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"), help="Deployment.")
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"), help="Azure OpenAI API key.")
    return parser.parse_args()


def load_canonical_fields(fields_path: Path) -> List[Dict]:
    payload = json.loads(fields_path.read_text(encoding="utf-8"))
    fields = payload.get("fields") if isinstance(payload, dict) else payload
    if not fields:
        raise ValueError("Canonical fields file is missing 'fields'.")
    normalized = normalize_fields(fields)
    normalized = coerce_fields_to_strings(normalized)
    if not normalized:
        raise ValueError("No usable fields found in canonical fields file.")
    return normalized


def build_prompt(content: str, field_names: List[str]) -> str:
    fields_text = ", ".join(field_names)
    return (
        "Extract exact text values for the listed fields from the OCR content. "
        "Return JSON only, with a top-level object containing a 'labels' object. "
        "Use the field names exactly as provided. "
        "Each field value must be an exact substring of the OCR content. "
        "Use null when the field is not present. "
        f"Fields: {fields_text}.\n\n"
        "OCR content:\n"
        f"{content}"
    )


def build_field_aliases(field_names: List[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    digit_collapse = re.compile(r"(?<=\D)_(?=\d)|(?<=\d)_(?=\D)")

    def add_alias(alias: str, canonical: str) -> None:
        if alias and alias not in aliases:
            aliases[alias] = canonical

    for name in field_names:
        add_alias(name, name)
        add_alias(name.lower(), name)

        collapsed = digit_collapse.sub("", name)
        if collapsed != name:
            add_alias(collapsed, name)
            add_alias(collapsed.lower(), name)

        if name.endswith("_zip_code"):
            add_alias(name.replace("_zip_code", "_zip"), name)
            add_alias(name.replace("_zip_code", "_zipcode"), name)

        if "relationship_to_primary_insured" in name:
            shortened = name.replace("_primary_insured", "_insured")
            add_alias(shortened, name)
            add_alias(shortened.lower(), name)

        if name.startswith("additional_insured_"):
            proposed = f"proposed_{name}"
            add_alias(proposed, name)
            add_alias(proposed.lower(), name)

    return aliases


def normalize_gpt_labels(payload: Dict, field_names: List[str]) -> List[Tuple[str, str]]:
    if not isinstance(payload, dict):
        return []
    labels_map = payload.get("labels") if "labels" in payload else payload
    if not isinstance(labels_map, dict):
        return []
    aliases = build_field_aliases(field_names)
    collected: Dict[str, List[str]] = {name: [] for name in field_names}
    for raw_name, value in labels_map.items():
        if raw_name is None:
            continue
        raw_name = str(raw_name).strip()
        if not raw_name:
            continue
        canonical = aliases.get(raw_name) or aliases.get(raw_name.lower())
        if not canonical:
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    collected[canonical].append(str(item))
        else:
            collected[canonical].append(str(value))

    labels: List[Tuple[str, str]] = []
    for field_name in field_names:
        for value_text in collected[field_name]:
            labels.append((field_name, value_text))
    return labels


def dedupe_bounding_boxes(entry: Dict, used_boxes: set) -> bool:
    values = entry.get("value")
    if not isinstance(values, list):
        return False

    candidate_boxes = []
    local_boxes = set()
    for value_entry in values:
        boxes = value_entry.get("boundingBoxes") if isinstance(value_entry, dict) else None
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, list):
                continue
            key = tuple(round(value, 6) for value in box)
            if key in used_boxes or key in local_boxes:
                return False
            local_boxes.add(key)
            candidate_boxes.append(key)
    for key in candidate_boxes:
        used_boxes.add(key)
    return True


def build_labels_payload(
    file_name: str, fields_by_name: Dict, labels: List[Tuple[str, str]], words: List[Dict]
) -> Dict:
    entries = []
    used_boxes: set = set()
    for field_name, value_text in labels:
        field_type = "string"
        matched_words = find_value_match(words, value_text)
        if not matched_words:
            print(f"Label text not found in document '{file_name}': {field_name}='{value_text}'")
            continue
        entry = build_label_entry(field_name, field_type, value_text, matched_words)
        if not entry:
            continue
        entry = convert_label_to_studio(entry)
        if not dedupe_bounding_boxes(entry, used_boxes):
            print(f"Skipping duplicate bounding box in '{file_name}' for field '{field_name}'.")
            continue
        entries.append(entry)
    return {"document": file_name, "labels": entries}


def convert_label_to_studio(label: Dict) -> Dict:
    values = label.get("value")
    if isinstance(values, list):
        for value_entry in values:
            if not isinstance(value_entry, dict):
                continue
            if "page" not in value_entry and "pageNumber" in value_entry:
                value_entry["page"] = value_entry.pop("pageNumber")
    return label


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


def wrap_ocr_payload(ocr_payload: Dict) -> Dict:
    if "analyzeResult" in ocr_payload and "status" in ocr_payload:
        return ocr_payload
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "status": "succeeded",
        "createdDateTime": timestamp,
        "lastUpdatedDateTime": timestamp,
        "analyzeResult": ocr_payload,
    }


def save_labels(output_dir: Path, file_name: str, payload: Dict) -> Path:
    output_path = output_dir / f"{file_name}.labels.json"
    output_path.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def main() -> None:
    load_dotenv()
    script_dir = Path(__file__).resolve().parent
    default_docs = script_dir / "docs"
    default_ocr = script_dir / "output" / "ocr"
    default_fields = script_dir / "output" / "canonical_fields.json"
    default_output = script_dir / "training_data"
    args = parse_args(default_docs, default_ocr, default_fields, default_output)

    docs_dir = Path(args.docs).expanduser().resolve()
    ocr_dir = Path(args.ocr).expanduser().resolve()
    fields_path = Path(args.fields).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_ref = "http://www.azure.com/schema/formrecognizer/labels.json"
    schema_path = output_dir / "labels.schema.json"
    if schema_path.exists():
        schema_path.unlink()

    fields = load_canonical_fields(fields_path)
    field_names = [field["name"] for field in fields if field.get("name")]
    fields_by_name = {field["name"]: field.get("type", "string") for field in fields}
    fields_payload = build_fields_payload(fields)
    output_fields_path = output_dir / "fields.json"
    output_fields_path.write_text(
        json.dumps(fields_payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )

    endpoint = resolve_azure_openai_endpoint(args.endpoint)
    client = create_openai_client(endpoint, args.api_key)

    document_paths = sorted(path for path in docs_dir.iterdir() if path.is_file())
    if not document_paths:
        raise FileNotFoundError(f"No documents found in {docs_dir}")

    for document_path in document_paths:
        ocr_path = ocr_dir / f"{document_path.name}.ocr.json"
        if not ocr_path.exists():
            print(f"OCR file missing for {document_path.name}: {ocr_path}")
            continue
        ocr_payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        content = ocr_payload.get("content", "")
        if not content:
            print(f"OCR content missing for {document_path.name}")
            continue

        prompt = build_prompt(content, field_names)
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
            print(f"Unable to parse labels for {document_path.name}")
            continue
        label_pairs = normalize_gpt_labels(parsed, field_names)
        words = build_word_index(ocr_payload)
        labels_payload = build_labels_payload(document_path.name, fields_by_name, label_pairs, words)
        labels_payload["$schema"] = schema_ref
        save_labels(output_dir, document_path.name, labels_payload)
        shutil.copy2(document_path, output_dir / document_path.name)
        wrapped_ocr = wrap_ocr_payload(ocr_payload)
        output_ocr_path = output_dir / ocr_path.name
        output_ocr_path.write_text(
            json.dumps(wrapped_ocr, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote labels for {document_path.name}")

    print(f"Training artifacts ready in {output_dir}")


if __name__ == "__main__":
    main()
