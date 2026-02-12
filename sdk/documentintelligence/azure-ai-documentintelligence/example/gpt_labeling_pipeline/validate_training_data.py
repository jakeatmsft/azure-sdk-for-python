# FILE: validate_training_data.py
#
# DESCRIPTION:
#     Sanity-check training artifacts (fields, labels, OCR) for custom extraction.
#
# USAGE:
#     python validate_training_data.py
#
#     Optional arguments:
#       --fields  Path to fields.json (defaults to training_data/fields.json).
#       --upload  Upload data after validation.

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

from pipeline_utils import SUPPORTED_EXTENSIONS, normalize_fields


ALLOWED_TYPES = {
    "string",
    # "number",
    # "integer",
    # "date",
    # "time",
    # "boolean",
    # "phoneNumber",
    # "selectionMark",
    # "signature",
    # "countryRegion",
    # "currency",
    # "address",
}

ALLOWED_LABEL_TYPES = {"Words", "SelectionMark", "Text"}


def parse_args(default_source: Path, default_fields: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate training artifacts for custom extraction.")
    parser.add_argument("--source", default=str(default_source), help="Training data directory.")
    parser.add_argument("--fields", default=str(default_fields), help="Path to fields.json.")
    parser.add_argument("--upload", action="store_true", help="Upload data after validation.")
    return parser.parse_args()


def load_fields(fields_path: Path) -> List[Dict]:
    payload = json.loads(fields_path.read_text(encoding="utf-8"))
    fields = payload.get("fields") if isinstance(payload, dict) else payload
    if fields is None:
        return []
    return normalize_fields(fields)


def gather_training_files(source_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    documents = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    labels = [
        path
        for path in sorted(source_dir.rglob("*.labels.json"))
        if not path.name.endswith(".labels.raw.json")
    ]
    ocr_files = [path for path in sorted(source_dir.rglob("*.ocr.json")) if path.is_file()]
    return documents, labels, ocr_files


def validate_fields(fields: List[Dict]) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []
    seen_names: Set[str] = set()
    for index, field in enumerate(fields, start=1):
        name = field.get("name") if isinstance(field, dict) else None
        if not name:
            errors.append(f"Field #{index} is missing a name.")
            continue
        if name in seen_names:
            errors.append(f"Duplicate field name '{name}'.")
        seen_names.add(name)

        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            warnings.append(f"Field '{name}' is not snake_case.")

        field_type = field.get("type", "string") if isinstance(field, dict) else "string"
        if field_type not in ALLOWED_TYPES:
            errors.append(f"Field '{name}' has unsupported type '{field_type}'.")
    return errors, warnings


def index_by_name(paths: List[Path], suffix: str) -> Tuple[Dict[str, Path], List[str]]:
    mapping: Dict[str, Path] = {}
    duplicates = []
    for path in paths:
        name = path.name[: -len(suffix)] if suffix else path.name
        if name in mapping:
            duplicates.append(name)
        mapping[name] = path
    return mapping, duplicates


def validate_labels(
    label_path: Path,
    fields_by_name: Dict[str, Dict],
    document_names: Set[str],
    ocr_contents: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    errors = []
    warnings = []
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if "$schema" not in payload:
        warnings.append(f"{label_path.name} missing '$schema' reference.")
    document_name = payload.get("document")
    if not document_name:
        errors.append(f"{label_path.name} missing 'document' field.")
        return errors, warnings
    if document_name not in document_names:
        errors.append(f"{label_path.name} document '{document_name}' not found in source files.")

    labels = payload.get("labels")
    if labels is None:
        errors.append(f"{label_path.name} missing 'labels' list.")
        return errors, warnings
    if not isinstance(labels, list):
        errors.append(f"{label_path.name} labels must be a list.")
        return errors, warnings

    for index, label in enumerate(labels, start=1):
        if not isinstance(label, dict):
            errors.append(f"{label_path.name} label #{index} is not an object.")
            continue
        field_name = label.get("label")
        if not field_name:
            errors.append(f"{label_path.name} label #{index} missing 'label'.")
            continue
        if field_name not in fields_by_name:
            errors.append(f"{label_path.name} label #{index} has unknown field '{field_name}'.")

        label_type = label.get("labelType")
        if not isinstance(label_type, str):
            errors.append(f"{label_path.name} label #{index} missing 'labelType'.")
        elif label_type not in ALLOWED_LABEL_TYPES:
            errors.append(
                f"{label_path.name} label #{index} has unsupported labelType '{label_type}'."
            )
        key_value = label.get("key")
        if key_value is not None and not isinstance(key_value, str):
            errors.append(f"{label_path.name} label #{index} key must be null or string.")

        values = label.get("value")
        if not isinstance(values, list) or not values:
            errors.append(f"{label_path.name} label #{index} missing value entries.")
            continue

        for value_index, value_entry in enumerate(values, start=1):
            if not isinstance(value_entry, dict):
                errors.append(
                    f"{label_path.name} label #{index} value #{value_index} is not an object."
                )
                continue
            boxes = value_entry.get("boundingBoxes")
            if not isinstance(boxes, list) or not boxes:
                errors.append(
                    f"{label_path.name} label #{index} value #{value_index} missing boundingBoxes."
                )
            else:
                for box_index, box in enumerate(boxes, start=1):
                    if not isinstance(box, list) or len(box) != 8:
                        errors.append(
                            f"{label_path.name} label #{index} value #{value_index} box #{box_index} has invalid polygon."
                        )
                        continue
                    for point in box:
                        if not isinstance(point, (int, float)) or point < 0 or point > 1:
                            errors.append(
                                f"{label_path.name} label #{index} value #{value_index} box #{box_index} has out-of-range coordinates."
                            )
                            break

            page_number = value_entry.get("pageNumber")
            if page_number is None:
                page_number = value_entry.get("page")
            if not isinstance(page_number, int) or page_number < 1:
                errors.append(
                    f"{label_path.name} label #{index} value #{value_index} missing valid page number."
                )
            text_value = value_entry.get("text")
            if text_value is None:
                errors.append(
                    f"{label_path.name} label #{index} value #{value_index} missing text."
                )
            elif isinstance(text_value, str):
                content = ocr_contents.get(document_name)
                if content and text_value not in content:
                    warnings.append(
                        f"{label_path.name} label #{index} value #{value_index} text not found in OCR content for '{document_name}'."
                    )
    return errors, warnings


def validate_ocr(ocr_path: Path) -> Tuple[List[str], List[str], Optional[str]]:
    errors = []
    warnings = []
    payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        errors.append(f"{ocr_path.name} is not a JSON object.")
        return errors, warnings, None
    payload = payload.get("analyzeResult", payload)
    content = payload.get("content")
    if not isinstance(content, str):
        errors.append(f"{ocr_path.name} missing OCR content string.")
    if not payload.get("pages"):
        errors.append(f"{ocr_path.name} missing pages array.")
    pages = payload.get("pages", []) if isinstance(payload.get("pages"), list) else []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"{ocr_path.name} page #{page_index} is not an object.")
            continue
        if page.get("pageNumber") is None:
            errors.append(f"{ocr_path.name} page #{page_index} missing pageNumber.")
        words = page.get("words")
        if words is None:
            warnings.append(f"{ocr_path.name} page #{page_index} missing words array.")
            continue
        if not isinstance(words, list):
            errors.append(f"{ocr_path.name} page #{page_index} words must be a list.")
            continue
        for word_index, word in enumerate(words, start=1):
            if not isinstance(word, dict):
                errors.append(f"{ocr_path.name} page #{page_index} word #{word_index} is not an object.")
                continue
            if not word.get("content"):
                warnings.append(
                    f"{ocr_path.name} page #{page_index} word #{word_index} missing content text."
                )
            span = word.get("span")
            if not isinstance(span, dict) or "offset" not in span or "length" not in span:
                errors.append(
                    f"{ocr_path.name} page #{page_index} word #{word_index} missing span offsets."
                )
            polygon = word.get("polygon")
            if polygon is None:
                warnings.append(
                    f"{ocr_path.name} page #{page_index} word #{word_index} missing polygon."
                )
            elif not isinstance(polygon, list) or len(polygon) != 8:
                errors.append(
                    f"{ocr_path.name} page #{page_index} word #{word_index} has invalid polygon."
                )
    return errors, warnings, content if isinstance(content, str) else None


def main() -> None:
    default_source = Path(__file__).resolve().parent / "training_data"
    default_fields = default_source / "fields.json"
    args = parse_args(default_source, default_fields)

    source_dir = Path(args.source).expanduser().resolve()
    fields_path = Path(args.fields).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Training data folder not found: {source_dir}")
    if not fields_path.exists():
        raise FileNotFoundError(f"fields.json not found: {fields_path}")

    fields = load_fields(fields_path)
    if not fields:
        print("ERROR: No fields found in fields.json.")
        sys.exit(1)

    errors: List[str] = []
    warnings: List[str] = []
    field_errors, field_warnings = validate_fields(fields)
    errors.extend(field_errors)
    warnings.extend(field_warnings)
    fields_by_name = {field.get("name"): field for field in fields if field.get("name")}

    documents, labels, ocr_files = gather_training_files(source_dir)
    doc_map, doc_duplicates = index_by_name(documents, "")
    label_map, label_duplicates = index_by_name(labels, ".labels.json")
    ocr_map, ocr_duplicates = index_by_name(ocr_files, ".ocr.json")

    if doc_duplicates:
        warnings.append(f"Duplicate document filenames found: {', '.join(sorted(set(doc_duplicates)))}")
    if label_duplicates:
        warnings.append(f"Duplicate label filenames found: {', '.join(sorted(set(label_duplicates)))}")
    if ocr_duplicates:
        warnings.append(f"Duplicate OCR filenames found: {', '.join(sorted(set(ocr_duplicates)))}")

    doc_names = set(doc_map.keys())
    missing_labels = sorted(doc_names - set(label_map.keys()))
    missing_ocr = sorted(doc_names - set(ocr_map.keys()))
    if missing_labels:
        errors.append(f"Missing labels for documents: {', '.join(missing_labels[:5])}")
    if missing_ocr:
        errors.append(f"Missing OCR for documents: {', '.join(missing_ocr[:5])}")

    ocr_contents: Dict[str, str] = {}
    for name, ocr_path in ocr_map.items():
        ocr_errors, ocr_warnings, content = validate_ocr(ocr_path)
        errors.extend(ocr_errors)
        warnings.extend(ocr_warnings)
        if content is not None:
            ocr_contents[name] = content

    for name, label_path in label_map.items():
        label_errors, label_warnings = validate_labels(
            label_path, fields_by_name, set(doc_map.keys()), ocr_contents
        )
        errors.extend(label_errors)
        warnings.extend(label_warnings)

    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)

    print(
        "Training data validated: "
        f"{len(documents)} docs, {len(labels)} labels, {len(ocr_files)} OCR files, "
        f"{len(fields)} fields."
    )

    if args.upload:
        upload_script = Path(__file__).resolve().parent / "upload_training_data.py"
        command = [sys.executable, str(upload_script), "--source", str(source_dir)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
