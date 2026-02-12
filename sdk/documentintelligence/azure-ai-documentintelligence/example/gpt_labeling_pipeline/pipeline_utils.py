"""
Shared helpers for the GPT labeling pipeline.
"""

from pathlib import Path
import json
import os
import re
from typing import Dict, List, Optional, Union

from openai import OpenAI


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LABEL_SCHEMA_FILENAME = "labels.schema.json"


def gather_documents(source_path: Union[str, Path], extensions: List[str]) -> List[Path]:
    source = Path(source_path).expanduser().resolve()
    extension_set = {extension.lower() for extension in extensions}
    if source.is_file():
        return [source] if source.suffix.lower() in extension_set else []
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")
    return [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in extension_set
    ]


def resolve_docintelligence_endpoint() -> str:
    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    account_name = os.getenv("DOCUMENTINTELLIGENCE_ACCOUNT_NAME")
    if account_name:
        if account_name.startswith("http://") or account_name.startswith("https://"):
            endpoint = account_name
        elif ".cognitiveservices.azure.com" in account_name or ".api.cognitive.microsoft.com" in account_name:
            endpoint = f"https://{account_name}"
        else:
            endpoint = f"https://{account_name}.cognitiveservices.azure.com/"

    if not endpoint:
        raise ValueError("Set DOCUMENTINTELLIGENCE_ENDPOINT or DOCUMENTINTELLIGENCE_ACCOUNT_NAME.")
    return endpoint.rstrip("/") + "/"


def resolve_azure_openai_endpoint(endpoint: Optional[str]) -> str:
    if not endpoint:
        raise ValueError("Provide --endpoint or set AZURE_OPENAI_ENDPOINT.")
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("openai/v1"):
        return f"{endpoint}/"
    if endpoint.endswith("openai/v1/"):
        return endpoint
    if endpoint.endswith("openai") or endpoint.endswith("openai/"):
        return f"{endpoint.rstrip('/')}/v1/"
    return f"{endpoint}/openai/v1/"


def create_openai_client(endpoint: str, api_key: Optional[str]) -> OpenAI:
    if not api_key:
        raise ValueError("Provide --api-key or set AZURE_OPENAI_API_KEY.")
    return OpenAI(base_url=endpoint, api_key=api_key)


def parse_json_from_response(text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_index = cleaned.find("{")
        end_index = cleaned.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return None
        try:
            return json.loads(cleaned[start_index : end_index + 1])
        except json.JSONDecodeError:
            return None


def normalize_fields(fields: Union[List, Dict]) -> List[Dict]:
    if isinstance(fields, list):
        return [
            {
                "name": field.get("name") or field.get("fieldKey"),
                "type": field.get("type", "string"),
            }
            for field in fields
            if isinstance(field, dict) and (field.get("name") or field.get("fieldKey"))
        ]

    return [
        {
            "name": name,
            "type": value.get("type", "string") if isinstance(value, dict) else "string",
        }
        for name, value in fields.items()
    ]


def coerce_fields_to_strings(fields: List[Dict]) -> List[Dict]:
    return [
        {
            "name": field.get("name"),
            "type": "string",
        }
        for field in fields
        if field.get("name")
    ]


def write_fields_json(output_dir: Path, fields: List[Dict]) -> Path:
    fields_payload = {"fields": fields}
    output_path = output_dir / "fields.json"
    output_path.write_text(json.dumps(fields_payload, separators=(",", ":"), ensure_ascii=False))
    return output_path


def build_word_index(ocr_payload: Dict) -> List[Dict]:
    words = []
    for page in ocr_payload.get("pages", []):
        page_number = page.get("pageNumber")
        page_width = page.get("width")
        page_height = page.get("height")
        for word in page.get("words", []):
            span = word.get("span") or {}
            words.append(
                {
                    "content": word.get("content", ""),
                    "offset": span.get("offset", 0),
                    "length": span.get("length", 0),
                    "polygon": word.get("polygon") or [],
                    "page": page_number or 1,
                    "page_width": page_width,
                    "page_height": page_height,
                }
            )
    return words


def normalize_token(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "", value)
    return value.lower()


def find_value_match(words: List[Dict], value_text: str) -> Optional[List[Dict]]:
    tokens = [normalize_token(token) for token in value_text.split() if normalize_token(token)]
    if not tokens:
        return None

    normalized_words = [normalize_token(word["content"]) for word in words]
    for index in range(len(normalized_words) - len(tokens) + 1):
        if normalized_words[index : index + len(tokens)] == tokens:
            return words[index : index + len(tokens)]

    filtered_words = [(index, token) for index, token in enumerate(normalized_words) if token]
    for start_index in range(len(filtered_words) - len(tokens) + 1):
        window = [token for _, token in filtered_words[start_index : start_index + len(tokens)]]
        if window == tokens:
            matched_indices = [index for index, _ in filtered_words[start_index : start_index + len(tokens)]]
            return [words[index] for index in matched_indices]
    return None


def build_bounding_region(matched_words: List[Dict]) -> Optional[List[float]]:
    x_values = []
    y_values = []
    for word in matched_words:
        polygon = word.get("polygon") or []
        if len(polygon) < 8:
            continue
        x_values.extend(polygon[::2])
        y_values.extend(polygon[1::2])
    if not x_values or not y_values:
        return None
    return [
        min(x_values),
        min(y_values),
        max(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
        min(x_values),
        max(y_values),
    ]


def normalize_polygon(polygon: List[float], width: Optional[float], height: Optional[float]) -> Optional[List[float]]:
    if not polygon or len(polygon) != 8:
        return None
    if not width or not height:
        return None
    normalized = []
    for index, value in enumerate(polygon):
        if index % 2 == 0:
            normalized.append(max(0.0, min(1.0, value / width)))
        else:
            normalized.append(max(0.0, min(1.0, value / height)))
    return normalized


def build_label_entry(
    field_name: str, field_type: str, value_text: str, matched_words: List[Dict]
) -> Optional[Dict]:
    if not matched_words:
        return None
    page_numbers = {word["page"] for word in matched_words}
    page_number = min(page_numbers) if page_numbers else 1
    polygon = build_bounding_region(matched_words)
    width = matched_words[0].get("page_width")
    height = matched_words[0].get("page_height")
    normalized_polygon = normalize_polygon(polygon, width, height) if polygon else None
    if not normalized_polygon:
        return None
    return {
        "label": field_name,
        "value": [
            {
                "boundingBoxes": [normalized_polygon],
                "pageNumber": page_number,
                "text": value_text,
            }
        ],
        "labelType": "Words",
    }


def resolve_labels_schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / LABEL_SCHEMA_FILENAME


def ensure_labels_schema(output_dir: Path) -> str:
    schema_path = resolve_labels_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Labels schema not found: {schema_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / LABEL_SCHEMA_FILENAME
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    target_path.write_text(
        json.dumps(schema_payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    return LABEL_SCHEMA_FILENAME
