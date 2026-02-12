import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a document with a custom model.")
    parser.add_argument("--model-id", help="Custom model ID to use for extraction.")
    parser.add_argument("--file", dest="file_path", help="Path to the document to analyze.")
    return parser.parse_args()


def resolve_endpoint() -> str:
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

    endpoint = endpoint.rstrip("/")
    if ".api.cognitive.microsoft.com" in endpoint:
        raise ValueError(
            "Token authentication requires a custom subdomain endpoint. "
            "Set DOCUMENTINTELLIGENCE_ACCOUNT_NAME to the resource name or use the resource endpoint."
        )
    if ".cognitiveservices.azure.com" not in endpoint:
        raise ValueError(
            "DOCUMENTINTELLIGENCE_ENDPOINT must be the resource endpoint "
            "(https://<resource>.cognitiveservices.azure.com)."
        )
    return f"{endpoint}/"


def resolve_model_id(model_id: Optional[str]) -> str:
    resolved = model_id or os.getenv("CUSTOM_BUILT_MODEL_ID") or os.getenv("DOCUMENTINTELLIGENCE_CUSTOM_MODEL_ID")
    if not resolved:
        raise ValueError("Provide --model-id or set CUSTOM_BUILT_MODEL_ID.")
    return resolved


def resolve_sample_file(file_path: Optional[str]) -> Path:
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        return path

    env_path = os.getenv("DOCUMENTINTELLIGENCE_SAMPLE_FILE")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        return path

    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    candidates = [path for path in sorted(docs_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(
            "No sample document found in example/docs. "
            "Provide --file or set DOCUMENTINTELLIGENCE_SAMPLE_FILE."
        )
    return candidates[0]


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def field_value_to_data(field) -> Any:
    if field.value_string is not None:
        return field.value_string
    if field.value_date is not None:
        return field.value_date.isoformat()
    if field.value_time is not None:
        return field.value_time.isoformat()
    if field.value_phone_number is not None:
        return field.value_phone_number
    if field.value_number is not None:
        return field.value_number
    if field.value_integer is not None:
        return field.value_integer
    if field.value_selection_mark is not None:
        return str(field.value_selection_mark)
    if field.value_signature is not None:
        return str(field.value_signature)
    if field.value_country_region is not None:
        return field.value_country_region
    if field.value_boolean is not None:
        return field.value_boolean
    if field.value_selection_group is not None:
        return list(field.value_selection_group)
    if field.value_currency is not None:
        currency = field.value_currency
        return compact_dict(
            {
                "amount": currency.amount,
                "currency_symbol": currency.currency_symbol,
                "currency_code": currency.currency_code,
            }
        )
    if field.value_address is not None:
        return compact_dict(field.value_address.as_dict())
    if field.value_array is not None:
        return [field_value_to_data(item) for item in field.value_array]
    if field.value_object is not None:
        return {key: field_value_to_data(value) for key, value in field.value_object.items()}
    return field.content


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def analyze_custom_document(model_id: str, file_path: Path) -> None:
    endpoint = resolve_endpoint()
    credential = DefaultAzureCredential()
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

    with file_path.open("rb") as file_handle:
        poller = client.begin_analyze_document(model_id=model_id, body=file_handle)
    result = poller.result()

    print(f"Analyzed with model: {result.model_id}")
    print(f"Document: {file_path}")

    if not result.documents:
        print("No documents returned in the analyze result.")
        return

    for index, document in enumerate(result.documents, start=1):
        confidence = f"{document.confidence:.2f}" if document.confidence is not None else "N/A"
        print(f"\nDocument #{index}")
        print(f"Type: {document.doc_type} (confidence: {confidence})")

        if not document.fields:
            print("No fields extracted.")
            continue

        for name, field in document.fields.items():
            value = field_value_to_data(field)
            field_confidence = f"{field.confidence:.2f}" if field.confidence is not None else "N/A"
            print(
                f"- {name}: {format_value(value)} "
                f"(type: {field.type}, confidence: {field_confidence})"
            )


def main() -> None:
    load_dotenv()
    args = parse_args()

    model_id = resolve_model_id(args.model_id)
    file_path = resolve_sample_file(args.file_path)

    try:
        analyze_custom_document(model_id, file_path)
    except HttpResponseError as error:
        if error.error and error.error.code:
            print(f"Service error ({error.error.code}): {error.error.message}")
        raise


if __name__ == "__main__":
    main()
