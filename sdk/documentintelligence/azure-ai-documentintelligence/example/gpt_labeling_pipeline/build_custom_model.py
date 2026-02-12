"""
FILE: build_custom_model.py

DESCRIPTION:
    Guide the custom extraction labeling flow in Document Intelligence Studio and
    optionally build a model from the uploaded training data.

USAGE:
    python build_custom_model.py

    Optional arguments:
      --list-limit      Limit number of blob names to print (0 for all).
      --build           Build the custom model after verification.
      --model-id        Optional model ID for the build.
      --allow-unlabeled Allow unlabeled documents under the prefix.
"""

import argparse
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Optional, Set, Tuple
from urllib.parse import urlparse

from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient
from azure.ai.documentintelligence.models import (
    AzureBlobContentSource,
    BuildDocumentModelRequest,
    DocumentBuildMode,
)
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv


LABEL_HINTS = ("label", "labels", "ocr", "annotation", "field")
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LABEL_SUFFIX = ".labels.json"
OCR_SUFFIX = ".ocr.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guide and build a custom extraction model.")
    parser.add_argument("--list-limit", type=int, default=200, help="Number of blob names to print.")
    parser.add_argument("--build", action="store_true", help="Build the model after verification.")
    parser.add_argument("--model-id", default=None, help="Optional model ID for the build.")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow unlabeled documents under the prefix (warnings only).",
    )
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


def resolve_container_url() -> str:
    container_url = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL")
    account_name = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME")
    if not container_url and account_name and container_name:
        container_url = f"https://{account_name}.blob.core.windows.net/{container_name}"
    if not container_url:
        raise ValueError("Set DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL or storage account/container name.")
    return validate_container_url(container_url.strip())


def resolve_prefix() -> Optional[str]:
    return os.getenv("DOCUMENTINTELLIGENCE_STORAGE_PREFIX")


def validate_container_url(container_url: str) -> str:
    parsed = urlparse(container_url)
    if parsed.scheme != "https":
        raise ValueError("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL must be an https URL.")
    if ".blob.core.windows.net" not in parsed.netloc:
        raise ValueError("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL must be a Blob container URL.")
    container_path = parsed.path.strip("/")
    if not container_path:
        raise ValueError("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL must include the container name.")
    if "/" in container_path:
        raise ValueError(
            "DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL must point to the container root. "
            "Use DOCUMENTINTELLIGENCE_STORAGE_PREFIX for subfolders."
        )
    return container_url


def list_blobs(
    container_url: str, prefix: Optional[str], limit: int, credential: DefaultAzureCredential
) -> Tuple[int, list, list]:
    container_client = ContainerClient.from_container_url(container_url, credential=credential)
    list_prefix = prefix.strip("/") + "/" if prefix else ""

    total = 0
    names = []
    label_candidates = []
    for blob in container_client.list_blobs(name_starts_with=list_prefix):
        total += 1
        name = blob.name
        if limit == 0 or len(names) < limit:
            names.append(name)
        if any(hint in name.lower() for hint in LABEL_HINTS):
            label_candidates.append(name)
    return total, names, label_candidates


def summarize_training_artifacts(
    container_url: str, prefix: Optional[str], credential: DefaultAzureCredential
) -> Tuple[Set[str], Set[str], Set[str], bool]:
    container_client = ContainerClient.from_container_url(container_url, credential=credential)
    list_prefix = prefix.strip("/") + "/" if prefix else ""
    doc_names: Set[str] = set()
    label_names: Set[str] = set()
    ocr_names: Set[str] = set()
    fields_found = False

    for blob in container_client.list_blobs(name_starts_with=list_prefix):
        filename = PurePosixPath(blob.name).name
        if filename == "fields.json":
            fields_found = True
            continue
        if filename.endswith(LABEL_SUFFIX):
            label_names.add(filename[: -len(LABEL_SUFFIX)])
            continue
        if filename.endswith(OCR_SUFFIX):
            ocr_names.add(filename[: -len(OCR_SUFFIX)])
            continue
        if Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS:
            doc_names.add(filename)

    return doc_names, label_names, ocr_names, fields_found


def verify_training_data(
    container_url: str, prefix: Optional[str], credential: DefaultAzureCredential, allow_unlabeled: bool
) -> None:
    doc_names, label_names, ocr_names, fields_found = summarize_training_artifacts(
        container_url, prefix, credential
    )
    prefix_display = prefix or "<container root>"

    if not doc_names and not label_names and not ocr_names:
        raise ValueError(
            "No blobs found for the training prefix. "
            f"Check DOCUMENTINTELLIGENCE_STORAGE_PREFIX and container contents under '{prefix_display}'."
        )
    if not fields_found:
        raise ValueError("Missing fields.json in the training prefix.")
    if not doc_names:
        raise ValueError("No source documents found in the training prefix.")

    missing_labels = sorted(doc_names - label_names)
    missing_ocr = sorted(doc_names - ocr_names)
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        message = f"Missing .labels.json for: {preview}"
        if allow_unlabeled:
            print(f"Warning: {message}")
        else:
            raise ValueError(message)
    if missing_ocr:
        preview = ", ".join(missing_ocr[:5])
        message = f"Missing .ocr.json for: {preview}"
        if allow_unlabeled:
            print(f"Warning: {message}")
        else:
            raise ValueError(message)

    orphan_labels = sorted(label_names - doc_names)
    orphan_ocr = sorted(ocr_names - doc_names)
    if orphan_labels:
        preview = ", ".join(orphan_labels[:5])
        print(f"Warning: labels without documents: {preview}")
    if orphan_ocr:
        preview = ", ".join(orphan_ocr[:5])
        print(f"Warning: OCR without documents: {preview}")

    labeled_docs = doc_names & label_names & ocr_names
    if not labeled_docs:
        raise ValueError("No fully labeled documents found (doc + labels + OCR).")

    print(
        "Training set summary: "
        f"{len(doc_names)} docs, {len(label_names)} labels, {len(ocr_names)} OCR files."
    )
    if allow_unlabeled and (missing_labels or missing_ocr):
        print(
            "Proceeding with build despite unlabeled documents. "
            "Consider moving unlabeled files to a separate prefix."
        )


def build_model(
    endpoint: str,
    container_url: str,
    prefix: Optional[str],
    model_id: Optional[str],
    credential: DefaultAzureCredential,
) -> str:
    prefix_display = prefix or "<container root>"
    print(f"Starting model build for {container_url} (prefix: {prefix_display}).")
    admin_client = DocumentIntelligenceAdministrationClient(endpoint=endpoint, credential=credential)
    request = BuildDocumentModelRequest(
        model_id=model_id or str(uuid.uuid4()),
        build_mode=DocumentBuildMode.TEMPLATE,
        azure_blob_source=AzureBlobContentSource(container_url=container_url, prefix=prefix),
        description="custom field extraction model",
    )
    poller = admin_client.begin_build_document_model(request)
    print("Build submitted. Waiting for completion...")
    model = poller.result()
    print(f"Built model: {model.model_id}")
    print("Build completed successfully.")
    return model.model_id


def main() -> None:
    load_dotenv()
    args = parse_args()

    endpoint = resolve_endpoint()
    container_url = resolve_container_url()
    prefix = resolve_prefix()
    credential = DefaultAzureCredential()

    total, names, label_candidates = list_blobs(container_url, prefix, args.list_limit, credential)

    print(f"Total blobs under prefix: {total}")
    if names:
        print("Blob names:")
        for name in names:
            print(f"- {name}")

    if total > len(names) and args.list_limit:
        print(f"... truncated, showing first {len(names)} blobs")

    if label_candidates:
        print("Label/ocr artifacts found:")
        for name in label_candidates:
            print(f"- {name}")
    else:
        print("No label/ocr artifacts detected in the prefix.")

    if args.build:
        try:
            verify_training_data(container_url, prefix, credential, args.allow_unlabeled)
            build_model(endpoint, container_url, prefix, args.model_id, credential)
        except HttpResponseError as error:
            if error.error is not None:
                if error.error.code == "InvalidImage":
                    print(f"Received an invalid image error: {error.error}")
                if error.error.code == "InvalidRequest":
                    print(f"Received an invalid request error: {error.error}")
                if error.error.code == "TrainingContentMissing":
                    print(
                        "Training data was not found by the service. Verify the container URL and "
                        "prefix point to labeled training data and that the Document Intelligence "
                        "resource has access to the storage container."
                    )
                if error.error.code == "InvalidContentSourceFormat":
                    print(
                        "The service could not read training content. This usually means the "
                        "Document Intelligence resource lacks access to the container or the "
                        "container URL needs a SAS token. Ensure Storage Blob Data Reader access "
                        "(or provide a SAS URL) and confirm storage firewall settings."
                    )
                raise
            if "Invalid request".casefold() in error.message.casefold():
                print(f"Uh-oh! Seems there was an invalid request: {error}")
            raise


if __name__ == "__main__":
    main()
