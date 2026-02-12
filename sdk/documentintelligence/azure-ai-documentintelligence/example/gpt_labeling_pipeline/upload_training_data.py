# FILE: upload_training_data.py
#
# DESCRIPTION:
#     Upload training artifacts (docs, *.ocr.json, *.labels.json, fields.json)
#     to Azure Blob Storage using Entra ID authentication.
#
# USAGE:
#     python upload_training_data.py
#
#     Set the environment variables with your own values before running:
#     1) DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL - URL of the container to upload into.
#        -OR-
#        DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME and DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME.
#     2) DOCUMENTINTELLIGENCE_STORAGE_PREFIX - optional prefix for uploaded blobs.
#     Variables can also be provided via a .env file in the working directory.

import argparse
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv


def parse_args(default_source: Path, default_prefix: Optional[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload training data to Blob Storage.")
    parser.add_argument(
        "--source",
        default=str(default_source),
        help="Directory containing training data to upload.",
    )
    parser.add_argument(
        "--prefix",
        default=default_prefix,
        help="Optional prefix for uploaded blobs.",
    )
    return parser.parse_args()


def resolve_container_url() -> str:
    container_url = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL")
    account_name = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME")
    if not container_url and account_name and container_name:
        container_url = f"https://{account_name}.blob.core.windows.net/{container_name}"

    if not container_url:
        raise ValueError(
            "Please set DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL or the storage account/container name."
        )
    return validate_container_url(container_url.strip())


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


def upload_folder(container_client: ContainerClient, source_dir: Path, prefix: Optional[str]) -> int:
    files = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.name != "labels.schema.json"
    ]
    if not files:
        raise ValueError(f"No files found in {source_dir}.")

    list_prefix = f"{prefix.strip('/')}/" if prefix else ""
    existing = [blob.name for blob in container_client.list_blobs(name_starts_with=list_prefix)]
    for blob_name in existing:
        container_client.delete_blob(blob_name)

    uploaded = 0
    for path in files:
        relative_path = path.relative_to(source_dir).as_posix()
        blob_name = f"{prefix.strip('/')}/{relative_path}" if prefix else relative_path
        with path.open("rb") as stream:
            container_client.upload_blob(name=blob_name, data=stream, overwrite=True)
        uploaded += 1
    return uploaded


def main() -> None:
    load_dotenv()
    default_source = Path(__file__).resolve().parent / "training_data"
    default_prefix = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_PREFIX")
    args = parse_args(default_source, default_prefix)

    source_dir = Path(args.source).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    container_url = resolve_container_url()
    credential = DefaultAzureCredential()
    container_client = ContainerClient.from_container_url(container_url, credential=credential)

    uploaded = upload_folder(container_client, source_dir, args.prefix)
    print(f"Uploaded {uploaded} files to {container_url}")


if __name__ == "__main__":
    main()
