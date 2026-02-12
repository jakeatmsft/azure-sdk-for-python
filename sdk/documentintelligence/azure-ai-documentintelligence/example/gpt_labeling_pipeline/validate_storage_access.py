# FILE: validate_storage_access.py
#
# DESCRIPTION:
#     Validate that the current managed identity (or DefaultAzureCredential)
#     can access the training container used for custom extraction builds.
#
# USAGE:
#     python validate_storage_access.py --use-managed-identity
#
#     Optional arguments:
#       --container-url  Full container URL (overrides env vars).
#       --prefix         Optional blob prefix to list.
#       --list-limit     Number of blob names to print.
#       --download       Download the first blob to verify read access.
#       --mi-client-id   User-assigned managed identity client ID.

import argparse
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate managed identity access to Blob Storage.")
    parser.add_argument(
        "--container-url",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL"),
        help="Storage container URL.",
    )
    parser.add_argument(
        "--account-name",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME"),
        help="Storage account name (used with --container-name).",
    )
    parser.add_argument(
        "--container-name",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME"),
        help="Storage container name (used with --account-name).",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_PREFIX"),
        help="Optional blob prefix to list.",
    )
    parser.add_argument("--list-limit", type=int, default=5, help="Number of blob names to print.")
    parser.add_argument("--download", action="store_true", help="Download first blob to verify read access.")
    parser.add_argument(
        "--mi-client-id",
        default=os.getenv("DOCUMENTINTELLIGENCE_MANAGED_IDENTITY_CLIENT_ID"),
        help="User-assigned managed identity client ID.",
    )
    parser.add_argument(
        "--use-managed-identity",
        action="store_true",
        help="Force ManagedIdentityCredential instead of DefaultAzureCredential.",
    )
    return parser.parse_args()


def resolve_container_url(container_url: Optional[str], account_name: Optional[str], container_name: Optional[str]) -> str:
    resolved = container_url
    if not resolved and account_name and container_name:
        resolved = f"https://{account_name}.blob.core.windows.net/{container_name}"

    if not resolved:
        raise ValueError(
            "Provide --container-url or set DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL, "
            "or set DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME and DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME."
        )
    return validate_container_url(resolved.strip())


def validate_container_url(container_url: str) -> str:
    parsed = urlparse(container_url)
    if parsed.scheme != "https":
        raise ValueError("Container URL must be an https URL.")
    if ".blob.core.windows.net" not in parsed.netloc:
        raise ValueError("Container URL must be a Blob container URL.")

    container_path = parsed.path.strip("/")
    if not container_path:
        raise ValueError("Container URL must include the container name.")
    if "/" in container_path:
        raise ValueError("Container URL must point to the container root; use --prefix for subfolders.")
    return container_url


def create_credential(use_managed_identity: bool, client_id: Optional[str]):
    if use_managed_identity or client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential()


def list_blobs(container_client: ContainerClient, prefix: Optional[str], list_limit: int):
    list_prefix = prefix.strip("/") + "/" if prefix else ""
    names = []
    total = 0
    for blob in container_client.list_blobs(name_starts_with=list_prefix):
        total += 1
        if list_limit == 0 or len(names) < list_limit:
            names.append(blob.name)
        if list_limit and len(names) >= list_limit:
            continue
    return total, names


def main() -> None:
    load_dotenv()
    args = parse_args()
    container_url = resolve_container_url(args.container_url, args.account_name, args.container_name)
    credential = create_credential(args.use_managed_identity, args.mi_client_id)

    try:
        container_client = ContainerClient.from_container_url(container_url, credential=credential)
        container_client.get_container_properties()
    except HttpResponseError as error:
        print(f"Access check failed: {error.message}")
        raise

    total, names = list_blobs(container_client, args.prefix, args.list_limit)
    prefix_display = args.prefix or "<container root>"
    print(f"Access confirmed. Prefix: {prefix_display}. Total blobs: {total}")
    if names:
        print("Sample blobs:")
        for name in names:
            print(f"- {name}")

    if args.download and names:
        blob_name = names[0]
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.download_blob().readall()
        print(f"Downloaded blob successfully: {blob_name}")
    elif args.download:
        print("No blobs available to download.")


if __name__ == "__main__":
    main()
