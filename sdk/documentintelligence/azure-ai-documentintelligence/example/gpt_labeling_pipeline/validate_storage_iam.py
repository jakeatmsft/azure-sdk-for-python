"""
FILE: validate_storage_iam.py

DESCRIPTION:
    Check IAM role assignments for the Document Intelligence managed identity
    against the storage container scope.

USAGE:
    python validate_storage_iam.py --resource-group <rg>
"""

import argparse
import os
import shutil
from typing import List
from urllib.parse import urlparse
import subprocess
import sys

from dotenv import load_dotenv


DEFAULT_REQUIRED_ROLES = ["Storage Blob Data Reader"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate IAM roles for Document Intelligence storage access.")
    parser.add_argument(
        "--resource-group",
        default=os.getenv("DOCUMENTINTELLIGENCE_RESOURCE_GROUP") or os.getenv("AZURE_RESOURCE_GROUP"),
        help="Resource group containing the Document Intelligence resource.",
    )
    parser.add_argument(
        "--docintel-name",
        default=os.getenv("DOCUMENTINTELLIGENCE_ACCOUNT_NAME"),
        help="Document Intelligence resource name.",
    )
    parser.add_argument(
        "--storage-account",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_ACCOUNT_NAME"),
        help="Storage account name.",
    )
    parser.add_argument(
        "--storage-resource-group",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_RESOURCE_GROUP")
        or os.getenv("DOCUMENTINTELLIGENCE_RESOURCE_GROUP")
        or os.getenv("AZURE_RESOURCE_GROUP"),
        help="Resource group containing the storage account.",
    )
    parser.add_argument(
        "--container-name",
        default=os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_NAME"),
        help="Storage container name.",
    )
    parser.add_argument(
        "--required-role",
        action="append",
        default=None,
        help="Role name required for access (can be specified multiple times).",
    )
    return parser.parse_args()


def resolve_az_executable() -> str:
    override = os.getenv("AZURE_CLI_PATH")
    if override:
        return override
    for candidate in ("az", "az.cmd", "az.exe"):
        path = shutil.which(candidate)
        if path:
            return path
    return "az"


def run_az(command: List[str]) -> str:
    az_path = resolve_az_executable()
    try:
        result = subprocess.run([az_path, *command], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Azure CLI not found. Install Azure CLI, ensure it is on PATH, "
            "or set AZURE_CLI_PATH to the full path to az/az.cmd."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        message = stderr or "Azure CLI command failed."
        raise RuntimeError(message) from error
    return result.stdout.strip()


def normalize_docintel_name(name: str) -> str:
    if name.startswith("http://") or name.startswith("https://"):
        hostname = urlparse(name).netloc
        return hostname.split(".")[0]
    if ".cognitiveservices.azure.com" in name or ".api.cognitive.microsoft.com" in name:
        hostname = name.replace("https://", "").replace("http://", "")
        return hostname.split(".")[0]
    return name


def parse_roles(output: str) -> List[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def require_values(args: argparse.Namespace) -> None:
    missing = []
    if not args.resource_group:
        missing.append("--resource-group")
    if not args.docintel_name:
        missing.append("--docintel-name")
    if not args.storage_account:
        missing.append("--storage-account")
    if not args.storage_resource_group:
        missing.append("--storage-resource-group")
    if not args.container_name:
        missing.append("--container-name")
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    require_values(args)

    docintel_name = normalize_docintel_name(args.docintel_name)
    required_roles = args.required_role or DEFAULT_REQUIRED_ROLES

    principal_id = run_az(
        [
            "cognitiveservices",
            "account",
            "show",
            "-g",
            args.resource_group,
            "-n",
            docintel_name,
            "--query",
            "identity.principalId",
            "-o",
            "tsv",
        ]
    )

    storage_scope = run_az(
        [
            "storage",
            "account",
            "show",
            "-g",
            args.storage_resource_group,
            "-n",
            args.storage_account,
            "--query",
            "id",
            "-o",
            "tsv",
        ]
    )
    container_scope = f"{storage_scope}/blobServices/default/containers/{args.container_name}"

    container_roles_output = run_az(
        [
            "role",
            "assignment",
            "list",
            "--assignee",
            principal_id,
            "--scope",
            container_scope,
            "--query",
            "[].roleDefinitionName",
            "-o",
            "tsv",
        ]
    )
    storage_roles_output = run_az(
        [
            "role",
            "assignment",
            "list",
            "--assignee",
            principal_id,
            "--scope",
            storage_scope,
            "--query",
            "[].roleDefinitionName",
            "-o",
            "tsv",
        ]
    )

    container_roles = parse_roles(container_roles_output)
    storage_roles = parse_roles(storage_roles_output)
    combined_roles = sorted(set(container_roles + storage_roles))

    print(f"Document Intelligence principal ID: {principal_id}")
    print(f"Container scope roles: {', '.join(container_roles) if container_roles else '<none>'}")
    print(f"Storage scope roles: {', '.join(storage_roles) if storage_roles else '<none>'}")

    missing_roles = [role for role in required_roles if role not in combined_roles]
    if missing_roles:
        print(f"Missing required roles: {', '.join(missing_roles)}")
        sys.exit(1)

    print("Required roles found.")


if __name__ == "__main__":
    main()
