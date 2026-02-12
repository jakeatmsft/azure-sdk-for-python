# FILE: generate_ocr_data.py
#
# DESCRIPTION:
#     Generate OCR (prebuilt-layout) outputs for documents and store *.ocr.json
#     files used by downstream GPT labeling steps.
#
# USAGE:
#     python generate_ocr_data.py
#
#     Optional arguments:
#       --source     Directory or file with documents.
#       --output     Directory to write *.ocr.json files.
#       --max-files  Limit number of documents processed.
#       --extensions File extensions to include (with leading dots).

import argparse
import json
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from pipeline_utils import SUPPORTED_EXTENSIONS, gather_documents, resolve_docintelligence_endpoint


def parse_args(default_source: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCR outputs for labeling.")
    parser.add_argument("--source", default=str(default_source), help="Directory or file to analyze.")
    parser.add_argument("--output", default=str(default_output), help="Directory for *.ocr.json files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for documents processed.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(SUPPORTED_EXTENSIONS),
        help="File extensions to include (with leading dots).",
    )
    return parser.parse_args()


def analyze_document(client: DocumentIntelligenceClient, document_path: Path) -> dict:
    with document_path.open("rb") as handle:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            handle,
            content_type="application/octet-stream",
        )
    result = poller.result()
    return result.as_dict()


def main() -> None:
    load_dotenv()
    default_source = Path(__file__).resolve().parent / "docs"
    default_output = Path(__file__).resolve().parent / "output" / "ocr"
    args = parse_args(default_source, default_output)

    source_path = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = gather_documents(source_path, args.extensions)
    if not documents:
        raise ValueError("No documents found for the provided extensions.")
    if args.max_files is not None:
        documents = documents[: args.max_files]

    endpoint = resolve_docintelligence_endpoint()
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=DefaultAzureCredential())

    for document_path in documents:
        payload = analyze_document(client, document_path)
        output_path = output_dir / f"{document_path.name}.ocr.json"
        output_path.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote OCR: {output_path}")


if __name__ == "__main__":
    main()
