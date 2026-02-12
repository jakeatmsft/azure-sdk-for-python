# Document Intelligence SDK tutorial project

This project follows the SDK tutorial flow with a small CLI that demonstrates
layout extraction, general document analysis, custom model build, and custom
model analysis using `DocumentIntelligenceClient` and
`DocumentIntelligenceAdministrationClient`.

## Prerequisites
- Python 3.8 or later
- An Azure Document Intelligence resource
- A Blob Storage container with labeled training data for custom extraction

## Setup
```bash
pip install azure-ai-documentintelligence azure-identity python-dotenv
```

Set the environment variables:
- `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_ACCOUNT_NAME`
- `DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_URL`
- `DOCUMENTINTELLIGENCE_STORAGE_PREFIX` (optional)
- `DOCUMENTINTELLIGENCE_SAMPLE_FILE` (optional, defaults to first file in `example/docs`)

Authentication uses `DefaultAzureCredential`, so make sure you are signed in with
`az login` or provide `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET`.

## Usage
```bash
python tutorial_project.py layout
python tutorial_project.py general
python tutorial_project.py build
python tutorial_project.py analyze-custom --model-id <model-id>
```

Use `--file` to point at a specific document path for the layout/general/custom
analysis commands.
