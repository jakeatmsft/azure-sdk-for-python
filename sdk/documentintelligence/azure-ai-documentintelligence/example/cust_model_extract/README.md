# Custom model extract

Use this script to run a custom extraction model against a document and print
the fields it finds.

## Prerequisites
- Python 3.8+
- Azure Document Intelligence resource with a custom extraction model

Install dependencies:
```bash
pip install azure-ai-documentintelligence azure-identity python-dotenv
```

## Environment variables
- `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_ACCOUNT_NAME`
- `CUSTOM_BUILT_MODEL_ID` (or pass `--model-id`)
- `DOCUMENTINTELLIGENCE_SAMPLE_FILE` (optional)

Authentication uses `DefaultAzureCredential`, so make sure you are signed in with
`az login` or set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET`.

## Usage
```bash
python custom_model_extract.py --model-id <model-id>
python custom_model_extract.py --model-id <model-id> --file /path/to/document.pdf
```
