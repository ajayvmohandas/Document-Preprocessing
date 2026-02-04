# Document Preprocessing API

FastAPI service that preprocesses images directly and PDFs by splitting pages, preprocessing each page, and merging them back into a single PDF.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

## Usage

- `POST /preprocess` with `multipart/form-data` key `file`
- `GET /health` for a simple health check

Example (PowerShell):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/preprocess `
  -Method Post -Form @{ file = Get-Item ".\input\sample.pdf" } `
  -OutFile ".\output\processed.pdf"
```
