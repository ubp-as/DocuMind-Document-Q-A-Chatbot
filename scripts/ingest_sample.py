#!/usr/bin/env python3
"""
scripts/ingest_sample.py
------------------------
Bulk-ingest a directory of PDF/TXT files into DocuMind via the API.

Usage:
    python scripts/ingest_sample.py --dir ./my_docs --api http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

import httpx

SUPPORTED = {".pdf", ".txt", ".md"}


def ingest_directory(directory: str, api_base: str) -> None:
    docs_path = Path(directory)
    if not docs_path.is_dir():
        print(f"ERROR: '{directory}' is not a directory")
        sys.exit(1)

    files = [f for f in docs_path.iterdir() if f.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files found in '{directory}'")
        return

    print(f"Found {len(files)} file(s) to ingest...\n")

    success, failed = 0, 0
    with httpx.Client(base_url=api_base, timeout=120) as client:
        for fp in files:
            print(f"  → Uploading {fp.name}...", end=" ", flush=True)
            with fp.open("rb") as f:
                resp = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (fp.name, f, _mime(fp))},
                )
            if resp.status_code == 201:
                data = resp.json()
                print(f"✓  ({data['chunk_count']} chunks)")
                success += 1
            else:
                print(f"✗  HTTP {resp.status_code}: {resp.text}")
                failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed.")


def _mime(path: Path) -> str:
    return {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown"}.get(
        path.suffix.lstrip(".").lower(), "application/octet-stream"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest documents into DocuMind")
    parser.add_argument("--dir", required=True, help="Directory of files to ingest")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    ingest_directory(args.dir, args.api)
