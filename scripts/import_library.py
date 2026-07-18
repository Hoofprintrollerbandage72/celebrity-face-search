#!/usr/bin/env python3
from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import httpx


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a folder tree: library/<person name>/*.jpg"
    )
    parser.add_argument("library", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--license", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.library.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    with httpx.Client(base_url=args.api, timeout=120) as client:
        existing = {person["name"]: person for person in client.get("/api/library/persons").raise_for_status().json()}
        for person_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            person = existing.get(person_dir.name)
            if person is None:
                response = client.post("/api/library/persons", json={"name": person_dir.name})
                response.raise_for_status()
                person = response.json()
                existing[person_dir.name] = person
                print(f"Created: {person_dir.name}")

            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                with image_path.open("rb") as handle:
                    response = client.post(
                        f"/api/library/persons/{person['id']}/images",
                        files={"image": (image_path.name, handle, mime)},
                        data={"license_code": args.license},
                    )
                if response.is_success:
                    print(f"  Imported: {image_path.name}")
                else:
                    print(f"  Skipped: {image_path.name} ({response.text})")


if __name__ == "__main__":
    main()

