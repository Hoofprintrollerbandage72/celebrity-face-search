#!/usr/bin/env python3
from __future__ import annotations

import argparse
import mimetypes
import urllib.parse
import urllib.request
from pathlib import Path

import httpx


REFERENCES = [
    {
        "name": "Barack Obama",
        "external_id": "Q76",
        "filename": "Official portrait of Barack Obama.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Official_portrait_of_Barack_Obama.jpg",
        "license_code": "CC BY 3.0 / Pete Souza",
    },
    {
        "name": "Donald Trump",
        "external_id": "Q22686",
        "filename": "Donald Trump official portrait.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Donald_Trump_official_portrait.jpg",
        "license_code": "Public Domain / US Government",
    },
]

QUERY = {
    "filename": "Donald Trump official portrait, 2025 (headshot).jpg",
    "source_url": "https://commons.wikimedia.org/wiki/File:Donald_Trump_official_portrait,_2025_(headshot).jpg",
    "expected_external_id": "Q22686",
}


def commons_download_url(filename: str) -> str:
    encoded = urllib.parse.quote(filename, safe="")
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{encoded}?width=1000"


def download(filename: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 10_000:
        return
    request = urllib.request.Request(
        commons_download_url(filename),
        headers={"User-Agent": "celebrity-face-search-mvp/0.1 (demo dataset)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
        "wb"
    ) as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed two license-tracked public figures and verify a real search"
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in [*REFERENCES, QUERY]:
        destination = output_dir / item["filename"]
        print(f"Preparing: {item['filename']}")
        download(item["filename"], destination)

    with httpx.Client(base_url=args.api, timeout=120, trust_env=False) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        health_payload = health.json()
        if not health_payload.get("real_face_recognition") or not health_payload.get(
            "engine_available"
        ):
            raise SystemExit(f"Real face engine is not ready: {health_payload}")

        persons = client.get("/api/library/persons").json()
        by_external_id = {person["external_id"]: person for person in persons}
        for item in REFERENCES:
            person = by_external_id.get(item["external_id"])
            if person is None:
                response = client.post(
                    "/api/library/persons",
                    json={
                        "name": item["name"],
                        "external_id": item["external_id"],
                    },
                )
                response.raise_for_status()
                person = response.json()
                by_external_id[item["external_id"]] = person
                print(f"Created person: {item['name']}")

            if person["image_count"] == 0:
                image_path = output_dir / item["filename"]
                mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                with image_path.open("rb") as handle:
                    response = client.post(
                        f"/api/library/persons/{person['id']}/images",
                        files={"image": (image_path.name, handle, mime)},
                        data={
                            "source_url": item["source_url"],
                            "license_code": item["license_code"],
                        },
                    )
                response.raise_for_status()
                print(f"Indexed reference: {item['name']}")

        query_path = output_dir / QUERY["filename"]
        with query_path.open("rb") as handle:
            response = client.post(
                "/api/search",
                files={"image": (query_path.name, handle, "image/jpeg")},
                data={"top_k": "2"},
            )
        if not response.is_success:
            raise SystemExit(
                f"Search request failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        candidates = payload["results"][0]["candidates"]
        if not candidates:
            raise SystemExit("Smoke test failed: no candidates returned")
        top = candidates[0]
        expected = by_external_id[QUERY["expected_external_id"]]
        print(
            f"Top-1: {top['name']} / cosine={top['similarity']:.4f} / "
            f"aggregate={top['aggregate_similarity']:.4f}"
        )
        if top["person_id"] != expected["id"]:
            raise SystemExit(
                f"Smoke test failed: expected {expected['name']}, got {top['name']}"
            )
        print(f"REAL_MVP_SMOKE_TEST=PASS query={query_path}")


if __name__ == "__main__":
    main()
