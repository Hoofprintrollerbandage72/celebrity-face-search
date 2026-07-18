#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "celebrity-face-search-mvp/0.1"}
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            total = int(response.headers.get("Content-Length", "0"))
            copied = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                if total:
                    print(
                        f"\r  {copied / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB",
                        end="",
                        flush=True,
                    )
        print()
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download verified OpenCV face models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("OPENCV_MODEL_DIR", "./data/models")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, metadata in MODELS.items():
        destination = model_dir / filename
        if destination.is_file() and sha256(destination) == metadata["sha256"]:
            print(f"Ready: {filename}")
            continue
        destination.unlink(missing_ok=True)
        print(f"Downloading: {filename}")
        download(metadata["url"], destination)
        actual = sha256(destination)
        if actual != metadata["sha256"]:
            destination.unlink(missing_ok=True)
            raise SystemExit(
                f"Checksum mismatch for {filename}: expected {metadata['sha256']}, got {actual}"
            )
        print(f"Verified: {filename}")
    print(f"Models are ready in {model_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

