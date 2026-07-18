#!/usr/bin/env python3
"""Resumable CelebA and VGGFace2 importer for the local face library.

The default is five usable reference faces per identity. CelebA identities are
anonymous numeric IDs; VGGFace2 display names come from identity_meta.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.db import Database
from app.face_engine import FaceEngineError, build_face_engine
from app.repository import Repository, VectorIndex


CELEBA_DATASET = "flwrlabs/celeba"
CELEBA_CONFIG = "img_align+identity+attr"
CELEBA_IDENTITIES = 10_177
CELEBA_SPLITS = {"train": 162_770, "valid": 19_867, "test": 19_962}
CELEBA_SHARD_SIZES = {
    "test-00000-of-00003.parquet": 391_177_633,
    "test-00001-of-00003.parquet": 383_959_773,
    "test-00002-of-00003.parquet": 383_093_383,
    "train-00000-of-00019.parquet": 500_402_070,
    "train-00001-of-00019.parquet": 498_251_284,
    "train-00002-of-00019.parquet": 494_123_817,
    "train-00003-of-00019.parquet": 490_434_458,
    "train-00004-of-00019.parquet": 494_342_449,
    "train-00005-of-00019.parquet": 503_422_722,
    "train-00006-of-00019.parquet": 494_063_518,
    "train-00007-of-00019.parquet": 492_511_619,
    "train-00008-of-00019.parquet": 497_445_734,
    "train-00009-of-00019.parquet": 503_059_054,
    "train-00010-of-00019.parquet": 498_101_294,
    "train-00011-of-00019.parquet": 501_186_917,
    "train-00012-of-00019.parquet": 493_796_347,
    "train-00013-of-00019.parquet": 503_616_679,
    "train-00014-of-00019.parquet": 490_108_044,
    "train-00015-of-00019.parquet": 489_097_029,
    "train-00016-of-00019.parquet": 497_714_156,
    "train-00017-of-00019.parquet": 489_160_281,
    "train-00018-of-00019.parquet": 489_124_107,
    "valid-00000-of-00003.parquet": 387_827_665,
    "valid-00001-of-00003.parquet": 385_032_335,
    "valid-00002-of-00003.parquet": 383_642_321,
}
CELEBA_LICENSE = "CelebA Dataset Release Agreement; non-commercial research only"
VGG_REPO = "ProgramComputer/VGGFace2"
VGG_LICENSE = "CC-BY-NC-4.0 (mirror metadata; verify original dataset terms)"
VGG_META_URL = (
    "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/"
    "meta/identity_meta.csv?download=true"
)
VGG_ARCHIVES = {
    "test": ("vggface2_test.tar.gz", 2_028_325_313),
    "train": ("vggface2_train.tar.gz", 37_909_212_476),
}


def runtime() -> tuple[Settings, Repository]:
    settings = Settings.from_env()
    settings.ensure_directories()
    repository = Repository(Database(settings.data_dir / "faces.sqlite3"), settings.data_dir)
    return settings, repository


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)


def parse_vgg_people(metadata_path: Path) -> list[dict]:
    people: list[dict] = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_id = row["Class_ID"].strip()
            raw_name = row[" Name"].strip().strip('"')
            display_name = raw_name.replace("_", " ")
            split = "train" if row[" Flag"].strip() == "1" else "test"
            gender = row[" Gender"].strip()
            people.append(
                {
                    "external_id": f"vggface2:{class_id}",
                    "name": display_name,
                    "aliases": [
                        f"dataset:vggface2",
                        f"class:{class_id}",
                        f"split:{split}",
                        f"gender:{gender}",
                        "license:non-commercial",
                    ],
                }
            )
    return people


def bootstrap(dataset: str) -> None:
    settings, repository = runtime()
    if dataset in {"celeba", "all"}:
        people = [
            {
                "external_id": f"celeba:{identity}",
                "name": f"CelebA #{identity:05d}",
                "aliases": [
                    "dataset:celeba",
                    "identity:anonymous",
                    "license:non-commercial-research-only",
                ],
            }
            for identity in range(1, CELEBA_IDENTITIES + 1)
        ]
        created, existing = repository.ensure_people(people)
        print(json.dumps({"dataset": "celeba", "created": created, "existing": existing}))

    if dataset in {"vggface2", "all"}:
        metadata_path = settings.data_dir / "datasets" / "vggface2" / "identity_meta.csv"
        if not metadata_path.is_file():
            download_file(VGG_META_URL, metadata_path)
        people = parse_vgg_people(metadata_path)
        created, existing = repository.ensure_people(people)
        print(
            json.dumps(
                {"dataset": "vggface2", "created": created, "existing": existing}
            )
        )


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def dataset_people(repository: Repository, prefix: str) -> tuple[dict[str, dict], dict[str, int]]:
    people = {
        person["external_id"]: person
        for person in repository.list_persons()
        if (person["external_id"] or "").startswith(prefix)
    }
    counts = {external_id: int(person["image_count"]) for external_id, person in people.items()}
    return people, counts


def choose_face(engine, image_path: Path):
    faces = engine.extract(image_path)
    if not faces:
        raise FaceEngineError("没有检测到可识别人脸")
    return max(faces, key=lambda face: face.box["width"] * face.box["height"])


def fetch_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = httpx.get(url, follow_redirects=True, timeout=60)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"图片下载失败：{last_error}")


def fetch_bytes_safe(url: str) -> bytes | None:
    try:
        return fetch_bytes(url)
    except RuntimeError:
        return None


def import_celeba(
    images_per_identity: int, reset_scan: bool = False, max_rows: int | None = None
) -> None:
    settings, repository = runtime()
    bootstrap("celeba")
    people, counts = dataset_people(repository, "celeba:")
    engine = build_face_engine(settings)
    if not engine.available:
        raise SystemExit("OpenCV SFace 模型未就绪，请先运行 scripts/setup_models.py")

    state_path = settings.data_dir / "import_jobs" / "celeba.json"
    state = {} if reset_scan else load_state(state_path)
    offsets = state.get("offsets", {})
    imported = 0
    rejected = 0
    rows_scanned = 0

    parquet_root = (
        settings.data_dir / "datasets" / "celeba" / "img_align+identity+attr"
    )
    if not parquet_root.is_dir():
        raise SystemExit(
            "CelebA Parquet 尚未下载，请先运行："
            "python scripts/import_research_datasets.py download-celeba"
        )

    import pyarrow.parquet as pq

    for split, total_rows in CELEBA_SPLITS.items():
        offset = int(offsets.get(split, 0))
        if offset >= total_rows or not any(
            value < images_per_identity for value in counts.values()
        ):
            continue

        parquet_files = sorted(parquet_root.glob(f"{split}-*.parquet"))
        if not parquet_files:
            raise SystemExit(f"缺少 CelebA {split} Parquet 文件：{parquet_root}")
        records = []
        split_rows_scanned = 0
        row_index = 0
        stop_split = False

        def process_row(row: dict, current_row_index: int) -> bool:
            nonlocal imported, rejected, rows_scanned, split_rows_scanned
            external_id = f"celeba:{int(row['celeb_id'])}"
            rows_scanned += 1
            split_rows_scanned += 1
            if external_id not in people or counts[external_id] >= images_per_identity:
                return max_rows is not None and rows_scanned >= max_rows

            image_payload = row["image"]
            content = image_payload.get("bytes")
            if content is None and image_payload.get("path"):
                content = fetch_bytes_safe(image_payload["path"])
            if content is None:
                rejected += 1
                return max_rows is not None and rows_scanned >= max_rows

            identity = external_id.split(":", 1)[1]
            suffix = Path(image_payload.get("path") or "image.png").suffix or ".png"
            filename = f"celeba_{split}_{current_row_index:06d}{suffix}"
            person_dir = settings.data_dir / "references" / "celeba" / identity
            person_dir.mkdir(parents=True, exist_ok=True)
            permanent_path = person_dir / filename
            with tempfile.NamedTemporaryFile(
                dir=settings.data_dir / "tmp", suffix=suffix, delete=False
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            try:
                face = choose_face(engine, temporary_path)
                shutil.move(str(temporary_path), permanent_path)
                records.append(
                    {
                        "person_id": people[external_id]["id"],
                        "original_filename": filename,
                        "stored_path": permanent_path,
                        "source_url": (
                            f"https://huggingface.co/datasets/{CELEBA_DATASET}"
                            f"?split={split}&row={current_row_index}"
                        ),
                        "license_code": CELEBA_LICENSE,
                        "embedding": face.embedding,
                    }
                )
                counts[external_id] += 1
            except (FaceEngineError, OSError):
                rejected += 1
                permanent_path.unlink(missing_ok=True)
            finally:
                temporary_path.unlink(missing_ok=True)

            if len(records) >= 100:
                imported += repository.add_references_bulk(records)
                records.clear()
            current_offset = current_row_index + 1
            if current_offset % 1000 == 0:
                imported += repository.add_references_bulk(records)
                records.clear()
                offsets[split] = current_offset
                state.update(
                    {
                        "dataset": "celeba",
                        "images_per_identity": images_per_identity,
                        "offsets": offsets,
                        "imported_this_run": imported,
                        "rejected_this_run": rejected,
                    }
                )
                save_state(state_path, state)
                print(
                    json.dumps(
                        {
                            "dataset": "celeba",
                            "split": split,
                            "offset": current_offset,
                            "identities_complete": sum(
                                value >= images_per_identity for value in counts.values()
                            ),
                            "identities_total": len(counts),
                            "references_imported_this_run": imported,
                            "rejected": rejected,
                        }
                    ),
                    flush=True,
                )
            return (
                (max_rows is not None and rows_scanned >= max_rows)
                or all(value >= images_per_identity for value in counts.values())
            )

        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(
                batch_size=100, columns=["image", "celeb_id"]
            ):
                for row in batch.to_pylist():
                    current_row_index = row_index
                    row_index += 1
                    if current_row_index < offset:
                        continue
                    if process_row(row, current_row_index):
                        stop_split = True
                        break
            if stop_split:
                break

        imported += repository.add_references_bulk(records)
        final_offset = min(total_rows, offset + split_rows_scanned)
        offsets[split] = final_offset
        state.update(
            {
                "dataset": "celeba",
                "images_per_identity": images_per_identity,
                "offsets": offsets,
                "imported_this_run": imported,
                "rejected_this_run": rejected,
            }
        )
        save_state(state_path, state)
        if max_rows is not None and rows_scanned >= max_rows:
            print_status(repository, settings)
            return

    print_status(repository, settings)


def download_celeba_parquet() -> None:
    settings, _ = runtime()

    destination = settings.data_dir / "datasets" / "celeba"
    destination.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"dataset": "celeba", "download_bytes": 11_734_694_689},
            ensure_ascii=False,
        ),
        flush=True,
    )
    for basename, expected_size in CELEBA_SHARD_SIZES.items():
        filename = f"img_align+identity+attr/{basename}"
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyarrow.parquet as pq

            if target.stat().st_size != expected_size:
                raise ValueError("size mismatch")
            pq.ParquetFile(target)
            print(json.dumps({"existing": filename}), flush=True)
            continue
        except Exception:
            pass
        url = (
            f"https://huggingface.co/datasets/{CELEBA_DATASET}/resolve/main/"
            f"{quote(filename, safe='/')}?download=true"
        )
        download_segmented(url, target, expected_size)
        pq.ParquetFile(target)
        print(json.dumps({"downloaded": filename}), flush=True)
    files = list((destination / "img_align+identity+attr").glob("*.parquet"))
    if len(files) != 25:
        raise SystemExit(f"CelebA 文件校验失败：expected=25, actual={len(files)}")
    print(json.dumps({"dataset": "celeba", "parquet_files": len(files)}), flush=True)


def download_vgg_archives(which: str) -> None:
    settings, _ = runtime()
    destination_dir = settings.data_dir / "datasets" / "vggface2"
    destination_dir.mkdir(parents=True, exist_ok=True)
    targets = VGG_ARCHIVES.items() if which == "all" else [(which, VGG_ARCHIVES[which])]
    for split, (filename, expected_size) in targets:
        destination = destination_dir / filename
        url = (
            f"https://huggingface.co/datasets/{VGG_REPO}/resolve/main/data/"
            f"{filename}?download=true"
        )
        print(json.dumps({"dataset": "vggface2", "download": split, "bytes": expected_size}))
        if not destination.is_file() or destination.stat().st_size != expected_size:
            download_segmented(
                url,
                destination,
                expected_size,
                segment_size=16 * 1024 * 1024,
                workers=10,
            )
        actual_size = destination.stat().st_size
        if actual_size != expected_size:
            raise SystemExit(
                f"{filename} 大小校验失败：expected={expected_size}, actual={actual_size}"
            )


def download_segmented(
    url: str,
    destination: Path,
    total_size: int,
    segment_size: int = 4 * 1024 * 1024,
    workers: int = 10,
) -> None:
    """Download immutable large files by verified byte ranges."""
    parts_dir = destination.parent / f".{destination.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    start = 0
    index = 0
    while start < total_size:
        end = min(total_size - 1, start + segment_size - 1)
        ranges.append((index, start, end))
        start = end + 1
        index += 1

    def fetch_segment(job: tuple[int, int, int]) -> tuple[int, int]:
        part_index, part_start, part_end = job
        expected = part_end - part_start + 1
        part_path = parts_dir / f"{part_index:05d}.part"
        if part_path.is_file() and part_path.stat().st_size == expected:
            return part_index, expected
        temporary = parts_dir / f"{part_index:05d}.tmp"
        for attempt in range(15):
            temporary.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--http1.1",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--retry-delay",
                    "2",
                    "--range",
                    f"{part_start}-{part_end}",
                    "--output",
                    str(temporary),
                    url,
                ],
                check=False,
            )
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size == expected:
                temporary.replace(part_path)
                return part_index, expected
            if attempt < 14:
                time.sleep(min(30, 2 ** min(attempt, 4)))
        raise RuntimeError(f"分片下载失败：{part_index} ({part_start}-{part_end})")

    completed = 0
    last_reported = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_segment, job) for job in ranges]
        for future in as_completed(futures):
            _, size = future.result()
            completed += size
            if completed - last_reported >= 256 * 1024 * 1024 or completed == total_size:
                print(
                    json.dumps(
                        {
                            "file": destination.name,
                            "downloaded_verified_bytes": completed,
                            "total_bytes": total_size,
                        }
                    ),
                    flush=True,
                )
                last_reported = completed

    assembled = destination.parent / f".{destination.name}.assembling"
    with assembled.open("wb") as output:
        for part_index, part_start, part_end in ranges:
            part_path = parts_dir / f"{part_index:05d}.part"
            expected = part_end - part_start + 1
            if not part_path.is_file() or part_path.stat().st_size != expected:
                raise RuntimeError(f"组装前分片校验失败：{part_path}")
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if assembled.stat().st_size != total_size:
        raise RuntimeError("归档组装后大小校验失败")
    assembled.replace(destination)


def import_vgg(images_per_identity: int, which: str) -> None:
    settings, repository = runtime()
    bootstrap("vggface2")
    people, counts = dataset_people(repository, "vggface2:")
    engine = build_face_engine(settings)
    if not engine.available:
        raise SystemExit("OpenCV SFace 模型未就绪，请先运行 scripts/setup_models.py")

    archive_dir = settings.data_dir / "datasets" / "vggface2"
    targets = VGG_ARCHIVES.items() if which == "all" else [(which, VGG_ARCHIVES[which])]
    imported = 0
    rejected = 0
    for split, (filename, expected_size) in targets:
        archive_path = archive_dir / filename
        if not archive_path.is_file() or archive_path.stat().st_size != expected_size:
            raise SystemExit(f"归档不存在或未下载完整：{archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            records = []
            visited = 0
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                visited += 1
                parts = Path(member.name).parts
                class_id = next((part for part in parts if part.startswith("n") and part[1:].isdigit()), None)
                external_id = f"vggface2:{class_id}" if class_id else ""
                if external_id not in people or counts[external_id] >= images_per_identity:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                suffix = Path(member.name).suffix.lower() or ".jpg"
                with tempfile.NamedTemporaryFile(
                    dir=settings.data_dir / "tmp", suffix=suffix, delete=False
                ) as temporary:
                    shutil.copyfileobj(extracted, temporary)
                    temporary_path = Path(temporary.name)
                person_dir = settings.data_dir / "references" / "vggface2" / class_id
                person_dir.mkdir(parents=True, exist_ok=True)
                permanent_path = person_dir / Path(member.name).name
                try:
                    face = choose_face(engine, temporary_path)
                    shutil.move(str(temporary_path), permanent_path)
                    records.append(
                        {
                            "person_id": people[external_id]["id"],
                            "original_filename": Path(member.name).name,
                            "stored_path": permanent_path,
                            "source_url": (
                                f"https://huggingface.co/datasets/{VGG_REPO}/blob/main/"
                                f"data/{filename}#{member.name}"
                            ),
                            "license_code": VGG_LICENSE,
                            "embedding": face.embedding,
                        }
                    )
                    counts[external_id] += 1
                except (FaceEngineError, OSError):
                    rejected += 1
                    permanent_path.unlink(missing_ok=True)
                finally:
                    temporary_path.unlink(missing_ok=True)

                if len(records) >= 100:
                    imported += repository.add_references_bulk(records)
                    records.clear()
                    print(
                        json.dumps(
                            {
                                "dataset": "vggface2",
                                "split": split,
                                "archive_members_seen": visited,
                                "references_imported_this_run": imported,
                                "rejected": rejected,
                            }
                        ),
                        flush=True,
                    )
            imported += repository.add_references_bulk(records)
    print_status(repository, settings)


def print_status(repository: Repository, settings: Settings) -> None:
    index = VectorIndex(repository.database)
    print(
        json.dumps(
            {
                "datasets": repository.dataset_stats(),
                "indexed_references": index.size,
                "data_dir": str(settings.data_dir),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = commands.add_parser("bootstrap", help="录入全部身份元数据")
    bootstrap_parser.add_argument("--dataset", choices=["celeba", "vggface2", "all"], default="all")

    celeba_parser = commands.add_parser("celeba", help="流式导入 CelebA 参考脸")
    celeba_parser.add_argument("--images-per-identity", type=int, default=5)
    celeba_parser.add_argument("--reset-scan", action="store_true")
    celeba_parser.add_argument(
        "--max-rows", type=int, help="本次最多扫描多少行；用于小批验收"
    )

    commands.add_parser("download-celeba", help="断点下载 CelebA Parquet")

    download_parser = commands.add_parser("download-vgg", help="断点下载 VGGFace2 归档")
    download_parser.add_argument("--which", choices=["test", "train", "all"], default="all")

    vgg_parser = commands.add_parser("vggface2", help="从本地归档导入 VGGFace2 参考脸")
    vgg_parser.add_argument("--images-per-identity", type=int, default=5)
    vgg_parser.add_argument("--which", choices=["test", "train", "all"], default="all")

    commands.add_parser("status", help="输出数据集人物和参考图计数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        bootstrap(args.dataset)
    elif args.command == "celeba":
        import_celeba(args.images_per_identity, args.reset_scan, args.max_rows)
    elif args.command == "download-celeba":
        download_celeba_parquet()
    elif args.command == "download-vgg":
        download_vgg_archives(args.which)
    elif args.command == "vggface2":
        import_vgg(args.images_per_identity, args.which)
    elif args.command == "status":
        settings, repository = runtime()
        print_status(repository, settings)


if __name__ == "__main__":
    main()
