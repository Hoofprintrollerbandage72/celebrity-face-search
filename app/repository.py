from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.db import Database
from app.face_engine import normalize


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IndexedReference:
    image_id: str
    person_id: str
    person_name: str
    original_filename: str
    source_url: str | None
    source_page_url: str | None
    license_code: str | None


class VectorIndex:
    def __init__(self, database: Database):
        self.database = database
        self._lock = threading.RLock()
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._references: list[IndexedReference] = []
        self.reload()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._references)

    def reload(self) -> None:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT ri.id AS image_id, ri.person_id, p.name AS person_name,
                       ri.original_filename, ri.source_url, ri.source_page_url,
                       ri.license_code, ri.embedding, ri.embedding_dimension
                FROM reference_images ri
                JOIN persons p ON p.id = ri.person_id
                ORDER BY ri.created_at
                """
            ).fetchall()

        references: list[IndexedReference] = []
        vectors: list[np.ndarray] = []
        dimension: int | None = None
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.size != int(row["embedding_dimension"]):
                continue
            if dimension is None:
                dimension = vector.size
            if vector.size != dimension:
                continue
            vectors.append(normalize(vector))
            references.append(
                IndexedReference(
                    image_id=row["image_id"],
                    person_id=row["person_id"],
                    person_name=row["person_name"],
                    original_filename=row["original_filename"],
                    source_url=row["source_url"],
                    source_page_url=row["source_page_url"],
                    license_code=row["license_code"],
                )
            )

        with self._lock:
            self._references = references
            self._matrix = (
                np.vstack(vectors).astype(np.float32)
                if vectors
                else np.empty((0, 0), dtype=np.float32)
            )

    def search(self, query: np.ndarray, top_k: int) -> list[dict]:
        query = normalize(query)
        with self._lock:
            if not self._references:
                return []
            if query.size != self._matrix.shape[1]:
                raise ValueError(
                    "查询向量维度与人物库不一致，可能更换了模型；请重新生成人物库向量"
                )
            scores = self._matrix @ query
            candidate_count = min(len(scores), max(50, top_k * 10))
            if candidate_count == len(scores):
                indices = np.argsort(scores)[::-1]
            else:
                partial = np.argpartition(scores, -candidate_count)[-candidate_count:]
                indices = partial[np.argsort(scores[partial])[::-1]]

            grouped: dict[str, dict] = {}
            for index in indices:
                reference = self._references[int(index)]
                score = float(scores[int(index)])
                item = grouped.setdefault(
                    reference.person_id,
                    {
                        "person_id": reference.person_id,
                        "name": reference.person_name,
                        "scores": [],
                        "best_reference_image_id": reference.image_id,
                        "best_reference_filename": reference.original_filename,
                        "source_url": reference.source_url,
                        "source_page_url": reference.source_page_url,
                        "license_code": reference.license_code,
                    },
                )
                item["scores"].append(score)
                if score > max(item["scores"][:-1], default=-1.0):
                    item["best_reference_image_id"] = reference.image_id
                    item["best_reference_filename"] = reference.original_filename
                    item["source_url"] = reference.source_url
                    item["source_page_url"] = reference.source_page_url
                    item["license_code"] = reference.license_code

            results: list[dict] = []
            for item in grouped.values():
                sorted_scores = sorted(item.pop("scores"), reverse=True)
                best = sorted_scores[0]
                mean_top3 = float(np.mean(sorted_scores[:3]))
                item["similarity"] = round(best, 6)
                item["aggregate_similarity"] = round(0.7 * best + 0.3 * mean_top3, 6)
                item["reference_matches"] = min(3, len(sorted_scores))
                results.append(item)

            results.sort(key=lambda value: value["aggregate_similarity"], reverse=True)
            return results[:top_k]


class Repository:
    def __init__(self, database: Database, data_dir: Path):
        self.database = database
        self.data_dir = data_dir

    def create_person(
        self, name: str, external_id: str | None = None, aliases: list[str] | None = None
    ) -> dict:
        person_id = str(uuid.uuid4())
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO persons(id, external_id, name, aliases_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, external_id or None, name.strip(), json.dumps(aliases or []), utc_now()),
            )
        return self.get_person(person_id)

    def get_person_by_external_id(self, external_id: str) -> dict | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT id FROM persons WHERE external_id = ?", (external_id,)
            ).fetchone()
        return self.get_person(row["id"]) if row is not None else None

    def get_or_create_person(
        self, name: str, external_id: str, aliases: list[str] | None = None
    ) -> tuple[dict, bool]:
        existing = self.get_person_by_external_id(external_id)
        if existing is not None:
            return existing, False
        try:
            return self.create_person(name, external_id, aliases), True
        except sqlite3.IntegrityError:
            existing = self.get_person_by_external_id(external_id)
            if existing is None:
                raise
            return existing, False

    def ensure_people(self, people: list[dict]) -> tuple[int, int]:
        """Create dataset people in one transaction and return (created, existing)."""
        if not people:
            return 0, 0
        created = 0
        with self.database.connection() as connection:
            for person in people:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO persons(
                        id, external_id, name, aliases_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        person["external_id"],
                        person["name"].strip(),
                        json.dumps(person.get("aliases", [])),
                        utc_now(),
                    ),
                )
                created += cursor.rowcount
        return created, len(people) - created

    def get_person(self, person_id: str) -> dict:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT p.*, COUNT(ri.id) AS image_count
                FROM persons p
                LEFT JOIN reference_images ri ON ri.person_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (person_id,),
            ).fetchone()
        if row is None:
            raise KeyError(person_id)
        return {
            "id": row["id"],
            "external_id": row["external_id"],
            "name": row["name"],
            "aliases": json.loads(row["aliases_json"]),
            "image_count": int(row["image_count"]),
            "created_at": row["created_at"],
        }

    def person_count(self) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM persons").fetchone()
        return int(row["count"])

    def list_persons(
        self, limit: int | None = None, offset: int = 0, query: str | None = None
    ) -> list[dict]:
        where = ""
        parameters: list[object] = []
        if query:
            where = "WHERE p.name LIKE ? OR p.external_id LIKE ?"
            pattern = f"%{query.strip()}%"
            parameters.extend([pattern, pattern])
        pagination = ""
        if limit is not None:
            pagination = "LIMIT ? OFFSET ?"
            parameters.extend([limit, offset])
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT p.*, COUNT(ri.id) AS image_count
                FROM persons p
                LEFT JOIN reference_images ri ON ri.person_id = p.id
                {where}
                GROUP BY p.id
                ORDER BY p.name COLLATE NOCASE
                {pagination}
                """,
                parameters,
            ).fetchall()
        return [
            {
                "id": row["id"],
                "external_id": row["external_id"],
                "name": row["name"],
                "aliases": json.loads(row["aliases_json"]),
                "image_count": int(row["image_count"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_person(self, person_id: str) -> list[Path]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT stored_path FROM reference_images WHERE person_id = ?", (person_id,)
            ).fetchall()
            cursor = connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            if cursor.rowcount == 0:
                raise KeyError(person_id)
        return [Path(row["stored_path"]) for row in rows]

    def add_reference(
        self,
        person_id: str,
        original_filename: str,
        stored_path: Path,
        embedding: np.ndarray,
        source_url: str | None,
        license_code: str | None,
        source_page_url: str | None = None,
    ) -> dict:
        self.get_person(person_id)
        image_id = str(uuid.uuid4())
        vector = normalize(embedding).astype(np.float32)
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO reference_images(
                    id, person_id, original_filename, stored_path, source_url,
                    source_page_url, license_code, embedding, embedding_dimension, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    person_id,
                    original_filename,
                    str(stored_path),
                    source_url or None,
                    source_page_url or None,
                    license_code or None,
                    vector.tobytes(),
                    int(vector.size),
                    utc_now(),
                ),
            )
        return {
            "id": image_id,
            "person_id": person_id,
            "filename": original_filename,
            "source_url": source_url,
            "source_page_url": source_page_url,
            "license_code": license_code,
        }

    def get_reference_by_source(self, person_id: str, source_url: str) -> dict | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT id, original_filename, source_url, source_page_url, license_code
                FROM reference_images
                WHERE person_id = ? AND source_url = ?
                """,
                (person_id, source_url),
            ).fetchone()
        return dict(row) if row is not None else None

    def reference_count(self, person_id: str, source_prefix: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM reference_images WHERE person_id = ?"
        parameters: list[str] = [person_id]
        if source_prefix:
            query += " AND source_url LIKE ?"
            parameters.append(f"{source_prefix}%")
        with self.database.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["count"])

    def add_references_bulk(self, records: list[dict]) -> int:
        """Insert already-validated references in one transaction.

        A dataset source URL is idempotent per person, so interrupted import jobs can
        be resumed without duplicating reference rows.
        """
        if not records:
            return 0
        inserted = 0
        with self.database.connection() as connection:
            for record in records:
                vector = normalize(record["embedding"]).astype(np.float32)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO reference_images(
                        id, person_id, original_filename, stored_path, source_url,
                        source_page_url, license_code, embedding, embedding_dimension,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        record["person_id"],
                        record["original_filename"],
                        str(record["stored_path"]),
                        record.get("source_url") or None,
                        record.get("source_page_url") or None,
                        record.get("license_code") or None,
                        vector.tobytes(),
                        int(vector.size),
                        utc_now(),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def dataset_stats(self) -> list[dict]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    CASE
                        WHEN p.external_id LIKE 'celeba:%' THEN 'celeba'
                        WHEN p.external_id LIKE 'vggface2:%' THEN 'vggface2'
                        ELSE 'custom'
                    END AS dataset,
                    COUNT(DISTINCT p.id) AS persons,
                    COUNT(ri.id) AS reference_images
                FROM persons p
                LEFT JOIN reference_images ri ON ri.person_id = p.id
                GROUP BY dataset
                ORDER BY dataset
                """
            ).fetchall()
        return [
            {
                "dataset": row["dataset"],
                "persons": int(row["persons"]),
                "reference_images": int(row["reference_images"]),
            }
            for row in rows
        ]

    def get_reference_path(self, image_id: str) -> Path:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT stored_path FROM reference_images WHERE id = ?", (image_id,)
            ).fetchone()
        if row is None:
            raise KeyError(image_id)
        return Path(row["stored_path"])
