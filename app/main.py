from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from app.config import Settings
from app.db import Database
from app.face_engine import FaceEngineError, build_face_engine
from app.repository import Repository, VectorIndex
from app.source_import import SourceDownloadError, download_remote_image


ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class PersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)
    aliases: list[str] = Field(default_factory=list)


class RemoteReferenceSource(BaseModel):
    image_url: str = Field(min_length=8, max_length=2048)
    source_page_url: str | None = Field(default=None, max_length=2048)
    license_code: str | None = Field(default=None, max_length=200)

    @field_validator("image_url", "source_page_url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("图片源和来源页面仅支持 http 或 https 地址")
        return value.strip()


class QuickSourceImport(BaseModel):
    person_id: str | None = None
    person: PersonCreate | None = None
    sources: list[RemoteReferenceSource] = Field(min_length=1, max_length=20)


async def save_validated_upload(
    upload: UploadFile, destination_dir: Path, max_bytes: int
) -> tuple[Path, str]:
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="图片超过上传大小限制")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    temporary = destination_dir / f"validate-{uuid.uuid4().hex}"
    temporary.write_bytes(content)
    try:
        with Image.open(temporary) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="仅支持有效的 JPEG、PNG 或 WEBP 图片") from exc

    suffix = ALLOWED_FORMATS.get(image_format or "")
    if suffix is None:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="仅支持 JPEG、PNG 或 WEBP 图片")
    final_path = destination_dir / f"{uuid.uuid4().hex}{suffix}"
    temporary.replace(final_path)
    return final_path, upload.filename or f"upload{suffix}"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.data_dir / "faces.sqlite3")
    repository = Repository(database, settings.data_dir)
    index = VectorIndex(database)
    engine = build_face_engine(settings)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings
    app.state.repository = repository
    app.state.index = index
    app.state.engine = engine

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok" if engine.available else "degraded",
            "engine": engine.name,
            "engine_available": engine.available,
            "real_face_recognition": engine.real_face_recognition,
            "reference_images": index.size,
            "persons": repository.person_count(),
        }

    @app.get("/api/library/persons")
    def list_persons(
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None, max_length=200),
    ) -> list[dict]:
        return repository.list_persons(limit=limit, offset=offset, query=q)

    @app.get("/api/library/stats")
    def library_stats() -> dict:
        return {"datasets": repository.dataset_stats(), "indexed_references": index.size}

    @app.post("/api/library/reload")
    def reload_library_index() -> dict:
        index.reload()
        return {"status": "ok", "indexed_references": index.size}

    @app.post("/api/library/persons", status_code=201)
    def create_person(payload: PersonCreate) -> dict:
        try:
            return repository.create_person(payload.name, payload.external_id, payload.aliases)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="external_id 已存在") from exc

    @app.delete(
        "/api/library/persons/{person_id}", status_code=204, response_class=Response
    )
    def delete_person(person_id: str) -> Response:
        try:
            paths = repository.delete_person(person_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="人物不存在") from exc
        for path in paths:
            path.unlink(missing_ok=True)
        index.reload()
        return Response(status_code=204)

    @app.post("/api/library/persons/{person_id}/images", status_code=201)
    async def add_reference_image(
        person_id: str,
        image: Annotated[UploadFile, File()],
        source_url: Annotated[str | None, Form()] = None,
        source_page_url: Annotated[str | None, Form()] = None,
        license_code: Annotated[str | None, Form()] = None,
    ) -> dict:
        try:
            repository.get_person(person_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="人物不存在") from exc

        path, original_filename = await save_validated_upload(
            image,
            settings.data_dir / "tmp",
            settings.max_upload_mb * 1024 * 1024,
        )
        try:
            faces = engine.extract(path)
            if len(faces) != 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"参考图必须恰好包含一张人脸，当前检测到 {len(faces)} 张",
                )
            permanent_path = settings.data_dir / "references" / path.name
            shutil.move(str(path), permanent_path)
            result = repository.add_reference(
                person_id=person_id,
                original_filename=original_filename,
                stored_path=permanent_path,
                embedding=faces[0].embedding,
                source_url=source_url,
                license_code=license_code,
                source_page_url=source_page_url,
            )
            index.reload()
            return result
        except FaceEngineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)

    @app.get("/api/library/source-guide")
    def source_guide() -> dict:
        return {
            "workflow": [
                "选择已有人物，或填写姓名创建新人物",
                "每行填写一个公开可访问的图片直链，可在竖线后补充来源页面",
                "确认图片授权后点击下载并关联",
            ],
            "line_format": "https://example.com/portrait.jpg | https://example.com/source-page",
            "requirements": {
                "protocols": ["https", "http"],
                "formats": ["JPEG", "PNG", "WEBP"],
                "max_sources_per_request": 20,
                "exactly_one_face": True,
                "private_network_blocked": True,
            },
            "notice": "请仅导入有合法处理依据和可核验来源的图片。系统记录来源，不代替授权审查。",
        }

    @app.post("/api/library/quick-source-import")
    def quick_source_import(payload: QuickSourceImport) -> dict:
        if bool(payload.person_id) == bool(payload.person):
            raise HTTPException(
                status_code=422,
                detail="person_id 与 person 必须且只能填写一个",
            )

        person_created = False
        if payload.person_id:
            try:
                person = repository.get_person(payload.person_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="人物不存在") from exc
        else:
            assert payload.person is not None
            if payload.person.external_id:
                person, person_created = repository.get_or_create_person(
                    payload.person.name,
                    payload.person.external_id,
                    payload.person.aliases,
                )
            else:
                person = repository.create_person(
                    payload.person.name,
                    aliases=payload.person.aliases,
                )
                person_created = True

        imported: list[dict] = []
        skipped: list[dict] = []
        failed: list[dict] = []
        max_bytes = settings.max_upload_mb * 1024 * 1024
        source_directory = settings.data_dir / "references" / "source-imports" / person["id"]
        source_directory.mkdir(parents=True, exist_ok=True)

        for source_number, source in enumerate(payload.sources, start=1):
            image_url = source.image_url.strip()
            existing = repository.get_reference_by_source(person["id"], image_url)
            if existing is not None:
                skipped.append(
                    {
                        "source_number": source_number,
                        "image_url": image_url,
                        "reason": "该图片源已关联",
                        "reference_id": existing["id"],
                    }
                )
                continue

            downloaded_path: Path | None = None
            permanent_path: Path | None = None
            try:
                downloaded = download_remote_image(
                    image_url,
                    settings.data_dir / "tmp",
                    max_bytes,
                )
                downloaded_path = downloaded.path
                faces = engine.extract(downloaded.path)
                if len(faces) != 1:
                    raise SourceDownloadError(
                        f"参考图必须恰好包含一张人脸，当前检测到 {len(faces)} 张"
                    )
                permanent_path = source_directory / downloaded.path.name
                shutil.move(str(downloaded.path), permanent_path)
                result = repository.add_reference(
                    person_id=person["id"],
                    original_filename=downloaded.original_filename,
                    stored_path=permanent_path,
                    embedding=faces[0].embedding,
                    source_url=image_url,
                    source_page_url=source.source_page_url,
                    license_code=source.license_code,
                )
                result["source_number"] = source_number
                result["downloaded_from"] = downloaded.final_url
                imported.append(result)
            except (SourceDownloadError, FaceEngineError, sqlite3.IntegrityError) as exc:
                if permanent_path is not None:
                    permanent_path.unlink(missing_ok=True)
                failed.append(
                    {
                        "source_number": source_number,
                        "image_url": image_url,
                        "error": str(exc),
                    }
                )
            finally:
                if downloaded_path is not None:
                    downloaded_path.unlink(missing_ok=True)

        if imported:
            index.reload()
        person = repository.get_person(person["id"])
        return {
            "status": "completed" if not failed else "completed_with_errors",
            "person": person,
            "person_created": person_created,
            "summary": {
                "requested": len(payload.sources),
                "imported": len(imported),
                "skipped": len(skipped),
                "failed": len(failed),
            },
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "indexed_references": index.size,
        }

    @app.get("/api/library/images/{image_id}")
    def get_reference_image(image_id: str) -> FileResponse:
        try:
            path = repository.get_reference_path(image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="参考图片不存在") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="参考图片文件已丢失")
        return FileResponse(path)

    @app.post("/api/search")
    async def search(
        image: Annotated[UploadFile, File()],
        top_k: Annotated[int, Form(ge=1, le=20)] = settings.default_top_k,
    ) -> dict:
        if index.size == 0:
            raise HTTPException(status_code=409, detail="人物库为空，请先录入人物和参考图")
        path, _ = await save_validated_upload(
            image,
            settings.data_dir / "tmp",
            settings.max_upload_mb * 1024 * 1024,
        )
        try:
            faces = engine.extract(path)
            results = []
            for face_number, face in enumerate(faces, start=1):
                try:
                    candidates = index.search(face.embedding, top_k)
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                results.append(
                    {
                        "face_number": face_number,
                        "box": face.box,
                        "detection_confidence": round(face.confidence, 6),
                        "candidates": candidates,
                    }
                )
            return {
                "engine": engine.name,
                "real_face_recognition": engine.real_face_recognition,
                "faces_detected": len(faces),
                "results": results,
                "notice": "相似度仅表示模型特征接近程度，不代表身份确认。",
            }
        except FaceEngineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)

    static_dir = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
