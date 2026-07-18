from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.config import Settings
from app.main import create_app
from app.source_import import DownloadedImage, SourceDownloadError, validate_public_http_url


def png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 80), color).save(output, format="PNG")
    return output.getvalue()


def client_for(tmp_path) -> TestClient:
    settings = Settings(
        app_name="test",
        data_dir=tmp_path,
        face_engine="demo",
        deepface_model="Facenet512",
        deepface_detector="opencv",
        max_upload_mb=2,
        default_top_k=5,
    )
    return TestClient(create_app(settings))


def add_person_with_image(client: TestClient, name: str, color: tuple[int, int, int]) -> dict:
    person = client.post("/api/library/persons", json={"name": name}).json()
    response = client.post(
        f"/api/library/persons/{person['id']}/images",
        files={"image": (f"{name}.png", png(color), "image/png")},
    )
    assert response.status_code == 201, response.text
    return person


def test_health_and_empty_library(tmp_path) -> None:
    client = client_for(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["real_face_recognition"] is False
    assert response.json()["engine_available"] is True
    assert response.json()["reference_images"] == 0


def test_create_index_and_search(tmp_path) -> None:
    client = client_for(tmp_path)
    red = add_person_with_image(client, "Red Person", (230, 20, 20))
    add_person_with_image(client, "Blue Person", (20, 20, 230))

    response = client.post(
        "/api/search",
        data={"top_k": "2"},
        files={"image": ("query.png", png((225, 25, 25)), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["faces_detected"] == 1
    assert payload["results"][0]["candidates"][0]["person_id"] == red["id"]


def test_rejects_invalid_image(tmp_path) -> None:
    client = client_for(tmp_path)
    person = client.post("/api/library/persons", json={"name": "Test"}).json()
    response = client.post(
        f"/api/library/persons/{person['id']}/images",
        files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_delete_person_rebuilds_index(tmp_path) -> None:
    client = client_for(tmp_path)
    person = add_person_with_image(client, "Delete Me", (100, 120, 140))
    response = client.delete(f"/api/library/persons/{person['id']}")
    assert response.status_code == 204
    assert client.get("/api/health").json()["reference_images"] == 0


def test_dataset_stats_filter_and_reload(tmp_path) -> None:
    client = client_for(tmp_path)
    person = client.post(
        "/api/library/persons",
        json={"name": "Anonymous One", "external_id": "celeba:1"},
    ).json()
    response = client.post(
        f"/api/library/persons/{person['id']}/images",
        files={"image": ("one.png", png((120, 80, 60)), "image/png")},
        data={"license_code": "research-only"},
    )
    assert response.status_code == 201
    stats = client.get("/api/library/stats").json()
    assert stats["datasets"][0] == {
        "dataset": "celeba",
        "persons": 1,
        "reference_images": 1,
    }
    assert client.get("/api/library/persons", params={"q": "celeba:1"}).json()[0][
        "id"
    ] == person["id"]
    assert client.post("/api/library/reload").json()["indexed_references"] == 1


def test_quick_source_import_downloads_links_and_skips_duplicates(
    tmp_path, monkeypatch
) -> None:
    def fake_download(url, destination_dir, max_bytes):
        path = destination_dir / "downloaded.png"
        path.write_bytes(png((210, 70, 40)))
        return DownloadedImage(path, "portrait.png", url)

    monkeypatch.setattr(main_module, "download_remote_image", fake_download)
    client = client_for(tmp_path)
    request = {
        "person": {
            "name": "Source Person",
            "external_id": "custom:source-person",
            "aliases": ["SP"],
        },
        "sources": [
            {
                "image_url": "https://images.example.test/portrait.png",
                "source_page_url": "https://example.test/profile",
                "license_code": "CC BY 4.0",
            }
        ],
    }
    response = client.post("/api/library/quick-source-import", json=request)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["person_created"] is True
    assert payload["summary"] == {
        "requested": 1,
        "imported": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert payload["person"]["image_count"] == 1

    duplicate = client.post("/api/library/quick-source-import", json=request).json()
    assert duplicate["person_created"] is False
    assert duplicate["summary"]["skipped"] == 1
    assert duplicate["person"]["image_count"] == 1

    search = client.post(
        "/api/search",
        files={"image": ("query.png", png((210, 70, 40)), "image/png")},
    ).json()
    candidate = search["results"][0]["candidates"][0]
    assert candidate["source_page_url"] == "https://example.test/profile"
    assert candidate["source_url"] == "https://images.example.test/portrait.png"
    assert candidate["license_code"] == "CC BY 4.0"


def test_quick_source_import_reports_blocked_private_source(tmp_path) -> None:
    client = client_for(tmp_path)
    response = client.post(
        "/api/library/quick-source-import",
        json={
            "person": {"name": "Blocked Source"},
            "sources": [{"image_url": "http://127.0.0.1/private.png"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["summary"]["failed"] == 1
    assert "内网" in response.json()["failed"][0]["error"]


def test_source_guide_and_private_url_guard(tmp_path) -> None:
    client = client_for(tmp_path)
    guide = client.get("/api/library/source-guide")
    assert guide.status_code == 200
    assert guide.json()["requirements"]["max_sources_per_request"] == 20
    try:
        validate_public_http_url("http://localhost/image.jpg")
    except SourceDownloadError as exc:
        assert "本机" in str(exc) or "内网" in str(exc)
    else:
        raise AssertionError("localhost source should be blocked")
