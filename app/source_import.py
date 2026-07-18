from __future__ import annotations

import ipaddress
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from PIL import Image, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class SourceDownloadError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadedImage:
    path: Path
    original_filename: str
    final_url: str


def validate_public_http_url(url: str) -> str:
    """Validate a remote URL before every request and redirect."""
    try:
        parsed = urlparse(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise SourceDownloadError("图片地址格式无效") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SourceDownloadError("图片地址仅支持 http 或 https")
    if parsed.username or parsed.password:
        raise SourceDownloadError("图片地址不能包含用户名或密码")
    if port not in {None, 80, 443}:
        raise SourceDownloadError("图片地址仅允许使用 80 或 443 端口")
    if parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith(".local"):
        raise SourceDownloadError("不允许下载本机或内网地址")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname, port or (443 if parsed.scheme == "https" else 80)
            )
        }
    except socket.gaierror as exc:
        raise SourceDownloadError("图片地址无法解析") from exc

    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise SourceDownloadError("不允许下载本机、内网或保留地址")
    return parsed.geturl()


def _safe_filename(url: str, suffix: str) -> str:
    candidate = Path(unquote(urlparse(url).path)).name
    stem = Path(candidate).stem[:120] if candidate else "remote-reference"
    safe_stem = "".join(character for character in stem if character.isalnum() or character in "-_.")
    return f"{safe_stem or 'remote-reference'}{suffix}"


def download_remote_image(
    url: str,
    destination_dir: Path,
    max_bytes: int,
    *,
    timeout_seconds: float = 20.0,
    max_redirects: int = 5,
) -> DownloadedImage:
    destination_dir.mkdir(parents=True, exist_ok=True)
    current_url = url.strip()
    temporary = destination_dir / f"remote-{uuid.uuid4().hex}.download"

    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "CelebrityFaceSearch/0.2 (+reference-source-import)"},
        ) as client:
            for redirect_number in range(max_redirects + 1):
                current_url = validate_public_http_url(current_url)
                try:
                    with client.stream("GET", current_url) as response:
                        if response.status_code in REDIRECT_STATUSES:
                            location = response.headers.get("location")
                            if not location:
                                raise SourceDownloadError("图片源返回了无目标的跳转")
                            if redirect_number >= max_redirects:
                                raise SourceDownloadError("图片源跳转次数过多")
                            current_url = urljoin(current_url, location)
                            continue
                        if response.status_code != 200:
                            raise SourceDownloadError(
                                f"图片源下载失败（HTTP {response.status_code}）"
                            )
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                declared_size = int(content_length)
                            except ValueError:
                                declared_size = 0
                            if declared_size > max_bytes:
                                raise SourceDownloadError("远程图片超过大小限制")

                        downloaded = 0
                        with temporary.open("wb") as output:
                            for chunk in response.iter_bytes():
                                downloaded += len(chunk)
                                if downloaded > max_bytes:
                                    raise SourceDownloadError("远程图片超过大小限制")
                                output.write(chunk)
                        if downloaded == 0:
                            raise SourceDownloadError("远程图片内容为空")
                        break
                except httpx.HTTPError as exc:
                    raise SourceDownloadError("连接图片源失败") from exc
            else:  # pragma: no cover - the redirect guard exits first
                raise SourceDownloadError("图片源跳转次数过多")

        try:
            with Image.open(temporary) as image:
                image.verify()
                image_format = image.format
        except (UnidentifiedImageError, OSError) as exc:
            raise SourceDownloadError("远程内容不是有效的 JPEG、PNG 或 WEBP 图片") from exc

        suffix = ALLOWED_FORMATS.get(image_format or "")
        if suffix is None:
            raise SourceDownloadError("远程图片仅支持 JPEG、PNG 或 WEBP")
        final_path = destination_dir / f"{uuid.uuid4().hex}{suffix}"
        temporary.replace(final_path)
        return DownloadedImage(
            path=final_path,
            original_filename=_safe_filename(current_url, suffix),
            final_url=current_url,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
