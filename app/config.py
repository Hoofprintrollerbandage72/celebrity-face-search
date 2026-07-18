from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    data_dir: Path
    face_engine: str
    deepface_model: str
    deepface_detector: str
    max_upload_mb: int
    default_top_k: int
    opencv_model_dir: Path | None = None
    opencv_detection_threshold: float = 0.75

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "公众人物人脸检索"),
            data_dir=Path(os.getenv("APP_DATA_DIR", "./data")).expanduser().resolve(),
            face_engine=os.getenv("FACE_ENGINE", "opencv_sface").strip().lower(),
            deepface_model=os.getenv("DEEPFACE_MODEL", "Facenet512").strip(),
            deepface_detector=os.getenv("DEEPFACE_DETECTOR", "opencv").strip(),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
            default_top_k=int(os.getenv("SEARCH_TOP_K", "5")),
            opencv_model_dir=(
                Path(os.environ["OPENCV_MODEL_DIR"]).expanduser().resolve()
                if os.getenv("OPENCV_MODEL_DIR")
                else None
            ),
            opencv_detection_threshold=float(
                os.getenv("OPENCV_DETECTION_THRESHOLD", "0.75")
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "references").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tmp").mkdir(parents=True, exist_ok=True)
        (self.opencv_model_dir or (self.data_dir / "models")).mkdir(
            parents=True, exist_ok=True
        )
