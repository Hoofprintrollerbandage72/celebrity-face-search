from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from app.config import Settings


class FaceEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray
    box: dict[str, int]
    confidence: float


class FaceEngine(Protocol):
    name: str
    real_face_recognition: bool

    @property
    def available(self) -> bool: ...

    def extract(self, image_path: Path) -> list[DetectedFace]: ...


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise FaceEngineError("模型返回了空的人脸向量")
    return vector / norm


class DemoImageEngine:
    """A tiny image-similarity engine for UI/API development only.

    It deliberately treats the whole image as one face. It must never be
    represented as a real identity-recognition model.
    """

    name = "demo-image-similarity"
    real_face_recognition = False

    @property
    def available(self) -> bool:
        return True

    def extract(self, image_path: Path) -> list[DetectedFace]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((32, 32))
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
            features: list[np.ndarray] = []
            for channel in range(3):
                histogram, _ = np.histogram(
                    pixels[:, :, channel], bins=32, range=(0.0, 1.0), density=True
                )
                features.append(histogram.astype(np.float32))
            grayscale = pixels.mean(axis=2)
            blocks = grayscale.reshape(8, 4, 8, 4).mean(axis=(1, 3)).flatten()
            vector = normalize(np.concatenate([*features, blocks]))
            width, height = image.size
        return [
            DetectedFace(
                embedding=vector,
                box={"x": 0, "y": 0, "width": width, "height": height},
                confidence=1.0,
            )
        ]


class DeepFaceEngine:
    name = "deepface"
    real_face_recognition = True

    def __init__(self, model_name: str, detector_backend: str):
        self.model_name = model_name
        self.detector_backend = detector_backend

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("deepface") is not None

    def extract(self, image_path: Path) -> list[DetectedFace]:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise FaceEngineError(
                "DeepFace 未安装。请使用 requirements-deepface.txt 或 Docker 启动。"
            ) from exc

        try:
            results = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
        except Exception as exc:
            raise FaceEngineError(f"没有检测到可识别人脸：{exc}") from exc

        faces: list[DetectedFace] = []
        for result in results:
            area = result.get("facial_area") or {}
            faces.append(
                DetectedFace(
                    embedding=normalize(np.asarray(result["embedding"], dtype=np.float32)),
                    box={
                        "x": int(area.get("x", 0)),
                        "y": int(area.get("y", 0)),
                        "width": int(area.get("w", 0)),
                        "height": int(area.get("h", 0)),
                    },
                    confidence=float(result.get("face_confidence", 1.0)),
                )
            )
        return faces


class OpenCVSFaceEngine:
    name = "opencv-sface"
    real_face_recognition = True

    def __init__(
        self,
        detector_model: Path,
        recognition_model: Path,
        detection_threshold: float = 0.75,
        max_dimension: int = 1000,
    ):
        self.detector_model = detector_model
        self.recognition_model = recognition_model
        self.detection_threshold = detection_threshold
        self.max_dimension = max_dimension
        self._detector = None
        self._recognizer = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return (
            importlib.util.find_spec("cv2") is not None
            and self.detector_model.is_file()
            and self.recognition_model.is_file()
        )

    def _load(self):
        if not self.available:
            raise FaceEngineError(
                "OpenCV 人脸模型尚未就绪，请运行 python3 scripts/setup_models.py"
            )
        if self._detector is not None and self._recognizer is not None:
            return
        import cv2

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_model),
                "",
                (320, 320),
                self.detection_threshold,
                0.3,
                5000,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognition_model), ""
            )
        except Exception as exc:
            self._detector = None
            self._recognizer = None
            raise FaceEngineError(f"OpenCV 人脸模型加载失败：{exc}") from exc

    def extract(self, image_path: Path) -> list[DetectedFace]:
        with self._lock:
            self._load()
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FaceEngineError("OpenCV 无法读取图片")

            original_height, original_width = image.shape[:2]
            scale = min(1.0, self.max_dimension / max(original_width, original_height))
            if scale < 1.0:
                image = cv2.resize(
                    image,
                    (int(original_width * scale), int(original_height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            height, width = image.shape[:2]
            self._detector.setInputSize((width, height))
            _, detected = self._detector.detect(image)
            if detected is None or len(detected) == 0:
                raise FaceEngineError("没有检测到可识别人脸")

            faces: list[DetectedFace] = []
            for face in detected:
                try:
                    aligned = self._recognizer.alignCrop(image, face)
                    feature = self._recognizer.feature(aligned).flatten()
                except Exception:
                    continue
                x, y, face_width, face_height = face[:4]
                faces.append(
                    DetectedFace(
                        embedding=normalize(feature),
                        box={
                            "x": max(0, int(x / scale)),
                            "y": max(0, int(y / scale)),
                            "width": int(face_width / scale),
                            "height": int(face_height / scale),
                        },
                        confidence=float(face[14]),
                    )
                )
            if not faces:
                raise FaceEngineError("检测到人脸，但无法生成有效特征")
            return faces


def build_face_engine(settings: Settings) -> FaceEngine:
    if settings.face_engine in {"opencv", "opencv_sface", "sface"}:
        model_dir = settings.opencv_model_dir or (settings.data_dir / "models")
        return OpenCVSFaceEngine(
            detector_model=model_dir / "face_detection_yunet_2023mar.onnx",
            recognition_model=model_dir / "face_recognition_sface_2021dec.onnx",
            detection_threshold=settings.opencv_detection_threshold,
        )
    if settings.face_engine == "deepface":
        return DeepFaceEngine(settings.deepface_model, settings.deepface_detector)
    if settings.face_engine == "demo":
        return DemoImageEngine()
    raise ValueError(f"不支持的 FACE_ENGINE：{settings.face_engine}")
