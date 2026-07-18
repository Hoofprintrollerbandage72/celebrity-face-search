#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c 'import cv2, fastapi, numpy, PIL' >/dev/null 2>&1; then
  "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

export APP_DATA_DIR=${APP_DATA_DIR:-"$PROJECT_DIR/data"}
export OPENCV_MODEL_DIR=${OPENCV_MODEL_DIR:-"$APP_DATA_DIR/models"}
export FACE_ENGINE=${FACE_ENGINE:-opencv_sface}

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" scripts/setup_models.py --model-dir "$OPENCV_MODEL_DIR"
exec "$VENV_DIR/bin/uvicorn" app.main:app --host 127.0.0.1 --port "${PORT:-8000}"

