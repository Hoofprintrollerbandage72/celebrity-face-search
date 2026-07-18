#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/.venv/bin/python"
STATUS_DIR="$PROJECT_DIR/data/import_jobs"
STATUS_FILE="$STATUS_DIR/full_dataset_import.status"
mkdir -p "$STATUS_DIR"

write_status() {
  "$PYTHON" - "$STATUS_FILE" "$1" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
status = sys.argv[2]
path.write_text(
    json.dumps(
        {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()},
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
PY
}

fail_status() {
  code=$?
  if [ "$code" -ne 0 ]; then
    write_status failed
  fi
  exit "$code"
}
trap fail_status EXIT INT TERM

write_status running
"$PYTHON" scripts/setup_models.py
"$PYTHON" scripts/import_research_datasets.py bootstrap --dataset all

"$PYTHON" -u scripts/import_research_datasets.py download-vgg --which train &
VGG_DOWNLOAD_PID=$!
"$PYTHON" -u scripts/import_research_datasets.py download-celeba &
CELEBA_DOWNLOAD_PID=$!

wait "$VGG_DOWNLOAD_PID"
wait "$CELEBA_DOWNLOAD_PID"

gzip -t data/datasets/vggface2/vggface2_train.tar.gz
"$PYTHON" -u scripts/import_research_datasets.py vggface2 \
  --which train --images-per-identity 5
"$PYTHON" -u scripts/import_research_datasets.py celeba \
  --images-per-identity 5

curl --silent --show-error --fail --request POST \
  http://127.0.0.1:8000/api/library/reload >/dev/null || true
"$PYTHON" scripts/import_research_datasets.py status
write_status complete
trap - EXIT INT TERM
