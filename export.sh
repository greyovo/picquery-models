#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec uv run --locked python "$ROOT/models/ml-mobileclip/export_mobileclip_models.py" "$@"
