#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m python.analyze_spot --input-file fixtures/leduc/root_k.json --format text
