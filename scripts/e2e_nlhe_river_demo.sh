#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m python.analyze_spot --input-file fixtures/nlhe-river/hero_aa_value_call.json --format text
