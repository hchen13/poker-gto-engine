#!/usr/bin/env bash
# Thin wrapper around install.py for `./install.sh` muscle memory.
# Pass any flags through, e.g. `./install.sh --yes --skip-rust`.

set -euo pipefail
cd "$(dirname "$0")"
exec python3 install.py "$@"
