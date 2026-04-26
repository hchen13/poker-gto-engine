#!/bin/bash
set -e
cd "$(dirname "$0")"

BIN=./target/release/precompute_bucketed_parallel
LINES=(3bet_called 4bet_called limped sr_called)
STACKS=(200bb 500bb)

echo "=== full precompute start: $(date) ==="
for stack in "${STACKS[@]}"; do
    for line in "${LINES[@]}"; do
        cfg="fixtures/precompute/full_${line}_${stack}.json"
        echo ""
        echo "--- starting ${stack} ${line} at $(date) ---"
        $BIN "$cfg"
        echo "--- done ${stack} ${line} at $(date) ---"
    done
done
echo ""
echo "=== full precompute complete: $(date) ==="
