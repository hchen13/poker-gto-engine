#!/bin/bash
# Single post-processing script for 3bet_called_ip3bet precompute.
# Does three things in one loop (every 60s):
#   1. Rolling gzip — compress completed .json files (keeps peak disk low)
#   2. Progress notifications — push every 5% milestone for each run
#   3. Finalize — when both runs done + all gzipped, run the two patch binaries
#
# Safe to restart (idempotent). Exits only after finalize completes.

set -e
cd "$(dirname "$0")/.."

DIRS=(
  precompute_out/full_3bet_called_ip3bet_200bb
  precompute_out/full_3bet_called_ip3bet_500bb
)
LABELS=(200BB 500BB)

last_pct_a=-5
last_pct_b=-5

notify() {
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Pop\"" 2>/dev/null || true
}

get_progress() {
    # echoes "completed total state" for a dir
    local dir=$1
    local p=$dir/progress.json
    [ -f "$p" ] || { echo "0 1 pending"; return; }
    python3 -c "
import json
try:
    d = json.load(open('$p'))
    print(d.get('completed',0), d.get('total',1), d.get('state','pending'))
except: print('0 1 pending')
"
}

no_uncompressed() {
    local dir=$1
    [ -d "$dir" ] || return 0
    local n=$(find "$dir" -maxdepth 1 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -eq 0 ]
}

while true; do
    # --- 1. Rolling gzip ---
    for dir in "${DIRS[@]}"; do
        [ -d "$dir" ] || continue
        files=$(find "$dir" -maxdepth 1 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null || true)
        if [ -n "$files" ]; then
            echo "$files" | xargs -P 4 -I{} gzip -f "{}"
        fi
    done

    # --- 2. Progress notifications ---
    read c_a t_a s_a <<< $(get_progress "${DIRS[0]}")
    read c_b t_b s_b <<< $(get_progress "${DIRS[1]}")
    pct_a=$((100 * c_a / (t_a > 0 ? t_a : 1)))
    pct_b=$((100 * c_b / (t_b > 0 ? t_b : 1)))

    # Push at each 5% bucket (so crossing 5% pushes, crossing 10% pushes, ...)
    step_a=$((pct_a / 5 * 5))
    step_b=$((pct_b / 5 * 5))
    if [ $step_a -gt $last_pct_a ] && [ $step_a -gt 0 ]; then
        notify "Poker GTO — ${LABELS[0]}" "${pct_a}% (${c_a}/${t_a})"
        last_pct_a=$step_a
    fi
    if [ $step_b -gt $last_pct_b ] && [ $step_b -gt 0 ]; then
        notify "Poker GTO — ${LABELS[1]}" "${pct_b}% (${c_b}/${t_b})"
        last_pct_b=$step_b
    fi

    # --- 3. Finalize once everything is done ---
    if [ "$s_a" = "done" ] && [ "$s_b" = "done" ] \
       && no_uncompressed "${DIRS[0]}" && no_uncompressed "${DIRS[1]}"; then
        echo "[$(date '+%H:%M:%S')] both runs done, all gzipped. Running patches..."
        ./target/release/patch_villain_buckets 2>&1 | tail -3
        ./target/release/patch_bucket_arrays 2>&1 | tail -3
        echo "[$(date '+%H:%M:%S')] done. Final sizes:"
        du -sh precompute_out/full_3bet_called_ip3bet_*
        notify "Poker GTO" "3bet_called_ip3bet ready (patches + gzip complete)"
        break
    fi

    sleep 60
done
