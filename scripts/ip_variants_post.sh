#!/bin/bash
# Rolling gzip + 5% desktop notifications + finalize patches, for the
# 4 new *_ip_caller variant precomputes.

set -e
cd "$(dirname "$0")/.."

DIRS=(
  precompute_out/full_sr_called_ip_caller_200bb
  precompute_out/full_sr_called_ip_caller_500bb
  precompute_out/full_4bet_called_ip_caller_200bb
  precompute_out/full_4bet_called_ip_caller_500bb
)
LABELS=(sr-200 sr-500 4bet-200 4bet-500)

LAST_PCT=(-5 -5 -5 -5)

notify() {
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Pop\"" 2>/dev/null || true
}

get_progress() {
    local p=$1/progress.json
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
    local n=$(find "$1" -maxdepth 1 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -eq 0 ]
}

while true; do
    # Rolling gzip
    for dir in "${DIRS[@]}"; do
        [ -d "$dir" ] || continue
        files=$(find "$dir" -maxdepth 1 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null || true)
        [ -n "$files" ] && echo "$files" | xargs -P 4 -I{} gzip -f "{}"
    done

    # Progress notifications
    all_done=1
    for i in "${!DIRS[@]}"; do
        read c t s <<< $(get_progress "${DIRS[$i]}")
        pct=$((100 * c / (t > 0 ? t : 1)))
        step=$((pct / 5 * 5))
        if [ $step -gt ${LAST_PCT[$i]} ] && [ $step -gt 0 ]; then
            notify "Poker GTO — ${LABELS[$i]}" "${pct}% (${c}/${t})"
            LAST_PCT[$i]=$step
        fi
        if [ "$s" != "done" ] || ! no_uncompressed "${DIRS[$i]}"; then
            all_done=0
        fi
    done

    if [ $all_done -eq 1 ]; then
        echo "[$(date '+%H:%M:%S')] all 4 runs done + gzipped. Running patches..."
        ./target/release/patch_villain_buckets 2>&1 | tail -3
        ./target/release/patch_bucket_arrays 2>&1 | tail -3
        echo "[$(date '+%H:%M:%S')] done. Final sizes:"
        for dir in "${DIRS[@]}"; do du -sh "$dir" 2>/dev/null; done
        notify "Poker GTO" "sr_called + 4bet_called IP variants ready (patches + gzip complete)"
        break
    fi
    sleep 60
done
