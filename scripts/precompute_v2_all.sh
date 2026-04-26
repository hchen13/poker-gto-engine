#!/bin/bash
# Run all 14 canonical-enumeration precomputes with rolling gzip, 5% progress
# notifications, and auto-patch when complete. Queue model: MAX_PARALLEL runs
# at a time; when one finishes, start the next.
#
# Ordering: fastest-first (narrow ranges / small trees) to validate pipeline
# before committing to the long sr_called runs.

set -e
cd "$(dirname "$0")/.."

MAX_PARALLEL=${MAX_PARALLEL:-3}

# Queue ordered fastest → slowest (by observed precompute time)
QUEUE=(
  # 4bet: narrowest ranges, smallest tree
  full_4bet_called_200bb
  full_4bet_called_ip_caller_200bb
  full_4bet_called_500bb
  full_4bet_called_ip_caller_500bb
  # 3bet
  full_3bet_called_200bb
  full_3bet_called_ip3bet_200bb
  full_3bet_called_500bb
  full_3bet_called_ip3bet_500bb
  # sr_called (widest)
  full_sr_called_200bb
  full_sr_called_ip_caller_200bb
  full_sr_called_500bb
  full_sr_called_ip_caller_500bb
  # limped (widest by far — run last so rolling gzip keeps disk low throughout)
  full_limped_200bb
  full_limped_500bb
)

V2_ROOT=precompute_out_v2
mkdir -p "$V2_ROOT"

notify() {
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Pop\"" 2>/dev/null || true
}

# Launch one precompute run. Prints stderr/stdout to a per-run log.
launch_run() {
    local name=$1
    local cfg=fixtures/precompute/v2/$name.json
    local logf=$V2_ROOT/$name/run.log
    mkdir -p "$(dirname "$logf")"
    ./target/release/precompute_bucketed_parallel "$cfg" > "$logf" 2>&1 &
    echo $!
}

# Track PIDs and names in parallel arrays
declare -a PIDS
declare -a NAMES

count_running() {
    local n=0
    for pid in "${PIDS[@]}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && n=$((n+1))
    done
    echo $n
}

# Fill initial slots
next_idx=0
echo "[$(date '+%H:%M:%S')] scheduling ${#QUEUE[@]} precompute runs, max_parallel=$MAX_PARALLEL"
while [ $(count_running) -lt $MAX_PARALLEL ] && [ $next_idx -lt ${#QUEUE[@]} ]; do
    name=${QUEUE[$next_idx]}
    pid=$(launch_run "$name")
    PIDS+=("$pid")
    NAMES+=("$name")
    echo "[$(date '+%H:%M:%S')] started $name (pid $pid)"
    next_idx=$((next_idx+1))
done

# Supervisor loop: rolling gzip, progress notifications, queue advancement,
# exits when all done.
declare -a LAST_PCT=()
for _ in "${QUEUE[@]}"; do LAST_PCT+=("-5"); done

notify "Poker GTO v2" "canonical precompute started (${#QUEUE[@]} runs)"

while true; do
    # Rolling gzip on all v2 dirs (atomic fs::write means .json files are complete)
    find "$V2_ROOT" -maxdepth 2 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null | xargs -P 4 -I{} gzip -f "{}" 2>/dev/null || true

    # Progress notifications
    for i in "${!NAMES[@]}"; do
        name=${NAMES[$i]}
        progress=$V2_ROOT/$name/progress.json
        [ -f "$progress" ] || continue
        pct=$(python3 -c "
import json
try:
    d = json.load(open('$progress'))
    t = d.get('total', 1); c = d.get('completed', 0)
    print(int(100 * c / max(t, 1)))
except: print(0)
" 2>/dev/null || echo 0)
        step=$((pct / 5 * 5))
        if [ $step -gt ${LAST_PCT[$i]} ] && [ $step -gt 0 ]; then
            notify "Poker GTO v2 — $name" "${pct}%"
            LAST_PCT[$i]=$step
        fi
    done

    # Check for finished runs, start next in queue
    all_done=1
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        [ -z "$pid" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            name=${NAMES[$i]}
            echo "[$(date '+%H:%M:%S')] $name finished"
            notify "Poker GTO v2 — $name" "done"
            PIDS[$i]=""
            # Start next in queue if any
            if [ $next_idx -lt ${#QUEUE[@]} ]; then
                new_name=${QUEUE[$next_idx]}
                new_pid=$(launch_run "$new_name")
                PIDS+=("$new_pid")
                NAMES+=("$new_name")
                LAST_PCT+=("-5")
                echo "[$(date '+%H:%M:%S')] started $new_name (pid $new_pid)"
                next_idx=$((next_idx+1))
                all_done=0
            fi
        else
            all_done=0
        fi
    done
    if [ $next_idx -lt ${#QUEUE[@]} ]; then all_done=0; fi

    if [ $all_done -eq 1 ]; then
        # One more gzip sweep to catch any stragglers
        find "$V2_ROOT" -maxdepth 2 -name 'flop_*.json' -not -name '*.gz' 2>/dev/null | xargs -P 4 -I{} gzip -f "{}" 2>/dev/null || true
        break
    fi

    sleep 60
done

echo "[$(date '+%H:%M:%S')] all 14 runs complete. Running patches on v2..."

# Temporarily point patch binaries at v2 by copying v2 dirs into a patch location
# Actually: patch binaries scan precompute_out/. Easiest is to temporarily move
# v1 aside and symlink v2 in place, patch, then move v1 back.
# Simpler alternative: patch binaries look for dirs matching full_* pattern;
# we can rename v2 subdirs to not clash with v1, patch finds both, or just
# run patches against v2_ROOT by setting PRECOMPUTE_DIR (not currently supported).
# For now, do atomic swap: backup v1 → move v2 → patch → keep v2 as primary.

BACKUP_ROOT=precompute_out_v1_backup
if [ -d precompute_out ]; then
    mv precompute_out "$BACKUP_ROOT"
fi
mv "$V2_ROOT" precompute_out

echo "[$(date '+%H:%M:%S')] running patch_villain_buckets..."
./target/release/patch_villain_buckets 2>&1 | tail -5

echo "[$(date '+%H:%M:%S')] running patch_bucket_arrays..."
./target/release/patch_bucket_arrays 2>&1 | tail -5

echo "[$(date '+%H:%M:%S')] done. Final sizes:"
du -sh precompute_out/* 2>/dev/null | head -20

notify "Poker GTO v2" "All 14 runs complete. Canonical iso-class precompute is live. v1 backed up."
