#!/bin/bash
# Monitor full precompute progress and send macOS notifications every 5%.
# Total: 8 runs × 1755 flops = 14040 flops. 5% = 702 flops.

OUTDIR="/Users/claire/projects/poker-gto-engine/precompute_out"
RUNS=(
  "full_3bet_called_200bb"
  "full_4bet_called_200bb"
  "full_limped_200bb"
  "full_sr_called_200bb"
  "full_3bet_called_500bb"
  "full_4bet_called_500bb"
  "full_limped_500bb"
  "full_sr_called_500bb"
)
TOTAL=14040
STEP=702   # 5%
LAST_NOTIFIED=0

notify() {
  osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\""
}

notify "GTO Precompute" "Full run started — 14,040 flops total (~16h)"

while true; do
  count=0
  for run in "${RUNS[@]}"; do
    d="$OUTDIR/$run"
    if [ -d "$d" ]; then
      n=$(ls "$d"/*.json 2>/dev/null | grep -v progress | wc -l | tr -d ' ')
      count=$((count + n))
    fi
  done

  pct=$((count * 100 / TOTAL))
  milestone=$((count / STEP))

  if [ "$milestone" -gt "$LAST_NOTIFIED" ] && [ "$count" -gt 0 ]; then
    LAST_NOTIFIED=$milestone
    notify "GTO Precompute ${pct}%" "${count}/${TOTAL} flops done — ~$((( TOTAL - count ) * 20 / 8 / 60))min remaining"
  fi

  if [ "$count" -ge "$TOTAL" ]; then
    notify "GTO Precompute 完成!" "全部 14,040 张 flop 计算完毕"
    break
  fi

  sleep 60
done
