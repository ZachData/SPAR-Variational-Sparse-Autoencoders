#!/usr/bin/env bash
# A2's sigma_init dose-response sweep (PROJECT.md Next steps A2): 4 new arms x 13
# seeds, matched to the run_overnight.sh pattern (safe to interrupt and re-run;
# completed runs are skipped via RUN_COMPLETE.json).
#
#   ./falsification/run_a2_sweep.sh
set -uo pipefail

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13)
ARMS=(a2_sigma_init_m1 a2_sigma_init_m3 a2_sigma_init_m4 a2_sigma_init_m5)

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/a2_sweep_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "arm\tseed\tstatus\tminutes\n" > "$SUMMARY"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/sweep.log"; }

log "A2 sweep starting. ${#ARMS[@]} arms x ${#SEEDS[@]} seeds = $(( ${#ARMS[@]} * ${#SEEDS[@]} )) runs."
log "Logs: $LOGDIR"

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN_START=$(date +%s)
    log "RUN $arm seed=$seed"
    if python falsification/run_arm.py --arm "$arm" --seed "$seed" \
         >"$LOGDIR/${arm}_seed${seed}.log" 2>&1; then
      STATUS=ok
    else
      STATUS=FAILED
      log "  ^ FAILED (continuing). See $LOGDIR/${arm}_seed${seed}.log"
      tail -5 "$LOGDIR/${arm}_seed${seed}.log" | sed 's/^/      /' | tee -a "$LOGDIR/sweep.log"
    fi
    RUN_MIN=$(( ($(date +%s) - RUN_START) / 60 ))
    printf "%s\t%s\t%s\t%s\n" "$arm" "$seed" "$STATUS" "$RUN_MIN" >> "$SUMMARY"
    log "  -> $STATUS in ${RUN_MIN}min"
  done
done

log "A2 sweep complete. Summary:"
column -t "$SUMMARY" 2>/dev/null | tee -a "$LOGDIR/sweep.log" || cat "$SUMMARY"
log "Runs completed: $(find experiments -path '*/a2_sigma_init_*/*/RUN_COMPLETE.json' 2>/dev/null | wc -l)"
