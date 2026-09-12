#!/usr/bin/env bash
# A4's JumpReLU discreteness companion (PROJECT.md, Claim #3's sharpest
# remaining test): 7 arms x 13 seeds, matched to run_a2_sweep.sh /
# run_a3_sweep.sh's pattern (safe to interrupt and re-run; completed runs are
# skipped via RUN_COMPLETE.json).
#
#   ./falsification/run_a4_sweep.sh
set -uo pipefail

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13)
ARMS=(a4_jumprelu_baseline a4_jumprelu_sigma_init_m1 a4_jumprelu_sampling_only
      a4_jumprelu_sigma_init_m3 a4_jumprelu_sigma_init_m4 a4_jumprelu_sigma_init_m5
      a4_jumprelu_sigma_low_init)

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/a4_sweep_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "arm\tseed\tstatus\tminutes\n" > "$SUMMARY"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/sweep.log"; }

log "A4 sweep starting. ${#ARMS[@]} arms x ${#SEEDS[@]} seeds = $(( ${#ARMS[@]} * ${#SEEDS[@]} )) runs."
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

log "A4 sweep complete. Summary:"
column -t "$SUMMARY" 2>/dev/null | tee -a "$LOGDIR/sweep.log" || cat "$SUMMARY"
log "Runs completed: $(find experiments -path '*/a4_jumprelu_*/*/RUN_COMPLETE.json' 2>/dev/null | wc -l)"
