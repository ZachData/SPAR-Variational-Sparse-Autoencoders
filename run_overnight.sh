#!/usr/bin/env bash
# Unattended experiment sweep. Safe to interrupt and re-run: completed runs are
# skipped via their RUN_COMPLETE.json marker.
#
#   ./run_overnight.sh                # ~10 hour budget (default)
#   ./run_overnight.sh --hours 14
#   ./run_overnight.sh --dry-run      # print the plan, train nothing
#
# Deliberately NOT `set -e`: one failed run must not abort the sweep. Failures are
# logged and the sweep continues to the next run.
set -uo pipefail

BUDGET_HOURS=10
DRY_RUN=0
SEEDS=(1 2 3 4 5 6)
# Priority order. If the budget runs out, later arms are simply not started, so
# the most valuable work is always the work that completes.
# e3_masked_kl_relu is the ReLU arm of the E3 factor: vsae_topk.py applies
# F.relu(mu) and the masked trainer does not, so one arm alone confounds the KL
# mask with the ReLU. Both are run and the difference between them IS the ReLU's
# contribution (REMEDIATION.md F9b).
#
# The E2 beta pilot is deliberately NOT here. It is stage 1 of a two-stage design
# and runs at seed 101, disjoint from these; see ./run_e2_pilot.sh.
ARMS=(baseline e2_learned_var e1_penalty e1_vsae_ref e3_masked_kl e3_masked_kl_relu)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) BUDGET_HOURS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"
# Without this the allocator fragments and evaluation OOMs on a 10GB card that
# is also driving a display; the OOM is caught and silently NaNs the loss metrics.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/sweep_${STAMP}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "arm\tseed\tstatus\tminutes\n" > "$SUMMARY"

START_EPOCH=$(date +%s)
DEADLINE=$(( START_EPOCH + BUDGET_HOURS * 3600 ))

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/sweep.log"; }

log "Sweep starting. Budget ${BUDGET_HOURS}h, deadline $(date -d "@$DEADLINE" '+%F %T' 2>/dev/null || date -r "$DEADLINE" '+%F %T')"
log "Logs: $LOGDIR"

# ---- Gate 1: framework tests. Cheap, and a failure means the analysis is broken.
log "Running framework tests..."
if ! python -m pytest falsification/tests/ -q >"$LOGDIR/pytest.log" 2>&1; then
  log "ABORT: framework tests failed. See $LOGDIR/pytest.log"
  exit 1
fi
log "Framework tests passed."

# ---- Gate 2: arm configs validate without torch.
if ! python falsification/run_arm.py --check >"$LOGDIR/argcheck.log" 2>&1; then
  log "ABORT: arm config validation failed. See $LOGDIR/argcheck.log"
  cat "$LOGDIR/argcheck.log"; exit 1
fi
log "Arm configs valid."

# ---- Gate 3: preflight (torch, CUDA, wiring, data, disk).
# Skipped for --dry-run so the plan can be reviewed on a machine without torch.
if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY RUN - skipping preflight (needs torch). Planned order:"
  for arm in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do echo "    $arm seed=$s"; done; done
  log "Total planned runs: $(( ${#ARMS[@]} * ${#SEEDS[@]} ))"
  exit 0
fi
if ! python falsification/preflight.py >"$LOGDIR/preflight.log" 2>&1; then
  log "ABORT: preflight failed. See $LOGDIR/preflight.log"
  cat "$LOGDIR/preflight.log"; exit 1
fi
log "Preflight clear."

# ---- Training. Time-aware: never start a run that cannot finish in budget.
LAST_MIN=30   # optimistic first estimate; replaced by the first real measurement
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    NOW=$(date +%s)
    REMAIN_MIN=$(( (DEADLINE - NOW) / 60 ))
    if (( REMAIN_MIN < LAST_MIN )); then
      log "STOP: ${REMAIN_MIN}min left, last run took ${LAST_MIN}min. Not starting $arm seed=$seed."
      log "Re-run this script later; completed runs are skipped automatically."
      break 2
    fi

    RUN_START=$(date +%s)
    log "RUN $arm seed=$seed  (${REMAIN_MIN}min left in budget)"
    if python falsification/run_arm.py --arm "$arm" --seed "$seed" \
         >"$LOGDIR/${arm}_seed${seed}.log" 2>&1; then
      STATUS=ok
    else
      STATUS=FAILED
      log "  ^ FAILED (continuing). See $LOGDIR/${arm}_seed${seed}.log"
      tail -5 "$LOGDIR/${arm}_seed${seed}.log" | sed 's/^/      /' | tee -a "$LOGDIR/sweep.log"
    fi
    RUN_MIN=$(( ($(date +%s) - RUN_START) / 60 ))
    [[ $STATUS == ok ]] && LAST_MIN=$(( RUN_MIN + 2 ))
    printf "%s\t%s\t%s\t%s\n" "$arm" "$seed" "$STATUS" "$RUN_MIN" >> "$SUMMARY"
    log "  -> $STATUS in ${RUN_MIN}min"
  done
done

# ---- Analysis over every completed checkpoint. Delegated to run_analysis.sh.
# The loop that used to live here was wrong in three ways (see
# falsification/FINDINGS_2026-09-02.md item 0); the dangerous one was silent:
# the analyzer names its output dir after the checkpoint dir, get_experiment_name()
# omits the seed, so with the default --output-dir every seed of an arm overwrote
# the previous one and you were left with one summary per arm instead of six.
log "Training phase done. Starting feature-usage analysis."
./run_analysis.sh 2>&1 | tee -a "$LOGDIR/sweep.log"

log "Sweep complete. Summary:"
column -t "$SUMMARY" 2>/dev/null | tee -a "$LOGDIR/sweep.log" || cat "$SUMMARY"
log "Runs completed: $(find experiments -name RUN_COMPLETE.json 2>/dev/null | wc -l)"
