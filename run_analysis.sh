#!/usr/bin/env bash
# Feature-usage analysis over every completed run.
#
# Separate from run_overnight.sh because the analysis block there had three bugs
# (see falsification/FINDINGS_2026-09-02.md, item 0). The important one: the
# analyzer names its output directory after the checkpoint directory, and
# get_experiment_name() omits the seed, so every seed of an arm resolves to the
# SAME output path. Writing them all to the default ./comprehensive_histogram_analysis
# makes each seed overwrite the last, leaving one summary per arm instead of six.
# Passing --output-dir per seed is what keeps the seeded design intact.
#
#   ./run_analysis.sh              # analyse every completed run, skipping done ones
#   ./run_analysis.sh --force      # re-analyse even if a summary already exists
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Identical for every checkpoint: features_used is sample-size dependent, so
# cross-arm comparability requires the same value everywhere.
N_SAMPLES=1000000

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/analysis_${STAMP}"
mkdir -p "$LOGDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/analysis.log"; }

ok=0; failed=0; skipped=0
log "Analysing with --n-samples $N_SAMPLES"

while IFS= read -r marker; do
  RUN=$(dirname "$marker")                      # e.g. experiments/baseline/seed3
  AE=$(find "$RUN" -name ae.pt | head -1)
  if [[ -z "$AE" ]]; then
    log "  no ae.pt under $RUN - skipping"; continue
  fi
  CKPT=$(dirname "$AE")                         # the trainer_0/ that load_dictionary needs
  NAME=$(echo "$RUN" | tr '/' '_')

  # Skip only if the summary is NEWER than the checkpoint it describes. A plain
  # existence check is a trap: retraining an arm (e.g. e1_vsae_ref after the
  # use_april_update_mode fix) leaves the previous generation's summary sitting
  # next to the new ae.pt, and every downstream table would then silently mix
  # measurements of checkpoints that no longer exist with ones that do. Comparing
  # mtimes re-analyses exactly what changed and nothing else -- which matters,
  # because --force costs ~6.5 min x every run on disk.
  if (( ! FORCE )); then
    SUMMARY=$(ls -t "$RUN"/*/comprehensive_summary_*.json 2>/dev/null | head -1)
    if [[ -n "$SUMMARY" ]]; then
      if [[ "$SUMMARY" -nt "$AE" ]]; then
        log "  skip (already analysed): $RUN"; skipped=$((skipped+1)); continue
      fi
      log "  STALE summary (older than ae.pt), re-analysing: $RUN"
    fi
  fi

  START=$(date +%s)
  log "  analysing $RUN"
  if python analysis_scripts/online_histogram_analyzer.py \
       --model-path "$CKPT" --output-dir "$RUN" \
       --n-samples "$N_SAMPLES" --no-individual \
       >"$LOGDIR/${NAME}.log" 2>&1; then
    log "    ok in $(( ($(date +%s) - START) / 60 ))min"
    ok=$((ok+1))
  else
    log "    FAILED - see $LOGDIR/${NAME}.log"
    tail -3 "$LOGDIR/${NAME}.log" | sed 's/^/      /' | tee -a "$LOGDIR/analysis.log"
    failed=$((failed+1))
  fi
done < <(find experiments -name RUN_COMPLETE.json 2>/dev/null | sort)

log "Analysis done. ok=$ok failed=$failed skipped=$skipped"
log "Summaries: $(find experiments -name 'comprehensive_summary_*.json' | wc -l)"
