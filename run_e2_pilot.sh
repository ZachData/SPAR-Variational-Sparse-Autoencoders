#!/usr/bin/env bash
# E2 stage 1: the beta pilot, then the pre-registered selection rule.
#
# Separate from run_overnight.sh because this is a SELECTION step, not an
# inference one. kl_coeff is not a shared scale across var_flag -- at 0 the KL is
# 0.5*||mu||^2, at 1 the variance term contributes ~220 of a ~225 total loss -- so
# "matched beta" compared two different interventions and posterior-collapsed all
# six seeds at beta=1.0. One seed per beta locates a beta that trains; six fresh
# seeds at that beta do the inference (REMEDIATION.md F6).
#
# The pilot runs at seed 101. The confirmatory arm runs at seeds 1-6. They are
# disjoint on purpose: no checkpoint may inform both selection and inference.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=101
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/e2_pilot_${STAMP}"
mkdir -p "$LOGDIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/pilot.log"; }

log "E2 beta pilot, seed $SEED"
for BETA in 0.0001 0.001 0.01 0.1 1; do
  ARM="e2_beta_pilot_${BETA}"
  log "  RUN $ARM"
  if python falsification/run_arm.py --arm "$ARM" --seed "$SEED" \
       >"$LOGDIR/${ARM}.log" 2>&1; then
    log "    ok"
  else
    log "    FAILED - see $LOGDIR/${ARM}.log"
    tail -5 "$LOGDIR/${ARM}.log" | sed 's/^/      /' | tee -a "$LOGDIR/pilot.log"
  fi
done

log "Applying the pre-registered selection rule."
python falsification/select_e2_beta.py 2>&1 | tee -a "$LOGDIR/pilot.log"
