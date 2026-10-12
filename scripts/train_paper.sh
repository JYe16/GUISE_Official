#!/usr/bin/env bash
# AXIS with the paper's per-model settings: anchor, then adapter.
#
#   bash scripts/train_paper.sh {qwen3-1.7b|qwen3-4b|glm-4-9b} [seed] [extra axis.train args]
#
# Inputs are read from $DATA (default data/), outputs go to $RUNS/<model>/$TAG
# (defaults runs/ and axis_s<seed>). Example ablation:
#   TAG=random_s0 bash scripts/train_paper.sh qwen3-1.7b 0 --anchor-mode random
set -euo pipefail
cd "$(dirname "$0")/.."

QWEN_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
NAME=${1:-}
case "$NAME" in
  qwen3-1.7b) MODEL=Qwen/Qwen3-1.7B;       LAYER=24; TARGETS=$QWEN_TARGETS ;;
  qwen3-4b)   MODEL=Qwen/Qwen3-4B;         LAYER=19; TARGETS=$QWEN_TARGETS ;;
  glm-4-9b)   MODEL=zai-org/GLM-4-9B-0414; LAYER=21; TARGETS=q_proj,k_proj,v_proj,o_proj,gate_up_proj,down_proj ;;
  *) echo "usage: $0 {qwen3-1.7b|qwen3-4b|glm-4-9b} [seed] [extra axis.train args]"; exit 1 ;;
esac
SEED=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))

DATA=${DATA:-data}
OUT=${RUNS:-runs}/$NAME
TAG=${TAG:-axis_s$SEED}

if [ ! -f "$OUT/anchor.npz" ]; then
  python -m axis.anchor --model "$MODEL" --layer "$LAYER" \
    --corpus "$DATA/parallel_corpus.json" --split "$DATA/split.json" --out "$OUT/anchor.npz"
fi

python -m axis.train --model "$MODEL" --anchor "$OUT/anchor.npz" \
  --pairs "$DATA/pref_train.jsonl" --rot "$DATA/rot_train.json" \
  --lora-targets "$TARGETS" --seed "$SEED" --out "$OUT/$TAG" "$@"
