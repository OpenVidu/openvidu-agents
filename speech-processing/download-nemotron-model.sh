#!/bin/bash
# Download the NVIDIA Nemotron 3.5 streaming ASR checkpoint used by the
# `nemotron` local STT provider. Run this after cloning, BEFORE building the
# nemotron Docker image (build-nemotron.sh expects nemotron-models/ to exist).
#
# The checkpoint (~2.4 GB, F32) is git-ignored and never committed. It is baked
# into the image at build time so the runtime can run fully offline
# (HF_HUB_OFFLINE=1).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="nvidia/nemotron-3.5-asr-streaming-0.6b"
TARGET_DIR="$SCRIPT_DIR/nemotron-models/nemotron-3.5-asr-streaming-0.6b"

echo "========================================"
echo "Downloading Nemotron STT model"
echo "  model:  $MODEL_ID"
echo "  target: $TARGET_DIR"
echo "========================================"
echo ""

# Pick a concrete Python interpreter (NOT the huggingface-cli PATH shim, which
# may be a pyenv shim pointing at an uninstalled version). Override with PYTHON=.
pick_python() {
    if [ -n "$PYTHON" ]; then echo "$PYTHON"; return; fi
    for c in "$SCRIPT_DIR/.venv/bin/python" python3 python; do
        if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return; fi
    done
}
PY="$(pick_python)"
if [ -z "$PY" ]; then
    echo "[ERROR] No Python interpreter found. Set PYTHON=/path/to/python and re-run."
    exit 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

# huggingface_hub is required (we call snapshot_download programmatically so we
# do not depend on the huggingface-cli / hf shim resolving a pyenv version).
# If it is missing, install it into the picked interpreter so that a plain
# ./download-nemotron-model.sh works everywhere (dev machines and CI). The
# --break-system-packages retry covers PEP 668 externally-managed Pythons
# (e.g. Ubuntu 24.04+ system python3).
if ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
    echo "huggingface_hub is not installed for $PY — installing it..."
    "$PY" -m pip install -U huggingface_hub 2>/dev/null \
        || "$PY" -m pip install -U --break-system-packages huggingface_hub \
        || {
            echo ""
            echo "[ERROR] Could not install huggingface_hub for this interpreter."
            echo "        Install it manually:   $PY -m pip install -U huggingface_hub"
            echo "        Or point PYTHON at an interpreter that already has it, e.g.:"
            echo "        PYTHON=/path/to/python $0"
            exit 1
        }
fi

mkdir -p "$TARGET_DIR"

echo "→ Downloading via huggingface_hub.snapshot_download (only the .nemo) ..."
# We fetch ONLY the *.nemo checkpoint. It is self-contained - it bundles the
# tokenizer, vocab, model_config and weights - so NeMo's restore_from(<.nemo>)
# loads it fully offline in the image. The HF repo also ships an HF-transformers
# copy (model.safetensors + *.json configs) and docs (~2.5 GB) that our NeMo
# code path never uses, so we skip them to keep the image small. If the model is
# gated, authenticate first:  $PY -m huggingface_hub login  (or export HF_TOKEN=...)
MODEL_ID="$MODEL_ID" TARGET_DIR="$TARGET_DIR" "$PY" - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

target = os.environ["TARGET_DIR"]
path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=target,
    allow_patterns=["*.nemo"],  # only the self-contained NeMo checkpoint
)
print("snapshot downloaded to:", path)

# Idempotently slim: if this dir previously held a full snapshot (from an older
# version of this script), drop everything that is not the .nemo checkpoint.
# Keep `.cache/` (small HF download metadata) so re-runs skip the 2.3 GB refetch.
if not any(f.endswith(".nemo") for f in os.listdir(target)):
    raise SystemExit("ERROR: no .nemo checkpoint was downloaded; refusing to prune.")
removed = 0
for name in os.listdir(target):
    if name == ".cache" or name.endswith(".nemo"):
        continue
    p = os.path.join(target, name)
    try:
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
        removed += 1
    except OSError as e:
        print(f"  (could not remove {name}: {e})")
if removed:
    print(f"Pruned {removed} unused non-.nemo item(s) to slim the model dir.")
PY

echo ""
echo "========================================"
echo "✓ Nemotron model downloaded"
echo "========================================"
echo "Model available in:"
echo "  - $TARGET_DIR"
echo ""
