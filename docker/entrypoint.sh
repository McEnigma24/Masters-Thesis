#!/usr/bin/env bash
# Montowany z hosta — nie jest bake'owany w obrazie.
set -euo pipefail

cd /workspace

# venv w mountowanym katalogu projektu (widoczny też z WSL, persystuje między runami)
export UV_PROJECT_ENVIRONMENT="/workspace/.venv"
export PATH="/workspace/.venv/bin:${PATH}"
export PYTHONPATH="/workspace/neural-receiver:/workspace${PYTHONPATH:+:${PYTHONPATH}}"

# Pierwszy start: jeśli jest pyproject.toml, a brak venv — sync
if [[ -f pyproject.toml ]] && [[ ! -x .venv/bin/python ]]; then
  echo "[entrypoint] Brak .venv — uruchamiam: uv sync"
  uv sync
fi

# Kernel Jupyter dla notebooków lekcji (venv w kontenerze)
if [[ -x .venv/bin/python ]] && [[ ! -f .venv/share/jupyter/kernels/magisterka-sionna/kernel.json ]]; then
  echo "[entrypoint] Rejestruję kernel: magisterka-sionna"
  .venv/bin/python -m ipykernel install --sys-prefix \
    --name magisterka-sionna \
    --display-name "Magisterka Sionna (CUDA / Docker)"
fi

exec "$@"
