# Dev-container — CUDA + UV (NVIDIA only)

Jeden obraz Docker = całe środowisko (lekcje 00–07, baseline, trening CNN, ONNX, PLGrid).  
**Kod projektu nigdy nie jest kopiowany do obrazu** — tylko bind-mount `.:/workspace`.

## Architektura

```
┌──────────────── WSL (host) ─────────────────┐
│  Docker Engine                              │
│  NVIDIA Container Toolkit                   │
│  sterownik NVIDIA (Windows → WSL2)          │
│                                             │
│  ./scripts/drun run python …  ──────────┐   │
└────────────────────────────────────────│───┘
                                         │ mount
┌────────────────────────────────────────▼───┐
│  kontener: nvidia/cuda:12.9.2-cudnn-devel  │
│  apt: python3.12, build-essential, …       │
│  uv → /workspace/.venv (na dysku WSL)      │
│  PyTorch cu129 + Sionna + Jupyter          │
└────────────────────────────────────────────┘
```

## Co jest gdzie

| Element | Gdzie | Dlaczego |
|---------|-------|----------|
| Docker, nvidia-container-toolkit | **WSL (host)** | jedyny sposób na GPU w kontenerze |
| Sterownik NVIDIA | **Windows** | WSL2 nie ma własnego sterownika GPU |
| Python, uv, Sionna, PyTorch | **kontener → `.venv`** | powtarzalne, identyczne na PLGrid |
| Kod, notebooki, eksperymenty | **mount `/workspace`** | edytujesz normalnie w Cursorze |
| Cache UV | **volume `uv-cache`** | szybsze `uv sync` |

## Setup WSL (raz)

```bash
sudo ./scripts/setup-wsl-host.sh
# wyloguj się i zaloguj (grupa docker), potem:
docker compose build
./scripts/drun sync
./scripts/drun verify
```

## Codzienne użycie — `uv run` przez Docker

Wrapper `./scripts/drun` odpala `uv` w kontenerze z GPU:

```bash
./scripts/drun sync                                    # uv sync
./scripts/drun run python docker/verify_env.py         # uv run …
./scripts/drun run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
./scripts/drun run python neural-receiver/src/utils/setup.py
```

### Alias — docker „niewidoczny”

Dodaj do `~/.bashrc`:

```bash
alias uv='/home/womackow/CODE_SPACE/agh/sem2/magisterka/scripts/drun'
```

Potem:

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Wygląda jak lokalne `uv`, ale wszystko leci w kontenerze z CUDA.

## Cursor / notebooki

Cursor **nie ma** natywnego Dev Containers — patrz **[CURSOR.md](CURSOR.md)**.

Skrót:

```bash
# terminal 1 — zostaw włączone
./scripts/drun jupyter

# Cursor: otwórz notebook → Select Kernel → Existing Jupyter Server
# URL http://127.0.0.1:8888  →  kernel magisterka-sionna
```

Folder `.devcontainer/` działa w **VS Code** (+ rozszerzenie Dev Containers) lub z opcjonalnym rozszerzeniem third-party w Cursorze.

## Notebooki — Remote Jupyter (Cursor) / przeglądarka

`.venv` na WSL ma symlinki do ścieżek **w kontenerze** — Cursor na hoście nie uruchomi tego Pythona bez Jupytera w tle.

### Cursor (rekomendowane)

1. `./scripts/drun jupyter` w osobnym terminalu
2. Notebook w Cursorze → kernel z serwera zdalnego **magisterka-sionna**
3. Szczegóły: [CURSOR.md](CURSOR.md)

### Przeglądarka

```bash
./scripts/drun jupyter
# → http://127.0.0.1:8888/
```

## CUDA — priorytet nr 1

- Obraz bazowy: `nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04` (oficjalny NVIDIA)
- PyTorch: `torch 2.13+cu129` z indeksu w `pyproject.toml`
- GPU w compose: overlay `docker-compose.gpu.yml` (auto przez `./scripts/drun`)
- `sync` / `lock` — bez GPU (działa zawsze)
- `verify` i trening — wymagają GPU (`setup-wsl-host.sh`)

```bash
./scripts/drun verify
# PyTorch: True, GPU: NVIDIA …
```

## PLGrid (Singularity / Apptainer)

Ten sam obraz — zbuduj raz lokalnie, przenieś na klaster:

```bash
# lokalnie
docker compose build
docker save magisterka-sionna:dev | gzip > magisterka-sionna.tar.gz

# na PLGrid (po skopiowaniu archiwum)
singularity build magisterka-sionna.sif docker-archive://magisterka-sionna.tar.gz

# job — mount projektu, GPU
singularity exec --nv \
  -B "$PWD:/workspace" \
  magisterka-sionna.sif \
  bash -lc 'cd /workspace && uv sync && uv run python docker/verify_env.py'
```

Na PLGrid host musi mieć moduł CUDA + `--nv`; reszta jest w obrazie.

## Komendy pomocnicze

| Komenda | Opis |
|---------|------|
| `./scripts/drun sync` | `uv sync` — instalacja deps do `.venv` |
| `./scripts/drun lock` | `uv lock` — aktualizacja lockfile |
| `./scripts/drun verify` | test CUDA + Sionna |
| `./scripts/dsh` | shell w kontenerze (debug) |
| `docker compose build` | przebudowa obrazu (apt, uv binary) |

## Zmiana zależności

1. Edytuj `pyproject.toml`
2. `./scripts/drun lock` (opcjonalnie, dla reproducibility)
3. `./scripts/drun sync`
4. Commit `pyproject.toml` + `uv.lock`
