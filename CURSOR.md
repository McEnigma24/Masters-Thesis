# Notebooki w Cursorze (bez Dev Containers)

Cursor **nie ma** wbudowanego „Reopen in Container” jak VS Code + rozszerzenie Microsoft Dev Containers.  
W marketplace widać głównie rozszerzenia third-party — można je testować, ale **nie są potrzebne** do naszego setupu.

## Jak to działa u nas

```
Terminal WSL                    Kontener Docker
─────────────                   ─────────────────
./scripts/drun jupyter  ──────► Jupyter Lab :8888
                                kernel: magisterka-sionna
                                Python + CUDA + Sionna
         ▲
         │  protokół Jupyter (HTTP)
         │
    Cursor (notebook .ipynb)
```

Cursor **nie uruchamia** Pythona lokalnie z WSL — łączy się z serwerem Jupyter **w kontenerze**.  
Komórki wykonują się tam, gdzie masz GPU.

## Krok po kroku

### 1. Uruchom serwer (osobny terminal, zostaw włączony)

```bash
cd ~/CODE_SPACE/agh/sem2/magisterka
./scripts/drun jupyter
```

Powinno wstać bez błędu: `http://127.0.0.1:8888/` (bez tokena).

### 2. Otwórz notebook w Cursorze

Np. `neural-receiver/notebooks/00_setup.ipynb`

### 3. Wybierz kernel

**Select Kernel** (prawy górny róg) → jedna z opcji:

- **Existing Jupyter Server** → URL: `http://127.0.0.1:8888` (bez tokena)  
  potem kernel: **Magisterka Sionna (CUDA / Docker)**

albo (jeśli Cursor sam połączy się z `.vscode/settings.json`):

- kernel **magisterka-sionna** z listy zdalnych

### 4. Run cell

W outputcie powinno być m.in. `cuda_available: True` (po `./scripts/drun verify`).

## Ustawienia workspace (już są)

Plik `.vscode/settings.json`:

```json
"jupyter.jupyterServerType": "remote",
"jupyter.remote.jupyterServerUrl": "http://127.0.0.1:8888"
```

## Alternatywy

| Sposób | Kiedy |
|--------|--------|
| **`./scripts/drun jupyter` + Cursor** | Codzienna praca z notebookami (rekomendowane) |
| **Przeglądarka** | Ten sam URL co wyżej, bez Cursora |
| **`./scripts/drun run python …`** | Skrypty, nie notebooki |
| **Rozszerzenie Dev Container** | Opcjonalnie, third-party — nie wymagane |

## Czego nie robić

- Nie wybieraj lokalnego **Python 3.x (WSL)** — to nie ma Sionny ani CUDA z kontenera.
- Nie licz na `${workspaceFolder}/.venv/bin/python` poza kontenerem — symlinki wskazują ścieżki **wewnątrz** obrazu Docker.

## Dev Containers (opcjonalnie)

Folder `.devcontainer/` zostaje pod **VS Code** lub jeśli zainstalujesz kompatybilne rozszerzenie (np. „Open Remote - Devcontainer”).  
W samym Cursorze bez rozszerzenia — używaj Remote Jupyter jak wyżej.
