# Neural Receiver — lekcje PHY (Sionna 2.x)

Ścieżka nauki od **QPSK + AWGN** do **CNN Neural Receiver**, krok po kroku.  
Każda lekcja buduje na poprzedniej i wyjaśnia, **które bloki klasycznego odbiornika zostaną zastąpione siecią neuronową**.

## Struktura

```
neural-receiver/
├── notebooks/          # 8 lekcji (00–07)
├── src/
│   ├── utils/          # setup, metryki, wykresy
│   ├── channels/       # (rozszerzenia na później)
│   ├── receivers/      # (rozszerzenia na później)
│   └── models/         # NeuralReceiverCNN
├── experiments/        # wyniki BLER, checkpointy
├── scripts/
└── requirements.txt
```

## Kolejność lekcji

| # | Notebook | Cel | Co zrozumiesz | Co zastąpi NN |
|---|----------|-----|---------------|---------------|
| 0 | `00_setup.ipynb` | Środowisko | Sionna, PyTorch, GPU | — |
| 1 | `01_qpsk_awgn.ipynb` | Fundament | bits, mapper, AWGN, LLR, BER | — |
| 2 | `02_ofdm_basics.ipynb` | OFDM | resource grid, piloty, IFFT/FFT | **wejście CNN** |
| 3 | `03_fading_channel.ipynb` | Kanał | AWGN → flat → TDL | estymacja H |
| 4 | `04_mimo_basics.ipynb` | MIMO | anteny, streamy, wymiary | detekcja MIMO |
| 5 | `05_5g_pusch.ipynb` | 5G NR | LDPC 5G, komponenty Sionna | LDPC **zostaje** |
| 6 | `06_classical_receiver.ipynb` | Baseline | LS + LMMSE + demap + LDPC | **cały ten łańcuch** |
| 7 | `07_neural_receiver.ipynb` | CNN | trening BCE, BLER vs classical | CNN → LLR |

## Instalacja

### Rekomendowane: dev-container (CUDA + uv)

Całe środowisko jest w Dockerze — na WSL instalujesz tylko Docker + NVIDIA Container Toolkit.

```bash
# z katalogu magisterka/ (nad neural-receiver/)
sudo ./scripts/setup-wsl-host.sh
docker compose build
./scripts/drun sync
./scripts/drun verify
./scripts/drun run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Szczegóły: [DOCKER.md](../DOCKER.md).

### Lokalny venv (bez Dockera)

Tylko jeśli masz Python 3.11+ i CUDA lokalnie — patrz `pyproject.toml` w katalogu głównym.

Weryfikacja (w kontenerze):

```bash
./scripts/drun run python -c "from src.utils.setup import print_environment; print_environment()"
```

## Filozofia nauki

1. **Nie zaczynaj od CNN** — najpierw musisz wiedzieć, co sieć ma robić.
2. **Każda linia kodu ma sens** — notebooki pokazują `shape` tensorów na każdym etapie.
3. **Baseline jest obowiązkowy** — lekcja 6 daje krzywą BLER vs SNR do porównania.
4. **Modularność** — sieć produkuje LLR; LDPC decoder zostaje klasyczny (jak w literaturze NVIDIA/ETH).

## Mapa pipeline (docelowa praca magisterska)

```
                    Transmitter (klasyczny)
Bits → LDPC → QAM → Resource Grid → OFDM → 3GPP Channel
                                                    │
                                                    ▼
              ┌─────────────────────────────────────────────┐
              │  Classical          │  Neural (CNN)         │
              │  LS Estimation      │                       │
              │  LMMSE Equalizer    │  Conv2D na grid RX     │
              │  Demapper           │                       │
              └──────────┬──────────┴──────────┬────────────┘
                         │                     │
                         └──────── LLR ────────┘
                                    │
                                    ▼
                              LDPC Decoder
                                    │
                                    ▼
                                  Bits
```

## Regeneracja notebooków

Jeśli edytujesz generator:

```bash
python scripts/generate_notebooks.py
```

## Oficjalne materiały Sionna

- [Hello World](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Hello_World.html)
- [Tutorial Part 1–4](https://nvlabs.github.io/sionna/phy/tutorials.html)
- [MIMO OFDM over CDL](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/MIMO_OFDM_Transmissions_over_CDL.html)

Te lekcje są **uproszczonym, polskojęzycznym mostem** do tych tutoriali — z naciskiem na to, co jest istotne dla Neural Receiver w pracy magisterskiej.
