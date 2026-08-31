#!/usr/bin/env python3
"""Generuje notebooki lekcji w katalogu notebooks/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

KERNEL_NAME = "magisterka-sionna"
KERNEL_DISPLAY = "Magisterka Sionna (CUDA / Docker)"


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": KERNEL_DISPLAY,
            "language": "python",
            "name": KERNEL_NAME,
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "vscode": {
            "kernelspec": {
                "name": KERNEL_NAME,
                "display_name": KERNEL_DISPLAY,
            }
        },
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def save(name: str, cells: list[dict]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


COMMON_IMPORTS = """import sys
from pathlib import Path

# Dodaj src/ do PYTHONPATH
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    raise ImportError(
        "Brak Sionny. Uruchom z katalogu magisterka/: ./scripts/drun sync"
    ) from e

from src.utils.setup import print_environment, get_device

sn.phy.config.seed = 42
device = get_device()
print_environment()
"""


def lesson_00() -> None:
    save(
        "00_setup.ipynb",
        [
            md(
                """# Lekcja 0 — Konfiguracja środowiska

## Cel
- Zainstalować Sionna 2.x (PyTorch) i Jupyter
- Zweryfikować GPU/CUDA
- Zrozumieć **pipeline PHY**, który później rozbudujemy

## Dlaczego zaczynamy od setupu?
Neural Receiver to **zamiennik części klasycznego odbiornika**. Żeby wiedzieć *co* zastępujemy, musisz najpierw zbudować prosty łańcuch:

```
Bits → Mapper → Kanał → Demapper → Bits
```

Dopiero potem dodamy OFDM, fading, LDPC i CNN.

## Środowisko (Docker + uv)

Wszystkie lekcje uruchamiasz w dev-containerze z CUDA:

```bash
# z katalogu magisterka/ (nad neural-receiver/)
./scripts/drun verify          # CUDA + Sionna OK?
./scripts/drun jupyter         # Jupyter Lab w kontenerze → http://127.0.0.1:8888
```

W **Cursor**: *Dev Containers → Reopen in Container* — notebooki używają kernela
`Magisterka Sionna (CUDA / Docker)` automatycznie.

Kernel musi być: **magisterka-sionna** (nie lokalny Python z WSL).
"""
            ),
            code(COMMON_IMPORTS),
            md(
                """## Ćwiczenie
1. Uruchom komórkę powyżej — czy `cuda_available` jest `True`?
2. Kernel w prawym górnym rogu: **Magisterka Sionna (CUDA / Docker)**.
3. Przejdź do **01_qpsk_awgn.ipynb**.
"""
            ),
        ],
    )


def lesson_01() -> None:
    save(
        "01_qpsk_awgn.ipynb",
        [
            md(
                """# Lekcja 1 — QPSK + AWGN

## Cel nauki
| Pojęcie | Co to jest | Rola w pipeline |
|---------|-----------|-----------------|
| **Bits** | 0/1 | Informacja źródłowa |
| **Mapper** | bits → symbole zespolone | Modulacja |
| **Konstelacja** | punkty w płaszczyźnie I/Q | Alfabet modulacji |
| **AWGN** | $y = x + n$ | Najprostszy kanał |
| **Demapper** | symbole → LLR | Soft detection |
| **BER** | odsetek błędnych bitów | Metryka jakości |

## Co zostanie zastąpione przez sieć?
W tej lekcji **nic** — to fundament. Sieć neuronowa w lekcji 7 zastąpi demapper (i wcześniejsze bloki OFDM), ale **LLR → LDPC** zostaje klasyczne.

## Równanie kanału
$$ y = x + n, \\quad n \\sim \\mathcal{CN}(0, \\sigma^2) $$
"""
            ),
            code(COMMON_IMPORTS),
            md("## Krok 1 — Źródło bitów"),
            code(
                """NUM_BITS_PER_SYMBOL = 2  # QPSK = 2 bity/symbol
BATCH_SIZE = 1000
BLOCK_LENGTH = 1024

binary_source = sn.phy.mapping.BinarySource()
bits = binary_source([BATCH_SIZE, BLOCK_LENGTH])

print("Shape bits:", bits.shape)       # [batch, block_length]
print("Typ:", bits.dtype)
print("Przykład (pierwsze 8 bitów):", bits[0, :8].tolist())
"""
            ),
            md("## Krok 2 — Konstelacja i mapper"),
            code(
                """constellation = sn.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
constellation.show()

mapper = sn.phy.mapping.Mapper(constellation=constellation)
# Mapper oczekuje BLOCK_LENGTH bitów podzielnych przez NUM_BITS_PER_SYMBOL
assert BLOCK_LENGTH % NUM_BITS_PER_SYMBOL == 0

x = mapper(bits)
print("Shape x (symbole TX):", x.shape)  # [batch, num_symbols]
print("Przykład symbolu:", x[0, 0])
"""
            ),
            md("## Krok 3 — Kanał AWGN"),
            code(
                """awgn = sn.phy.channel.AWGN()
EBN0_DB = 10.0
no = sn.phy.utils.ebnodb2no(
    ebno_db=EBN0_DB,
    num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
    coderate=1.0,  # brak kodowania
)

y = awgn(x, no)
print("Shape y:", y.shape)
print("Wariancja szumu no:", float(no))

from src.utils.plotting import plot_constellation
plot_constellation(x[:64], y[:64], title=f"QPSK @ Eb/N0={EBN0_DB} dB")
plt.show()
"""
            ),
            md("## Krok 4 — Demapper → LLR → hard bits"),
            code(
                """demapper = sn.phy.mapping.Demapper("app", constellation=constellation)
llr = demapper(y, no)

print("Shape LLR:", llr.shape)  # taki sam jak bits
print("LLR > 0 → bit=1, LLR < 0 → bit=0")
bits_hat = (llr > 0).float()

from src.utils.metrics import ber
print(f"BER @ {EBN0_DB} dB:", ber(bits, bits_hat))
"""
            ),
            md("## Krok 5 — Krzywa BER vs SNR"),
            code(
                """snr_range = np.arange(0, 12, 2)
ber_vals = []

for ebno_db in snr_range:
    no = sn.phy.utils.ebnodb2no(ebno_db, NUM_BITS_PER_SYMBOL, 1.0)
    b = binary_source([500, BLOCK_LENGTH])
    x = mapper(b)
    y = awgn(x, no)
    llr = demapper(y, no)
    ber_vals.append(ber(b, (llr > 0).float()))

from src.utils.plotting import plot_ber_curve
plot_ber_curve(snr_range, np.array(ber_vals), label="QPSK uncoded")
plt.show()
"""
            ),
            md(
                """## Podsumowanie
- **Tensor bits**: `[batch, block_length]` — wartości 0/1
- **Tensor x**: `[batch, num_symbols]` — liczby zespolone
- **LLR** to logit dla każdego bitu — sieć neuronowa w lekcji 7 też produkuje LLR

## Ćwiczenie
Zmień `NUM_BITS_PER_SYMBOL` na 4 (16-QAM). Jak zmienia się BER przy tym samym Eb/N0?

**Następna lekcja:** `02_ofdm_basics.ipynb`
"""
            ),
        ],
    )


def lesson_02() -> None:
    save(
        "02_ofdm_basics.ipynb",
        [
            md(
                """# Lekcja 2 — Podstawy OFDM

## Cel nauki
- **Resource grid** — slot czasowo-częstotliwościowy (jak „obraz” dla CNN)
- **Subcarriers** — równoległe nośne w częstotliwości
- **OFDM symbols** — bloki IFFT/FFT w czasie
- **Piloty (DMRS)** — znane symbole do estymacji kanału

## Resource grid
```
       frequency →
time  [ D  D  P  D  D  P ]
      [ D  D  D  D  D  D ]
```
D = data, P = pilot

## Co zastąpi sieć?
CNN w lekcji 7 bierze **cały odebrany grid** `[Real, Imag, Time, Freq]` jako wejście. Musisz rozumieć ten tensor zanim zbudujesz sieć.

## Pipeline
```
Bits → QAM → ResourceGridMapper → OFDMModulator → Kanał → OFDMDemodulator → Grid RX
```
"""
            ),
            code(COMMON_IMPORTS),
            md("## Krok 1 — Definicja Resource Grid"),
            code(
                """from sionna.phy.ofdm import ResourceGrid

rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=64,
    subcarrier_spacing=30e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=6,
    num_guard_carriers=[5, 6],
    dc_null=True,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)
rg.show()
print("Liczba RE danych:", rg.num_data_symbols)
print("Liczba RE pilotów:", rg.num_pilot_symbols)
"""
            ),
            md("## Krok 2 — Mapowanie symboli QAM na grid"),
            code(
                """from sionna.phy.ofdm import ResourceGridMapper
from sionna.phy.mapping import Mapper, Constellation

NUM_BITS_PER_SYMBOL = 4
constellation = Constellation("qam", NUM_BITS_PER_SYMBOL)
mapper = Mapper(constellation=constellation)
rg_mapper = ResourceGridMapper(rg)

batch_size = 4
num_data_symbols = rg.num_data_symbols
num_bits = num_data_symbols * NUM_BITS_PER_SYMBOL

bits = sn.phy.mapping.BinarySource()([batch_size, num_bits])
symbols = mapper(bits)
print("Shape symbols (Mapper):", symbols.shape)  # [batch, num_data_RE]

# ResourceGridMapper oczekuje [batch, num_tx, num_streams, num_data_RE]
symbols = symbols.reshape(batch_size, 1, 1, num_data_symbols)
x_rg = rg_mapper(symbols)

print("Shape grid TX:", x_rg.shape)
# [batch, num_tx, num_streams, num_ofdm_symbols, fft_size]
print("Przykład |x| na RE danych:", torch.abs(x_rg[0, 0, 0]).mean().item())
"""
            ),
            md("## Krok 3 — Modulacja / demodulacja OFDM"),
            code(
                """from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator

modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, 0, rg.cyclic_prefix_length)

x_time = modulator(x_rg)
print("Shape sygnał czasowy:", x_time.shape)

# Idealny kanał (brak zniekształceń) — tylko demodulacja
y_rg = demodulator(x_time)
print("Shape grid po demodulacji:", y_rg.shape)
"""
            ),
            md("## Krok 4 — Wizualizacja grida"),
            code(
                """# Magnitude pierwszego batcha, pierwszy stream
mag = torch.abs(x_rg[0, 0, 0]).cpu().numpy()
plt.figure(figsize=(10, 4))
plt.imshow(mag, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="|x|")
plt.xlabel("Subcarrier")
plt.ylabel("OFDM symbol")
plt.title("Resource Grid — magnitude (TX)")
plt.show()
"""
            ),
            md(
                """## Podsumowanie tensorów
| Etap | Shape | Typ |
|------|-------|-----|
| bits | `[B, num_bits]` | float 0/1 |
| symbols | `[B, 1, 1, num_data_RE]` | complex |
| grid | `[B, 1, 1, 14, 64]` | complex |
| time | `[B, 1, 1, num_samples]` | complex |

## Ćwiczenie
Porównaj `rg.num_data_symbols` z liczbą niezerowych elementów w gridzie.

**Następna lekcja:** `03_fading_channel.ipynb`
"""
            ),
        ],
    )


def lesson_03() -> None:
    save(
        "03_fading_channel.ipynb",
        [
            md(
                """# Lekcja 3 — Kanał z fadingiem

## Cel nauki
Progresja złożoności kanału (dokładnie jak w planie nauki):

1. **AWGN**: $y = x + n$
2. **Flat fading**: $y = h x + n$ (jeden współczynnik $h$)
3. **Frequency selective**: $y = H(f) \\cdot x + n$ (różne $h$ na subcarrierach)
4. **TDL/CDL** — modele 3GPP (realistyczny 5G)

## Pojęcia
- **Channel response** $H$ — jak kanał zniekształca sygnał
- **Multipath** — wiele ścieżek propagacji → selectivity w częstotliwości
- **Doppler** — ruch → zmiana $h$ w czasie

## Co zastąpi sieć?
Klasyczny receiver robi **Channel Estimation** (z pilotów) + **Equalization**.
Sieć neuronowa uczy się robić to implicitnie z całego grida.
"""
            ),
            code(COMMON_IMPORTS),
            md("## Etap 1 — AWGN (przypomnienie)"),
            code(
                """NUM_BITS = 2
constellation = sn.phy.mapping.Constellation("qam", NUM_BITS)
mapper = sn.phy.mapping.Mapper(constellation=constellation)
demapper = sn.phy.mapping.Demapper("app", constellation=constellation)
awgn = sn.phy.channel.AWGN()

bits = sn.phy.mapping.BinarySource()([100, 512])
x = mapper(bits)
no = sn.phy.utils.ebnodb2no(15.0, NUM_BITS, 1.0)
y = awgn(x, no)
llr = demapper(y, no)
print("BER AWGN:", float((bits != (llr > 0)).float().mean()))
"""
            ),
            md("## Etap 2 — Flat fading (1 współczynnik h)"),
            code(
                """from sionna.phy.channel import GenerateFlatFadingChannel

# Macierz kanału: [batch, num_rx_ant, num_tx_ant]
flat_gen = GenerateFlatFadingChannel(num_tx_ant=1, num_rx_ant=1)
h = flat_gen(batch_size=x.shape[0])  # h ~ CN(0,1)
print("Shape h:", h.shape)

# y = h*x + n  (flat fading — to samo h dla wszystkich symboli w batchu)
h_scalar = h[:, 0, 0].unsqueeze(-1)  # [batch, 1]
y_fade = h_scalar * x + awgn(torch.zeros_like(x), no)
print("Średnie |y| vs |x|:", torch.abs(y_fade).mean().item(), torch.abs(x).mean().item())
"""
            ),
            md("## Etap 3 — Frequency selective (OFDM + TDL)"),
            code(
                """from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL

rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=64,
    subcarrier_spacing=30e3,
    cyclic_prefix_length=6,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)
rg_mapper = ResourceGridMapper(rg)
modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, 0, rg.cyclic_prefix_length)

# TDL-A — profil delay spread (3GPP)
tdl = TDL(model="A", delay_spread=30e-9, carrier_frequency=3.5e9, min_speed=0.0)

NUM_BPS = 2
mapper_q = sn.phy.mapping.Mapper(constellation=sn.phy.mapping.Constellation("qam", NUM_BPS))
bits_ofdm = sn.phy.mapping.BinarySource()([8, rg.num_data_symbols * NUM_BPS])
sym = mapper_q(bits_ofdm).reshape(8, 1, 1, rg.num_data_symbols)
x_grid = rg_mapper(sym)
x_time = modulator(x_grid)

# OFDMChannel łączy TDL + AWGN w domenie częstotliwości
from sionna.phy.channel import OFDMChannel

ofdm_channel = OFDMChannel(tdl, rg, add_awgn=True, normalize_channel=True, return_channel=True)
no_t = sn.phy.utils.ebnodb2no(10.0, NUM_BPS, 1.0)
y_grid, h_freq = ofdm_channel(x_grid, no_t)

print("Shape y_grid (RX):", y_grid.shape)
print("Shape h_freq (idealna wiedza o kanale):", h_freq.shape)
"""
            ),
            md(
                """## Podsumowanie
- **Flat fading** — ten sam $h$ dla wszystkich subcarrierów
- **Selective fading** — $h$ zależy od częstotliwości → potrzebne piloty
- **TDL/CDL** — parametryzowane modele do treningu i testu generalizacji

## Ćwiczenie
Narysuj `|h_freq|` dla pierwszego batcha jako heatmapę (subcarrier × OFDM symbol).

**Następna lekcja:** `04_mimo_basics.ipynb`
"""
            ),
        ],
    )


def lesson_04() -> None:
    save(
        "04_mimo_basics.ipynb",
        [
            md(
                """# Lekcja 4 — Podstawy MIMO

## Cel nauki
- **num_tx / num_rx** — anteny nadawcze i odbiorcze
- **Streams** — niezależne strumienie danych
- **MIMO w OFDM** — grid rozszerzony o wymiar anten/streamów

## Równanie (uproszczone)
$$ \\mathbf{y} = \\mathbf{H} \\mathbf{x} + \\mathbf{n} $$

gdzie $\\mathbf{H}$ to macierz kanału MIMO.

## Co zastąpi sieć?
Przy MIMO klasyczny receiver robi jeszcze **detekcję MIMO** (np. LMMSE).
Sieć może uczyć się separacji strumieni z tensora `[Real, Imag, Antennas, Time, Freq]`.
"""
            ),
            code(COMMON_IMPORTS),
            md("## Resource grid z wieloma streamami"),
            code(
                """from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

NUM_STREAMS = 2
rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=64,
    subcarrier_spacing=30e3,
    num_tx=1,
    num_streams_per_tx=NUM_STREAMS,
    cyclic_prefix_length=6,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)
rg.show()

rg_mapper = ResourceGridMapper(rg)
NUM_BPS = 2
mapper = sn.phy.mapping.Mapper(
    constellation=sn.phy.mapping.Constellation("qam", NUM_BPS)
)

batch = 4
bits = sn.phy.mapping.BinarySource()(
    [batch, NUM_STREAMS * rg.num_data_symbols * NUM_BPS]
)
# Sionna oczekuje [batch, num_tx, num_streams, num_data_symbols]
sym = mapper(bits).reshape(batch, 1, NUM_STREAMS, rg.num_data_symbols)
x_rg = rg_mapper(sym)
print("Grid MIMO shape:", x_rg.shape)
"""
            ),
            md("## Kanał MIMO (Rayleigh block fading)"),
            code(
                """from sionna.phy.channel import RayleighBlockFading, OFDMChannel

# 1 gNB (2 anteny RX), 1 UE (2 anteny TX / streamy)
rayleigh = RayleighBlockFading(
    num_rx=1,
    num_rx_ant=2,
    num_tx=1,
    num_tx_ant=NUM_STREAMS,
)
mimo_channel = OFDMChannel(
    channel_model=rayleigh,
    resource_grid=rg,
    add_awgn=True,
    normalize_channel=True,
)

no_mimo = sn.phy.utils.ebnodb2no(10.0, NUM_BPS, 1.0)
y_mimo = mimo_channel(x_rg, no_mimo)
print("RX grid shape:", y_mimo.shape)
"""
            ),
            md(
                """## Podsumowanie
MIMO zwiększa wymiarowość tensorów — to ważne przy projektowaniu wejścia CNN.

## Ćwiczenie
Porównaj liczbę pilotów w gridzie dla 1 vs 2 streamów (`rg.num_pilot_symbols`).

**Następna lekcja:** `05_5g_pusch.ipynb`
"""
            ),
        ],
    )


def lesson_05() -> None:
    save(
        "05_5g_pusch.ipynb",
        [
            md(
                """# Lekcja 5 — 5G NR PUSCH (komponenty Sionna)

## Cel nauki
Nie budujemy PUSCH od zera — używamy gotowych bloków Sionna:
- **LDPC 5G** encoder/decoder
- **ResourceGrid** z parametrami zbliżonymi do NR
- **OFDMChannel** z modelem TDL/CDL

## PUSCH w skrócie
Physical Uplink Shared Channel — kanał danych użytkownika w górę (UE → gNB).

## Pipeline docelowy pracy magisterskiej
```
Bits → LDPC → QAM → Grid → OFDM → 3GPP Channel → Receiver → LLR → LDPC decode
```

## Co zostaje klasyczne po wprowadzeniu CNN?
**LDPC decoder** — sieć produkuje LLR, dekoder zostaje standardowy (modularność!).
"""
            ),
            code(COMMON_IMPORTS),
            md("## LDPC 5G — encoder i decoder"),
            code(
                """k = 512   # informacyjne bity
n = 1024  # długość kodoword
coderate = k / n

encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sn.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

bits = sn.phy.mapping.BinarySource()([16, k])
codewords = encoder(bits)
print("bits:", bits.shape, "codewords:", codewords.shape)
"""
            ),
            md("## Transmisja zakodowana przez AWGN"),
            code(
                """NUM_BPS = 2
constellation = sn.phy.mapping.Constellation("qam", NUM_BPS)
mapper = sn.phy.mapping.Mapper(constellation=constellation)
demapper = sn.phy.mapping.Demapper("app", constellation=constellation)
awgn = sn.phy.channel.AWGN()

x = mapper(codewords)
ebno_db = 3.0
no = sn.phy.utils.ebnodb2no(ebno_db, NUM_BPS, coderate)
y = awgn(x, no)
llr = demapper(y, no)
bits_hat = decoder(llr)

from src.utils.metrics import ber, bler
print(f"BER @ {ebno_db} dB:", ber(codewords, bits_hat))
print(f"BLER @ {ebno_db} dB:", bler(bits, bits_hat))
"""
            ),
            md(
                """## Uwaga
Pełny PUSCH z OFDM + TDL + klasycznym receiverem budujemy w **06_classical_receiver.ipynb**.

## Ćwiczenie
Porównaj BLER z i bez LDPC przy Eb/N0 = 0…6 dB.

**Następna lekcja:** `06_classical_receiver.ipynb`
"""
            ),
        ],
    )


def lesson_06() -> None:
    save(
        "06_classical_receiver.ipynb",
        [
            md(
                """# Lekcja 6 — Klasyczny odbiornik (baseline)

## Cel
Zbudować **Classical Baseline** — krzywa **BLER vs SNR** do porównania z CNN.

## Łańcuch odbiornika
```
RX Grid → LS Channel Estimation → LMMSE Equalization → Demapper → LLR → LDPC
```

## Dlaczego to kluczowe?
Bez baseline wynik „BLER = 0.03” nic nie znaczy. Potrzebujesz:
```
SNR = 5 dB → Classical BLER = 0.08, Neural BLER = 0.04  ✓
```

## Co dokładnie zastąpi sieć neuronowa?
| Blok klasyczny | Rola | Zastąpiony przez CNN? |
|----------------|------|----------------------|
| Channel Estimation (LS) | $\\hat{H}$ z pilotów | **TAK** |
| LMMSE Equalizer | usuwa ISI/fading | **TAK** |
| Demapper | symbole → LLR | **TAK** |
| LDPC Decoder | LLR → bits | **NIE** |
"""
            ),
            code(COMMON_IMPORTS),
            md("## End-to-end classical link (OFDM + TDL + LDPC)"),
            code(
                """from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    LSChannelEstimator,
    LMMSEEqualizer,
)
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.mimo import StreamManagement

# Parametry linku
NUM_BPS = 2
K = 256
N = 512
coderate = K / N

encoder = sn.phy.fec.ldpc.LDPC5GEncoder(K, N)
decoder = sn.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)
constellation = sn.phy.mapping.Constellation("qam", NUM_BPS)
mapper = sn.phy.mapping.Mapper(constellation=constellation)
demapper = sn.phy.mapping.Demapper("app", constellation=constellation)

rg = ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=64,
    subcarrier_spacing=30e3,
    cyclic_prefix_length=6,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[2, 11],
)
rg_mapper = ResourceGridMapper(rg)

tdl = TDL(model="A", delay_spread=30e-9, carrier_frequency=3.5e9)
ofdm_ch = OFDMChannel(tdl, rg, add_awgn=True, normalize_channel=True, return_channel=True)

stream_mgmt = StreamManagement([[0]], 1)
ch_est = LSChannelEstimator(rg, interpolation_type="nn")
equalizer = LMMSEEqualizer(rg, stream_mgmt)

print("Gotowe bloki klasycznego odbiornika.")
"""
            ),
            code(
                """def classical_transmit_receive(batch_size: int, ebno_db: float):
    no = sn.phy.utils.ebnodb2no(ebno_db, NUM_BPS, coderate)
    bits = sn.phy.mapping.BinarySource()([batch_size, K])
    cw = encoder(bits)
    x_sym = mapper(cw)
    x_grid = rg_mapper(x_sym.reshape(batch_size, 1, 1, -1))

    # Kanał OFDM (freq domain) — zwraca odebrany resource grid
    y_grid, _ = ofdm_ch(x_grid, no)

    # Klasyczny odbiornik: estymacja → equalizacja → demapping → LDPC
    h_hat, err_var = ch_est(y_grid, no)
    x_hat, _ = equalizer(y_grid, h_hat, err_var, no)
    llr = demapper(x_hat.reshape(batch_size, -1), no)
    bits_hat = decoder(llr)
    return bits, bits_hat

# Szybki test
b, bh = classical_transmit_receive(32, ebno_db=5.0)
from src.utils.metrics import bler
print("BLER @ 5 dB:", bler(b, bh))
"""
            ),
            md("## Symulacja BLER vs SNR"),
            code(
                """snr_db = np.arange(0, 11, 2)
bler_vals = []

for ebno in snr_db:
    b, bh = classical_transmit_receive(64, ebno)
    bler_vals.append(bler(b, bh))

plt.figure(figsize=(7, 5))
plt.semilogy(snr_db, bler_vals, "o-", label="Classical (LS + LMMSE + LDPC)")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BLER")
plt.grid(True, which="both")
plt.legend()
plt.title("Baseline — zapisz tę krzywą do porównania z CNN")
plt.show()
"""
            ),
            md(
                """## Podsumowanie
To jest Twój **Experiment 1 — Classical baseline**.

## Ćwiczenie
Zapisz wyniki `snr_db` i `bler_vals` do pliku `experiments/classical_bler.npy`.

**Następna lekcja:** `07_neural_receiver.ipynb`
"""
            ),
        ],
    )


def lesson_07() -> None:
    save(
        "07_neural_receiver.ipynb",
        [
            md(
                """# Lekcja 7 — Neural Receiver (CNN → LLR → LDPC)

## Cel
Zastąpić trzy bloki klasyczne **jednym CNN**:
```
RX Grid [B, 2, T, F] → CNN → LLR → LDPC → Bits
```

## Architektura (uproszczona)
```
Conv2D(2→32) → ReLU → Conv2D → ReLU → Conv2D → GAP → Linear → LLR
```

## Loss
Nie BLER (niedifferentiowalny), tylko **Binary Cross-Entropy** na LLR:
$$ \\mathcal{L} = \\text{BCE}(\\text{bits}, \\text{LLR}) $$

## Co NIE jest neuronowe
- **Transmitter** (LDPC, mapper, grid) — generuje dane treningowe
- **LDPC decoder** — zostaje klasyczny
"""
            ),
            code(COMMON_IMPORTS),
            code(
                """import torch.nn as nn
import torch.nn.functional as F
from src.models.neural_receiver import NeuralReceiverCNN

# Przykładowe wymiary (dopasuj do swojego resource grid)
BATCH = 8
T, F = 14, 64
NUM_BITS = 512

model = NeuralReceiverCNN(num_bits=NUM_BITS, hidden_channels=32).to(device)
y_fake = torch.randn(BATCH, 2, T, F, device=device)
llr = model(y_fake)
print("LLR shape:", llr.shape)
"""
            ),
            md("## Trening (szkielet pętli)"),
            code(
                """optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# W prawdziwym treningu: generujesz (y_grid, bits) z pipeline Sionna
bits_train = torch.randint(0, 2, (BATCH, NUM_BITS), device=device).float()
llr_out = model(y_fake)

loss = F.binary_cross_entropy_with_logits(llr_out, bits_train)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print("Loss po 1 kroku:", loss.item())
"""
            ),
            md("## Porównanie z baseline"),
            code(
                """# Po wytrenowaniu: ta sama pętla SNR co w lekcji 06
# neural_bler vs classical_bler na jednym wykresie

plt.figure(figsize=(8, 5))
plt.semilogy([0, 2, 4, 6, 8, 10], [0.5, 0.3, 0.15, 0.05, 0.02, 0.01], "--", label="Classical (przykład)")
plt.semilogy([0, 2, 4, 6, 8, 10], [0.45, 0.25, 0.10, 0.03, 0.01, 0.005], "-", label="CNN (docelowo Twój wynik)")
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("BLER")
plt.legend()
plt.title("Classical vs Neural — cel pracy magisterskiej")
plt.grid(True, which="both")
plt.show()
"""
            ),
            md(
                """## Podsumowanie całej ścieżki nauki

| Lekcja | Zrozumiałeś | Zastąpione przez NN |
|--------|-------------|---------------------|
| 01 QPSK+AWGN | bity, LLR, BER | — |
| 02 OFDM | resource grid | wejście CNN |
| 03 Fading | H(f), piloty | estymacja H |
| 04 MIMO | wymiary tensora | detekcja MIMO |
| 05 PUSCH | LDPC 5G | — (decoder zostaje) |
| 06 Classical | LS + LMMSE + demap | **cały ten łańcuch** |
| 07 Neural | CNN → LLR | trening + ewaluacja |

## Kolejne kroki (poza lekcjami)
1. Pełny trening CNN na OFDM+TDL
2. Generalization (train CDL-A, test CDL-B/C)
3. Latency / FLOPs / deployment (ONNX, TensorRT)

Gratulacje — masz mapę od zera do Neural Receiver!
"""
            ),
        ],
    )


def main() -> None:
    lesson_00()
    lesson_01()
    lesson_02()
    lesson_03()
    lesson_04()
    lesson_05()
    lesson_06()
    lesson_07()


if __name__ == "__main__":
    main()
