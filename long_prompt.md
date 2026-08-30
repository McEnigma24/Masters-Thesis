23. I najważniejsze: czy to jest za trudne?

Tak. Ale w dobrym sensie.

To nie jest temat:

„mam 3 miesiące i muszę zrobić projekt”.

To jest temat, przy którym możesz mieć:

minimum

Sionna + CNN neural receiver + BLER evaluation.

normalną magisterkę

Sionna + CNN/Transformer + generalization + latency.

bardzo dobrą magisterkę
TensorRT + CUDA + FP16/INT8 + real-time optimization.
absurdalnie dobrą
MU-MIMO + dynamic MCS + hardware-aware architecture search + real/OAI validation.

I to jest właśnie sytuacja, której szukasz: masz bezpieczny minimalny scope, ale bardzo wysoki ceiling.

Co więcej, aktualna literatura pokazuje, że real-time jest właśnie jednym z głównych problemów badawczych: praca NVIDIA/ETH z 2025 r. celowała w <1 ms na A100, a Sionna Research Kit ma już gotową ścieżkę od treningu do TensorRT/OAI.

I jeszcze jedna rzecz jest dla mnie bardzo ważna: 3GPP ma formalny work item dotyczący AI/ML dla NR air interface, a Rel-19 wprowadza kolejne elementy AI/ML do specyfikacji. Czyli nie jest to „AI wsadzone na siłę do 5G”, tylko rzeczywisty kierunek rozwoju RAN.

Ja bym więc tego Neural Receivera na Twoim miejscu absolutnie nie skreślał — wręcz dałbym go teraz na #1, a CSI Prediction + Link Adaptation na #2 jako bezpieczniejszy fallback.

Jeśli chcesz to dalej zawęzić, wybrałbym teraz jedną z trzech wersji:
CNN vs Transformer,
real-time CUDA/TensorRT, albo
universal receiver dla różnych MCS.

Branched from Pomysły na pracę magisterską

Today 5:11 PM
skupmy się na tym



minimum
Sionna + CNN neural receiver + BLER evaluation

normalną magisterkę
Sionna + CNN/Transformer + generalization + latency

bardzo dobrą magisterkę
TensorRT + CUDA + FP16/INT8 + real-time optimization


rozpisz mi tutaj co każdy z kroków oznacza w praktyce

Jasne. Rozbijmy to nie jako abstrakcyjne poziomy, tylko dosłownie: co byś robił w kodzie, jakie dane przepływają przez system i co jest wynikiem każdego etapu.

I od razu jedna ważna rzecz: ja bym tę pracę budował warstwowo. Każdy kolejny etap wykorzystuje poprzedni. Dzięki temu nawet jeśli zatrzymasz się w połowie, masz kompletną pracę.

Cała architektura pracy

Docelowo wyglądałoby to mniej więcej tak:

                    5G NR Simulator
                       Sionna
                         │
                         │ IQ / OFDM grid
                         ▼
              ┌──────────────────────┐
              │   Neural Receiver    │
              │                      │
              │ CNN / Transformer    │
              └──────────┬───────────┘
                         │
                         │ LLR
                         ▼
                     LDPC Decoder
                         │
                         ▼
                      Bits

         ┌────────────────────────────────┐
         │          Evaluation            │
         │                                │
         │ BLER │ BER │ Latency │ Memory  │
         └────────────────────────────────┘

                         │
                         ▼
                   Deployment

                  ONNX / TensorRT
                         │
                  FP32 / FP16 / INT8
                         │
                         ▼
                   GPU Benchmark
ETAP 0 — zanim w ogóle zacznie się neural network

To nie jest jeszcze część „minimum”, ale praktycznie musisz to zrobić.

Budujesz klasyczny 5G NR link-level simulation

W Sionnie tworzysz coś takiego:

Random bits
    │
    ▼
LDPC Encoder
    │
    ▼
QAM Mapper
    │
    ▼
Resource Grid Mapping
    │
    ▼
OFDM Modulation
    │
    ▼
5G Channel
    │
    ▼
AWGN / Fading / Doppler
    │
    ▼
Receiver

Na początku receiver może być całkowicie klasyczny:

Received signal
       │
       ▼
Channel Estimation
       │
       ▼
Equalization
       │
       ▼
Demapper
       │
       ▼
LDPC Decoder
       │
       ▼
Decoded bits
Po co?

Musisz mieć baseline.

Jeżeli później powiesz:

„Mój neural receiver ma BLER 0.03”

to samo w sobie nic nie znaczy.

Musisz porównać:

SNR = 5 dB

Classical Receiver:
BLER = 0.08

Neural Receiver:
BLER = 0.04

Dopiero wtedy masz wynik.

POZIOM 1 — Minimum
Sionna + CNN Neural Receiver + BLER Evaluation

To jest pierwsza pełnoprawna wersja pracy.

Krok 1 — generujesz dane

Nie musisz budować datasetu jak ImageNet.

To jest fajna rzecz w komunikacji.

Dataset może być generowany w locie.

Czyli podczas treningu:

for batch in training:

    bits = generate_random_bits()

    signal = transmitter(bits)

    received_signal = channel(signal)

    prediction = neural_receiver(received_signal)

    loss = ...

    optimizer.step()

Za każdym razem dostajesz nową realizację kanału.

Co dokładnie wchodzi do Neural Receivera?

To zależy od implementacji.

Załóżmy uproszczony resource grid:

              Frequency
      ┌────────────────────────┐
Time  │  D  D  P  D  D  D  P   │
      │  D  D  D  D  D  D  D   │
      │  P  D  D  D  P  D  D   │
      │  D  D  D  D  D  D  D   │
      └────────────────────────┘

D = Data
P = Pilot / DMRS

Receiver dostaje kompleksowe wartości:

$$ Y = H \cdot X + N $$

czyli received resource grid.

Ponieważ neural network zwykle nie operuje natywnie na liczbach zespolonych, robisz np.:

Input shape:

[batch,
 real/imag,
 time,
 frequency]

czyli:

[B, 2, 14, N_subcarriers]

Przykładowo:

Real(Y)
Imag(Y)

stają się dwoma kanałami CNN.

To trochę przypomina obraz:

RGB image:

[R, G, B]

Tutaj:

OFDM grid:

[Real, Imag]
Krok 2 — CNN

Budujesz prosty CNN.

Np.:

Input
  │
  ▼
Conv2D
  │
ReLU
  │
Conv2D
  │
ReLU
  │
Conv2D
  │
  ▼
Output LLR

Możesz zacząć naprawdę mało ambitnie:

Conv 3x3, 32 channels
Conv 3x3, 64 channels
Conv 3x3, 64 channels
Output layer

Najważniejsze jest:

CNN patrzy na lokalne zależności w czasie i częstotliwości.

Na przykład:

        neighboring subcarriers
                │
                ▼

      x  x  x  x  x
      x  x [X] x  x
      x  x  x  x  x

           ▲
           │
     convolution kernel

I model może nauczyć się:

jak kanał zmienia się między subcarrierami,
gdzie są piloty,
jak interpolować channel state,
jak usuwać zakłócenia.
Krok 3 — co jest outputem?

Tutaj ja nie robiłbym outputu = bits.

To byłoby trudniejsze i mniej modularne.

Robiłbym:

Received grid
      │
      ▼
Neural Receiver
      │
      ▼
LLRs
      │
      ▼
Standard 5G LDPC Decoder
      │
      ▼
Bits

LLR:

$$ LLR = \log \frac{P(bit = 1 | y)} {P(bit = 0 | y)} $$

Czyli neural network mówi:

bit 1:
LLR = +4.2

bit 2:
LLR = -0.8

bit 3:
LLR = +7.1

W uproszczeniu:

duży dodatni → raczej 1
duży ujemny → raczej 0
blisko 0 → nie jestem pewien

To jest świetne, bo:

neural network robi soft detection,
LDPC nadal pozostaje klasyczny,
architektura jest kompatybilna z istniejącym receiverem.
Krok 4 — trening

Loss niekoniecznie liczysz bezpośrednio po BLER.

Bo BLER:

block correct / incorrect

nie jest differentiable.

Więc typowo:

transmitted bits
        │
        ▼
     Neural Rx
        │
        ▼
       LLR
        │
        ▼
Binary Cross Entropy Loss

Czyli:

$$ Loss = BCE(bits, LLR) $$

Sieć uczy się generować dobre soft-information dla decoder.

Krok 5 — test

Po treningu robisz:

for snr in [-5, -4, ..., 20]:

    generate many transmissions

    decode

    calculate BLER

Dostajesz:

BLER
 │
 │ Classical
 │   ╲
 │    ╲
 │     ╲
 │ Neural
 │    ╲
 │     ╲
 └────────────────── SNR

I Twoje podstawowe pytanie brzmi:

Czy CNN receiver osiąga lepszy BLER niż klasyczny receiver?

Co jest wynikiem minimum?

Masz już pełną pracę eksperymentalną:

Classical pipeline
Channel estimation
→ Equalization
→ Demapping

vs.

Neural pipeline
CNN

i porównujesz:

BLER vs SNR,
BER vs SNR,
ewentualnie różne kanały.

To jest minimalny zamknięty eksperyment.

POZIOM 2 — Normalna magisterka
CNN vs Transformer + Generalization + Latency

Tutaj nie zmieniasz całej pracy.

Po prostu rozwijasz etap pierwszy.

Część A — porównanie architektur

Masz baseline:

Classical Receiver

Potem:

CNN Receiver

Dodajesz:

Transformer Receiver

Czyli:

                    Received Grid
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
           CNN                  Transformer
             │                       │
             └───────────┬───────────┘
                         ▼
                        LLR
CNN

CNN widzi lokalne zależności.

x x x x
x x x x
x x X x
x x x x

Bardzo dobrze działa dla lokalnych struktur.

Transformer

Transformer robi attention.

Może teoretycznie powiedzieć:

Ten resource element
      │
      ├──────── attention ────────┐
      │                           │
      ▼                           ▼
element tutaj              pilot daleko tutaj

Czyli może modelować bardziej globalne zależności.

Pytanie badawcze:

Czy global attention rzeczywiście pomaga w odbiorniku OFDM?

I to nie jest wcale oczywiste.

Może się okazać:

Transformer:
BLER trochę lepszy

ale

latency ×5

I to jest świetny wynik.

Bo nie musisz udowodnić, że Transformer wygrywa.

Możesz udowodnić:

CNN jest bardziej opłacalny dla real-time receiver.

To też jest wynik naukowy.

Część B — generalization

To moim zdaniem jedna z najważniejszych części.

Nie chcesz:

Train:
SNR = 10 dB
CDL-A

Test:
SNR = 10 dB
CDL-A

bo wtedy model może być po prostu świetnie dopasowany.

Robisz eksperyment:

Scenario 1 — Seen conditions

Training:

SNR: 0–20 dB
Channel: CDL-A
Speed: 30 km/h

Testing:

CDL-A
SNR: 0–20 dB
30 km/h
Scenario 2 — Unseen SNR

Train:

0–20 dB

Test:

-5 dB
25 dB
Scenario 3 — Unseen channel

Train:

CDL-A

Test:

CDL-C
Scenario 4 — Unseen mobility

Train:

3–30 km/h

Test:

100–300 km/h

I wtedy możesz zrobić tabelę:

Train	Test	CNN	Transformer	Classical
CDL-A	CDL-A
CDL-A	CDL-C
Low speed	High speed

To jest już bardzo konkretna analiza robustness/generalization.

Część C — latency

I teraz zaczynasz mierzyć nie tylko jakość.

Dla każdego modelu:

CNN Small
CNN Large
Transformer Small
Transformer Large

mierzysz:

Parametry
number of parameters
FLOPs
computational complexity
Memory
VRAM
Latency
time of single inference

Na przykład:

Model	Params	BLER	Latency
CNN Small	0.5M	0.05	0.08 ms
CNN Large	10M	0.03	0.6 ms
Transformer	10M	0.025	2.4 ms

I nagle możesz powiedzieć:

Transformer daje najlepszy BLER, ale nie spełnia real-time constraints.

To jest właśnie przejście od:

„porównuję modele ML”

do:

„projektuję model dla systemu komunikacyjnego”

POZIOM 3 — Bardzo dobra magisterka
TensorRT + CUDA + FP16/INT8 + Real-Time Optimization

Tutaj przestajesz patrzeć na model wyłącznie jako:

PyTorch model

i zaczynasz traktować go jako coś, co musi działać w prawdziwym systemie.

Krok 1 — masz model w PyTorch

Powiedzmy:

CNN Receiver

BLER = 0.03
Latency = 1.2 ms

Problem:

PyTorch nie jest idealnym runtime dla realnego receivera.

Masz:

Python overhead,
general-purpose runtime,
dynamic graph components,
nie zawsze optymalny execution.

Więc eksportujesz:

PyTorch
   │
   ▼
ONNX
Krok 2 — TensorRT

TensorRT bierze model:

ONNX graph

i robi optymalizacje:

Conv
+
BatchNorm
+
ReLU

może np. zamienić na:

Fused CUDA kernel

Czyli zamiast:

Kernel 1
Kernel 2
Kernel 3

masz:

One optimized kernel

To redukuje:

memory transfers,
kernel launch overhead,
latency.

Pipeline:

PyTorch
   │
   ▼
ONNX
   │
   ▼
TensorRT Engine
   │
   ▼
GPU inference
Krok 3 — Precision

Normalnie model działa:

FP32

czyli:

32-bit float

Ale GPU Tensor Cores bardzo dobrze działają dla:

FP16

albo:

INT8

Więc robisz:

FP32
 │
 ├── FP16
 │
 └── INT8

i testujesz:

Precision	BLER	Latency	Memory
FP32
FP16
INT8

To jest bardzo ciekawy eksperyment.

Może wyjść:

FP32:
0.8 ms

FP16:
0.25 ms
praktycznie ten sam BLER

INT8:
0.15 ms
trochę gorszy BLER

I wtedy masz bardzo konkretny trade-off.

Krok 4 — Quantization

INT8 nie oznacza tylko:

„zamieniłem float na int”.

Musisz odpowiednio przeskalować wartości.

Masz:

FP32 weights

i robisz:

INT8 weights

Ale źle zrobiona quantization może zniszczyć model.

Więc porównujesz np.:

Post Training Quantization
Train FP32
   │
   ▼
Quantize INT8

vs.

Quantization Aware Training
Train model
while simulating quantization

I pytasz:

Czy QAT pozwala zachować BLER przy INT8 inference?

To jest bardzo dobry dodatkowy eksperyment.

Krok 5 — CUDA optimization

Tutaj możesz pójść jeszcze głębiej.

Masz neural receiver:

Received OFDM Grid
      │
      ▼
GPU memory
      │
      ▼
TensorRT
      │
      ▼
LLR

Ale real-time system może mieć problem nie tylko z samym inference.

Może być:

CPU → GPU copy

który kosztuje:

0.2 ms

Inference:

0.3 ms

I nagle:

total = 0.5 ms

Więc zaczynasz patrzeć na:

Host-to-device copy,
device-to-host copy,
CUDA streams,
asynchronous execution,
memory allocation,
pinned memory.

Możesz zrobić:

CPU
 │
 ├── prepares slot N+1
 │
 ▼
GPU
 │
 ├── processes slot N
 │
 ▼
CPU
 │
 └── decodes slot N-1

Czyli pipeline.

Przykład

Bez optymalizacji:

Slot arrives

CPU preprocessing
      0.2 ms

CPU → GPU copy
      0.3 ms

Inference
      0.4 ms

GPU → CPU copy
      0.2 ms

Total:
      1.1 ms

Po optymalizacji:

Pinned memory
CUDA streams
FP16
TensorRT

Total:
0.35 ms

I to już jest naprawdę interesujący wynik.

Co oznacza Real-Time?

To trzeba byłoby zdefiniować bardzo konkretnie.

W 5G NR masz różne numerologie.

Przykładowo slot duration zależy od SCS:

$$ T_{slot} = \frac{1 ms}{2^\mu} $$

czyli:

SCS	Slot
15 kHz	1 ms
30 kHz	0.5 ms
60 kHz	0.25 ms
120 kHz	0.125 ms

Oczywiście nie oznacza to automatycznie, że neural receiver musi wykorzystać dokładnie cały czas slotu — bo są jeszcze inne processing deadlines — ale daje Ci bardzo konkretny punkt odniesienia.

Możesz zdefiniować:

Target deployment: SCS = 30 kHz.

Czyli:

Target:
receiver processing < 0.5 ms

I wtedy każdy model oceniasz:

CNN Small:
0.12 ms ✓

CNN Large:
0.42 ms ✓

Transformer:
1.8 ms ✗

To nadaje pracy bardzo mocny engineeringowy charakter.

Co dokładnie byłoby CUDA w tej pracy?

To jest ważne, bo TensorRT samo w sobie robi większość CUDA za Ciebie.

Nie musisz pisać własnego CNN w CUDA, żeby powiedzieć, że praca ma część GPU.

Ale jeżeli chcesz mocniej wejść w CUDA, możesz zrobić np.:

Custom preprocessing kernel

Received signal:

complex64

Musisz zrobić:

Real / Imag
Normalization
Resource extraction

Zamiast:

Python preprocessing

robisz:

CUDA kernel

Czyli:

Received IQ
     │
     ▼
Custom CUDA preprocessing
     │
     ▼
TensorRT Neural Network
     │
     ▼
LLR

To byłoby bardzo fajne.

Możesz też badać batching

To jest bardzo ciekawy temat.

GPU jest wydajne, kiedy robisz:

batch = 128

Ale receiver działa real-time.

Nie możesz czekać:

"poczekam aż zbiorę 128 UE"

bo latency zabije system.

Więc porównujesz:

Batch size = 1
Batch size = 2
Batch size = 4
Batch size = 8
Batch size = 32

i:

Latency per sample
Throughput

To może dać taki wynik:

Batch 1:
Latency świetne
Throughput średni

Batch 32:
Throughput świetny
Latency fatalne

I znowu:

real-time ≠ maksymalny throughput.

Jak wyglądałaby finalna struktura eksperymentów?

Ja bym zrobił dokładnie tak:

Experiment 1 — Classical baseline
Classical receiver

Metryka:

BLER vs SNR
Experiment 2 — Neural baseline
CNN Neural Receiver

Porównanie:

Classical vs CNN
Experiment 3 — Architecture comparison
CNN Small
CNN Large
Transformer Small
Transformer Large

Porównanie:

BLER
Params
FLOPs
Memory
Latency
Experiment 4 — Generalization

Train:

CDL-A

Test:

CDL-A
CDL-B
CDL-C

oraz:

different SNR
different Doppler
different MCS
Experiment 5 — Deployment
PyTorch FP32
ONNX Runtime
TensorRT FP32
TensorRT FP16
TensorRT INT8

Porównanie:

Latency
Throughput
Memory
BLER degradation
Experiment 6 — Real-time pipeline
Input
 │
 ▼
CUDA preprocessing
 │
 ▼
TensorRT
 │
 ▼
LLR output

Mierzysz:

End-to-end latency

Nie tylko inference latency.

To jest ważne.

Co byłoby Twoim faktycznym wkładem?

I to jest chyba najważniejsze pytanie.

Nie:

„użyłem Sionny i wytrenowałem CNN”.

Bo to już istnieje.

Potencjalny wkład byłby:

1. Architecture comparison

Jakie architektury najlepiej działają dla neural receivera?

2. Generalization study

Jak neural receiver radzi sobie poza warunkami treningowymi?

3. Performance-complexity trade-off

Ile jakości zyskujemy za koszt większego modelu?

4. Hardware-aware optimization

Jak dobrać architekturę pod konkretne real-time constraints?

5. Deployment

Czy model, który dobrze działa w Pythonie, nadal jest użyteczny po faktycznym deployment?

I dla mnie właśnie punkty 3–5 są najciekawsze.

Bo możesz mieć pracę, która nie mówi:

„Neural networks są dobre w odbiornikach.”

Tylko:

„Analizuję kompromis między jakością odbioru, zdolnością generalizacji i opóźnieniem inferencji oraz pokazuję, jak zoptymalizować neural receiver pod wymagania real-time 5G NR.”

To już brzmi jak bardzo sensowny temat magisterski.

Gdybym miał teraz zaproponować scope, który jest ambitny, ale jeszcze realny
Must-have
Sionna
standardowy 5G NR/PUSCH baseline
CNN Neural Receiver
BLER vs SNR
klasyczny receiver vs CNN
Strong thesis
CNN vs Transformer
unseen channel generalization
inference latency
model size/FLOPs analysis
Excellent thesis
ONNX/TensorRT
FP32 vs FP16
INT8 jako opcjonalny eksperyment
end-to-end latency
batch-size analysis
CUDA preprocessing/pipeline

To ostatnie jest moim zdaniem najlepszym miejscem, żeby wykorzystać dostęp do dużych GPU w PLGrid: nie tylko „wytrenować większy model”, ale zrobić duży sweep architektur i konfiguracji, a potem faktycznie zoptymalizować najlepszych kandydatów pod deployment.

ok zacznijmy od sionna, jak to zkomfiguwać jak tego użyć ?

Jasne. I od razu ważny update względem starszych tutoriali: Sionna 2.x przeszła z TensorFlow na PyTorch. Aktualna wersja Sionna PHY/SYS wymaga Python 3.11+ i PyTorch 2.9+, a instalacja jest przez pip install sionna.

Dla naszego celu — Neural Receiver — to nawet lepiej, bo będziesz naturalnie pracował w PyTorch.

Ja bym zaczął bardzo metodycznie.

1. Czym właściwie jest Sionna?

W naszym przypadku interesuje nas głównie:

Sionna PHY

Czyli biblioteka do budowania takich pipeline'ów:

Bits
 ↓
LDPC Encoder
 ↓
QAM Mapper
 ↓
OFDM Resource Grid
 ↓
5G Channel
 ↓
Received Signal
 ↓
Receiver
 ↓
Bits

Sionna ma trzy główne części:

Sionna PHY — link-level simulation ← nas interesuje
Sionna SYS — system-level simulation
Sionna RT — ray tracing propagacji radiowej

Na początku nie potrzebujemy SYS ani RT.

2. Gdzie to instalować?

Ja bym Tobie polecił:

Opcja A — WSL Ubuntu

Jeżeli dalej używasz WSL2 do developmentu, to będzie bardzo wygodne.

Struktura:

Windows
│
└── WSL Ubuntu
      │
      ├── Python
      ├── PyTorch
      ├── CUDA
      └── Sionna

Docelowo PLGrid będzie osobnym środowiskiem, ale lokalnie możesz:

pisać kod,
robić małe testy,
debugować pipeline.

A potem:

local PC
   │
   │ git push
   ▼
PLGrid
   │
   ▼
GPU training

To byłby moim zdaniem najlepszy workflow.

3. Załóżmy świeże środowisko

Najpierw sprawdź:

python3 --version

Aktualna Sionna wymaga Python 3.11+.

Ja bym zrobił osobny projekt:

mkdir neural-receiver
cd neural-receiver

Potem virtual environment:

python3 -m venv .venv

Aktywacja:

source .venv/bin/activate

I upgrade:

pip install --upgrade pip
4. PyTorch

Najpierw sprawdź, czy masz GPU NVIDIA dostępne w WSL:

nvidia-smi

Jeżeli działa, instalujemy PyTorch z CUDA zgodnie z aktualnym selektorem instalacji PyTorch.

Aktualna dokumentacja Sionna zaleca PyTorch 2.9+ i Ubuntu 24.04, ale starsze wersje mogą nadal działać.

Po instalacji PyTorch sprawdź:

import torch

print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

Powinieneś dostać coś w stylu:

2.x.x
True
NVIDIA RTX ...
5. Instalacja Sionna

Teraz:

pip install sionna

To jest oficjalnie rekomendowana instalacja.

Jeżeli nie interesuje Cię teraz ray tracing, możesz też użyć:

pip install sionna-no-rt

Ja na początek wybrałbym właśnie:

pip install sionna-no-rt

bo nas interesuje PHY.

6. Test instalacji

Stwórz:

test_sionna.py

i:

import torch
import sionna

print("Sionna:", sionna.__version__)
print("PyTorch:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

Uruchamiasz:

python test_sionna.py
7. Ale zanim zrobimy Neural Receiver...

Ja bym absolutnie nie zaczynał od neural networka.

Najpierw musimy nauczyć się Sionny na bardzo prostym przykładzie.

Pierwszy cel:

QPSK → AWGN → Classical Receiver

Czyli:

Random Bits
     │
     ▼
QPSK Modulator
     │
     ▼
AWGN Channel
     │
     ▼
Demapper
     │
     ▼
Received Bits

Bez:

OFDM,
MIMO,
5G,
LDPC,
CNN.

Po prostu:

$$ y = x + n $$
8. Pierwszy eksperyment, który bym zrobił

Chcemy wygenerować:

0 1 1 0 0 1 1 1

zamapować na QPSK:

bits
 ↓
QPSK symbols

dodać noise:

QPSK
 ↓
AWGN
 ↓
Noisy QPSK

i potem odzyskać bity.

Na końcu liczymy:

$$ BER = \frac{\text{incorrect bits}} {\text{all bits}} $$

i robimy:

BER
 │
 │\
 │ \
 │  \
 │   \
 │    \
 └──────────── SNR

To będzie Twój pierwszy kontakt z pipeline'em Sionny.

9. Dopiero potem OFDM

Następny krok:

Bits
 ↓
QAM
 ↓
Resource Grid
 ↓
OFDM
 ↓
Channel
 ↓
OFDM Demodulation
 ↓
Receiver

Czyli zaczynamy mieć coś bardziej podobnego do 5G.

Resource grid:

       frequency
    ─────────────────→

t   [ ][ ][ ][ ][ ][ ]
i   [ ][ ][ ][ ][ ][ ]
m   [ ][ ][ ][ ][ ][ ]
e   [ ][ ][ ][ ][ ][ ]

Każdy element to:

Resource Element

czyli:

$$ 1 \text{ OFDM symbol} \times 1 \text{ subcarrier} $$
10. Następnie robimy kanał

Najpierw:

AWGN
y = x + n

Potem:

Flat fading
y = hx+n

Potem:

Frequency selective fading
y = H(f)x+n

I dopiero potem:

3GPP CDL / TDL

czyli kanały bardziej realistyczne dla 5G.

11. Następnie Classical Receiver

To będzie bardzo ważny moment.

Zrobimy:

Transmitter
    │
    ▼
5G Channel
    │
    ▼
Receiver

Receiver:

Received OFDM Grid
        │
        ▼
Channel Estimation
        │
        ▼
Equalization
        │
        ▼
Demapper
        │
        ▼
LLR
        │
        ▼
LDPC Decoder

I to będzie nasz:

Classical Baseline
12. Dopiero wtedy Neural Receiver

Wtedy robimy coś takiego:

              ┌──────────────────────┐
              │ Classical Receiver   │
              │                      │
RX Grid ──────► Channel Estimation   │
              │ Equalization         │
              │ Demapping            │
              └──────────┬───────────┘
                         │
                        LLR


              ┌──────────────────────┐
              │ Neural Receiver      │
              │                      │
RX Grid ──────► CNN                  │
              │                      │
              └──────────┬───────────┘
                         │
                        LLR

I wtedy możemy bardzo dobrze porównywać oba systemy.

Mój proponowany plan nauki Sionny

Dosłownie zrobiłbym sobie repo:

neural-receiver/
│
├── notebooks/
│
│   ├── 01_qpsk_awgn.ipynb
│   │
│   ├── 02_ofdm_basics.ipynb
│   │
│   ├── 03_fading_channel.ipynb
│   │
│   ├── 04_mimo_basics.ipynb
│   │
│   ├── 05_5g_pusch.ipynb
│   │
│   ├── 06_classical_receiver.ipynb
│   │
│   └── 07_neural_receiver.ipynb
│
├── src/
│
│   ├── channels/
│   ├── receivers/
│   ├── models/
│   └── utils/
│
├── experiments/
│
├── requirements.txt
│
└── README.md
Kolejność, którą polecam
Etap 1

Zainstaluj Sionna i PyTorch

Cel:

Sionna działa
GPU działa
Etap 2

QPSK + AWGN

Cel:

Zrozumieć:

bits,
mapper,
constellation,
AWGN,
demapper,
BER.
Etap 3

OFDM

Cel:

Zrozumieć:

resource grid,
subcarriers,
OFDM symbols,
piloty.
Etap 4

Fading channel

Cel:

Zrozumieć:

channel response,
multipath,
Doppler.
Etap 5

5G NR PUSCH

Cel:

Nie budować wszystkiego ręcznie, tylko wykorzystać gotowe komponenty Sionna.

Etap 6

Classical Receiver

Cel:

Uzyskać baseline:

BLER vs SNR
Etap 7

Neural Receiver

Dopiero tutaj:

CNN
↓
LLR
↓
LDPC
Co bym zrobił teraz konkretnie?

Nie próbowałbym od razu pisać własnego kodu.

Sionna ma oficjalne tutoriale, więc najlepszy pierwszy krok to odpalić istniejący przykład i go rozebrać na części. Oficjalna dokumentacja Sionna

Ja proponuję, żebyśmy teraz poszli dosłownie krok po kroku:

Krok 1:

konfigurujemy środowisko Sionna u Ciebie,

Krok 2:

piszemy pierwszy minimalny przykład QPSK → AWGN → BER,

Krok 3:

rozbieramy każdą linię kodu i patrzymy na tensory.

To będzie najlepszy sposób, żebyś faktycznie zrozumiał Sionnę, zamiast od razu wkleić gotowy notebook od NVIDIA.