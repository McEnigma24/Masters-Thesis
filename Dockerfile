# syntax=docker/dockerfile:1
#
# Dev-container image — ZERO kopii kodu projektu.
# Cały repo montujemy jako volume: .:/workspace
#
# Build:  docker compose build
# Sync:   ./scripts/drun sync
# Run:    ./scripts/drun run python ...
# Shell:  ./scripts/dsh
#
# CUDA / cuDNN / NVIDIA only — bez wsparcia AMD.

FROM ghcr.io/astral-sh/uv:0.6.14 AS uv-bin

FROM nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TZ=Europe/Warsaw

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/neural-receiver:/workspace \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# System (apt) — tylko to, czego nie da się sensownie wpychnąć do venv
# ---------------------------------------------------------------------------
RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    software-properties-common \
    tzdata \
    build-essential \
    pkg-config \
    cmake \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libegl1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    xauth \
    xvfb \
    openssh-client \
    vim \
    less \
    htop \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    ${PYTHON_VERSION} \
    ${PYTHON_VERSION}-venv \
    ${PYTHON_VERSION}-dev \
 && ln -snf /usr/share/zoneinfo/Europe/Warsaw /etc/localtime \
 && echo "Europe/Warsaw" > /etc/timezone \
 && ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python \
 && ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# UV — menedżer venv (venv żyje w mountowanym /workspace/.venv)
COPY --from=uv-bin /uv /usr/local/bin/uv

WORKDIR /workspace

# Nic z projektu nie kopiujemy. Entrypoint też jest mountowany.
# Domyślna komenda — nadpisywana przez docker compose / drun.
CMD ["bash"]
