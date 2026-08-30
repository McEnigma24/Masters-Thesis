"""Sprawdzenie środowiska Sionna + PyTorch."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Zwraca CUDA jeśli dostępne, w przeciwnym razie CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_environment() -> dict[str, str | bool]:
    """Weryfikuje instalację i zwraca podsumowanie."""
    import sionna

    device = get_device()
    info: dict[str, str | bool] = {
        "sionna_version": sionna.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def print_environment() -> None:
    """Wypisuje status środowiska w czytelnej formie."""
    info = check_environment()
    print("=== Środowisko neural-receiver ===")
    print(f"Sionna:  {info['sionna_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"CUDA:    {info['cuda_available']}")
    print(f"Device:  {info['device']}")
    if info.get("gpu_name"):
        print(f"GPU:     {info['gpu_name']}")
