#!/usr/bin/env python3
"""Weryfikacja CUDA + Sionna — uruchom: ./scripts/drun verify"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print("=== PyTorch / CUDA ===")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"CUDA ver: {torch.version.cuda}")
    else:
        print("BŁĄD: CUDA niedostępne — sprawdź sterownik Windows + nvidia-container-toolkit")
        return 1

    import sionna

    print("\n=== Sionna ===")
    print(f"Sionna:   {sionna.__version__}")

    import sionna.phy  # noqa: F401

    print("sionna.phy: OK")

    print("\n=== Gotowe ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
