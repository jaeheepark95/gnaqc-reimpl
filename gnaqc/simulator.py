"""Qiskit Aer simulator utilities for GNAQC.

Always uses the tensor-network method on GPU (cuQuantum). The project assumes
`qiskit-aer-gpu` with a CUDA-capable device is installed; if not, simulator
construction will fail loudly rather than silently degrade.
"""

from __future__ import annotations

from typing import Any

from qiskit_aer import AerSimulator

_SIM_CONFIG: dict[str, str] = {"method": "tensor_network", "device": "GPU"}


def create_ideal_simulator(backend: Any) -> AerSimulator:
    """Create a noiseless tensor-network GPU simulator for a backend."""
    return AerSimulator.from_backend(backend, noise_model=None, **_SIM_CONFIG)


def create_noisy_simulator(backend: Any) -> AerSimulator:
    """Create a noisy tensor-network GPU simulator for a backend."""
    return AerSimulator.from_backend(backend, **_SIM_CONFIG)
