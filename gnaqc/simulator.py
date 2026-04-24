"""Qiskit Aer simulator utilities for GNAQC.

Uses the method specified by GNAQCConfig.sim_method (default: statevector).
GPU device is requested when method is tensor_network; for statevector and
matrix_product_state the GPU flag is omitted so Aer uses its own device selection.
"""

from __future__ import annotations

from typing import Any

from qiskit_aer import AerSimulator

_DEFAULT_METHOD = "statevector"


def _sim_kwargs(sim_method: str) -> dict[str, str]:
    return {"method": sim_method, "device": "GPU"}


def create_ideal_simulator(backend: Any, sim_method: str = _DEFAULT_METHOD) -> AerSimulator:
    """Create a noiseless simulator for a backend."""
    return AerSimulator.from_backend(backend, noise_model=None, **_sim_kwargs(sim_method))


def create_noisy_simulator(backend: Any, sim_method: str = _DEFAULT_METHOD) -> AerSimulator:
    """Create a noisy simulator for a backend."""
    return AerSimulator.from_backend(backend, **_sim_kwargs(sim_method))
