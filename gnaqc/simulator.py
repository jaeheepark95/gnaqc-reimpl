"""Qiskit Aer simulator utilities for GNAQC.

Provides ideal and noisy simulator creation with automatic method detection.
"""

from __future__ import annotations

import logging
from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

_SIM_CONFIG: dict[str, str] | None = None


def _detect_sim_config() -> dict[str, str]:
    """Auto-detect the best available simulation method.

    Tries tensor_network (GPU then CPU), falls back to automatic method.
    """
    global _SIM_CONFIG
    if _SIM_CONFIG is not None:
        return _SIM_CONFIG

    # Try tensor_network + GPU (cuQuantum)
    try:
        sim = AerSimulator(method="tensor_network", device="GPU")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sim.run(qc, shots=1).result()
        _SIM_CONFIG = {"method": "tensor_network", "device": "GPU"}
        logger.info("Simulation method: tensor_network + GPU (cuQuantum)")
        return _SIM_CONFIG
    except Exception:
        pass

    # Try tensor_network + CPU
    try:
        sim = AerSimulator(method="tensor_network", device="CPU")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sim.run(qc, shots=1).result()
        _SIM_CONFIG = {"method": "tensor_network", "device": "CPU"}
        logger.info("Simulation method: tensor_network + CPU")
        return _SIM_CONFIG
    except Exception:
        pass

    # Fallback to automatic
    _SIM_CONFIG = {}
    logger.info("Simulation method: automatic (Aer default)")
    return _SIM_CONFIG


def create_ideal_simulator(backend: Any) -> AerSimulator:
    """Create a noiseless simulator for a backend."""
    sim_config = _detect_sim_config()
    return AerSimulator.from_backend(backend, noise_model=None, **sim_config)


def create_noisy_simulator(backend: Any) -> AerSimulator:
    """Create a noisy simulator for a backend."""
    sim_config = _detect_sim_config()
    return AerSimulator.from_backend(backend, **sim_config)
