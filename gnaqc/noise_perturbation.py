"""Noise perturbation for GNAQC training.

The GNAQC paper uses 150 days of daily calibration data to expose the model
to diverse noise profiles, teaching it to *read* noise properties and select
layouts accordingly. Without such data, we simulate calibration diversity by
perturbing the FakeBackendV2's noise parameters within realistic bounds.

Perturbation scales (approximating day-to-day IBM hardware variation):
    - 2Q gate error / 1Q gate error / readout error: +/- 30%
    - T1 / T2: +/- 20%
    - Gate durations: +/- 10%

Topology (coupling map) is NEVER perturbed — only noise parameters.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from qiskit.transpiler import InstructionProperties


DEFAULT_PERTURB_SCALES = {
    "2q_error": 0.30,      # +/- 30%
    "1q_error": 0.30,      # +/- 30%
    "readout_error": 0.30, # +/- 30%
    "t1": 0.20,            # +/- 20%
    "t2": 0.20,            # +/- 20%
    "duration": 0.10,      # +/- 10%
}


def perturb_backend_noise(
    backend: Any,
    rng: np.random.Generator,
    scales: dict[str, float] | None = None,
) -> Any:
    """Return a new backend with perturbed noise parameters.

    Creates a deep copy of the backend and scales each noise-related property
    by an independent uniform random factor in [1 - scale, 1 + scale]. Topology
    is preserved exactly.

    Args:
        backend: Original FakeBackendV2 instance.
        rng: numpy random generator for reproducibility.
        scales: Perturbation scales dict. Defaults to DEFAULT_PERTURB_SCALES.

    Returns:
        A deep-copied backend with perturbed noise. The original backend is untouched.
    """
    if scales is None:
        scales = DEFAULT_PERTURB_SCALES

    perturbed = copy.deepcopy(backend)
    target = perturbed.target
    num_qubits = target.num_qubits

    # --- Perturb qubit properties (T1, T2) ---
    # qubit_properties is a list of QubitProperties objects; we modify in place
    for q in range(num_qubits):
        qp = target.qubit_properties[q]
        if qp is None:
            continue
        if qp.t1 is not None:
            qp.t1 = float(qp.t1 * _sample_scale(rng, scales["t1"]))
        if qp.t2 is not None:
            qp.t2 = float(qp.t2 * _sample_scale(rng, scales["t2"]))
        # frequency NOT perturbed (physical qubit property, not calibration)

    # --- Perturb gate/measurement properties ---
    for op_name in list(target.operation_names):
        op_map = target[op_name]
        for qargs, props in list(op_map.items()):
            if props is None:
                continue

            is_2q = len(qargs) == 2
            if op_name == "measure":
                err_scale_key = "readout_error"
            elif is_2q:
                err_scale_key = "2q_error"
            else:
                err_scale_key = "1q_error"

            err_scale = scales[err_scale_key]
            dur_scale = scales["duration"]

            new_error = (
                float(props.error * _sample_scale(rng, err_scale))
                if props.error is not None else None
            )
            new_duration = (
                float(props.duration * _sample_scale(rng, dur_scale))
                if props.duration is not None else None
            )
            # Clamp error to [0, 1]
            if new_error is not None:
                new_error = min(max(new_error, 0.0), 1.0)
            # Clamp duration to positive
            if new_duration is not None:
                new_duration = max(new_duration, 0.0)

            op_map[qargs] = InstructionProperties(
                duration=new_duration,
                error=new_error,
            )

    return perturbed


def _sample_scale(rng: np.random.Generator, scale: float) -> float:
    """Sample a scaling factor from U(1 - scale, 1 + scale)."""
    return 1.0 + rng.uniform(-scale, scale)
