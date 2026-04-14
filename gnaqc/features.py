"""Feature extraction for GNAQC.

Backend: 14-dim node features (row-normalized) + N x N edge matrix (doubly-stochastic).
Circuit: per-qubit feature matrix (gate counts, measurement, CNOT partners), ancilla-padded.

References:
    - LeCompte et al., IEEE TQE 2023, Section IV, Tables 1-2, Figures 8-9.
    - Gong & Cheng, CVPR 2019 (doubly-stochastic edge normalization).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    Unroll3qOrMore,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

from gnaqc.backend import get_two_qubit_gate_name


# ---------------------------------------------------------------------------
# Backend Features
# ---------------------------------------------------------------------------

def extract_backend_node_features(backend) -> np.ndarray:
    """Extract 14-dim node features per physical qubit (Table 1).

    Features:
        0: SX gate error         1: SX gate length
        2: X gate error          3: X gate length
        4: Readout error         5: T1
        6: T2                    7: Frequency (scaled to GHz)
        8: ID gate error         9: ID gate length
       10: Measurement length   11: CNOT error (node avg)
       12: CNOT length (node avg) 13: Readout length

    Args:
        backend: FakeBackendV2 instance.

    Returns:
        (N, 14) numpy array, **row-normalized**.
    """
    target = backend.target
    num_qubits = target.num_qubits

    features = np.zeros((num_qubits, 14), dtype=np.float64)

    # Get 2Q gate name for this backend (cx, ecr, or cz)
    gate_2q = get_two_qubit_gate_name(backend)

    # Precompute per-node CNOT error/duration averages from edges
    cx_props = target[gate_2q]
    node_cx_errors: dict[int, list[float]] = {q: [] for q in range(num_qubits)}
    node_cx_durations: dict[int, list[float]] = {q: [] for q in range(num_qubits)}
    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        error = props.error if props.error is not None else 0.0
        duration = props.duration if props.duration is not None else 0.0
        node_cx_errors[a].append(error)
        node_cx_errors[b].append(error)
        node_cx_durations[a].append(duration)
        node_cx_durations[b].append(duration)

    for q in range(num_qubits):
        qp = target.qubit_properties[q]

        # SX gate (0, 1)
        if "sx" in target.operation_names and (q,) in target["sx"]:
            sx_props = target["sx"][(q,)]
            if sx_props is not None:
                features[q, 0] = sx_props.error or 0.0
                features[q, 1] = sx_props.duration or 0.0

        # X gate (2, 3)
        if "x" in target.operation_names and (q,) in target["x"]:
            x_props = target["x"][(q,)]
            if x_props is not None:
                features[q, 2] = x_props.error or 0.0
                features[q, 3] = x_props.duration or 0.0

        # Readout error (4)
        if "measure" in target.operation_names and (q,) in target["measure"]:
            m_props = target["measure"][(q,)]
            if m_props is not None:
                features[q, 4] = m_props.error or 0.0

        # T1 (5), T2 (6)
        features[q, 5] = qp.t1 if qp.t1 is not None else 0.0
        features[q, 6] = qp.t2 if qp.t2 is not None else 0.0

        # Frequency (7) — scale from Hz to GHz
        freq = qp.frequency if qp.frequency is not None else 0.0
        features[q, 7] = freq * 1e-9  # Hz -> GHz

        # ID gate (8, 9)
        if "id" in target.operation_names and (q,) in target["id"]:
            id_props = target["id"][(q,)]
            if id_props is not None:
                features[q, 8] = id_props.error or 0.0
                features[q, 9] = id_props.duration or 0.0

        # Measurement length (10)
        if "measure" in target.operation_names and (q,) in target["measure"]:
            m_props = target["measure"][(q,)]
            if m_props is not None:
                features[q, 10] = m_props.duration or 0.0

        # CNOT error avg (11), CNOT length avg (12)
        if node_cx_errors[q]:
            features[q, 11] = np.mean(node_cx_errors[q])
        if node_cx_durations[q]:
            features[q, 12] = np.mean(node_cx_durations[q])

        # Readout length (13)
        features[q, 13] = features[q, 10]

    # Row normalization (paper: "normalize the matrix by row to accelerate convergence")
    features = _row_normalize(features)

    return features


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix by its L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def extract_backend_edge_matrix(backend) -> np.ndarray:
    """Build N x N edge matrix from 2Q gate error rates, doubly-stochastic normalized.

    Self-loops added before normalization (standard GCN practice: A_tilde = A + I)
    to ensure Sinkhorn-Knopp convergence on sparse coupling maps.

    Args:
        backend: FakeBackendV2 instance.

    Returns:
        (N, N) numpy array, doubly-stochastic normalized.
    """
    target = backend.target
    num_qubits = target.num_qubits
    gate_2q = get_two_qubit_gate_name(backend)
    cx_props = target[gate_2q]

    edge_matrix = np.zeros((num_qubits, num_qubits), dtype=np.float64)

    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        error = props.error if props.error is not None else 0.0
        edge_matrix[a, b] = error
        edge_matrix[b, a] = error

    # Add self-loops: diagonal = average of connected edge errors per node
    for q in range(num_qubits):
        neighbors = edge_matrix[q, :]
        nonzero = neighbors[neighbors > 0]
        if len(nonzero) > 0:
            edge_matrix[q, q] = np.mean(nonzero)
        else:
            edge_matrix[q, q] = 1e-4

    # Doubly-stochastic normalization via Sinkhorn-Knopp
    edge_matrix = _doubly_stochastic_normalize(edge_matrix)

    return edge_matrix


def _doubly_stochastic_normalize(
    matrix: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Sinkhorn-Knopp doubly-stochastic normalization.

    Reference: Gong & Cheng, CVPR 2019 (edge-aware GNN).
    """
    result = matrix.copy()

    if result.sum() == 0:
        return result

    for _ in range(max_iter):
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        result = result / row_sums

        col_sums = result.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        result = result / col_sums

        row_sums_check = result.sum(axis=1)
        col_sums_check = result.sum(axis=0)
        nonzero_rows = row_sums_check[row_sums_check > 0]
        nonzero_cols = col_sums_check[col_sums_check > 0]
        if len(nonzero_rows) > 0 and len(nonzero_cols) > 0:
            if (np.abs(nonzero_rows - 1.0).max() < tol and
                    np.abs(nonzero_cols - 1.0).max() < tol):
                break

    return result


# ---------------------------------------------------------------------------
# Circuit Features
# ---------------------------------------------------------------------------

def get_intermediate_circuit(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    """Get circuit after gate decomposition but before qubit allocation.

    Paper Section IV-B: "We do not use the original logical circuit...
    Instead, we acquire the intermediate circuit during the compilation process
    at the point where qubit mapping normally occurs."
    """
    basis_gates = backend.configuration().basis_gates

    pm = PassManager()
    pm.append(Unroll3qOrMore())
    pm.append(BasisTranslator(SessionEquivalenceLibrary, basis_gates))
    return pm.run(circuit)


def extract_circuit_features(
    circuit: QuantumCircuit,
    num_physical: int,
    look_ahead: int = 1,
) -> np.ndarray:
    """Extract per-qubit circuit features and pad to num_physical qubits.

    Features per qubit (Table 2):
        0: SX count          1: X count
        2: RZ count          3: ID count
        4: Measurement flag  5: CNOT count
        6..6+LA-1: CNOT partner indices (-1 if none)

    Args:
        circuit: Intermediate circuit (basis gates, no layout applied).
        num_physical: Number of physical qubits (for ancilla padding).
        look_ahead: Number of CNOT partners to record (default 1).

    Returns:
        (num_physical, F) numpy array where F = 6 + look_ahead.
    """
    num_logical = circuit.num_qubits
    num_features = 6 + look_ahead
    features = np.zeros((num_physical, num_features), dtype=np.float64)

    cnot_partners: dict[int, list[int]] = {q: [] for q in range(num_logical)}

    for instruction in circuit.data:
        op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        if isinstance(op, Measure):
            for q in qubits:
                features[q, 4] = 1.0
            continue

        gate_name = op.name.lower()

        if len(qubits) == 1:
            q = qubits[0]
            if gate_name == "sx":
                features[q, 0] += 1.0
            elif gate_name == "x":
                features[q, 1] += 1.0
            elif gate_name == "rz":
                features[q, 2] += 1.0
            elif gate_name == "id":
                features[q, 3] += 1.0

        elif len(qubits) == 2 and gate_name in ("cx", "cnot", "ecr", "cz"):
            q0, q1 = qubits
            features[q0, 5] += 1.0
            features[q1, 5] += 1.0
            cnot_partners[q0].append(q1)
            cnot_partners[q1].append(q0)

    # Fill CNOT partner columns
    for q in range(num_logical):
        partners = cnot_partners[q]
        for la in range(look_ahead):
            col = 6 + la
            if la < len(partners):
                features[q, col] = float(partners[la])
            else:
                features[q, col] = -1.0

    # Ancilla: CNOT partner = -1
    for q in range(num_logical, num_physical):
        for la in range(look_ahead):
            features[q, 6 + la] = -1.0

    return features
