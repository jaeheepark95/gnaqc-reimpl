"""Generate paper benchmark circuits (DJ, BV, Simon's, QFT, QPE, Grover's).

Reproduces the circuit set from LeCompte et al., IEEE TQE 2023:
  - 6 standard quantum algorithms
  - Sizes: 3, 4, 5, 6, 7, 15, 27 qubits total per circuit
  - Saved as QASM 2.0 to circuits_paper/

Oracle choices (canonical/standard):
  BV:     hidden string s = 11...1
  DJ:     balanced oracle (parity of all input bits)
  Simon:  secret period s = 11...1
  QFT:    standard QFT with bit-reversal swaps
  QPE:    T gate unitary, eigenstate |1>, n-1 counting + 1 target qubit
  Grover: marked state |11...1>, iterations capped at 3

Serialization note:
  QFT/QPE use QFT library gate which qasm2.dumps exports as "gate_IQFT_dg" — a
  named custom gate that BasisTranslator cannot translate. We call decompose(reps=1)
  on QFT/QPE circuits to unfold library gates into h+cp+swap before saving.
  Grover is built with bare mcx gates (no GroverOperator wrapper); Unroll3qOrMore
  in get_intermediate_circuit() handles mcx decomposition at training time.

Usage:
    python generate_paper_circuits.py [--dst circuits_paper] [--sizes 3 4 5 6 7 15 27]
"""

from __future__ import annotations

import argparse
import math
import os

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import QFT

SIZES = [3, 4, 5, 6, 7, 15, 27]


def make_bv(n: int) -> QuantumCircuit:
    """Bernstein-Vazirani: n total qubits (n-1 input + 1 ancilla), s = 11...1."""
    assert n >= 2
    qc = QuantumCircuit(n, n - 1, name=f"bv_{n}q")
    qc.x(n - 1)
    qc.h(n - 1)
    for i in range(n - 1):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, n - 1)
    for i in range(n - 1):
        qc.h(i)
    qc.measure(range(n - 1), range(n - 1))
    return qc


def make_dj(n: int) -> QuantumCircuit:
    """Deutsch-Jozsa: n total qubits (n-1 input + 1 output), balanced oracle (parity)."""
    assert n >= 2
    qc = QuantumCircuit(n, n - 1, name=f"dj_{n}q")
    qc.x(n - 1)
    qc.h(n - 1)
    for i in range(n - 1):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, n - 1)
    for i in range(n - 1):
        qc.h(i)
    qc.measure(range(n - 1), range(n - 1))
    return qc


def make_simon(n: int) -> QuantumCircuit:
    """Simon's: n total qubits (k=floor(n/2) input + ceil(n/2) workspace), s = 11...1."""
    assert n >= 2
    k = n // 2
    w = n - k
    qc = QuantumCircuit(n, k, name=f"simon_{n}q")
    for i in range(k):
        qc.h(i)
    # Oracle: copy input to workspace, then XOR workspace with s=11...1 via q[0] control
    for i in range(min(k, w)):
        qc.cx(i, k + i)
    for i in range(min(k, w)):
        qc.cx(0, k + i)
    for i in range(k):
        qc.h(i)
    qc.measure(range(k), range(k))
    return qc


def make_qft(n: int) -> QuantumCircuit:
    """QFT: n total qubits, with bit-reversal swaps.

    decompose(reps=1) unfolds the QFT library gate into h+cp+swap so that
    BasisTranslator in get_intermediate_circuit() can handle the result.
    """
    assert n >= 1
    qft_gate = QFT(n, do_swaps=True, insert_barriers=False)
    qc = QuantumCircuit(n, n, name=f"qft_{n}q")
    qc.compose(qft_gate, inplace=True)
    qc.measure(range(n), range(n))
    return qc.decompose(reps=1)


def make_qpe(n: int) -> QuantumCircuit:
    """QPE: n total qubits (n-1 counting + 1 target). Unitary: T gate, eigenstate |1>.

    T gate has phase pi/4, so T^(2^i) has phase (2^i * pi/4) mod 2pi.
    For i >= 3 this wraps to 0 (identity) — only i=0,1,2 produce non-trivial CP gates.

    decompose(reps=1) unfolds the QFT_inv library gate into h+cp+swap before saving.
    """
    assert n >= 2
    t = n - 1
    qc = QuantumCircuit(n, t, name=f"qpe_{n}q")
    qc.x(t)  # eigenstate |1> on target qubit
    for i in range(t):
        qc.h(i)
    # Controlled-T^(2^i): phase = (2^i * pi/4) mod 2pi; zero for i >= 3
    for i in range(t):
        phase = (2 ** i * math.pi / 4) % (2 * math.pi)
        if phase > 1e-10:
            qc.cp(phase, i, t)
    # Inverse QFT on counting register (decomposed to avoid named library gate in QASM)
    qft_inv = QFT(t, do_swaps=True, inverse=True, insert_barriers=False)
    qc.compose(qft_inv, qubits=range(t), inplace=True)
    qc.measure(range(t), range(t))
    return qc.decompose(reps=1)


def make_grover(n: int) -> QuantumCircuit:
    """Grover's: n total qubits, marked state = |11...1>, iterations capped at 3.

    MCX synthesis produces unitary_xxxxxxx custom gates in qasm2.dumps output that
    BasisTranslator cannot translate by name. We pre-transpile to {'cx','u','measure'}
    at optimization_level=1 (includes HighLevelSynthesis) so the saved QASM contains
    only standard-named gates loadable by get_intermediate_circuit().
    Optimal iters: 7Q→11, 15Q→142, 27Q→9000+ — capped at 3.
    """
    assert n >= 1
    num_iter = min(max(1, round(math.pi / 4 * math.sqrt(2 ** n))), 3)
    qc = QuantumCircuit(n, n, name=f"grover_{n}q")
    qc.h(range(n))
    for _ in range(num_iter):
        # Oracle: phase kickback for |11...1>
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        # Diffuser: 2|0><0| - I
        qc.h(range(n))
        qc.x(range(n))
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    qc.measure(range(n), range(n))
    # Synthesize MCX to cx+u before saving; opt_level=1 uses HighLevelSynthesis
    # which avoids the unitary_xxxxxxx custom gate artefacts from qasm2.dumps
    return transpile(qc, basis_gates=["cx", "u", "measure"], optimization_level=1)


GENERATORS: dict[str, callable] = {
    "bv": make_bv,
    "dj": make_dj,
    "simon": make_simon,
    "qft": make_qft,
    "qpe": make_qpe,
    "grover": make_grover,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper benchmark circuits for GNAQC reproducibility"
    )
    parser.add_argument("--dst", default="circuits_paper", help="Output directory")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=SIZES,
        help="Qubit sizes to generate (default: 3 4 5 6 7 15 27)",
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    rows = []
    for algo, gen_fn in GENERATORS.items():
        for n in args.sizes:
            fname = f"{algo}_{n}q.qasm"
            fpath = os.path.join(args.dst, fname)
            try:
                qc = gen_fn(n)
                qasm_str = qasm2.dumps(qc)
                with open(fpath, "w") as f:
                    f.write(qasm_str)
                cx = qc.num_nonlocal_gates()
                rows.append((fname, qc.num_qubits, qc.num_clbits, cx, "ok"))
            except Exception as e:
                rows.append((fname, n, "-", "-", f"SKIP: {e}"))

    print(f"\n{'file':<26} {'qubits':>6} {'clbits':>6} {'cx':>6}  status")
    print("-" * 60)
    for fname, nq, nc, cx, status in rows:
        print(f"{fname:<26} {str(nq):>6} {str(nc):>6} {str(cx):>6}  {status}")

    ok = sum(1 for r in rows if r[-1] == "ok")
    print(f"\n{ok}/{len(rows)} circuits generated → {args.dst}/")


if __name__ == "__main__":
    main()
