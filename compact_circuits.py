"""Compact QASM circuits by removing idle (never-used) qubits.

Many RevLib/Pozzi benchmarks declare qreg q[16] but only use 3-7 qubits.
This inflates:
  - AerSimulator statevector: 2^16 instead of 2^actual_used
  - GNAQC flat_reward count: 16 "real logicals" instead of actual count

Usage:
    python compact_circuits.py [--src circuits] [--dst circuits_compact]

Prints a before/after table and writes compacted QASM 2.0 files.
"""

from __future__ import annotations

import argparse
import os

from qiskit import QuantumCircuit, qasm2


def compact(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a new circuit with idle qubits removed.

    A qubit is considered used if it appears in any instruction other than
    barrier. Classical bits are retained only if they appear in a measure.
    """
    # Collect used qubits and classical bits
    used_q_indices: set[int] = set()
    used_c_indices: set[int] = set()
    for inst in qc.data:
        if inst.operation.name == "barrier":
            continue
        for q in inst.qubits:
            used_q_indices.add(qc.find_bit(q).index)
        if inst.operation.name == "measure":
            for c in inst.clbits:
                used_c_indices.add(qc.find_bit(c).index)

    # If already compact, return as-is
    if len(used_q_indices) == qc.num_qubits and len(used_c_indices) == qc.num_clbits:
        return qc

    # Build index remappings (preserve original ordering)
    q_sorted = sorted(used_q_indices)
    c_sorted = sorted(used_c_indices)
    q_map = {old: new for new, old in enumerate(q_sorted)}
    c_map = {old: new for new, old in enumerate(c_sorted)}

    new_qc = QuantumCircuit(len(q_sorted), len(c_sorted), name=qc.name)

    for inst in qc.data:
        if inst.operation.name == "barrier":
            continue
        new_qs = [new_qc.qubits[q_map[qc.find_bit(q).index]] for q in inst.qubits]
        new_cs = [new_qc.clbits[c_map[qc.find_bit(c).index]] for c in inst.clbits]
        new_qc.append(inst.operation, new_qs, new_cs)

    return new_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact idle-qubit QASM circuits")
    parser.add_argument("--src", default="circuits", help="Source circuit directory")
    parser.add_argument("--dst", default="circuits_compact", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    rows = []
    for fname in sorted(os.listdir(args.src)):
        if not fname.endswith(".qasm"):
            continue
        src_path = os.path.join(args.src, fname)
        qc = qasm2.load(src_path)
        qc_c = compact(qc)

        dst_path = os.path.join(args.dst, fname)
        with open(dst_path, "w") as f:
            f.write(qasm2.dumps(qc_c))

        changed = qc.num_qubits != qc_c.num_qubits or qc.num_clbits != qc_c.num_clbits
        rows.append((
            fname,
            qc.num_qubits, qc_c.num_qubits,
            qc.num_clbits, qc_c.num_clbits,
            "→ compacted" if changed else "(no change)",
        ))

    # Print summary table
    print(f"{'file':<28} {'q_before':>8} {'q_after':>7} {'c_before':>8} {'c_after':>7}  note")
    print("-" * 75)
    for fname, qb, qa, cb, ca, note in rows:
        print(f"{fname:<28} {qb:>8} {qa:>7} {cb:>8} {ca:>7}  {note}")

    changed_count = sum(1 for r in rows if "compacted" in r[-1])
    print(f"\n{changed_count}/{len(rows)} circuits compacted → {args.dst}/")


if __name__ == "__main__":
    main()
