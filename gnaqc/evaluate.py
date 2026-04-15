"""Evaluation script for GNAQC.

Compares GNAQC layouts against 4 baseline methods (trivial, dense, noise_adaptive, sabre)
using BOTH Hellinger fidelity (paper's metric) and PST (common in qubit-mapping literature).

Usage:
    python -m gnaqc.evaluate \
        --checkpoint runs/<RUN>/checkpoints/nairobi_best.pt \
        --backend nairobi \
        --circuit-dir circuits
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import torch
from qiskit import QuantumCircuit, transpile

from gnaqc.backend import get_backend
from gnaqc.config import GNAQCConfig
from gnaqc.environment import QubitAllocationEnv, ensure_measurements
from gnaqc.fidelity import compute_hellinger_fidelity, compute_pst
from gnaqc.model import GNAQCNetwork
from gnaqc.simulator import create_ideal_simulator, create_noisy_simulator
from gnaqc.train import load_training_circuits

logger = logging.getLogger(__name__)

# Baseline layout methods available in Qiskit
BASELINE_METHODS = ["trivial", "dense", "noise_adaptive", "sabre"]


def evaluate_gnaqc_layout(
    model: GNAQCNetwork,
    circuit: QuantumCircuit,
    backend,
    backend_name: str,
    cfg: GNAQCConfig,
    device: torch.device,
) -> dict:
    """Generate GNAQC layout and evaluate with Hellinger fidelity."""
    env = QubitAllocationEnv(cfg, device)
    state = env.reset(circuit, backend, backend_name)

    # Greedy sequential placement (epsilon=0)
    while not env.done:
        invalid_mask = env.invalid_action_mask()
        action = model.get_action(
            state["node_features"], state["edge_matrix"],
            state["circuit_features"], state["mapping_vector"],
            invalid_mask,
        )
        state, reward, done = env.step(action)

    # Extract layout
    layout_list = [None] * env.num_logical
    for phys_idx in range(env.num_physical):
        logical_idx = int(env.mapping_vector[phys_idx].item())
        if 0 <= logical_idx < env.num_logical:
            layout_list[logical_idx] = phys_idx

    # Compute fidelity
    # Use the raw high-level circuit so transpile(opt=3) can fuse/cancel gates
    # before decomposition — same reasoning as environment._compute_terminal_reward.
    meas_circuit = ensure_measurements(env.raw_circuit)

    compiled = transpile(
        meas_circuit, backend=backend, initial_layout=layout_list,
        routing_method=cfg.routing_method, optimization_level=3,
        seed_transpiler=cfg.seed_transpiler,
    )

    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)
    ideal_counts = ideal_sim.run(compiled, shots=cfg.eval_shots).result().get_counts()
    noisy_counts = noisy_sim.run(compiled, shots=cfg.eval_shots).result().get_counts()
    hellinger = compute_hellinger_fidelity(ideal_counts, noisy_counts)
    pst = compute_pst(ideal_counts, noisy_counts)

    ops = compiled.count_ops()
    cx_count = sum(ops.get(g, 0) for g in ("cx", "ecr", "cz"))

    return {
        "method": "gnaqc",
        "layout": str(layout_list),
        "hellinger": hellinger,
        "pst": pst,
        "depth": compiled.depth(),
        "cx_count": cx_count,
    }


def evaluate_baseline_layout(
    circuit: QuantumCircuit,
    backend,
    layout_method: str,
    cfg: GNAQCConfig,
) -> dict:
    """Evaluate a Qiskit baseline layout method."""
    meas_circuit = ensure_measurements(circuit)

    compiled = transpile(
        meas_circuit, backend=backend, layout_method=layout_method,
        routing_method=cfg.routing_method, optimization_level=3,
        seed_transpiler=cfg.seed_transpiler,
    )

    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)
    ideal_counts = ideal_sim.run(compiled, shots=cfg.eval_shots).result().get_counts()
    noisy_counts = noisy_sim.run(compiled, shots=cfg.eval_shots).result().get_counts()
    hellinger = compute_hellinger_fidelity(ideal_counts, noisy_counts)
    pst = compute_pst(ideal_counts, noisy_counts)

    ops = compiled.count_ops()
    cx_count = sum(ops.get(g, 0) for g in ("cx", "ecr", "cz"))

    return {
        "method": layout_method,
        "hellinger": hellinger,
        "pst": pst,
        "depth": compiled.depth(),
        "cx_count": cx_count,
    }


def evaluate(
    checkpoint_path: str,
    backend_name: str,
    circuit_dir: str = "circuits",
    cfg: GNAQCConfig | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Full evaluation: GNAQC + 4 baselines on benchmark circuits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg is None:
        cfg = GNAQCConfig()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"runs/eval/{timestamp}_{backend_name}"
    os.makedirs(output_dir, exist_ok=True)

    backend = get_backend(backend_name)
    num_physical = backend.target.num_qubits
    logger.info(f"Backend: {backend_name} ({num_physical}Q)")

    model = GNAQCNetwork(cfg, num_physical).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    circuits = load_training_circuits(circuit_dir)

    results = []
    for circ_name, circuit in circuits:
        if circuit.num_qubits > num_physical:
            logger.info(f"Skipping {circ_name} ({circuit.num_qubits}Q > {num_physical}Q)")
            continue

        logger.info(f"\n--- {circ_name} ({circuit.num_qubits}Q) ---")

        # GNAQC
        gnaqc_result = evaluate_gnaqc_layout(model, circuit, backend, backend_name, cfg, device)
        gnaqc_result["circuit"] = circ_name
        gnaqc_result["backend"] = backend_name
        gnaqc_result["num_qubits"] = circuit.num_qubits
        results.append(gnaqc_result)
        logger.info(
            f"  GNAQC:            hellinger={gnaqc_result['hellinger']:.4f}  "
            f"pst={gnaqc_result['pst']:.4f}"
        )

        # Baselines
        for method in BASELINE_METHODS:
            try:
                bl_result = evaluate_baseline_layout(circuit, backend, method, cfg)
                bl_result["circuit"] = circ_name
                bl_result["backend"] = backend_name
                bl_result["num_qubits"] = circuit.num_qubits
                results.append(bl_result)
                logger.info(
                    f"  {method:16s}: hellinger={bl_result['hellinger']:.4f}  "
                    f"pst={bl_result['pst']:.4f}"
                )
            except Exception as e:
                logger.warning(f"  {method} failed: {e}")

    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/eval_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")

    if not df.empty:
        for metric in ("hellinger", "pst"):
            summary = df.groupby("method")[metric].agg(["mean", "std", "min", "max"]).round(4)
            summary = summary.sort_values("mean", ascending=False)
            logger.info(f"\n=== Summary ({metric}) ===\n{summary.to_string()}")

            pivot = df.pivot_table(index="circuit", columns="method", values=metric).round(4)
            logger.info(f"\n=== Per-circuit {metric} ===\n{pivot.to_string()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNAQC")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--backend", required=True, help="Backend name (e.g. nairobi)")
    parser.add_argument("--circuit-dir", type=str, default="circuits",
                        help="Directory containing .qasm benchmark circuits")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Evaluation shots (default 8192, aligned with GraphQMap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Evaluation: fixed calibration (no noise perturbation) for deterministic comparison
    cfg = GNAQCConfig(
        eval_shots=args.shots,
        seed_transpiler=args.seed,
        noise_perturb_enabled=False,
    )
    evaluate(
        checkpoint_path=args.checkpoint,
        backend_name=args.backend,
        circuit_dir=args.circuit_dir,
        cfg=cfg,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
