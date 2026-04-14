"""DQN Training loop for GNAQC.

Implements Q-learning with experience replay and target network.

Usage:
    python -m gnaqc.train --backends nairobi algiers --episodes 5000 --name baseline

References:
    - LeCompte et al., IEEE TQE 2023, Section V (Training).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qiskit import QuantumCircuit

from gnaqc.backend import get_backend
from gnaqc.config import GNAQCConfig
from gnaqc.environment import QubitAllocationEnv
from gnaqc.model import GNAQCNetwork, create_model, create_target_model

logger = logging.getLogger(__name__)

# Default benchmark circuits (subset used in the paper)
BENCHMARK_CIRCUITS = [
    "bv_n3", "bv_n4", "peres_3", "toffoli_3", "fredkin_3",
    "xor5_254", "3_17_13", "4mod5-v1_22", "mod5mils_65",
    "alu-v0_27", "decod24-v2_43", "4gt13_92",
]


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Single (s, a, r, s', done) transition."""
    state: dict[str, torch.Tensor]
    action: int
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def _collate_batch(
    transitions: list[Transition],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collate a list of transitions into batched tensors."""
    return {
        "node_features": torch.stack([t.state["node_features"] for t in transitions]),
        "edge_matrix": torch.stack([t.state["edge_matrix"] for t in transitions]),
        "circuit_features": torch.stack([t.state["circuit_features"] for t in transitions]),
        "mapping_vector": torch.stack([t.state["mapping_vector"] for t in transitions]),
        "actions": torch.tensor([t.action for t in transitions], dtype=torch.long, device=device),
        "rewards": torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device),
        "next_node_features": torch.stack([t.next_state["node_features"] for t in transitions]),
        "next_edge_matrix": torch.stack([t.next_state["edge_matrix"] for t in transitions]),
        "next_circuit_features": torch.stack([t.next_state["circuit_features"] for t in transitions]),
        "next_mapping_vector": torch.stack([t.next_state["mapping_vector"] for t in transitions]),
        "dones": torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Circuit Loading
# ---------------------------------------------------------------------------

def load_training_circuits(
    circuit_dir: str = "circuits",
) -> list[tuple[str, QuantumCircuit]]:
    """Load training circuits from QASM files.

    Looks in the given directory for .qasm files.

    Returns:
        List of (name, QuantumCircuit) tuples.
    """
    circuits = []
    circuit_path = Path(circuit_dir)
    if circuit_path.exists():
        for qasm_file in sorted(circuit_path.glob("*.qasm")):
            name = qasm_file.stem
            try:
                qc = QuantumCircuit.from_qasm_file(str(qasm_file))
                circuits.append((name, qc))
            except Exception as e:
                logger.warning(f"Failed to load {qasm_file}: {e}")

    if not circuits:
        logger.info("No circuits found in %s, generating default benchmarks", circuit_dir)
        circuits = _generate_default_circuits()

    logger.info(f"Loaded {len(circuits)} training circuits")
    return circuits


def _generate_default_circuits() -> list[tuple[str, QuantumCircuit]]:
    """Generate a set of simple benchmark circuits for training."""
    circuits = []

    # Bernstein-Vazirani (various sizes)
    for n in [3, 4, 5]:
        qc = QuantumCircuit(n)
        qc.h(range(n - 1))
        qc.x(n - 1)
        qc.h(n - 1)
        for i in range(n - 1):
            qc.cx(i, n - 1)
        qc.h(range(n - 1))
        qc.measure_all()
        circuits.append((f"bv_n{n}", qc))

    # QFT (various sizes)
    for n in [3, 4, 5]:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                qc.cp(np.pi / (2 ** (j - i)), i, j)
        qc.measure_all()
        circuits.append((f"qft_n{n}", qc))

    # GHZ states
    for n in [3, 4, 5]:
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        circuits.append((f"ghz_n{n}", qc))

    # Deutsch-Jozsa
    for n in [3, 4]:
        qc = QuantumCircuit(n)
        qc.x(n - 1)
        qc.h(range(n))
        for i in range(n - 1):
            qc.cx(i, n - 1)
        qc.h(range(n - 1))
        qc.measure_all()
        circuits.append((f"dj_n{n}", qc))

    return circuits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: GNAQCConfig, run_dir: str | None = None):
    """Run GNAQC DQN training.

    Args:
        cfg: Training configuration.
        run_dir: Directory for saving checkpoints and logs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Setup run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

    # Save config
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    # Load backends
    backends = {}
    for backend_name in cfg.backends:
        backends[backend_name] = get_backend(backend_name)
    logger.info(f"Loaded backends: {list(backends.keys())}")

    # Load circuits
    circuits = load_training_circuits()

    # Train per-backend (paper trains separate models for 7Q and 27Q)
    for be_name, be in backends.items():
        num_physical = be.target.num_qubits
        _train_single_backend(cfg, be_name, be, num_physical, circuits, device, run_dir)


def _train_single_backend(
    cfg: GNAQCConfig,
    backend_name: str,
    backend,
    num_physical: int,
    circuits: list[tuple[str, QuantumCircuit]],
    device: torch.device,
    run_dir: str,
):
    """Train GNAQC on a single backend."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training on backend: {backend_name} ({num_physical}Q)")
    logger.info(f"{'='*60}")

    valid_circuits = [
        (name, circ) for name, circ in circuits
        if circ.num_qubits <= num_physical
    ]
    if not valid_circuits:
        logger.warning(f"No circuits fit on {backend_name}, skipping")
        return

    logger.info(f"Training circuits: {len(valid_circuits)}")

    # Create model and target network
    model = create_model(cfg, num_physical).to(device)
    target_model = create_target_model(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Environment and replay buffer
    env = QubitAllocationEnv(cfg, device)
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    # Metrics tracking
    episode_rewards = []
    episode_fidelities = []
    best_avg_fidelity = 0.0
    log_file = open(f"{run_dir}/{backend_name}_training_log.csv", "w")
    log_file.write("episode,total_reward,terminal_fidelity,loss,epsilon\n")

    for episode in range(cfg.num_episodes):
        circ_name, circ = random.choice(valid_circuits)
        state = env.reset(circ, backend, backend_name)
        total_reward = 0.0
        episode_loss = 0.0

        while not env.done:
            # Epsilon-greedy (paper Section V-A)
            if random.random() < cfg.epsilon:
                valid = env.valid_actions()
                action = random.choice(valid) if valid else 0
            else:
                if cfg.use_action_masking:
                    invalid_mask = env.invalid_action_mask()
                else:
                    invalid_mask = None
                action = model.get_action(
                    state["node_features"],
                    state["edge_matrix"],
                    state["circuit_features"],
                    state["mapping_vector"],
                    invalid_mask,
                )

            next_state, reward, done = env.step(action)
            total_reward += reward

            replay_buffer.add(Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done,
            ))
            state = next_state

            # Mini-batch Q-learning update
            if len(replay_buffer) >= cfg.batch_size:
                batch_transitions = replay_buffer.sample(cfg.batch_size)
                batch = _collate_batch(batch_transitions, device)

                q_current = model(
                    batch["node_features"], batch["edge_matrix"],
                    batch["circuit_features"], batch["mapping_vector"],
                ).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_model(
                        batch["next_node_features"], batch["next_edge_matrix"],
                        batch["next_circuit_features"], batch["next_mapping_vector"],
                    ).max(dim=1).values

                q_target = batch["rewards"] + cfg.gamma * q_next * (1 - batch["dones"])
                loss = F.mse_loss(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss = loss.item()

        terminal_fidelity = max(0, total_reward - cfg.flat_reward * env.num_logical) / cfg.terminal_reward_scale
        episode_rewards.append(total_reward)
        episode_fidelities.append(terminal_fidelity)

        log_file.write(f"{episode},{total_reward:.4f},{terminal_fidelity:.4f},{episode_loss:.6f},{cfg.epsilon}\n")

        # Periodic target network update
        if (episode + 1) % cfg.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Periodic logging
        if (episode + 1) % 100 == 0:
            recent_fidelities = episode_fidelities[-100:]
            avg_fidelity = np.mean(recent_fidelities)
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(
                f"[{backend_name}] Episode {episode+1}/{cfg.num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | Avg Fidelity: {avg_fidelity:.4f} | "
                f"Loss: {episode_loss:.6f}"
            )

            if avg_fidelity > best_avg_fidelity:
                best_avg_fidelity = avg_fidelity
                torch.save(
                    model.state_dict(),
                    f"{run_dir}/checkpoints/{backend_name}_best.pt",
                )
                logger.info(f"  New best avg fidelity: {best_avg_fidelity:.4f}")

        if (episode + 1) % 1000 == 0:
            torch.save(
                model.state_dict(),
                f"{run_dir}/checkpoints/{backend_name}_ep{episode+1}.pt",
            )

    torch.save(model.state_dict(), f"{run_dir}/checkpoints/{backend_name}_final.pt")
    log_file.close()
    logger.info(f"\n[{backend_name}] Training complete. Best avg fidelity: {best_avg_fidelity:.4f}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNAQC")
    parser.add_argument("--backends", nargs="+", default=["nairobi", "algiers"],
                        help="Backend names (e.g. nairobi algiers)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gnn-hidden", type=int, default=64)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--name", type=str, default="", help="Run name suffix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--circuit-dir", type=str, default="circuits",
                        help="Directory containing .qasm training circuits")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = GNAQCConfig(
        backends=args.backends,
        num_episodes=args.episodes,
        lr=args.lr,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        gnn_hidden=args.gnn_hidden,
        shots=args.shots,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_suffix = f"_{args.name}" if args.name else ""
    run_dir = f"runs/{timestamp}{name_suffix}"

    train(cfg, run_dir)


if __name__ == "__main__":
    main()
