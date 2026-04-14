"""RL Environment for GNAQC qubit allocation.

The agent places logical qubits one at a time onto physical qubits.
Terminal reward is Hellinger fidelity x 100 from noisy simulation.

State = (backend_node_features, backend_edge_matrix, circuit_features, mapping_vector).
Action = logical_idx * N + physical_idx (N^2 discrete actions).

References:
    - LeCompte et al., IEEE TQE 2023, Section V (RL Setup).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile

from gnaqc.config import GNAQCConfig
from gnaqc.features import (
    extract_backend_edge_matrix,
    extract_backend_node_features,
    extract_circuit_features,
    get_intermediate_circuit,
)
from gnaqc.fidelity import compute_hellinger_fidelity
from gnaqc.simulator import create_ideal_simulator, create_noisy_simulator


class QubitAllocationEnv:
    """RL environment for sequential qubit placement.

    Differences from paper:
        - Mapping vector initialized to -1 (not 0) to avoid 0-based indexing conflict.
        - Action masking with -inf (paper uses 0 reward for invalid actions).
        - Ideal simulation cached per circuit (paper recomputes each episode).
    """

    def __init__(self, cfg: GNAQCConfig, device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        self.device = device

        # Caches (persist across episodes)
        self._backend_cache: dict[str, dict[str, Any]] = {}
        self._ideal_cache: dict[str, dict[str, int]] = {}
        self._simulator_cache: dict[str, tuple] = {}

        # Episode state (set in reset())
        self.backend = None
        self.backend_name: str = ""
        self.circuit: QuantumCircuit | None = None
        self.num_physical: int = 0
        self.num_logical: int = 0
        self.node_features: torch.Tensor | None = None
        self.edge_matrix: torch.Tensor | None = None
        self.circuit_features: torch.Tensor | None = None
        self.mapping_vector: torch.Tensor | None = None
        self.placed_logical: set[int] = set()
        self.placed_physical: set[int] = set()
        self.done: bool = False

    def _get_backend_data(self, backend, backend_name: str) -> dict[str, Any]:
        """Get or compute cached backend features."""
        if backend_name not in self._backend_cache:
            node_feat = extract_backend_node_features(backend)
            edge_mat = extract_backend_edge_matrix(backend)
            self._backend_cache[backend_name] = {
                "node_features": torch.tensor(node_feat, dtype=torch.float32, device=self.device),
                "edge_matrix": torch.tensor(edge_mat, dtype=torch.float32, device=self.device),
                "num_physical": backend.target.num_qubits,
            }
        return self._backend_cache[backend_name]

    def _get_simulators(self, backend, backend_name: str):
        """Get or create cached simulators for a backend."""
        if backend_name not in self._simulator_cache:
            ideal_sim = create_ideal_simulator(backend)
            noisy_sim = create_noisy_simulator(backend)
            self._simulator_cache[backend_name] = (ideal_sim, noisy_sim)
        return self._simulator_cache[backend_name]

    def _get_ideal_counts(
        self, circuit: QuantumCircuit, backend, backend_name: str
    ) -> dict[str, int]:
        """Get or compute cached ideal simulation counts.

        Ideal simulation is layout-independent, so we compute once per circuit
        and reuse across all episodes. This halves training time.
        """
        cache_key = f"{circuit.name}_{backend_name}"
        if cache_key not in self._ideal_cache:
            ideal_sim, _ = self._get_simulators(backend, backend_name)

            meas_circuit = circuit.copy()
            if meas_circuit.num_clbits == 0:
                meas_circuit.measure_all()

            compiled = transpile(
                meas_circuit,
                backend=backend,
                optimization_level=0,
                seed_transpiler=self.cfg.seed_transpiler,
            )
            result = ideal_sim.run(compiled, shots=self.cfg.shots).result()
            self._ideal_cache[cache_key] = result.get_counts()
        return self._ideal_cache[cache_key]

    def reset(
        self,
        circuit: QuantumCircuit,
        backend,
        backend_name: str,
    ) -> dict[str, torch.Tensor]:
        """Reset environment for a new episode.

        Args:
            circuit: Original quantum circuit (will be decomposed to intermediate form).
            backend: FakeBackendV2 instance.
            backend_name: Backend identifier for caching.

        Returns:
            Initial state dict.
        """
        self.backend = backend
        self.backend_name = backend_name
        self.done = False
        self.placed_logical = set()
        self.placed_physical = set()

        # Backend features (cached)
        bd = self._get_backend_data(backend, backend_name)
        self.node_features = bd["node_features"]
        self.edge_matrix = bd["edge_matrix"]
        self.num_physical = bd["num_physical"]

        # Circuit features (intermediate circuit)
        intermediate = get_intermediate_circuit(circuit, backend)
        self.circuit = intermediate
        self.num_logical = intermediate.num_qubits
        circuit_feat = extract_circuit_features(
            intermediate, self.num_physical, self.cfg.look_ahead
        )
        self.circuit_features = torch.tensor(
            circuit_feat, dtype=torch.float32, device=self.device
        )

        # Mapping vector: -1 = unassigned
        self.mapping_vector = torch.full(
            (self.num_physical,), -1.0, dtype=torch.float32, device=self.device
        )

        # Pre-cache ideal simulation
        self._get_ideal_counts(intermediate, backend, backend_name)

        return self._get_state()

    def _get_state(self) -> dict[str, torch.Tensor]:
        """Return current state as a dict of tensors."""
        return {
            "node_features": self.node_features,
            "edge_matrix": self.edge_matrix,
            "circuit_features": self.circuit_features,
            "mapping_vector": self.mapping_vector.clone(),
        }

    def step(self, action: int) -> tuple[dict[str, torch.Tensor], float, bool]:
        """Execute one placement action.

        Args:
            action: action_idx = logical_idx * N + physical_idx.

        Returns:
            (next_state, reward, done).
        """
        N = self.num_physical
        logical_idx = action // N
        physical_idx = action % N

        # Case 1: Already placed logical qubit -> 0 reward
        if logical_idx in self.placed_logical:
            return self._get_state(), 0.0, self.done

        # Case 2: Physical qubit already occupied -> 0 reward
        if physical_idx in self.placed_physical:
            return self._get_state(), 0.0, self.done

        # Case 3: Ancilla qubit
        is_ancilla = logical_idx >= self.num_logical

        # Place the qubit
        self.mapping_vector[physical_idx] = float(logical_idx)
        self.placed_logical.add(logical_idx)
        self.placed_physical.add(physical_idx)

        if is_ancilla:
            reward = 0.0
        else:
            reward = self.cfg.flat_reward  # +10

        # Check terminal: all logical qubits placed
        all_logical_placed = all(
            i in self.placed_logical for i in range(self.num_logical)
        )

        if all_logical_placed and not self.done:
            terminal_reward = self._compute_terminal_reward()
            reward = terminal_reward
            self.done = True

        return self._get_state(), reward, self.done

    def _compute_terminal_reward(self) -> float:
        """Compute Hellinger fidelity x 100 for the completed layout."""
        # Build layout list
        layout_list = [None] * self.num_logical
        for phys_idx in range(self.num_physical):
            logical_idx = int(self.mapping_vector[phys_idx].item())
            if 0 <= logical_idx < self.num_logical:
                layout_list[logical_idx] = phys_idx

        if any(p is None for p in layout_list):
            return 0.0

        meas_circuit = self.circuit.copy()
        if meas_circuit.num_clbits == 0:
            meas_circuit.measure_all()

        compiled = transpile(
            meas_circuit,
            backend=self.backend,
            initial_layout=layout_list,
            routing_method=self.cfg.routing_method,
            optimization_level=3,
            seed_transpiler=self.cfg.seed_transpiler,
        )

        _, noisy_sim = self._get_simulators(self.backend, self.backend_name)
        noisy_counts = noisy_sim.run(compiled, shots=self.cfg.shots).result().get_counts()

        ideal_counts = self._get_ideal_counts(
            self.circuit, self.backend, self.backend_name
        )

        fidelity = compute_hellinger_fidelity(ideal_counts, noisy_counts)
        return fidelity * self.cfg.terminal_reward_scale

    def invalid_action_mask(self) -> torch.Tensor:
        """Return boolean mask over N^2 actions. True = invalid."""
        N = self.num_physical
        mask = torch.zeros(N * N, dtype=torch.bool, device=self.device)

        for action_idx in range(N * N):
            logical_idx = action_idx // N
            physical_idx = action_idx % N
            if logical_idx in self.placed_logical or physical_idx in self.placed_physical:
                mask[action_idx] = True

        return mask

    def valid_actions(self) -> list[int]:
        """Return list of valid action indices."""
        N = self.num_physical
        valid = []
        unplaced_logical = set(range(N)) - self.placed_logical
        unoccupied_physical = set(range(N)) - self.placed_physical
        for l_idx in unplaced_logical:
            for p_idx in unoccupied_physical:
                valid.append(l_idx * N + p_idx)
        return valid
