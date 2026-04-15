"""RL Environment for GNAQC qubit allocation.

The agent places logical qubits one at a time onto physical qubits.
Terminal reward is Hellinger fidelity x 100 from noisy simulation.

State = (backend_node_features, backend_edge_matrix, circuit_features, mapping_vector).
Action = logical_idx * N + physical_idx (N^2 discrete actions).

Noise diversity: when cfg.noise_perturb_enabled is True, each call to reset()
applies a fresh perturbation of the backend's noise parameters (drawn from a
reproducible RNG). Topology is preserved. Ideal simulation is cached per circuit
since it is layout/noise-independent; noisy simulators and backend features are
recomputed whenever the noise instance changes.

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
from gnaqc.noise_perturbation import perturb_backend_noise
from gnaqc.simulator import create_ideal_simulator, create_noisy_simulator


class QubitAllocationEnv:
    """RL environment for sequential qubit placement.

    Differences from paper:
        - Mapping vector initialized to -1 (not 0) to avoid 0-based indexing conflict.
        - Action masking with -inf (paper uses 0 reward for invalid actions).
        - Ideal simulation cached per circuit (paper recomputes each episode).
        - Noise perturbation simulates 150-day calibration diversity.
    """

    def __init__(self, cfg: GNAQCConfig, device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        self.device = device

        # Ideal counts cache: layout/noise-independent, keyed by circuit name
        self._ideal_cache: dict[str, dict[str, int]] = {}
        # Ideal simulator per base backend (noise doesn't affect ideal)
        self._ideal_sim_cache: dict[str, Any] = {}

        # Noise RNG (reproducible across episodes)
        if cfg.noise_perturb_enabled:
            self._noise_rng = np.random.default_rng(cfg.noise_perturb_seed)
        else:
            self._noise_rng = None

        # Episode state
        self.base_backend = None         # original un-perturbed backend
        self.backend = None              # possibly-perturbed backend used this episode
        self.noisy_sim = None            # noisy simulator for this episode's backend
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

    # -----------------------------------------------------------------
    # Per-episode backend preparation
    # -----------------------------------------------------------------

    def _prepare_backend(self, base_backend, backend_name: str):
        """Perturb the base backend (if enabled) and set up features + simulator."""
        if self.cfg.noise_perturb_enabled and self._noise_rng is not None:
            backend = perturb_backend_noise(
                base_backend, self._noise_rng, self.cfg.perturbation_scales()
            )
        else:
            backend = base_backend

        node_feat = extract_backend_node_features(backend)
        edge_mat = extract_backend_edge_matrix(backend)

        self.backend = backend
        self.noisy_sim = create_noisy_simulator(backend)
        self.node_features = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        self.edge_matrix = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        self.num_physical = backend.target.num_qubits

    def _get_ideal_sim(self, backend_name: str):
        """Ideal simulator depends only on base backend (topology + basis gates)."""
        if backend_name not in self._ideal_sim_cache:
            self._ideal_sim_cache[backend_name] = create_ideal_simulator(self.base_backend)
        return self._ideal_sim_cache[backend_name]

    def _get_ideal_counts(
        self, circuit: QuantumCircuit, backend_name: str
    ) -> dict[str, int]:
        """Cached ideal counts (noise-independent). Computed once per circuit."""
        cache_key = f"{circuit.name}_{backend_name}"
        if cache_key not in self._ideal_cache:
            ideal_sim = self._get_ideal_sim(backend_name)

            meas_circuit = circuit.copy()
            if meas_circuit.num_clbits == 0:
                meas_circuit.measure_all()

            compiled = transpile(
                meas_circuit,
                backend=self.base_backend,
                optimization_level=0,
                seed_transpiler=self.cfg.seed_transpiler,
            )
            result = ideal_sim.run(compiled, shots=self.cfg.train_shots).result()
            self._ideal_cache[cache_key] = result.get_counts()
        return self._ideal_cache[cache_key]

    # -----------------------------------------------------------------
    # Gym-like interface
    # -----------------------------------------------------------------

    def reset(
        self,
        circuit: QuantumCircuit,
        backend,
        backend_name: str,
    ) -> dict[str, torch.Tensor]:
        """Reset environment for a new episode.

        Applies a fresh noise perturbation (if enabled). Ideal counts are
        cached per circuit; noisy simulator is rebuilt each episode.
        """
        self.base_backend = backend
        self.backend_name = backend_name
        self.done = False
        self.placed_logical = set()
        self.placed_physical = set()

        # Prepare (possibly-perturbed) backend + features + noisy simulator
        self._prepare_backend(backend, backend_name)

        # Circuit features from intermediate circuit
        intermediate = get_intermediate_circuit(circuit, self.base_backend)
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

        # Pre-cache ideal simulation (noise-independent)
        self._get_ideal_counts(intermediate, backend_name)

        return self._get_state()

    def _get_state(self) -> dict[str, torch.Tensor]:
        return {
            "node_features": self.node_features,
            "edge_matrix": self.edge_matrix,
            "circuit_features": self.circuit_features,
            "mapping_vector": self.mapping_vector.clone(),
        }

    def step(self, action: int) -> tuple[dict[str, torch.Tensor], float, bool]:
        """Execute one placement action."""
        N = self.num_physical
        logical_idx = action // N
        physical_idx = action % N

        if logical_idx in self.placed_logical:
            return self._get_state(), 0.0, self.done
        if physical_idx in self.placed_physical:
            return self._get_state(), 0.0, self.done

        is_ancilla = logical_idx >= self.num_logical

        self.mapping_vector[physical_idx] = float(logical_idx)
        self.placed_logical.add(logical_idx)
        self.placed_physical.add(physical_idx)

        if is_ancilla:
            reward = 0.0
        else:
            reward = self.cfg.flat_reward

        all_logical_placed = all(
            i in self.placed_logical for i in range(self.num_logical)
        )

        if all_logical_placed and not self.done:
            terminal_reward = self._compute_terminal_reward()
            reward = terminal_reward
            self.done = True

        return self._get_state(), reward, self.done

    def _compute_terminal_reward(self) -> float:
        """Hellinger fidelity x 100 for the completed layout under current noise."""
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

        noisy_counts = self.noisy_sim.run(
            compiled, shots=self.cfg.train_shots
        ).result().get_counts()

        ideal_counts = self._get_ideal_counts(self.circuit, self.backend_name)
        fidelity = compute_hellinger_fidelity(ideal_counts, noisy_counts)
        return fidelity * self.cfg.terminal_reward_scale

    def invalid_action_mask(self) -> torch.Tensor:
        N = self.num_physical
        mask = torch.zeros(N * N, dtype=torch.bool, device=self.device)
        for action_idx in range(N * N):
            logical_idx = action_idx // N
            physical_idx = action_idx % N
            if logical_idx in self.placed_logical or physical_idx in self.placed_physical:
                mask[action_idx] = True
        return mask

    def valid_actions(self) -> list[int]:
        N = self.num_physical
        valid = []
        unplaced_logical = set(range(N)) - self.placed_logical
        unoccupied_physical = set(range(N)) - self.placed_physical
        for l_idx in unplaced_logical:
            for p_idx in unoccupied_physical:
                valid.append(l_idx * N + p_idx)
        return valid
