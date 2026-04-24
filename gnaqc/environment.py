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
from qiskit_aer.noise import NoiseModel

from gnaqc.config import GNAQCConfig
from gnaqc.features import (
    extract_backend_edge_matrix,
    extract_backend_node_features,
    extract_circuit_features,
    get_intermediate_circuit,
)
from gnaqc.fidelity import compute_hellinger_fidelity
from gnaqc.noise_perturbation import perturb_backend_noise
from gnaqc.sim_worker import SimWorker, SimWorkerTimeout
from gnaqc.simulator import _sim_kwargs


def ensure_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy with measurements appended if the circuit has none.

    Mirrors GraphQMap's / Attention_Qubit_Mapping's `add_measurements()`:
    measures only the qubits that are actually used by any gate, reusing any
    pre-existing classical register from the QASM file. This keeps the output
    bitstring as short as possible — critical for tensor-network simulation,
    since `measure_all()` on circuits with an oversized `qreg` (e.g. qft_10
    declares q[16] but uses only 10 qubits) produces an N-bit output with
    2^N sampling bins and an unnecessarily large noise-propagation graph.

    Using `len(used)` clbits keeps the output distribution at 2^len(used).
    """
    if circuit.count_ops().get("measure", 0) > 0:
        return circuit.copy()
    from qiskit.circuit import ClassicalRegister
    used = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})
    new_qc = circuit.copy()
    if len(new_qc.clbits) == 0:
        new_qc.add_register(ClassicalRegister(len(used), "meas"))
    for idx, q in enumerate(used):
        if idx < len(new_qc.clbits):
            new_qc.measure(q, idx)
    return new_qc


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

        # Persistent subprocess worker for all AerSimulator calls — gives us
        # OS-level kill semantics for cuTensorNet hangs that survive in-process
        # timeouts.
        self._sim_worker = SimWorker(sim_config=_sim_kwargs(cfg.sim_method))

        # Noise RNG (reproducible across episodes)
        if cfg.noise_perturb_enabled:
            self._noise_rng = np.random.default_rng(cfg.noise_perturb_seed)
        else:
            self._noise_rng = None

        # Episode state
        self.base_backend = None         # original un-perturbed backend
        self.backend = None              # possibly-perturbed backend used this episode
        self.noise_model: NoiseModel | None = None  # per-episode noise model (sent to worker)
        self.backend_name: str = ""
        # `raw_circuit` preserves the high-level gate form (e.g. ccx, h) so
        # transpile(opt=3) can apply gate-level fusion/cancellation. `circuit`
        # is the intermediate (post-decomposition) form the paper's §IV-B
        # requires for circuit-feature extraction.
        self.raw_circuit: QuantumCircuit | None = None
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
        # Set by step() if the terminal noisy simulation raised (e.g. cuTensorNet
        # contractor returning CUTENSORNET_STATUS_INVALID_VALUE). Train loop uses
        # this to skip the episode instead of polluting replay with reward=0.
        self.crashed: bool = False
        self.crash_info: dict[str, Any] | None = None
        # Cached Hellinger fidelity from the terminal simulation (0 before done
        # or on crash). Exposed so callers can log it without reconstructing
        # from total_reward, which is ambiguous under the N^2 action space.
        self.terminal_fidelity: float = 0.0

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
        edge_mat = extract_backend_edge_matrix(
            backend, add_self_loops=self.cfg.edge_self_loops
        )

        self.backend = backend
        self.noise_model = NoiseModel.from_backend(backend)
        self.node_features = torch.tensor(node_feat, dtype=torch.float32, device=self.device)
        self.edge_matrix = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        self.num_physical = backend.target.num_qubits

    def _get_ideal_counts(
        self, circuit: QuantumCircuit, backend_name: str
    ) -> dict[str, int] | None:
        """Cached ideal counts (noise-independent). Computed once per circuit.

        Returns None if the ideal simulation times out or raises — caller is
        expected to mark the episode crashed and skip.
        """
        cache_key = f"{circuit.name}_{backend_name}"
        if cache_key in self._ideal_cache:
            return self._ideal_cache[cache_key]

        meas_circuit = ensure_measurements(circuit)
        compiled = transpile(
            meas_circuit,
            backend=self.base_backend,
            optimization_level=0,
            seed_transpiler=self.cfg.seed_transpiler,
        )

        try:
            counts = self._sim_worker.run(
                compiled,
                noise_model=None,
                shots=self.cfg.train_shots,
                timeout_s=self.cfg.train_sim_timeout_s,
            )
        except (SimWorkerTimeout, Exception):
            return None

        self._ideal_cache[cache_key] = counts
        return counts

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
        self.crashed = False
        self.crash_info = None
        self.terminal_fidelity = 0.0
        self.placed_logical = set()
        self.placed_physical = set()
        self._invalid_step_count = 0

        # Prepare (possibly-perturbed) backend + features + noisy simulator
        self._prepare_backend(backend, backend_name)

        # Keep raw circuit (high-level gates) for simulation-time transpile so
        # opt=3 can do gate-level fusion. Feature extraction still uses the
        # intermediate form per paper §IV-B.
        self.raw_circuit = circuit
        intermediate = get_intermediate_circuit(circuit, self.base_backend)
        self.circuit = intermediate
        self.num_logical = intermediate.num_qubits
        circuit_feat = extract_circuit_features(
            intermediate,
            self.num_physical,
            self.cfg.look_ahead,
            normalize_partners=self.cfg.normalize_circuit_partners,
        )
        self.circuit_features = torch.tensor(
            circuit_feat, dtype=torch.float32, device=self.device
        )

        # Mapping vector: -1 = unassigned
        self.mapping_vector = torch.full(
            (self.num_physical,), -1.0, dtype=torch.float32, device=self.device
        )

        # Pre-cache ideal simulation (noise-independent). If it hangs or fails,
        # mark the episode crashed so the train loop skips it (the while-loop
        # body is guarded by `not env.done`, so setting done=True aborts it).
        if self._get_ideal_counts(self.raw_circuit, backend_name) is None:
            self.crashed = True
            self.done = True
            self.crash_info = {
                "backend": self.backend_name,
                "num_logical": self.num_logical,
                "num_physical": self.num_physical,
                "layout": None,
                "error_type": "IdealSimTimeout",
                "error": f"ideal_sim exceeded {self.cfg.train_sim_timeout_s}s "
                         f"or raised (circuit={circuit.name})",
            }

        return self._get_state()

    def _get_state(self) -> dict[str, torch.Tensor]:
        return {
            "node_features": self.node_features,
            "edge_matrix": self.edge_matrix,
            "circuit_features": self.circuit_features,
            "mapping_vector": self.mapping_vector.clone(),
        }

    def step(self, action: int) -> tuple[dict[str, torch.Tensor], float, bool]:
        """Execute one placement action.

        Paper MDP (§V): action space = N_phys^2 over (logical, physical) pairs
        including ancilla logical indices. Episode terminates when every
        physical qubit slot is filled (fully-mapped circuit). Ancilla placement
        yields 0 reward; newly-placing a real logical qubit yields +flat_reward;
        the final placement additionally returns Hellinger fidelity * scale.
        """
        N = self.num_physical
        logical_idx = action // N
        physical_idx = action % N

        if logical_idx in self.placed_logical or physical_idx in self.placed_physical:
            # Paper §V-C: invalid action => 0 reward, no placement. Under
            # "zero_reward" mode we must also cap iterations to avoid an
            # unbounded loop of invalid picks.
            self._invalid_step_count += 1
            cap = self.cfg.max_invalid_steps or (self.num_physical * 2)
            if self._invalid_step_count >= cap:
                self.done = True
            return self._get_state(), 0.0, self.done

        is_ancilla = logical_idx >= self.num_logical

        self.mapping_vector[physical_idx] = float(logical_idx)
        self.placed_logical.add(logical_idx)
        self.placed_physical.add(physical_idx)

        if is_ancilla:
            reward = 0.0
        else:
            reward = self.cfg.flat_reward

        fully_mapped = len(self.placed_physical) == self.num_physical

        if fully_mapped and not self.done:
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

        # Use the raw high-level circuit here — opt=3 does gate-level fusion
        # and cancellation before decomposing to the backend basis, yielding
        # far shorter compiled circuits than feeding an already-decomposed
        # intermediate form. Shorter compiled circuit => smaller tensor network
        # => drastically fewer cuTensorNet contractor path-search failures.
        meas_circuit = ensure_measurements(self.raw_circuit)

        compiled = transpile(
            meas_circuit,
            backend=self.backend,
            initial_layout=layout_list,
            routing_method=self.cfg.routing_method,
            optimization_level=3,
            seed_transpiler=self.cfg.seed_transpiler,
        )

        # Subprocess-isolated sim with OS-level timeout. cuTensorNet's contractor
        # autotuner can pick a path that runs for many minutes while holding the
        # Python GIL, defeating in-process timeouts. Killing the worker subprocess
        # bypasses the GIL entirely.
        try:
            noisy_counts = self._sim_worker.run(
                compiled,
                noise_model=self.noise_model,
                shots=self.cfg.train_shots,
                timeout_s=self.cfg.train_sim_timeout_s,
            )
        except SimWorkerTimeout:
            self.crashed = True
            self.crash_info = {
                "backend": self.backend_name,
                "num_logical": self.num_logical,
                "num_physical": self.num_physical,
                "layout": list(layout_list),
                "error_type": "TimeoutError",
                "error": f"noisy_sim exceeded {self.cfg.train_sim_timeout_s}s",
            }
            return 0.0
        except Exception as e:
            # cuTensorNet occasionally fails with CUTENSORNET_STATUS_INVALID_VALUE
            # on specific (circuit, layout) combinations. Skip the episode
            # instead of learning from a fabricated reward.
            self.crashed = True
            self.crash_info = {
                "backend": self.backend_name,
                "num_logical": self.num_logical,
                "num_physical": self.num_physical,
                "layout": list(layout_list),
                "error_type": type(e).__name__,
                "error": str(e).splitlines()[0][:200] if str(e) else "",
            }
            return 0.0

        ideal_counts = self._get_ideal_counts(self.raw_circuit, self.backend_name)
        if ideal_counts is None:
            # Defensive: reset() pre-caches ideal counts and aborts the episode
            # on failure, so this branch should not be reachable. Guard against
            # future cache-policy changes (e.g. eviction) that could re-trigger
            # an ideal sim here and return None.
            self.crashed = True
            self.crash_info = {
                "backend": self.backend_name,
                "num_logical": self.num_logical,
                "num_physical": self.num_physical,
                "layout": list(layout_list),
                "error_type": "IdealSimUnavailable",
                "error": "ideal_counts cache miss + recompute failed",
            }
            return 0.0
        fidelity = compute_hellinger_fidelity(ideal_counts, noisy_counts)
        self.terminal_fidelity = fidelity
        return fidelity * self.cfg.terminal_reward_scale

    def close(self) -> None:
        """Shut down the simulation worker subprocess. Idempotent."""
        worker = getattr(self, "_sim_worker", None)
        if worker is not None:
            try:
                worker.shutdown()
            finally:
                self._sim_worker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def invalid_action_mask(self) -> torch.Tensor:
        """Mask invalid actions in the N*N action grid.

        Paper action space is N_phys^2 over (logical, physical) pairs INCLUDING
        ancilla logical indices (logical_idx in [num_logical, num_physical)).
        Invalid = logical already placed or physical already occupied.
        """
        N = self.num_physical
        mask = torch.zeros(N * N, dtype=torch.bool, device=self.device)
        for action_idx in range(N * N):
            logical_idx = action_idx // N
            physical_idx = action_idx % N
            if (
                logical_idx in self.placed_logical
                or physical_idx in self.placed_physical
            ):
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
