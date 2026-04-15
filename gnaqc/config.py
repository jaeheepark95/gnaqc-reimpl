"""Hyperparameter configuration for GNAQC.

Values marked [paper] are from the GNAQC paper (LeCompte et al., IEEE TQE 2023).
Values marked [standard] are standard DQN defaults not explicitly stated in the paper.
"""

from dataclasses import dataclass, field


@dataclass
class GNAQCConfig:
    # === Architecture ===
    node_feature_dim: int = 14          # [paper] Table 1: 14 backend node features
    circuit_feature_dim: int = 7        # 4 gate counts + 1 meas + 1 cnot count + 1 partner (LA=1)
    gnn_hidden: int = 64                # [standard]
    gnn_layers: int = 2                 # [paper] Fig.7: 2 stacked GNN layers
    circuit_hidden: int = 128           # [standard]
    state_hidden: int = 64              # [paper] Fig.7: State vector gets independent Dense layer
    combined_hidden: int = 256          # [standard]

    # === RL ===
    epsilon: float = 0.05               # [paper] Section V-A: Epsilon-Greedy epsilon=0.05
    gamma: float = 0.99                 # [standard] discount factor
    lr: float = 1e-3                    # [standard]
    replay_buffer_size: int = 10000     # [standard]
    batch_size: int = 32                # [standard]
    target_update_freq: int = 100       # [standard] episodes between target network sync
    num_episodes: int = 5000            # [standard]
    flat_reward: float = 10.0           # [paper] Section V-C: flat reward for placing a new logical qubit
    terminal_reward_scale: float = 100.0  # [paper] Section V-C: Hellinger fidelity x 100

    # === Circuit Features ===
    look_ahead: int = 1                 # [paper] default CNOT partner look-ahead window

    # === Training Backends ===
    backends: list[str] = field(default_factory=lambda: ["toronto", "rochester"])

    # === Simulation (train vs eval split) ===
    # Training uses fewer shots for speed; evaluation uses more for accuracy.
    train_shots: int = 1000             # fast noise-reward estimation during RL
    eval_shots: int = 8192              # aligned with GraphQMap for fair comparison
    # Abort a noisy-sim call if it exceeds this wall-clock budget. Typical
    # tensor-network sims finish in <15s on 27Q toronto; 120s is a safety
    # margin for the pathological-contraction-path case where cuTensorNet's
    # autotuner picks a plan that runs for many minutes without crashing.
    train_sim_timeout_s: float = 120.0
    routing_method: str = "sabre"       # [paper] Qiskit default routing
    seed_transpiler: int = 42           # deterministic routing for reproducibility

    # === Noise Perturbation (simulates daily calibration diversity) ===
    noise_perturb_enabled: bool = True  # disable for deterministic eval
    noise_perturb_seed: int = 2024      # base seed for per-episode noise RNG
    # Per-property perturbation scales (uniform in [1-s, 1+s])
    noise_scale_2q_error: float = 0.30     # [paper-inspired] +/- 30%
    noise_scale_1q_error: float = 0.30
    noise_scale_readout_error: float = 0.30
    noise_scale_t1: float = 0.20           # +/- 20%
    noise_scale_t2: float = 0.20
    noise_scale_duration: float = 0.10     # +/- 10%

    # === Action Masking ===
    # Paper uses 0 reward for invalid actions; we use -inf masking for faster convergence.
    use_action_masking: bool = True

    def perturbation_scales(self) -> dict[str, float]:
        """Return perturbation scales dict for noise_perturbation.perturb_backend_noise()."""
        return {
            "2q_error": self.noise_scale_2q_error,
            "1q_error": self.noise_scale_1q_error,
            "readout_error": self.noise_scale_readout_error,
            "t1": self.noise_scale_t1,
            "t2": self.noise_scale_t2,
            "duration": self.noise_scale_duration,
        }
