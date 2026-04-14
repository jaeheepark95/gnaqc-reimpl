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
    backends: list[str] = field(default_factory=lambda: ["nairobi", "algiers"])

    # === Simulation ===
    shots: int = 10000                  # [paper] Section III: 10,000 shots
    routing_method: str = "sabre"       # [paper] Qiskit default routing
    seed_transpiler: int = 42           # deterministic routing for reproducibility

    # === Action Masking ===
    # Paper uses 0 reward for invalid actions; we use -inf masking for faster convergence.
    # Set to False to match paper's original behavior (0 reward, no masking).
    use_action_masking: bool = True
