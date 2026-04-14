"""GNAQC Q-Network.

Architecture (Figure 7, LeCompte et al., IEEE TQE 2023):

    Backend Path:
        Node matrix (N, 14) -> 2x Edge-aware GNN layers -> Flatten -> (N*hidden,)

    Circuit Path:
        Circuit matrix (N, F) -> Flatten -> Dense+ReLU -> (circuit_hidden,)

    State Path:
        Mapping vector (N,) -> Dense+ReLU -> (state_hidden,)

    Combined:
        Concat [backend, circuit, state] -> Dense+ReLU -> Dense -> (N^2, ) Q-values

Edge-aware GNN (Equation 4):
    X^(k) = sigma(E_tilde @ X^(k-1) @ W)
where E_tilde is the doubly-stochastic edge matrix, sigma = ReLU.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnaqc.config import GNAQCConfig


class EdgeAwareGNNLayer(nn.Module):
    """Single edge-aware GNN layer: X' = ReLU(E @ X @ W).

    Implements Equation (4) from the paper using direct matrix multiplication.
    No PyG dependency needed.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (batch, N, in_dim) or (N, in_dim).
            edge_matrix: Doubly-stochastic edge matrix (batch, N, N) or (N, N).

        Returns:
            Updated node features, same shape as input with out_dim.
        """
        h = torch.matmul(edge_matrix, x)  # E @ X
        h = self.weight(h)                # (E @ X) @ W + bias
        return F.relu(h)


class GNAQCNetwork(nn.Module):
    """GNAQC Q-Network for qubit allocation.

    Outputs Q-values for N^2 actions (one per logical-physical qubit pair).
    """

    def __init__(self, cfg: GNAQCConfig, num_physical: int):
        super().__init__()
        self.num_physical = num_physical
        self.cfg = cfg

        N = num_physical
        hidden = cfg.gnn_hidden

        # --- Backend GNN path ---
        self.gnn_layers = nn.ModuleList()
        in_dim = cfg.node_feature_dim
        for _ in range(cfg.gnn_layers):
            self.gnn_layers.append(EdgeAwareGNNLayer(in_dim, hidden))
            in_dim = hidden

        # --- Circuit path ---
        circuit_flat_dim = N * cfg.circuit_feature_dim
        self.circuit_dense = nn.Sequential(
            nn.Linear(circuit_flat_dim, cfg.circuit_hidden),
            nn.ReLU(),
        )

        # --- State path (Fig.7: state vector gets its own Dense layer) ---
        self.state_dense = nn.Sequential(
            nn.Linear(N, cfg.state_hidden),
            nn.ReLU(),
        )

        # --- Combined path ---
        combined_input_dim = N * hidden + cfg.circuit_hidden + cfg.state_hidden
        self.combined = nn.Sequential(
            nn.Linear(combined_input_dim, cfg.combined_hidden),
            nn.ReLU(),
            nn.Linear(cfg.combined_hidden, N * N),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_matrix: torch.Tensor,
        circuit_features: torch.Tensor,
        state_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values for all N^2 actions.

        Args:
            node_features: (batch, N, 14) or (N, 14).
            edge_matrix: (batch, N, N) or (N, N).
            circuit_features: (batch, N, F) or (N, F).
            state_vector: (batch, N) or (N,).

        Returns:
            Q-values (batch, N^2) or (N^2,).
        """
        single = node_features.dim() == 2
        if single:
            node_features = node_features.unsqueeze(0)
            edge_matrix = edge_matrix.unsqueeze(0)
            circuit_features = circuit_features.unsqueeze(0)
            state_vector = state_vector.unsqueeze(0)

        batch_size = node_features.shape[0]

        # Backend GNN path
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_matrix)
        backend_flat = h.reshape(batch_size, -1)

        # Circuit path
        circuit_flat = circuit_features.reshape(batch_size, -1)
        circuit_repr = self.circuit_dense(circuit_flat)

        # State path
        state_repr = self.state_dense(state_vector)

        # Combined
        combined = torch.cat([backend_flat, circuit_repr, state_repr], dim=-1)
        q_values = self.combined(combined)

        if single:
            q_values = q_values.squeeze(0)

        return q_values

    def get_action(
        self,
        node_features: torch.Tensor,
        edge_matrix: torch.Tensor,
        circuit_features: torch.Tensor,
        state_vector: torch.Tensor,
        invalid_mask: torch.Tensor | None = None,
    ) -> int:
        """Select greedy action (argmax Q-value) with optional masking."""
        with torch.no_grad():
            q_values = self.forward(
                node_features, edge_matrix, circuit_features, state_vector
            )
            if invalid_mask is not None:
                q_values[invalid_mask] = float("-inf")
            return q_values.argmax().item()


def create_model(cfg: GNAQCConfig, num_physical: int) -> GNAQCNetwork:
    """Create GNAQC Q-network."""
    return GNAQCNetwork(cfg, num_physical)


def create_target_model(model: GNAQCNetwork) -> GNAQCNetwork:
    """Create a target network (deep copy, no grad)."""
    target = copy.deepcopy(model)
    for param in target.parameters():
        param.requires_grad = False
    return target
