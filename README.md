# GNAQC Reimplementation

PyTorch reimplementation of **GNAQC** (GNN-Assisted Compilation for Quantum Circuits):

> LeCompte et al., "Machine-Learning-Based Qubit Allocation for Error Reduction in Quantum Circuits",
> IEEE Transactions on Quantum Engineering, Vol. 4, 2023.
> DOI: [10.1109/TQE.2023.3301899](https://doi.org/10.1109/TQE.2023.3301899)

## Overview

GNAQC solves the **qubit allocation problem** — mapping logical qubits of a quantum circuit to physical qubits of a hardware backend to reduce execution error. It uses:

- **Edge-aware GNN** to process the backend's graph structure and error properties
- **Feedforward network** to process circuit features
- **Reinforcement learning (DQN)** to sequentially place qubits, rewarded by Hellinger fidelity from simulation

## Architecture

```
Backend (N×14 node features, N×N edge matrix)
  → 2× Edge-aware GNN layers [Eq.4: X' = ReLU(Ẽ @ X @ W)]
  → Flatten

Circuit (N×7 feature matrix, ancilla-padded)
  → Flatten → Dense + ReLU

State (N-dim mapping vector, -1=unassigned)
  → Dense + ReLU

Combined: Concat → Dense + ReLU → Dense → N² Q-values
```

## Setup

```bash
# Install PyTorch first (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

## Benchmark Circuits

The `circuits/` directory contains **25 Pozzi benchmark circuits** (original high-level QASM, not basis-gate-normalized). Gate decomposition to basis gates `{cx, rz, sx, x, id}` is performed internally via `get_intermediate_circuit()` at train/eval time, matching the paper's Section IV-B intent ("intermediate circuit after decomposition, before allocation").

Circuits (12 "train" + 13 "test" from the Pozzi/attention-qubit-mapping split, used together as a single training set):

| Train (12) | Qubits | Test (13) | Qubits |
|---|:---:|---|:---:|
| bv_n3, bv_n4 | 3-4 | ham3_102, miller_11 | 3 |
| peres_3, toffoli_3, fredkin_3 | 3 | decod24-v0_38, rd32-v0_66 | 4 |
| xor5_254, 3_17_13 | 3-5 | 4gt5_76, 4mod7-v0_94 | 5 |
| 4mod5-v1_22, mod5mils_65 | 5 | alu-v2_32, hwb4_49 | 4-5 |
| alu-v0_27 | 5 | ex1_226, decod24-bdd_294 | 6 |
| decod24-v2_43 | 4 | ham7_104, rd53_138 | 7-8 |
| 4gt13_92 | 5 | qft_10 | 16 |

## Noise Perturbation (Calibration Diversity)

The paper uses **150 days of daily calibration data** to expose the model to diverse noise profiles. Without access to such historical data, we approximate this by perturbing the FakeBackendV2's noise parameters on every `reset()`:

| Property | Perturbation range |
|----------|:---:|
| 2Q gate error | ±30% |
| 1Q gate error | ±30% |
| Readout error | ±30% |
| T1 | ±20% |
| T2 | ±20% |
| Gate duration | ±10% |

Topology (coupling map) and qubit frequency are **never perturbed**. Values are clamped to physically valid ranges (error ∈ [0,1], duration ≥ 0). A dedicated `numpy.random.default_rng` seeded by `cfg.noise_perturb_seed` ensures reproducibility.

Evaluation disables perturbation (`noise_perturb_enabled=False`) to use the backend's reference calibration for deterministic, reproducible comparison.

## Usage

### Training

```bash
# Train on Toronto(27Q) and Rochester(53Q), 5000 episodes each, with noise perturbation
python -m gnaqc.train --backends toronto rochester --episodes 5000

# Toronto only
python -m gnaqc.train --backends toronto --episodes 5000 --name toronto_v1

# Disable noise perturbation (matches paper's single-snapshot baseline)
python -m gnaqc.train --backends toronto --no-noise-perturb

# Custom settings
python -m gnaqc.train \
    --backends toronto \
    --episodes 10000 \
    --train-shots 1000 \
    --lr 0.001 \
    --epsilon 0.05 \
    --batch-size 32 \
    --gnn-hidden 64 \
    --seed 42
```

Training outputs are saved to `runs/<timestamp>_<name>/`:
- `config.json` — hyperparameters used
- `<backend>_training_log.csv` — per-episode metrics (reward, fidelity, loss)
- `checkpoints/<backend>_best.pt` — best model by avg fidelity
- `checkpoints/<backend>_final.pt` — final model

### Evaluation

```bash
# Evaluate on Toronto (eval uses 8192 shots, deterministic calibration)
python -m gnaqc.evaluate \
    --checkpoint runs/<RUN>/checkpoints/toronto_best.pt \
    --backend toronto \
    --shots 8192
```

Compares GNAQC against 4 Qiskit baseline layout methods:
- **Trivial**: sequential mapping (q0→0, q1→1, ...)
- **Dense**: place in highly-connected subgraphs
- **Noise-adaptive**: use most reliable connections
- **SABRE**: iterative routing-aware layout

Reports **two metrics** side by side:
- `hellinger` — Hellinger fidelity (paper's metric, Eq.3)
- `pst` — Probability of Successful Trial (common in qubit-mapping literature, used for cross-paper comparison)

### Custom Circuits

Replace or add `.qasm` files in `circuits/`; update `BENCHMARK_CIRCUITS` in `gnaqc/train.py` to match. Original (non-normalized) QASM is expected — decomposition to basis gates happens internally.

## Project Structure

```
gnaqc/
├── __init__.py
├── backend.py              # FakeBackendV2 registry and utilities
├── config.py               # Hyperparameters (paper-annotated)
├── environment.py          # RL environment (state, actions, rewards, noise perturbation)
├── evaluate.py             # Evaluation against baselines (Hellinger + PST)
├── features.py             # 14-dim backend features + 7-dim circuit features
├── fidelity.py             # Hellinger fidelity (Eq.3) + PST
├── model.py                # Q-network (Edge-aware GNN + FFN)
├── noise_perturbation.py   # Per-episode backend noise randomization
├── simulator.py            # Qiskit Aer simulator utilities
└── train.py                # DQN training loop
circuits/                   # 25 Pozzi benchmark QASM files (original, non-normalized)
```

## Implementation Details

### Backend Features (14-dim, Table 1)
SX/X/ID gate error and length, readout error, T1, T2, frequency (scaled to GHz), measurement length, CNOT error/length (node-averaged from edges), readout length. Row-normalized (L2) per the paper.

### Edge Matrix (N×N)
CNOT error rates, **doubly-stochastic** normalized via Sinkhorn-Knopp. Self-loops added before normalization (standard GCN practice `Ã = A + I`) to ensure convergence on sparse coupling maps.

### Circuit Features (Table 2)
Per-qubit: SX/X/RZ/ID gate counts, measurement flag, CNOT count, CNOT partner indices (look-ahead window). Extracted from the **intermediate circuit** (post-decomposition, pre-allocation), as the paper prescribes. Ancilla rows are zero-padded with CNOT partner = -1.

### Reward (Section V-C)
- New logical qubit placed (non-terminal): **+10** (flat reward)
- Ancilla placed / already-placed qubit / already-occupied physical: **0**
- Terminal (all mapped): **Hellinger fidelity × 100**

### DQN Training
- Epsilon-greedy (ε=0.05), γ=0.99
- Experience replay (buffer=10k), target network (update every 100 episodes)
- **Huber loss + gradient clipping (max_norm=10.0)** — replaces MSE. MSE caused Q-value divergence (loss 300 → 2M+ within 1200 episodes); Huber's linear region for large TD errors stabilizes learning
- Training shots: 1000 (fast reward estimation); Eval shots: 8192 (aligned with other qubit-mapping papers)
- Noise perturbation sampled fresh every episode; ideal counts cached per circuit (noise-independent)

### Differences from Paper

| Aspect | Paper | This Implementation | Reason |
|--------|-------|---------------------|--------|
| Framework | TensorFlow | PyTorch | Ecosystem preference |
| Invalid actions | 0 reward | Action masking (-inf) | Faster convergence |
| Mapping vector init | 0 (unassigned) | -1 (unassigned) | Avoid conflict with qubit index 0 |
| RZ gate features (Table 1 slots) | Unclear | Replaced by CNOT error/length (node-avg) | RZ is virtual gate on IBM HW (always 0 error/duration) |
| DQN loss | Not specified (likely MSE) | **Huber + grad clip** | Empirically required to prevent Q-value divergence |
| Ideal simulation | Every terminal step | Cached per circuit | Layout/noise-independent; 50%+ training speedup |
| Calibration data | 150 daily snapshots (real IBM) | FakeBackendV2 + **±30%/±20%/±10% perturbation** | No access to historical calibration archive |
| Train shots / Eval shots | 10000 / 10000 | **1000 / 8192** | Lower train shots for speed; eval shots aligned with GraphQMap |
| Eval backends | nairobi(7Q), algiers(27Q) | **toronto(27Q), rochester(53Q)** | Aligned with our project's evaluation setup |

## Citation

```bibtex
@article{lecompte2023gnaqc,
  author={LeCompte, Travis and Qi, Fang and Yuan, Xu and Tzeng, Nian-Feng and Najafi, M. Hassan and Peng, Lu},
  journal={IEEE Transactions on Quantum Engineering},
  title={Machine-Learning-Based Qubit Allocation for Error Reduction in Quantum Circuits},
  year={2023},
  volume={4},
  pages={1-14},
  doi={10.1109/TQE.2023.3301899}
}
```
