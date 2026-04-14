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

## Usage

### Training

```bash
# Train on FakeNairobi (7Q) and FakeAlgiers (27Q)
python -m gnaqc.train --backends nairobi algiers --episodes 5000 --name baseline

# Custom settings
python -m gnaqc.train \
    --backends nairobi \
    --episodes 10000 \
    --lr 0.001 \
    --epsilon 0.05 \
    --batch-size 32 \
    --gnn-hidden 64 \
    --shots 10000 \
    --seed 42
```

Training outputs are saved to `runs/<timestamp>/`:
- `config.json` — hyperparameters used
- `<backend>_training_log.csv` — per-episode metrics
- `checkpoints/<backend>_best.pt` — best model by avg fidelity
- `checkpoints/<backend>_final.pt` — final model

### Evaluation

```bash
python -m gnaqc.evaluate \
    --checkpoint runs/<RUN>/checkpoints/nairobi_best.pt \
    --backend nairobi \
    --shots 10000
```

Compares GNAQC against 4 Qiskit baseline layout methods:
- **Trivial**: sequential mapping (q0→0, q1→1, ...)
- **Dense**: place in highly-connected subgraphs
- **Noise-adaptive**: use most reliable connections
- **SABRE**: iterative routing-aware layout

### Custom Circuits

Place `.qasm` files in `circuits/` directory:

```bash
mkdir circuits
cp /path/to/your/circuit.qasm circuits/
python -m gnaqc.train --circuit-dir circuits --backends nairobi
```

If no circuits are found, default benchmarks (BV, QFT, GHZ, Deutsch-Jozsa at 3-5 qubits) are generated automatically.

## Project Structure

```
gnaqc/
├── __init__.py        # Package init
├── backend.py         # Backend registry and utilities
├── config.py          # Hyperparameters (paper-annotated)
├── environment.py     # RL environment (state, actions, rewards)
├── evaluate.py        # Evaluation against baselines
├── features.py        # Backend 14-dim + circuit feature extraction
├── fidelity.py        # Hellinger fidelity (Eq.3)
├── model.py           # Q-network (Edge-aware GNN + FFN)
├── simulator.py       # Qiskit Aer simulator utilities
└── train.py           # DQN training loop
```

## Implementation Details

### Backend Features (14-dim, Table 1)
SX/X/ID gate error and length, readout error, T1, T2, frequency (scaled), measurement length, CNOT error/length (node-averaged from edges), readout length. Row-normalized per the paper.

### Edge Matrix (N×N)
CNOT error rates, doubly-stochastic normalized via Sinkhorn-Knopp. Self-loops added for GCN convergence on sparse coupling maps.

### Circuit Features (Table 2)
Per-qubit: SX/X/RZ/ID gate counts, measurement flag, CNOT count, CNOT partner indices (look-ahead window). Extracted from the intermediate circuit (post-decomposition, pre-allocation).

### Reward (Section V-C)
- New logical qubit placed: +10 (flat reward)
- Ancilla / already placed: 0
- Terminal (all mapped): Hellinger fidelity × 100

### Differences from Paper

| Aspect | Paper | This Implementation | Reason |
|--------|-------|---------------------|--------|
| Framework | TensorFlow | PyTorch | Ecosystem preference |
| Invalid actions | 0 reward | Action masking (-inf) | Faster convergence |
| Mapping vector init | 0 (unassigned) | -1 (unassigned) | Avoid 0-based index conflict |
| RZ gate features | Included (0 values) | Replaced by CNOT error/length | RZ is virtual gate on IBM HW |
| Ideal simulation | Every episode | Cached per circuit | 50%+ training speedup |
| Calibration data | 150 daily snapshots | Single FakeBackendV2 | Simplified for reproducibility |

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
