# GNAQC Reimplementation

PyTorch reimplementation of **GNAQC** (GNN-Assisted Compilation for Quantum Circuits):

> LeCompte et al., "Machine-Learning-Based Qubit Allocation for Error Reduction in Quantum Circuits",
> IEEE Transactions on Quantum Engineering, Vol. 4, 2023.
> DOI: [10.1109/TQE.2023.3301899](https://doi.org/10.1109/TQE.2023.3301899)

Used as a **baseline** in the GraphQMap project (`/home/jaehee/workspace/projects/GraphQMap/`).

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
  → nn.Embedding(N+1, d) → Dense + ReLU

Combined: Concat → Dense + ReLU → Dense → N² Q-values
```

Action space: **N² (logical, physical) pairs** over all physical qubit slots. Episode terminates when all N physical slots are filled (N = num_physical of backend).

## Setup

```bash
# 1. Create environment (Python 3.10+)
conda create -n gnaqc python=3.10 -y && conda activate gnaqc

# 2. Install PyTorch with CUDA (version must match your system CUDA)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify GPU simulation is available
python -c "
import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
from qiskit_aer import AerSimulator
sim = AerSimulator(method='tensor_network', device='GPU')
print('tensor_network+GPU simulator OK')
"
```

**Version constraints** (all pin each other):
- `qiskit < 2.0` — qiskit-aer-gpu 0.15.1 requires qiskit 1.x API
- `qiskit-aer-gpu == 0.15.1` — GPU tensor-network simulation
- `cuquantum-cu12 == 24.x` — cuquantum 25.x breaks qiskit-aer-gpu 0.15.1
- **Do not install `qiskit-aer`** (non-GPU) — conflicts with `qiskit-aer-gpu`

## Benchmark Circuits

### circuits/ — original Pozzi benchmarks

The `circuits/` directory contains **25 Pozzi benchmark circuits** (original high-level QASM, not basis-gate-normalized). Gate decomposition to basis gates `{cx, rz, sx, x, id}` is performed internally via `get_intermediate_circuit()` at train/eval time, matching the paper's Section IV-B.

**Within each category, the same circuits are used for both training and evaluation** — no held-out split; train set == test set per category.

| Category A (12) | Qubits | Category B (13) | Qubits |
|---|:---:|---|:---:|
| bv_n3, bv_n4 | 3–4 | ham3_102, miller_11 | 3 |
| peres_3, toffoli_3, fredkin_3 | 3 | decod24-v0_38, rd32-v0_66 | 4 |
| xor5_254, 3_17_13 | 3–5 | 4gt5_76, 4mod7-v0_94 | 5 |
| 4mod5-v1_22, mod5mils_65 | 5 | alu-v2_32, hwb4_49 | 4–5 |
| alu-v0_27 | 5 | ex1_226, decod24-bdd_294 | 6 |
| decod24-v2_43 | 4 | ham7_104, rd53_138 | 7–8 |
| 4gt13_92 | 5 | qft_10 | 10 |

### circuits_compact/ — idle-qubit-stripped version

Several RevLib/Pozzi QASM files declare a large register (e.g. `qreg q[16]`) but only use a fraction of the qubits. This inflates:
- **Simulation cost**: statevector allocates 2^declared qubits (16 → 3 qubits = ×8192 slower)
- **Reward signal**: `flat_reward` counts 16 "real logicals" instead of 3, distorting training

`compact_circuits.py` strips idle (never-used) qubits and writes `circuits_compact/`:

```bash
python compact_circuits.py --src circuits --dst circuits_compact
```

Key reductions:

| Circuit | Before | After | Speedup |
|---------|:------:|:-----:|:-------:|
| ham3_102 | 16Q | 3Q | ×8192 |
| miller_11 | 16Q | 3Q | ×8192 |
| ham7_104 | 16Q | 7Q | ×512 |
| rd53_138 | 16Q | 8Q | ×256 |
| qft_10 | 16Q | 10Q | ×64 |

After compaction, all 25 circuits complete within **5s on statevector+GPU** (1000 shots, FakeToronto noise):

| Avg time | Circuits |
|----------|---------|
| < 1s | 17/25 |
| 1–3s | 6/25 |
| 3–5s | hwb4_49 (3.1s), ham7_104 (4.9s) |

**Recommended**: use `--circuit-dir circuits_compact` for all training runs.

## Noise Perturbation (Calibration Diversity)

The paper uses **150 days of daily calibration data** to expose the model to diverse noise profiles. Without access to such historical data, we approximate by perturbing the FakeBackendV2's noise parameters on every `reset()`:

| Property | Perturbation range |
|----------|:---:|
| 2Q gate error | ±30% |
| 1Q gate error | ±30% |
| Readout error | ±30% |
| T1 | ±20% |
| T2 | ±20% |
| Gate duration | ±10% |

Topology (coupling map) and qubit frequency are **never perturbed**. Values are clamped to physically valid ranges.

Evaluation always disables perturbation (`noise_perturb_enabled=False`) for deterministic comparison.

**Note on crashes**: When using `tensor_network` simulation with noise perturbation enabled, certain random Kraus operator combinations trigger `CUTENSORNET_STATUS_INVALID_VALUE`. These are caught; the episode is dropped from the replay buffer (see [Crash Handling](#simulator-crash-handling)). With `statevector` simulation, this class of crash does not occur.

## Usage

### Training

```bash
# Recommended: Toronto(27Q), statevector, no noise perturbation, compact circuits
python -m gnaqc.train \
    --backends toronto \
    --episodes 5000 \
    --flat-reward 0.0 \
    --no-noise-perturb \
    --sim-method statevector \
    --sim-timeout 30 \
    --circuit-dir circuits_compact \
    --name toronto_sv

# Paper-faithful: tensor_network, with noise perturbation
python -m gnaqc.train \
    --backends toronto \
    --episodes 5000 \
    --sim-method tensor_network \
    --sim-timeout 120 \
    --circuit-dir circuits \
    --name toronto_paper

# Rochester (53Q) — requires tensor_network; statevector infeasible at 53Q
python -m gnaqc.train \
    --backends rochester \
    --episodes 5000 \
    --sim-method tensor_network \
    --circuit-dir circuits_compact \
    --name rochester_v1
```

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backends` | `toronto rochester` | Backend(s) to train on |
| `--episodes` | 5000 | Training episodes per backend |
| `--flat-reward` | 10.0 | Per-placement reward for real logical qubits. Set to 0.0 for terminal-only reward (better convergence empirically) |
| `--sim-method` | `tensor_network` | AerSimulator method. `statevector` recommended for ≤27Q; `tensor_network` required for 53Q |
| `--sim-timeout` | 120.0 | Per-simulation timeout (s). Use 30 with `statevector` — normal sims finish in <5s |
| `--no-noise-perturb` | off | Disable per-episode noise perturbation |
| `--target-update-freq` | 100 | Episodes between target network syncs |
| `--circuit-dir` | `circuits` | Directory of QASM files. Use `circuits_compact` for best performance |
| `--train-shots` | 1000 | Shots for training-time reward estimation |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 32 | Replay batch size |
| `--gnn-hidden` | 64 | GNN hidden dim |
| `--invalid-action-mode` | `mask` | `mask` (−∞ invalid Q-values) or `zero_reward` (paper-faithful) |
| `--name` | `` | Run name suffix appended to timestamp |
| `--seed` | 42 | RNG seed |

Training outputs are saved to `runs/<timestamp>_<name>/`:
- `config.json` — all hyperparameters
- `<backend>_training_log.csv` — `episode,total_reward,terminal_fidelity,loss,epsilon`
- `<backend>_crashes.csv` — skipped episodes with circuit, layout, error info
- `checkpoints/<backend>_best.pt` — best model by rolling avg fidelity
- `checkpoints/<backend>_final.pt`, `<backend>_ep1000.pt`, ... — periodic snapshots

### Evaluation

```bash
python -m gnaqc.evaluate \
    --checkpoint runs/<RUN>/checkpoints/toronto_best.pt \
    --backend toronto \
    --sim-method statevector
```

Match `--sim-method` to the training run for a fair comparison.

Compares GNAQC against 4 Qiskit baseline layout methods (**Trivial, Dense, Noise-adaptive, SABRE**) with SABRE routing. Reports two metrics per circuit:
- `hellinger` — Hellinger fidelity (paper's metric, Eq.3)
- `pst` — Probability of Successful Trial (for cross-paper comparison)

## Project Structure

```
gnaqc/
├── __init__.py
├── backend.py              # FakeBackendV2 registry and utilities
├── config.py               # Hyperparameters (paper-annotated)
├── environment.py          # RL environment (state, actions, rewards)
├── evaluate.py             # Evaluation against baselines (Hellinger + PST)
├── features.py             # 14-dim backend features + 7-dim circuit features
├── fidelity.py             # Hellinger fidelity (Eq.3) + PST
├── model.py                # Q-network (Edge-aware GNN + FFN)
├── noise_perturbation.py   # Per-episode backend noise randomization
├── sim_worker.py           # Subprocess-isolated AerSimulator with OS timeout
├── simulator.py            # AerSimulator construction utilities
└── train.py                # DQN training loop
circuits/                   # 25 Pozzi benchmark QASM (original, non-normalized)
circuits_compact/           # Same circuits with idle qubits removed
compact_circuits.py         # Tool: strip idle qubits from QASM files
```

## Implementation Details

### Backend Features (14-dim, Table 1)

Paper-faithful order: `[E_ID, L_ID, E_RZ, L_RZ, E_SX, L_SX, E_X, L_X, T1, T2, F, E_M, P_01, P_10]`.
Per-qubit 1Q gate error/length for ID/RZ/SX/X, relaxation/dephasing times, frequency (GHz), aggregate measurement error `E_M`, and asymmetric readout errors `P_01`, `P_10`. RZ slots are genuinely 0 on IBM HW (virtual frame-change gate). Row-normalized (L2) per the paper.

### Edge Matrix (N×N)

CNOT error rates, **doubly-stochastic** normalized via Sinkhorn-Knopp. Self-loops configurable (`--edge-self-loops`; off by default, matching Eq.4).

### Circuit Features (Table 2)

Per-qubit: SX/X/RZ/ID gate counts, measurement flag, CNOT count, CNOT partner indices. Extracted from the **intermediate circuit** (post-decomposition, pre-allocation, §IV-B). Ancilla rows are zero-padded. CNOT partner indices are normalized by `num_physical` by default (`--no-normalize-partners` to disable).

### Reward (Section V-C)

| Step | Reward |
|------|--------|
| Real logical qubit placed (non-terminal) | `+flat_reward` (default 10.0; configurable) |
| Ancilla placed (non-terminal) | 0 |
| Terminal (all physical slots filled) | Hellinger fidelity × 100 |

`--flat-reward 0.0` (terminal-only) has empirically shown faster convergence and higher fidelity on Toronto — the flat reward can dominate when `num_logical` is small relative to the terminal signal (e.g. 3-logical circuit gives +30 flat vs ~50 terminal).

### DQN Training

- Epsilon-greedy (ε=0.05), γ=0.99
- Experience replay (buffer=10k), target network sync every `--target-update-freq` episodes (default 100)
- **Huber loss + gradient clipping (max_norm=10.0)** — MSE caused Q-value divergence (loss 300 → 2M+ within 1200 episodes)
- Training shots: 1000 (fast); Eval shots: 8192 (aligned with GraphQMap literature)
- Ideal counts cached per (circuit, backend) — layout- and noise-independent; ~50% training speedup

### Simulation Backend

`gnaqc/sim_worker.py` runs AerSimulator in a **persistent subprocess** with an OS-level timeout (`queue.Empty` at `--sim-timeout` seconds). The method is configurable:

| Method | When to use | Typical time (Toronto, 1000 shots) |
|--------|-------------|-----------------------------------|
| `statevector` | Toronto (27Q), any ≤27Q backend | <5s per circuit |
| `tensor_network` | Rochester (53Q), any >27Q backend | 10–60s; crash-prone with noise perturbation |
| `matrix_product_state` | Not tested | — |

On timeout, the subprocess is killed and respawned (next run incurs ~5–10s CUDA re-init). On crash, the episode is dropped (see below).

### Simulator Crash Handling

cuTensorNet can fail with `CUTENSORNET_STATUS_INVALID_VALUE` on specific (circuit, layout, noise) combinations when using `tensor_network` method. On failure:

1. `_compute_terminal_reward()` catches the exception, sets `env.crashed = True`
2. Training loop drops the **entire episode** — no replay-buffer insertion, no synthetic `reward=0`
3. The event is logged to `<backend>_crashes.csv` (episode, circuit, layout, error_type)

Injecting `reward=0` would teach the agent that crashed layouts are bad, biasing the policy against layouts that happen to trigger cuQuantum's contraction optimizer. Dropping is the correct behavior.

With `statevector` method, `CUTENSORNET_STATUS_INVALID_VALUE` crashes do not occur. Timeout-based crashes (`queue.Empty`) are still possible if a circuit is too large, but all 25 compacted Pozzi circuits complete in <5s on Toronto statevector.

### Circuit Measurement Handling

Some benchmark QASM files declare a classical register but emit no `measure` instructions. `environment.ensure_measurements()` scans for actual `Measure` ops (not `num_clbits`) and appends `measure_all()` when needed. Used in both training and evaluation paths.

## Differences from Paper

| Aspect | Paper | This Implementation | Reason |
|--------|-------|---------------------|--------|
| Framework | TensorFlow | PyTorch | Ecosystem preference |
| Invalid actions | 0 reward | **Action masking (−∞)** | Faster convergence; no non-terminating loops |
| Mapping vector init | 0 | **−1** (unassigned) | Avoid conflict with qubit index 0 |
| State encoding | Raw integer IDs → Dense | **`nn.Embedding(N+1, d)` → Dense** | Raw IDs leak O(N) magnitude vs. L2-normalized 14-dim features |
| DQN loss | Not specified | **Huber + grad clip (10.0)** | MSE caused Q-value divergence empirically |
| Flat reward | 10.0 fixed | **Configurable (`--flat-reward`)** | 0.0 (terminal-only) shows better convergence in ablation |
| CNOT partner normalization | Raw indices | **Divided by `num_physical`** | Raw indices dominate L2-normalized backend features in concat |
| Simulator method | Not specified | **Configurable (`--sim-method`)** | `statevector` is 2–10× faster for ≤27Q and eliminates cuTensorNet crashes |
| Ideal simulation | Every terminal step | **Cached per (circuit, backend)** | Layout/noise-independent; ~50% training speedup |
| Calibration data | 150 daily IBM snapshots | **±30%/±20%/±10% perturbation** | No access to historical calibration archive |
| Train / Eval shots | 10000 / 10000 | **1000 / 8192** | Lower train shots for speed; eval aligned with GraphQMap |
| Eval backends | nairobi (7Q), algiers (27Q) | **toronto (27Q), rochester (53Q)** | Aligned with project's evaluation scope |

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
