# CLAUDE.md

Claude Code working notes for this repository. Everything here is
implementation-level context that supplements README.md.

## Project Overview

**gnaqc-reimpl** — PyTorch reimplementation of GNAQC (LeCompte et al.,
IEEE TQE 2023). Used as a **baseline** in a separate GraphQMap project.
The goal is faithful reproduction of the paper's RL-based sequential
qubit-allocation model with a few deliberate deviations that are
documented in README.md and, in more detail, below.

Parent consumer project: `/home/jaehee/workspace/projects/GraphQMap/`.
A stale copy of this code exists at `GraphQMap/gnaqc/` but is **frozen** —
do NOT mirror edits there. All active development happens only in this
standalone repo. The parent will re-sync manually when it wants to pull
a new version of the baseline.

## Tech Stack

- Python 3.10+, PyTorch 2.5.1 (CUDA 12.1)
- Qiskit 1.x (must stay `<2.0` for qiskit-aer-gpu 0.15.1)
- qiskit-aer-gpu 0.15.1 (tensor_network + GPU via cuQuantum)
- cuquantum-cu12 24.x (25.x breaks qiskit-aer-gpu 0.15.1)
- qiskit-ibm-runtime (FakeBackendV2 registry)
- numpy, pandas

`requirements.txt` pins the incompatible combinations. `qiskit-aer`
(non-GPU) **must not be installed** alongside `qiskit-aer-gpu`.

## Project Structure

```
gnaqc/
├── __init__.py
├── backend.py              # FakeBackendV2 registry + get_backend(name)
├── config.py               # GNAQCConfig dataclass (paper-annotated)
├── environment.py          # QubitAllocationEnv — RL state/step/reward
├── evaluate.py             # GNAQC vs 4 Qiskit baselines, Hellinger + PST
├── features.py             # 14-dim backend node, N×N edge matrix, 7-dim circuit
├── fidelity.py             # compute_hellinger_fidelity (Eq.3) + compute_pst
├── model.py                # GNAQCNetwork (Edge-aware GNN + FFN + Dense)
├── noise_perturbation.py   # Per-episode calibration perturbation
├── simulator.py            # AerSimulator(tensor_network+GPU) — no fallback
└── train.py                # DQN training (replay buffer, target network)
circuits/                   # 25 Pozzi QASM (original, non-normalized)
runs/                       # Training outputs (gitignored)
```

## Key Commands

```bash
# Training (Toronto 27Q, 5000 episodes, noise perturbation ON)
python -m gnaqc.train --backends toronto --episodes 5000 --name toronto_v1

# Training without perturbation (deterministic calibration)
python -m gnaqc.train --backends toronto --episodes 5000 --no-noise-perturb \
    --name toronto_no_perturb

# Evaluation (fixed calibration, 8192 shots)
python -m gnaqc.evaluate \
    --checkpoint runs/<RUN>/checkpoints/toronto_best.pt \
    --backend toronto
```

### Outputs per run (`runs/<timestamp>_<name>/`)

- `config.json` — full GNAQCConfig dump
- `<backend>_training_log.csv` — `episode,total_reward,terminal_fidelity,loss,epsilon` (crashed episodes omitted)
- `<backend>_crashes.csv` — `episode,circuit,num_logical,layout,error_type,error` for skipped episodes
- `checkpoints/<backend>_best.pt` — best model by rolling avg fidelity
- `checkpoints/<backend>_final.pt`, `<backend>_ep1000.pt`, ... — periodic snapshots

## Faithful vs Deviated From Paper

Summary (details in README.md and in the code as comments):

**Faithful**
- Architecture: 2× edge-aware GNN (Eq.4) + circuit/state/concat FFN → N² Q-values (Fig.7)
- 14-dim backend node features (row-normalized) + doubly-stochastic edge matrix
- 7-dim per-qubit circuit features from the **intermediate circuit** (post-decomposition, pre-allocation, §IV-B)
- **Action space: N_phys² over (logical, physical) pairs including ancilla indices** (§V-A). Episode terminates when every physical qubit slot is filled.
- Reward: +10 per new real-logical placement, 0 for ancilla, Hellinger-fidelity × 100 terminal (replaces per-step reward on the final step)
- ε=0.05 ε-greedy, γ=0.99, DQN with replay buffer + target network
- Hellinger fidelity (Eq.3) as RL reward
- Edge matrix is the pure doubly-stochastic CNOT-error adjacency (no self-loops by default, matching Eq.4). Self-loop variant available via `cfg.edge_self_loops=True` / `--edge-self-loops`.
- Baseline comparison: trivial/dense/noise_adaptive/sabre layouts with SABRE routing

**Deviated (with reasons)**
- PyTorch instead of TensorFlow (ecosystem)
- **Invalid-action handling defaults to −∞ masking** (paper §V-C uses 0-reward). Mask prevents the agent from ever picking already-placed logicals or occupied physicals — faster convergence, no risk of non-terminating loops. Paper-faithful 0-reward path available via `--invalid-action-mode zero_reward` (adds a `max_invalid_steps` safety cap = `2 * num_physical`).
- Mapping vector init with −1 instead of 0 (avoid conflict with qubit index 0). State path uses `nn.Embedding(N+1, d)` — NOT raw linear on integer IDs — so the discrete qubit-index input doesn't leak O(N) magnitude into the L2-normalized feature fusion
- **Circuit CNOT-partner columns normalized by num_physical** (`cfg.normalize_circuit_partners=True`, default on). Paper leaves them raw; ablation shows +4% fid / more stable convergence vs raw indices, which otherwise dominate the L2-normalized backend path in the concat layer. Disable with `--no-normalize-partners`.
- Table 1 implemented faithfully: 14 features in paper order [E_ID, L_ID, E_RZ, L_RZ, E_SX, L_SX, E_X, L_X, T1, T2, F, E_M, P_01, P_10]. RZ slots read directly from Target (genuinely 0 on IBM — virtual frame-change gate). P_01/P_10 from `backend.properties().qubit_property('prob_meas1_prep0' / 'prob_meas0_prep1')`
- **Huber loss + grad_clip(10.0)** instead of unspecified loss — MSE caused Q-value divergence (300 → 2M+ within 1200 eps). Kept on by default
- Ideal simulation cached per circuit (noise-independent) — ~50% training speedup
- 150-day daily calibration data unavailable → **per-episode noise perturbation** (±30% error, ±20% T1/T2, ±10% duration); topology and qubit frequency preserved
- Train shots 1000 / Eval shots 8192 (speed vs. accuracy; eval matches GraphQMap)
- Evaluation backends: toronto(27Q), rochester(53Q) (project-specific, not paper's nairobi/algiers)

## Benchmark Circuits

25 Pozzi benchmarks in `circuits/` (12 Category-A + 13 Category-B).
**Per-category, train set == test set** (no held-out split — matches the
paper's setup where the same 6 algorithms are used for both training and
evaluation). Original, non-normalized QASM; basis-gate decomposition
happens inside `features.get_intermediate_circuit()`.

Note: many of these QASM files declare a `creg` but never emit `measure`
instructions. `environment.ensure_measurements()` detects actual
`Measure` ops (not `num_clbits`) and appends `measure_all()` when
needed. Used in both training and evaluation paths. This bug silently
returned empty counts on ~20/25 circuits before being fixed.

## Simulation Backend (Important)

`gnaqc/simulator.py` hardcodes `AerSimulator(method="tensor_network",
device="GPU")`. There is **no CPU / statevector / MPS fallback** —
`automatic` mode silently picks statevector-CPU on 27Q backends, turning
each episode into minutes. If the GPU path is unavailable, AerSimulator
construction raises immediately.

### cuTensorNet crash handling

On specific (circuit, layout, noise) combinations, cuTensorNet fails
with `CUTENSORNET_STATUS_INVALID_VALUE` during noisy terminal-reward
computation. We **catch the exception inside
`_compute_terminal_reward()`**, set `env.crashed = True`, and the
training loop drops the **entire episode** — no replay-buffer insertion,
no synthetic `reward=0` (which would teach the agent that crashed layouts
are bad, biasing the policy).

Crashed episodes are appended to `<backend>_crashes.csv` for offline
pattern analysis.

This was discovered empirically: with noise perturbation enabled on a
single FakeToronto backend, 38/5000 episodes crashed. Without
perturbation, 100/100 episodes completed cleanly. The perturbation
changes the NoiseModel's Kraus operators every episode, and certain
random combinations trip cuQuantum's contraction path optimizer.

## Noise Perturbation

`gnaqc/noise_perturbation.py` implements the paper's "150-day daily
calibration" approximation. Deep-copies the backend, then for each
qubit/edge independently samples a scaling factor from
`U(1 − scale, 1 + scale)` per property:

| Property | Scale |
|----------|:---:|
| 2Q gate error | ±30% |
| 1Q gate error | ±30% |
| Readout error | ±30% |
| T1 | ±20% |
| T2 | ±20% |
| Gate duration | ±10% |

Topology, coupling map, and qubit frequency are never touched.
Values are clamped to physically valid ranges.

**Open alternative** (not yet implemented): property-wise shuffle — for
each property, permute values across qubits/edges instead of scaling.
Guarantees every value is a real hardware value (better cuTensorNet
stability, hypothetically). See session notes in git history if needed.

## Critical Rules

- Never install `qiskit-aer` alongside `qiskit-aer-gpu` (they conflict)
- Never bump `cuquantum-cu12` to 25.x (breaks qiskit-aer-gpu 0.15.1)
- Never bump `qiskit` to 2.x (breaks qiskit-aer-gpu 0.15.1)
- Hellinger fidelity is the **training reward**; both Hellinger and PST
  are reported during evaluation. `compute_pst()` was added specifically
  for cross-comparison with GraphQMap / other qubit-mapping literature
- Training uses `cfg.train_shots` (1000); evaluation uses
  `cfg.eval_shots` (8192). The old `cfg.shots` field has been removed
- Ideal counts are cached per circuit (layout- and noise-independent);
  only noisy simulation runs per episode
- `ensure_measurements()` must be used anywhere we transpile a benchmark
  circuit before simulation — the raw QASM cannot be trusted
- If cuTensorNet crashes during training, **don't** return `reward=0`
  or `fidelity=0` to the replay buffer. Drop the episode.

## Deferred Tech Debt

- **`backend.configuration().basis_gates` (deprecated path, [features.py:228](gnaqc/features.py#L228))**
  `BackendV2.configuration()` is marked deprecated but does not emit a
  warning on the currently-pinned Qiskit 1.x. A naive migration to
  `target.operation_names` is **not safe**: on FakeToronto the
  configuration list includes `'reset'` while the Target does not, so
  switching would break any circuit containing `reset` (the
  BasisTranslator would try to decompose it). Revisit when the
  deprecation actually fires; the correct migration requires a curated
  non-gate op filter (`measure`, `delay`, `reset`, `barrier`, control
  flow) and a per-backend audit.

## Known Issues & Open Questions

- **Noise perturbation strategy is under review**. Scaling-based
  perturbation trains but may not reflect real day-to-day calibration
  drift well. Team is discussing alternatives (property-wise shuffle,
  archival calibration replay). Do not change `noise_perturbation.py`
  defaults without consensus.
- **cuTensorNet intermittent failures**: root cause unclear. Suspects
  include (a) noise perturbation generating ill-conditioned Kraus
  operators, (b) SABRE routing producing contraction-unfriendly
  topologies, (c) cuQuantum 24.x contractor autotune non-determinism.
  Handled defensively by skipping crashed episodes.
- **Rochester (53Q)**: action space 2809. Training expected to be
  significantly slower and more crash-prone than Toronto. Not yet
  attempted end-to-end with the N² action-space MDP. Statevector method
  is infeasible at 53Q, so tensor_network is the only option and crash
  handling is load-bearing.

## Environment Hygiene

Always run from the `gnaqc` conda env (`qiskit 1.4.5`, `qiskit-aer-gpu
0.15.1`, `cuquantum-cu12 24.11`). The `base` env has `qiskit 2.3.1` +
CPU-only `qiskit-aer` and will make `tensor_network` sims fail silently
via the crash-handling path, producing thousands of bogus "IdealSimTimeout"
rows. If a training log shows >5% crash rate or a flat ~0.13 fidelity
floor, verify `conda env list` / `which python` first.

## Related Commits (chronological)

- `c88c31b` — Initial implementation (RL env, model, training loop)
- `9d01432` — Huber loss + gradient clipping (fixes MSE Q-divergence)
- `0a702d1` — Align experimental setup with GraphQMap: 25 Pozzi
  benchmarks, noise perturbation, Toronto/Rochester backends, split
  train/eval shots
- `f06556f` — `ensure_measurements()` helper (fixes QASM-with-unused-creg)
- `5f197c1` — Pin GPU tensor_network simulator, remove `cfg.shots`
  back-compat alias
- `9e04a8a` — Skip crashed episodes instead of injecting fake reward=0

## When to Use EnterPlanMode

This project has several interlocking components (simulator, noise
perturbation, crash handling, RL environment). Non-trivial changes
(new reward shaping, alternate perturbation schemes, different
simulator methods, architecture changes) should go through
`EnterPlanMode` — past sessions have shown that seemingly-isolated
edits tend to break caching or cuTensorNet stability in subtle ways.
