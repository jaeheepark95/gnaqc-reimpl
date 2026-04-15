"""Fidelity metrics for GNAQC.

Implements two metrics:
    1. Hellinger fidelity (Eq.3 of LeCompte et al., IEEE TQE 2023) — used as RL reward
    2. PST (Probability of Successful Trial) — used for fair comparison with other
       compilation methods that report PST.

Both take (ground_truth_counts, test_counts) and return a float in [0, 1].
"""

from __future__ import annotations

import math


def compute_hellinger_fidelity(
    ground_truth_counts: dict[str, int],
    test_counts: dict[str, int],
) -> float:
    """Compute Hellinger fidelity between two count distributions (GNAQC paper Eq.3).

    F = 1 - (1/sqrt(2)) * sqrt(sum_i (sqrt(p_GT_i) - sqrt(p_T_i))^2)

    Args:
        ground_truth_counts: Ideal (noise-free) simulation counts.
        test_counts: Noisy simulation counts.

    Returns:
        Fidelity in [0, 1]. 1.0 = identical distributions, 0.0 = maximally different.
    """
    all_keys = set(ground_truth_counts) | set(test_counts)
    total_gt = sum(ground_truth_counts.values())
    total_t = sum(test_counts.values())

    if total_gt == 0 or total_t == 0:
        return 0.0

    hellinger_sum = 0.0
    for key in all_keys:
        p_gt = ground_truth_counts.get(key, 0) / total_gt
        p_t = test_counts.get(key, 0) / total_t
        hellinger_sum += (math.sqrt(p_gt) - math.sqrt(p_t)) ** 2

    hellinger_dist = (1.0 / math.sqrt(2.0)) * math.sqrt(hellinger_sum)
    return 1.0 - hellinger_dist


def compute_pst(
    ground_truth_counts: dict[str, int],
    test_counts: dict[str, int],
) -> float:
    """Compute PST (Probability of Successful Trial).

    PST = P(correct output bitstring in noisy execution), where the "correct"
    bitstring is the most-probable outcome in the ideal (noise-free) distribution.

    This is the metric commonly reported in qubit-mapping literature (e.g. SABRE,
    NoiseAdaptive, QAP). Included here for fair cross-comparison with such methods.

    For multi-register circuits (space-separated bitstrings), per-register PST
    is averaged.

    Args:
        ground_truth_counts: Ideal (noise-free) simulation counts.
        test_counts: Noisy simulation counts.

    Returns:
        PST in [0, 1]. Higher = more correct outputs.
    """
    if not ground_truth_counts or not test_counts:
        return 0.0

    ideal_result = max(ground_truth_counts, key=lambda k: ground_truth_counts[k])
    total_shots = sum(test_counts.values())

    if total_shots == 0:
        return 0.0

    if " " in ideal_result:
        # Multi-register circuit: compute per-register PST and average
        ideal_parts = ideal_result.split(" ")
        psts = []
        for idx, ideal_part in enumerate(ideal_parts):
            matching = 0
            for key, count in test_counts.items():
                parts = key.split(" ")
                if idx < len(parts) and ideal_part == parts[idx]:
                    matching += count
            psts.append(matching / total_shots)
        return sum(psts) / len(psts) if psts else 0.0

    return test_counts.get(ideal_result, 0) / total_shots
