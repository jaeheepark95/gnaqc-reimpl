"""Hellinger fidelity computation for GNAQC.

Implements Equation (3) from LeCompte et al., IEEE TQE 2023.

    F = 1 - H(p_GT, p_T)

where H is the Hellinger distance:

    H = (1/sqrt(2)) * sqrt(sum_i (sqrt(p_GT_i) - sqrt(p_T_i))^2)

This is NOT the same as PST (Probability of Successful Trial), which measures
the probability of a single correct bitstring. Hellinger fidelity compares
the *entire* output distribution against the ideal ground truth.
"""

from __future__ import annotations

import math


def compute_hellinger_fidelity(
    ground_truth_counts: dict[str, int],
    test_counts: dict[str, int],
) -> float:
    """Compute Hellinger fidelity between two count distributions.

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
