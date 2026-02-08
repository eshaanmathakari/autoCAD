"""
Score fusion for multi-signal reference matching.

Combines similarity scores from multiple embedding backends
using weighted sum or reciprocal rank fusion.
"""


def fuse_scores(
    scores_a: list[float],
    scores_b: list[float],
    weight_a: float = 0.7,
    weight_b: float = 0.3,
) -> list[float]:
    """Weighted sum fusion of two score lists."""
    return [a * weight_a + b * weight_b for a, b in zip(scores_a, scores_b)]


def reciprocal_rank_fusion(
    ranks_list: list[list[int]],
    k: int = 60,
) -> list[float]:
    """
    Reciprocal Rank Fusion (RRF): score = sum(1/(k + rank_i)).

    Parameter-free alternative to weighted sum. Robust across
    different score distributions.
    """
    if not ranks_list:
        return []
    n_items = len(ranks_list[0])
    scores = [0.0] * n_items
    for ranks in ranks_list:
        for idx, rank in enumerate(ranks):
            scores[idx] += 1.0 / (k + rank)
    return scores
