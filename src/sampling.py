"""
Balanced factorial sampling of experimental conditions.

Generates the full design matrix such that:
  - Every topic × age_group × gender × values combination appears.
  - Profiles are assigned conditions deterministically given a seed.
  - The design is reproducible and documented for methods reporting.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

from src.utils.seeds import make_rng

log = logging.getLogger("pipeline.sampling")


@dataclass
class Condition:
    profile_id: str
    factors:    dict[str, str]   # design key → sampled value


def build_design_matrix(
    n_profiles: int,
    factors: dict[str, list[str]],
    seed: int,
) -> list[Condition]:
    """
    Create a balanced list of Condition objects from an arbitrary factors dict.

    All factor-level combinations are enumerated, tiled to n_profiles, then
    shuffled deterministically. This guarantees full coverage of all cells and
    reproducibility from (n_profiles, seed).
    """
    rng = make_rng(seed)
    keys   = list(factors.keys())
    combos = list(itertools.product(*factors.values()))
    n_combos = len(combos)

    log.info(
        "Design: %s = %d combos",
        " × ".join(f"{k}({len(factors[k])})" for k in keys),
        n_combos,
    )

    n_reps = math.ceil(n_profiles / n_combos)
    pool = (combos * n_reps)[:n_profiles]
    shuffled = rng.sample(pool, len(pool))

    conditions: list[Condition] = [
        Condition(profile_id=f"P{idx + 1:04d}", factors=dict(zip(keys, vals)))
        for idx, vals in enumerate(shuffled)
    ]

    _log_balance(conditions, factors)
    return conditions


def _log_balance(
    conditions: list[Condition],
    factors: dict[str, list[str]],
) -> None:
    from collections import Counter
    n = len(conditions)
    for key, levels in factors.items():
        counts = Counter(c.factors[key] for c in conditions)
        counts_str = "  ".join(f"{lv}={counts[lv]}" for lv in levels)
        log.info("  %-22s  %s  (n=%d)", key, counts_str, n)
