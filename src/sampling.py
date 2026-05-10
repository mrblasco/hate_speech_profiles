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
    profile_id:        str
    topic:             str
    age_group:         str
    gender:            str
    values:            str
    religion:          str
    country_of_origin: str


def build_design_matrix(
    n_profiles: int,
    topics: list[str],
    age_groups: list[str],
    genders: list[str],
    values: list[str],
    religions: list[str],
    countries_of_origin: list[str],
    seed: int,
) -> list[Condition]:
    """
    Create a balanced list of Condition objects.

    Strategy
    --------
    All combinations of (topic, age_group, gender, values, religion,
    country_of_origin) are enumerated, tiled to reach at least n_profiles
    rows, then shuffled deterministically. The first n_profiles rows are
    returned.

    This guarantees:
      • All factor combinations appear.
      • Marginal distributions are as balanced as possible.
      • The design is fully reproducible from (n_profiles, seed).
    """
    rng = make_rng(seed)
    combos = list(itertools.product(
        topics, age_groups, genders, values, religions, countries_of_origin
    ))
    n_combos = len(combos)

    log.info(
        "Design: %d topics × %d age_groups × %d genders × %d values"
        " × %d religions × %d countries = %d combos",
        len(topics), len(age_groups), len(genders), len(values),
        len(religions), len(countries_of_origin), n_combos,
    )

    n_reps = math.ceil(n_profiles / n_combos)
    pool = (combos * n_reps)[:n_profiles]
    shuffled = rng.sample(pool, len(pool))

    conditions: list[Condition] = []
    for idx, (topic, age_group, gender, values_, religion, country_of_origin) in enumerate(shuffled):
        profile_id = f"P{idx + 1:04d}"
        conditions.append(Condition(
            profile_id=profile_id,
            topic=topic,
            age_group=age_group,
            gender=gender,
            values=values_,
            religion=religion,
            country_of_origin=country_of_origin,
        ))

    _log_balance(conditions, topics, age_groups, genders, values, religions, countries_of_origin)
    return conditions


def _log_balance(
    conditions: list[Condition],
    topics: list[str],
    age_groups: list[str],
    genders: list[str],
    values: list[str],
    religions: list[str],
    countries_of_origin: list[str],
) -> None:
    from collections import Counter
    n = len(conditions)
    for attr, levels in [
        ("topic",             topics),
        ("age_group",         age_groups),
        ("gender",            genders),
        ("values",            values),
        ("religion",          religions),
        ("country_of_origin", countries_of_origin),
    ]:
        counts = Counter(getattr(c, attr) for c in conditions)
        counts_str = "  ".join(f"{lv}={counts[lv]}" for lv in levels)
        log.info("  %-18s  %s  (n=%d)", attr, counts_str, n)
