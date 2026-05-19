"""
Policy-mode configuration: load explicit policy stances from YAML and
build PolicyCondition objects that flow through the standard pipeline.

Usage
-----
    python src/main.py --policies configs/policies.yaml --n_profiles 12

Each policy in the YAML defines the exact stance for the post and the
opposing stance that hate-speech comments will attack. The rest of the
demographics (age_group, gender, religion, country_of_origin) are
sampled balanced across policies.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.sampling import Condition
from src.utils.io import load_yaml
from src.utils.seeds import make_rng

log = logging.getLogger("pipeline.policies")


@dataclass
class PolicyCondition(Condition):
    """
    A Condition extended with explicit policy stance information.
    Inherits all demographic fields from Condition; policy fields are additive.
    """
    policy_id:             str = ""
    post_stance:           str = ""
    opposing_stance:       str = ""
    target_group_override: str = ""  # overrides TARGET_GROUPS lookup in posts.py


def load_policies(policies_path: Path) -> list[dict]:
    """Load and return only enabled policies from a policies.yaml file."""
    raw = load_yaml(policies_path)
    all_policies = raw.get("policies", [])
    enabled = [p for p in all_policies if p.get("enabled", True)]
    log.info("Loaded %d / %d enabled policies from %s",
             len(enabled), len(all_policies), policies_path)
    return enabled


def build_policy_conditions(
    policies: list[dict],
    age_groups: list[str],
    genders: list[str],
    religions: list[str],
    countries_of_origin: list[str],
    n_profiles: int,
    seed: int,
) -> list[PolicyCondition]:
    """
    Create PolicyCondition objects distributed across all enabled policies.

    For each policy, topic and values are fixed; age_group, gender, religion,
    and country_of_origin are balanced across profiles. n_profiles is shared
    evenly among policies (extra profiles go to earlier policies).

    Profile IDs use format POL_{POLICY_ID_UPPER}_{n:04d} to distinguish
    from standard P0001 IDs in merged outputs.
    """
    if not policies:
        raise ValueError("No enabled policies found in policies.yaml")

    rng = make_rng(seed)
    demo_combos = list(itertools.product(
        age_groups, genders, religions, countries_of_origin
    ))
    if not demo_combos:
        raise ValueError("No demographic combinations: check study_config.yaml design fields")

    n_policies = len(policies)
    base_per_policy = n_profiles // n_policies
    extra = n_profiles % n_policies

    log.info(
        "Policy mode: %d policies, %d profiles total (%d base + up to 1 extra per policy)",
        n_policies, n_profiles, base_per_policy,
    )

    all_conditions: list[PolicyCondition] = []
    global_idx = 0

    for i, policy in enumerate(policies):
        count = base_per_policy + (1 if i < extra else 0)
        if count == 0:
            continue

        n_reps = math.ceil(count / len(demo_combos))
        pool = (demo_combos * n_reps)[:count]
        shuffled = rng.sample(pool, len(pool))

        policy_id    = policy["id"]
        topic        = policy["topic"]
        values       = policy.get("values", "progressive")
        post_stance  = policy["post_stance"]
        opp_stance   = policy["opposing_stance"]
        tg_override  = policy.get("target_group", "")

        log.info(
            "  Policy %-30s  topic=%-15s values=%-12s  n=%d",
            policy_id, topic, values, count,
        )

        for age_group, gender, religion, country_of_origin in shuffled:
            global_idx += 1
            profile_id = f"POL_{policy_id.upper()}_{global_idx:04d}"
            all_conditions.append(PolicyCondition(
                profile_id=profile_id,
                topic=topic,
                age_group=age_group,
                gender=gender,
                values=values,
                religion=religion,
                country_of_origin=country_of_origin,
                policy_id=policy_id,
                post_stance=post_stance,
                opposing_stance=opp_stance,
                target_group_override=tg_override,
            ))

    return all_conditions
