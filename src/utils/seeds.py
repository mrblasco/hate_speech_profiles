"""
Deterministic seed derivation.

A single top-level seed fans out into per-item seeds by hashing the seed
together with a stable item identifier. This means every generated item has
a unique, reproducible seed without requiring a stateful global RNG.
"""

from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, *keys: str | int) -> int:
    """
    Derive a child seed from a base seed and one or more string/int keys.
    The result is deterministic and independent of insertion order of calls.
    """
    payload = f"{base_seed}:{'|'.join(str(k) for k in keys)}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:16], 16) % (2**31)


def make_rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng
