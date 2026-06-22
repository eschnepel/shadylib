"""effective.py – Power loss distribution across PV strings. No HA dependencies.

Computes the effective (net-of-losses) power per PV string given raw PV string
readings and system-level net import/export scalars.

Algorithm
---------
1. Compute total system loss::

       total_loss = max(0, sum(non-zero PV inputs) + net_import − net_export)

   where:
   - ``net_import``  is the scalar sum of all import sensors (power entering
     the BMS: grid draw, battery discharge).
   - ``net_export``  is the scalar sum of all export sensors (power leaving
     the BMS: grid feed-in, battery charge).

2. Distribute ``total_loss`` across non-zero PV strings using a
   **waterfall/cascade** algorithm in ascending order of PV value:

   - Compute ``fair_share = remaining_loss / active_string_count``.
   - For each string (smallest first): ``effective = max(0, pv − fair_share)``.
   - ``absorbed = pv − effective`` (actual loss taken).
   - Subtract ``absorbed`` from ``remaining_loss`` and remove the string from
     the active pool.
   - Repeat until all strings are processed.

   This ensures that strings that cannot absorb their fair share pass the
   remainder on to larger strings.

3. Floor at zero: effective values are never negative.
   If ``total_loss ≥ sum(all non-zero PV strings)``, all effective values are 0.

Public API
----------
    from shadylib import compute_effective_strings
"""

from __future__ import annotations


def compute_effective_strings(
    pv_values: list[float],
    *,
    net_import: float = 0.0,
    net_export: float = 0.0,
) -> list[float]:
    """Distribute system-level power losses across PV strings.

    Args:
        pv_values:   Raw PV readings per string (W or Wh, consistent unit).
                     Zero and negative values are treated as inactive strings.
        net_import:  Scalar sum of all import sensors — power entering the BMS
                     (grid draw, battery discharge).  Must be ≥ 0.
        net_export:  Scalar sum of all export sensors — power leaving the BMS
                     (grid feed-in, battery charge).  Must be ≥ 0.

    Returns:
        List of effective power values (same length and order as ``pv_values``).
        All values are ≥ 0.  Strings that were 0 in the input remain 0.
    """
    if not pv_values:
        return []

    # --- step 1: total loss ---
    pv_nonzero = [v for v in pv_values if v > 0.0]
    pv_sum = sum(pv_nonzero)

    total_loss = max(0.0, pv_sum + net_import - net_export)

    # No loss: return originals, floored at 0
    if total_loss <= 0.0:
        return [max(0.0, v) for v in pv_values]

    # All strings wiped out
    if total_loss >= pv_sum:
        return [0.0] * len(pv_values)

    # --- step 2: waterfall/cascade ---
    indexed = [(i, v) for i, v in enumerate(pv_values) if v > 0.0]
    indexed.sort(key=lambda x: x[1])

    effective_map: dict[int, float] = {}
    remaining_loss = total_loss

    for pos, (idx, val) in enumerate(indexed):
        active_count = len(indexed) - pos
        fair_share = remaining_loss / active_count
        eff = max(0.0, val - fair_share)
        absorbed = val - eff
        remaining_loss -= absorbed
        effective_map[idx] = eff

    # Build result in original order
    result: list[float] = []
    for i, v in enumerate(pv_values):
        if v > 0.0:
            result.append(effective_map.get(i, 0.0))
        else:
            result.append(0.0)

    return result
