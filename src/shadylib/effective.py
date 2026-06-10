"""effective.py – Power loss distribution across PV strings. No HA dependencies.

Computes the effective (net-of-losses) power per PV string given raw PV string
readings and system-level input/output power measurements.

Algorithm
---------
1. Extract signed contributions from system sensors:
   - ``grid_import``   and ``battery_export`` contribute to the *input* bucket
     (positive values only; negative values are ignored – they belong to the
     opposite channel).
   - ``grid_export``   and ``battery_import`` contribute to the *output* bucket
     (positive values only).

2. Compute total system loss::

       total_loss = sum(non-zero PV inputs)
                   + sum(system inputs)
                   − sum(system outputs)

3. Distribute ``total_loss`` across non-zero PV strings using a
   **waterfall/cascade** algorithm in ascending order of PV value:

   - Compute ``fair_share = remaining_loss / active_string_count``.
   - For each string (smallest first): ``effective = max(0, pv − fair_share)``.
   - ``absorbed = pv − effective`` (actual loss taken).
   - Subtract ``absorbed`` from ``remaining_loss`` and remove the string from
     the active pool.
   - Repeat until all strings are processed.

   This ensures that strings that cannot absorb their fair share pass the
   remainder on to larger strings.

4. Floor at zero: effective values are never negative.
   If ``total_loss ≥ sum(all non-zero PV strings)``, all effective values are 0.

Public API
----------
    from shadylib import compute_effective_strings, split_combined_sensor
"""

from __future__ import annotations


def split_combined_sensor(value: float) -> tuple[float, float]:
    """Split a bidirectional sensor reading into (input_part, output_part).

    Positive values → input, negative values (absolute) → output.

    Args:
        value: Current sensor reading (positive = into system, negative = out).

    Returns:
        ``(input_part, output_part)`` both ≥ 0.
    """
    if value >= 0.0:
        return value, 0.0
    return 0.0, -value


def compute_effective_strings(
    pv_values: list[float],
    *,
    grid_import: float = 0.0,
    grid_export: float = 0.0,
    battery_import: float = 0.0,
    battery_export: float = 0.0,
) -> list[float]:
    """Distribute system-level power losses across PV strings.

    Each system sensor parameter should already be split into its directional
    part (i.e. already ≥ 0).  Use :func:`split_combined_sensor` when a single
    bidirectional sensor covers both directions.

    Args:
        pv_values:      Raw PV readings per string (W or Wh, consistent unit).
                        Zero and negative values are treated as inactive strings.
        grid_import:    Power drawn *from* the grid (≥ 0, contribution to inputs).
        grid_export:    Power pushed *to* the grid (≥ 0, contribution to outputs).
        battery_import: Power used to charge the battery (≥ 0, output).
        battery_export: Power discharged from the battery (≥ 0, input).

    Returns:
        List of effective power values (same length and order as ``pv_values``).
        All values are ≥ 0.  Strings that were 0 in the input remain 0.
    """
    if not pv_values:
        return []

    # --- step 1: system I/O buckets (already directional, floor at 0) ---
    sys_input = max(0.0, grid_import) + max(0.0, battery_export)
    sys_output = max(0.0, grid_export) + max(0.0, battery_import)

    # --- step 2: total loss ---
    pv_nonzero = [v for v in pv_values if v > 0.0]
    pv_sum = sum(pv_nonzero)

    total_loss = pv_sum + sys_input - sys_output

    # No loss (or gain): return originals, floored at 0
    if total_loss <= 0.0:
        return [max(0.0, v) for v in pv_values]

    # All strings wiped out
    if total_loss >= pv_sum:
        return [0.0] * len(pv_values)

    # --- step 3: waterfall/cascade ---
    # Work on (original_index, value) pairs, sorted ascending by value
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
