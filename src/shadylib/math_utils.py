"""math_utils.py – Pure-Python helpers. No external dependencies.

Contains:
  - Output precision helpers (r, r6)
  - Datetime parsing (parse_dt)
  - 5-min bucket snapping (snap)
  - Hourly aggregation (aggregate_to_hours)
  - EM forecast normalisation (normalise_em_to_5min)
  - WLS solvers (wls2, wls2_origin_quad)
  - Recorder data quality filters (enforce_monotonic, filter_gap_successors)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

PRECISION = 2  # decimal places for all Wh output values
BUCKET_MIN = 5  # minutes per bucket

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def r(v: float) -> float:
    """Round to standard output precision (2 decimal places)."""
    return round(v, PRECISION)


def r6(v: float) -> float:
    """Round model coefficients to 6 decimal places."""
    return round(v, 6)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def parse_dt(iso_str: str) -> datetime:
    """Parse an ISO-8601 string to datetime. Returns datetime.min on failure."""
    try:
        return datetime.fromisoformat(iso_str)
    except ValueError:
        return datetime.min.replace(tzinfo=_UTC)


def snap(minute: int) -> int:
    """Round a minute value down to the nearest 5-minute boundary."""
    return (minute // BUCKET_MIN) * BUCKET_MIN


def aggregate_to_hours(slots: dict[str, float]) -> dict[str, float]:
    """Sum sub-hourly slots into full-hour buckets.

    Key is the ISO string of the hour's start (minute=0, second=0).
    Invalid timestamps are silently skipped.
    """
    hourly: dict[str, float] = {}
    for ts, wh in slots.items():
        try:
            dt = datetime.fromisoformat(ts)
            key = dt.replace(minute=0, second=0, microsecond=0).isoformat()
        except ValueError:
            continue
        hourly[key] = r(hourly.get(key, 0.0) + wh)
    return dict(sorted(hourly.items()))


def normalise_em_to_5min(
    em_slots: dict[str, float],
) -> dict[str, float]:
    """Distribute arbitrary-interval Energy Manager values into 5-minute slots.

    The EM delivers forecast values at irregular timestamps that reflect when
    the provider changes its value, not 5-minute boundaries.  Each EM value is
    valid from its own timestamp until the next EM timestamp (or indefinitely
    for the last entry).

    Each EM interval is distributed pro-rata across the 5-minute slots it
    overlaps.  Boundary slots receive a proportional fraction based on the
    overlap duration within the slot.

    Args:
        em_slots: {ISO-timestamp: Wh} – raw EM forecast, arbitrary resolution.
                  Values are already in Wh/slot-equivalent after unit
                  conversion by the caller (to_wh_per_slot).

    Returns:
        {ISO-timestamp: Wh} – one entry per 5-minute slot that receives any
        contribution.  Keys are ISO strings with second=0, microsecond=0 at
        the slot boundary (minute snapped down to nearest 5).  Slots with zero
        contribution are omitted.

    Notes:
        - Timezone information is preserved from the input timestamps.
        - If em_slots is empty, an empty dict is returned.
        - The last EM entry has no defined end; it is assigned to exactly the
          one 5-minute slot that contains its timestamp (no forward propagation
          beyond the known forecast horizon).
    """
    if not em_slots:
        return {}

    # Parse and sort entries
    parsed: list[tuple[datetime, float]] = []
    for iso_ts, wh in em_slots.items():
        try:
            dt = datetime.fromisoformat(iso_ts)
        except ValueError:
            continue
        parsed.append((dt, wh))

    if not parsed:
        return {}

    parsed.sort(key=lambda x: x[0])

    _5min = timedelta(minutes=BUCKET_MIN)
    result: dict[str, float] = {}

    def _slot_start(dt: datetime) -> datetime:
        """Floor dt to the nearest 5-minute boundary, preserving tzinfo."""
        return dt.replace(
            minute=(dt.minute // BUCKET_MIN) * BUCKET_MIN,
            second=0,
            microsecond=0,
        )

    def _add(slot_dt: datetime, wh: float) -> None:
        key = slot_dt.isoformat()
        result[key] = round(result.get(key, 0.0) + wh, PRECISION)

    for i, (start_dt, wh) in enumerate(parsed):
        if i + 1 < len(parsed):
            end_dt = parsed[i + 1][0]
        else:
            # Last entry: assign only to its own 5-minute slot
            _add(_slot_start(start_dt), wh)
            continue

        interval_secs = (end_dt - start_dt).total_seconds()
        if interval_secs <= 0:
            continue

        # Walk through all 5-minute slots that overlap [start_dt, end_dt)
        slot = _slot_start(start_dt)
        while slot < end_dt:
            slot_end = slot + _5min
            # Overlap of this EM interval with this slot
            overlap_start = max(start_dt, slot)
            overlap_end = min(end_dt, slot_end)
            overlap_secs = (overlap_end - overlap_start).total_seconds()
            if overlap_secs > 0:
                fraction = overlap_secs / interval_secs
                _add(slot, wh * fraction)
            slot = slot_end

    return dict(sorted(result.items()))


def normalise_to_5min_day(
    slots: dict[str, float],
    day_start: datetime,
) -> dict[str, float]:
    """Return a complete 288-slot dict for *day_start*'s calendar day.

    All timestamps in *slots* that fall on that day are snapped to the
    nearest 5-minute boundary (floor) and accumulated.  Every slot for
    the full 24 hours is present in the output; slots with no data are
    set to 0.0.

    This normalises away sub-5-min timestamps (e.g. 21:12:46) that some
    forecast providers emit, and fills night-time gaps so consumers always
    receive a complete, uniform series.

    Output keys are formatted as ``YYYY-MM-DDTHH:MM`` (no seconds, no UTC
    offset) to keep HA state-attribute serialisation well below the 16 kB
    per-attribute limit.

    Args:
        slots:     {ISO-timestamp: value} – any resolution, any timezone
        day_start: start of the target calendar day (must be tz-aware,
                   minute=0, second=0)

    Returns:
        Ordered dict of 288 entries covering 00:00–23:55 of day_start's day.
        Keys are ``YYYY-MM-DDTHH:MM`` strings in the timezone of *day_start*.
    """
    from datetime import timedelta, timezone

    day_end = day_start + timedelta(days=1)
    tz = day_start.tzinfo or timezone.utc

    # Build a zero-filled skeleton for the entire day.
    result: dict[str, float] = {}
    t = day_start
    while t < day_end:
        result[t.strftime("%Y-%m-%dT%H:%M")] = 0.0
        t += timedelta(minutes=BUCKET_MIN)

    # Accumulate incoming values into the correct 5-min bucket.
    for ts, wh in slots.items():
        try:
            dt_val = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=timezone.utc)
        dt_val = dt_val.astimezone(tz)

        if not (day_start <= dt_val < day_end):
            continue

        snapped = dt_val.replace(
            minute=(dt_val.minute // BUCKET_MIN) * BUCKET_MIN,
            second=0,
            microsecond=0,
        )
        key = snapped.strftime("%Y-%m-%dT%H:%M")
        if key in result:
            result[key] = round(result[key] + wh, PRECISION)

    return result


# ---------------------------------------------------------------------------
# WLS solvers
# ---------------------------------------------------------------------------


def wls2(
    xs: list[float], ys: list[float], ws: list[float]
) -> tuple[float, float] | None:
    """Weighted least squares linear regression: y ~ slope*x + intercept.

    Returns (slope, intercept) or None if the system is degenerate
    (e.g. zero total weight, all-same x values).
    """
    sw = sum(ws)
    if sw == 0:
        return None
    swx = sum(w * x for w, x in zip(ws, xs))
    swy = sum(w * y for w, y in zip(ws, ys))
    swxx = sum(w * x * x for w, x in zip(ws, xs))
    swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * swxx - swx**2
    if abs(denom) < 1e-12:
        return None
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    return slope, intercept


def wls2_origin_quad(
    xs: list[float], ys: list[float], ws: list[float]
) -> tuple[float, float] | None:
    """WLS quadratic through the origin: y ~ a*x² + b*x  (no free intercept).

    Fixing the intercept to zero is physically correct for solar correction
    (fc=0 → pv=0) and prevents the model from memorising the training mean
    as a constant offset.

    Solves the 2×2 normal equations:
      [Σw·x⁴  Σw·x³] [a]   [Σw·x²·y]
      [Σw·x³  Σw·x²] [b] = [Σw·x·y  ]

    Returns (a, b) or None if the system is degenerate.
    """
    swx2 = sum(w * x**2 for w, x in zip(ws, xs))
    swx3 = sum(w * x**3 for w, x in zip(ws, xs))
    swx4 = sum(w * x**4 for w, x in zip(ws, xs))
    swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    swx2y = sum(w * x**2 * y for w, x, y in zip(ws, xs, ys))

    det = swx4 * swx2 - swx3**2
    if abs(det) < 1e-12:
        return None

    a = (swx2y * swx2 - swxy * swx3) / det
    b = (swxy * swx4 - swx2y * swx3) / det
    return a, b


# ---------------------------------------------------------------------------
# Recorder data quality filters
# ---------------------------------------------------------------------------


def enforce_monotonic(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Discard rows that violate strictly non-decreasing mean values (f(n) ≤ f(n+1)).

    The HA recorder accumulates energy as a running total.  After a backup
    restore the counter may reset to a lower value, producing rows where
    ``mean[n] > mean[n+1]``.  Any such row – and every subsequent row until
    the series becomes monotonic again – is discarded.

    The check operates on the **raw** (not gap-filtered) series so that it
    can be applied before :func:`filter_gap_successors`.

    Args:
        rows: Sorted list of ``{"start": datetime, "mean": float}`` dicts.
              The list is not required to be sorted by this function, but
              the monotonicity check is only meaningful when it is.

    Returns:
        A new list containing only the rows that preserve f(n) ≤ f(n+1).
        The input list is not mutated.  Rows are returned in the original
        order.  An empty input or a single-row input is returned unchanged.
    """
    if len(rows) < 2:
        return list(rows)

    result: list[dict[str, Any]] = [rows[0]]
    last_valid_mean: float = float(rows[0]["mean"])

    for row in rows[1:]:
        mean: float = float(row["mean"])
        if mean >= last_valid_mean:
            result.append(row)
            last_valid_mean = mean
        # else: discard – counter reset detected

    return result


def filter_gap_successors(
    rows: list[dict[str, Any]],
    slot_minutes: int = 5,
) -> list[dict[str, Any]]:
    """Remove the direct successor of any gap larger than one slot interval.

    A gap is detected when the difference between two consecutive ``start``
    timestamps exceeds ``slot_minutes``.  The sample immediately following
    the gap is dropped because the HA recorder may have accumulated all
    missing values into it, producing an artificially inflated reading.

    Args:
        rows:         Sorted list of ``{"start": datetime, "mean": float}``
                      dicts.  Rows must be sorted by ``start`` in ascending
                      order.
        slot_minutes: Expected interval between samples (default 5).

    Returns:
        A new list with gap-successor rows removed.  The input list is not
        mutated.  An empty input or a single-row input is returned unchanged.
    """
    if len(rows) < 2:
        return list(rows)

    threshold = timedelta(minutes=slot_minutes)
    result: list[dict[str, Any]] = [rows[0]]

    for prev, curr in zip(rows, rows[1:]):
        prev_start = prev["start"]
        curr_start = curr["start"]
        if not isinstance(prev_start, datetime):
            prev_start = datetime.fromisoformat(str(prev_start))
        if not isinstance(curr_start, datetime):
            curr_start = datetime.fromisoformat(str(curr_start))
        gap = curr_start - prev_start
        if gap > threshold:
            continue  # discard gap successor
        result.append(curr)

    return result
