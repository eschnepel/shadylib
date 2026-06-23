"""correction.py – Applies bucket models to a raw forecast. No HA dependencies.

The apply_corrections() function takes a raw forecast dict and a set of
pre-fetched recorder statistics, builds per-string per-bucket models, and
returns both the combined corrected forecast and per-string forecasts.

The raw forecast from the Energy Manager uses arbitrary-interval timestamps.
Before prediction, the caller must normalise it to 5-minute Wh/slot values
using normalise_em_to_5min() so that prediction inputs match the scale on
which bucket models were trained (5-minute recorder means converted to
Wh/slot by to_wh_per_slot()).
"""

from __future__ import annotations

import logging
from datetime import datetime

from .math_utils import r, snap
from .models import (
    build_bucket_models,
    predict,
    BucketValue,
    BucketModels,
    InputHistory,
)

_LOGGER = logging.getLogger(__name__)


def apply_corrections(
    raw: dict[str, float],
    fc_rows: InputHistory,
    pv_sensors_rows: dict[str, InputHistory],
    algorithm: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, BucketModels]]:
    """Correct a raw forecast using per-string per-bucket models.

    Arguments:
        raw:              {ISO-ts: Wh/slot} – EM forecast pre-normalised to
                          5-minute slots by normalise_em_to_5min().
        fc_rows:          [{"start": datetime, "mean": float}] – forecast ref
                          (5-min recorder means, already in Wh/slot)
        pv_sensors_rows:  {entity_id: [{"start": datetime, "mean": float}]}
        algorithm:        "factor" | "linear" | "quadratic"

    Returns:
        (combined, string_forecasts, string_bucket_models)

        combined:             {ISO-ts: Wh} – sum of all string predictions
        string_forecasts:     {entity_id: {ISO-ts: Wh}} – per-string predictions
        string_bucket_models: {entity_id: BucketModels} – fitted models per string

    If all string models fail, returns (dict(raw), {}, {}).
    """
    combined: dict[str, float] = {}
    string_forecasts: dict[str, dict[str, float]] = {}
    string_bucket_models: dict[str, BucketModels] = {}

    for pv_sensor, pv_rows in pv_sensors_rows.items():
        models = build_bucket_models(fc_rows, pv_rows, algorithm)

        if not models:
            _LOGGER.warning(
                "No bucket models for %s"
                " (algorithm=%s, fc_rows=%d, pv_rows=%d)."
                " Enable DEBUG logging for shadylib.models to see root cause"
                " (timestamp mismatch, curtailment filter, or degenerate fit).",
                pv_sensor,
                algorithm,
                len(fc_rows),
                len(pv_rows),
            )
            continue

        _LOGGER.info(
            "Bucket models for %s: algorithm=%s  %d buckets fitted",
            pv_sensor,
            algorithm,
            len(models),
        )

        string_bucket_models[pv_sensor] = models
        string_slots = _predict_string(raw, models)
        string_forecasts[pv_sensor] = string_slots

        for ts, val in string_slots.items():
            combined[ts] = r(combined.get(ts, 0.0) + val)

    if not combined:
        _LOGGER.debug("All string models failed – falling back to raw forecast")
        return dict(raw), {}, {}

    return dict(sorted(combined.items())), string_forecasts, string_bucket_models


def _predict_string(
    raw: dict[str, float],
    models: BucketModels,
) -> dict[str, float]:
    """Apply bucket models to a single string for all 5-minute raw slots.

    raw must already be normalised to 5-minute Wh/slot values (via
    normalise_em_to_5min()) so each entry maps directly to one bucket.
    The model input and output are both in Wh/slot – no further scaling
    or sub-slot expansion is needed.
    """
    result: dict[str, float] = {}

    for iso_ts, raw_wh in raw.items():
        try:
            dt = datetime.fromisoformat(iso_ts)
        except ValueError:
            continue

        model = _nearest_model(models, dt.hour, snap(dt.minute))
        result[iso_ts] = r(max(0.0, predict(model, raw_wh))) if model else 0.0

    return result


def _nearest_model(
    models: BucketModels, hour: int, snapped_minute: int
) -> BucketValue | None:
    """Return the model for (hour, snapped_minute) or the nearest bucket
    within the same hour if an exact match doesn't exist."""
    exact = models.get((hour, snapped_minute))
    if exact is not None:
        return exact

    best_model: BucketValue | None = None
    best_dist = 60
    for (mh, mm), model in models.items():
        if mh == hour:
            dist = abs(mm - snapped_minute)
            if dist < best_dist:
                best_dist = dist
                best_model = model
    return best_model
