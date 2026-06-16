# shadylib

Solar forecast correction library – HA-independent core for the [Shady](https://github.com/eschnepel/shady_solar_forecast) Home Assistant integration.

## Install

```bash
pip install shadylib
```

## Modules

| Module | Contents |
|---|---|
| `math_utils` | `r`, `r6`, `snap`, `parse_dt`, `aggregate_to_hours`, `normalise_em_to_5min`, `normalise_to_5min_day`, `wls2`, `wls2_origin_quad`, `BUCKET_MIN`, `PRECISION` |
| `models` | `build_bucket_models`, `predict`, `BucketKey`, `BucketValue`, `BucketModels`, `InputHistory`, `PV_MIN_W` |
| `correction` | `apply_corrections` |
| `effective` | `compute_effective_strings`, `split_combined_sensor` |

## Public API

```python
from shadylib import (
    # Math & utilities
    r, r6, snap, parse_dt, aggregate_to_hours,
    normalise_em_to_5min, normalise_to_5min_day,
    wls2, wls2_origin_quad, BUCKET_MIN, PRECISION,
    # Model types
    BucketKey, BucketValue, BucketModels, InputHistory, PV_MIN_W,
    # Model building & prediction
    build_bucket_models, predict,
    # Correction pipeline
    apply_corrections,
    # Effective power loss distribution
    compute_effective_strings, split_combined_sensor,
)
```

## Quick Start

```python
from shadylib import build_bucket_models, predict, apply_corrections, normalise_em_to_5min

# Build per-5-min-bucket models from recorder statistics
# fc_rows / pv_rows: list[{"start": datetime, "mean": float}]
# means must be in Wh/slot (e.g. W × 5/60 via to_wh_per_slot)
models = build_bucket_models(fc_rows, pv_rows, algorithm="linear")

# Normalise the Energy Manager forecast (arbitrary-interval timestamps)
# to a complete 5-minute Wh/slot raster before correction.
# em_forecast: dict[str, float] – {ISO-timestamp: Wh}, arbitrary resolution
raw_5min = normalise_em_to_5min(em_forecast)

# Apply corrections across all configured PV strings
corrected, per_string = apply_corrections(
    raw_5min,              # dict[str, float] – EM forecast normalised to 5-min Wh/slot
    fc_rows,               # list[{"start": datetime, "mean": float}] – fc_sensor history
    {
        "sensor.pv_string_1": pv_rows_1,
        "sensor.pv_string_2": pv_rows_2,
    },
    algorithm="linear",    # "factor" | "linear" | "quadratic"
)
# corrected:   dict[str, float] – summed corrected forecast (5-min Wh/slot)
# per_string:  dict[str, dict[str, float]] – per-string forecasts
```

## EM Forecast Normalisation

Energy Manager providers deliver forecasts at arbitrary intervals — hourly,
half-hourly, or at non-boundary minutes. `normalise_em_to_5min` distributes
each EM value pro-rata across the 5-minute slots it overlaps, including
partial boundary slots:

```python
from shadylib import normalise_em_to_5min

# Hourly EM provider: one value per hour
em = {
    "2025-06-15T06:00:00+00:00": 228.0,   # Wh, valid until 07:00
    "2025-06-15T07:00:00+00:00": 480.0,   # Wh, valid until 08:00
    "2025-06-15T08:00:00+00:00": 0.0,     # sentinel / next slot
}
slots = normalise_em_to_5min(em)
# → {"2025-06-15T06:00:00+00:00": 19.0,
#    "2025-06-15T06:05:00+00:00": 19.0,
#    ...  (12 slots à 19.0 Wh for the 06:xx hour)
#    "2025-06-15T07:00:00+00:00": 40.0,
#    ...  (12 slots à 40.0 Wh for the 07:xx hour) }
```

The last entry has no defined end and is assigned only to its own 5-minute
slot. Callers should append a sentinel entry (e.g. next day midnight at 0 Wh)
to ensure the last real forecast slot is fully expanded.

## Effective String Computation

`compute_effective_strings` distributes system-level losses (grid import/export,
battery import/export) across PV strings using a waterfall cascade. This allows
the correction models to be trained on loss-corrected data rather than raw
inverter output.

```python
from shadylib import compute_effective_strings, split_combined_sensor

effective = compute_effective_strings(
    pv_values={"sensor.string_1": 800.0, "sensor.string_2": 600.0},
    grid_import=0.0,
    grid_export=150.0,
    battery_import=200.0,
    battery_export=0.0,
)
# effective: dict[str, float] – loss-adjusted power per string
```
