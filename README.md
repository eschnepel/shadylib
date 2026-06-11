# shadylib

Solar forecast correction library – HA-independent core for the [Shady](https://github.com/eschnepel/shady_solar_forecast) Home Assistant integration.

## Install

```bash
pip install shadylib
```

## Modules

| Module | Contents |
|---|---|
| `math_utils` | `r`, `r6`, `snap`, `parse_dt`, `aggregate_to_hours`, `normalise_to_5min_day`, `wls2`, `wls2_origin_quad`, `BUCKET_MIN`, `PRECISION` |
| `models` | `build_bucket_models`, `predict`, `BucketKey`, `BucketValue`, `BucketModels`, `InputHistory`, `PV_MIN_W` |
| `correction` | `apply_corrections` |
| `effective` | `compute_effective_strings`, `split_combined_sensor` |

## Public API

```python
from shadylib import (
    # Math & utilities
    r, r6, snap, parse_dt, aggregate_to_hours, normalise_to_5min_day,
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
from shadylib import build_bucket_models, predict, apply_corrections

# fc_rows / pv_rows: list of (datetime, float) tuples in Wh/slot
# raw_forecast: dict[str, float] – {ISO-timestamp: Wh} at any provider resolution
corrected, per_string = apply_corrections(
    raw_forecast,
    fc_rows,
    {
        "sensor.pv_string_1": pv_rows_1,
        "sensor.pv_string_2": pv_rows_2,
    },
    algorithm="linear",    # "factor" | "linear" | "quadratic"
)
# corrected:   dict[str, float] – summed corrected forecast, same timestamps as input
# per_string:  dict[str, dict[str, float]] – per-string forecasts
```

Each slot in `raw_forecast` is matched to its exact 5-min bucket model and corrected individually. The output retains the input's timestamps and resolution unchanged.

## Effective String Computation

`compute_effective_strings` distributes system-level losses (grid import/export,
battery import/export) across PV strings using a waterfall cascade. This allows
the correction models to be trained on loss-corrected data rather than raw
inverter output.

```python
from shadylib import compute_effective_strings

effective = compute_effective_strings(
    pv_values={"sensor.string_1": 800.0, "sensor.string_2": 600.0},
    grid_import=0.0,
    grid_export=150.0,
    battery_import=200.0,
    battery_export=0.0,
)
# effective: dict[str, float] – loss-adjusted power per string
```
