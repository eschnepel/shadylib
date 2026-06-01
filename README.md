# shadylib

Solar forecast correction library – HA-independent core for the [Shady](https://github.com/eschnepel/shady_solar_forecast) Home Assistant integration.

## Install

```bash
pip install shadylib
```

## Modules

| Module | Contents |
|---|---|
| `math_utils` | `r`, `r6`, `snap`, `parse_dt`, `aggregate_to_hours`, `wls2`, `wls2_origin_quad` |
| `models` | `build_bucket_models`, `predict`, `BucketKey`, `BucketModels` |
| `correction` | `apply_corrections` |

## Quick Start

```python
from shadylib import build_bucket_models, predict, apply_corrections

# Build per-5-min-bucket models from recorder statistics
models = build_bucket_models(fc_rows, pv_rows, algorithm="linear")

# Apply to raw forecast
corrected, per_string = apply_corrections(
    raw_forecast,
    fc_rows,
    {"sensor.pv_string_1": pv_rows},
    algorithm="linear",
)
```
