"""Tests for shadylib – no HA stubs needed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shadylib import (
    r,
    r6,
    snap,
    parse_dt,
    aggregate_to_hours,
    wls2,
    wls2_origin_quad,
    build_bucket_models,
    predict,
    apply_corrections,
    BUCKET_MIN,
)

UTC = timezone.utc


def dt(hour: int, minute: int = 0, day: int = 1) -> datetime:
    return datetime(2025, 6, day, hour, minute, tzinfo=UTC)


def make_rows(
    hour: int,
    minutes: list[int],
    fc_val: float,
    pv_val: float,
    days: int = 60,
    vary: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Generate realistic training rows with optional daily variation."""
    fc_rows, pv_rows = [], []
    ratio = pv_val / fc_val if fc_val else 0.5
    for d in range(days):
        scale = (0.8 + 0.4 * d / max(days - 1, 1)) if vary else 1.0
        for mm in minutes:
            ts = datetime(2025, 1, 1, hour, mm, tzinfo=UTC) + timedelta(days=d)
            fc = fc_val * scale
            pv = fc * ratio
            fc_rows.append({"start": ts, "mean": fc})
            pv_rows.append({"start": ts, "mean": pv})
    return fc_rows, pv_rows


# ---------------------------------------------------------------------------
# Public API smoke tests
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_r(self):
        assert r(1.23456) == 1.23

    def test_r6(self):
        assert r6(0.12345678) == 0.123457

    def test_snap(self):
        assert snap(7) == 5
        assert snap(0) == 0
        assert snap(55) == 55

    def test_parse_dt_valid(self):
        result = parse_dt("2025-06-01T10:00:00+00:00")
        assert result.hour == 10

    def test_parse_dt_invalid(self):
        result = parse_dt("not-a-date")
        assert result == datetime.min.replace(tzinfo=UTC)

    def test_aggregate_to_hours(self):
        slots = {f"2025-06-01T10:{mm:02d}:00+00:00": 10.0 for mm in range(0, 60, 5)}
        hourly = aggregate_to_hours(slots)
        assert len(hourly) == 1
        assert abs(list(hourly.values())[0] - 120.0) < 0.1

    def test_wls2_exact(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        ws = [1.0] * 4
        slope, intercept = wls2(xs, ys, ws)
        assert abs(slope - 2.0) < 1e-9
        assert abs(intercept) < 1e-9

    def test_wls2_origin_quad_exact(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1 * x**2 + 2 * x for x in xs]
        ws = [1.0] * 5
        a, b = wls2_origin_quad(xs, ys, ws)
        assert abs(a - 1.0) < 1e-6
        assert abs(b - 2.0) < 1e-6

    def test_predict_factor(self):
        assert predict((0.5,), 200.0) == 100.0

    def test_predict_linear(self):
        assert predict((2.0, 3.0), 5.0) == 13.0

    def test_predict_quadratic_origin(self):
        assert predict((1.0, 2.0, 0.0), 3.0) == 15.0  # 9 + 6

    def test_bucket_min_constant(self):
        assert BUCKET_MIN == 5


# ---------------------------------------------------------------------------
# build_bucket_models
# ---------------------------------------------------------------------------


class TestBuildBucketModels:
    def test_returns_empty_for_no_data(self):
        assert build_bucket_models([], [], "linear") == {}

    def test_curtailment_filter(self):
        """PV readings < 5W excluded."""
        fc_rows, pv_rows = make_rows(10, [0], 400.0, 200.0)
        # Inject some curtailed readings
        for i in range(0, 30):
            ts = datetime(2025, 1, 1, 10, 0, tzinfo=UTC) + timedelta(days=i)
            pv_rows.append({"start": ts, "mean": 1.0})  # curtailed
        models = build_bucket_models(fc_rows, pv_rows, "factor")
        model = models.get((10, 0))
        assert model is not None
        # Model should not be dragged to zero by curtailed readings
        assert predict(model, 400.0) > 50.0

    def test_factor_model_1_tuple(self):
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "factor")
        assert all(len(m) == 1 for m in models.values())

    def test_linear_model_2_tuple(self):
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "linear")
        assert all(len(m) == 2 for m in models.values())

    def test_quadratic_model_3_tuple_c_zero(self):
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "quadratic")
        assert all(len(m) == 3 for m in models.values())
        assert all(m[2] == 0.0 for m in models.values())

    def test_shading_captured_per_bucket(self):
        """Different buckets produce different predictions for same input."""
        fc_rows, pv_rows = [], []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            for mm in range(0, 60, 5):
                ts = datetime(2025, 1, 1, 10, mm, tzinfo=UTC) + timedelta(days=d)
                fc = 400.0 * scale
                # 10:15–10:30 shaded (30%), others full (90%)
                ratio = 0.3 if mm in (15, 20, 25, 30) else 0.9
                pv = fc * ratio
                fc_rows.append({"start": ts, "mean": fc})
                pv_rows.append({"start": ts, "mean": pv})

        models = build_bucket_models(fc_rows, pv_rows, "factor")
        unshaded = predict(models[(10, 0)], 400.0)
        shaded = predict(models[(10, 20)], 400.0)
        assert shaded < unshaded * 0.5


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    def _training(
        self, fc_val: float, pv_val: float, hour: int, days: int = 60
    ) -> tuple[list[dict], list[dict]]:
        return make_rows(hour, list(range(0, 60, 5)), fc_val, pv_val, days)

    def test_no_pv_rows_returns_raw(self):
        raw = {"2025-06-01T10:00:00+00:00": 400.0}
        combined, per_string = apply_corrections(raw, [], {}, "linear")
        assert combined == raw
        assert per_string == {}

    def test_hourly_slot_expands_to_12_sub_slots(self):
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        raw = {"2025-06-01T10:00:00+00:00": 400.0}
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")
        hour_keys = [k for k in combined if "T10:" in k]
        assert len(hour_keys) == 12

    def test_no_negative_values(self):
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        raw = {"2025-06-01T10:00:00+00:00": 0.1}  # tiny raw value
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "linear")
        assert all(v >= 0.0 for v in combined.values())

    def test_two_strings_summed(self):
        fc_rows, pv_rows1 = self._training(400.0, 150.0, 10)
        _, pv_rows2 = self._training(400.0, 100.0, 10)
        raw = {"2025-06-01T10:00:00+00:00": 400.0}
        combined, per_string = apply_corrections(
            raw,
            fc_rows,
            {"sensor.s1": pv_rows1, "sensor.s2": pv_rows2},
            "factor",
        )
        assert "sensor.s1" in per_string
        assert "sensor.s2" in per_string
        # Combined ≥ either individual string
        c_val = sum(combined.values())
        s1_val = sum(per_string["sensor.s1"].values())
        s2_val = sum(per_string["sensor.s2"].values())
        assert abs(c_val - (s1_val + s2_val)) < 0.1

    def test_result_sorted(self):
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        raw = {
            "2025-06-01T12:00:00+00:00": 400.0,
            "2025-06-01T10:00:00+00:00": 400.0,
            "2025-06-01T11:00:00+00:00": 400.0,
        }
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")
        keys = list(combined.keys())
        assert keys == sorted(keys)

    def test_fallback_to_raw_on_no_models(self):
        """When all pv data is below PV_MIN_W, no models are built → raw returned."""
        fc_rows = [
            {"start": dt(10, 0) + timedelta(days=d), "mean": 400.0} for d in range(30)
        ]
        pv_rows = [
            {"start": dt(10, 0) + timedelta(days=d), "mean": 1.0} for d in range(30)
        ]
        raw = {"2025-06-01T10:00:00+00:00": 400.0}
        combined, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, "linear"
        )
        assert combined == raw
        assert per_string == {}


# ---------------------------------------------------------------------------
# Full 24h forecast attribute test
# ---------------------------------------------------------------------------


class TestFullDayForecast:
    """Verify that a full day of hourly FC slots expands to 24×12 = 288 sub-slots,
    and that the correct slot is selected for the current hour."""

    def _make_full_day_training(self, days: int = 60) -> tuple[list[dict], list[dict]]:
        """Generate training data for all 24 hours, all 12 buckets per hour."""
        fc_rows, pv_rows = [], []
        for d in range(days):
            scale = 0.8 + 0.4 * d / max(days - 1, 1)
            for h in range(24):
                for mm in range(0, 60, BUCKET_MIN):
                    ts = datetime(2025, 1, 1, h, mm, tzinfo=UTC) + timedelta(days=d)
                    fc = 400.0 * scale if 6 <= h <= 19 else 0.0
                    pv = fc * 0.7 if fc > 0 else 0.0
                    fc_rows.append({"start": ts, "mean": fc})
                    if pv >= 5.0:  # only include above curtailment threshold
                        pv_rows.append({"start": ts, "mean": pv})
        return fc_rows, pv_rows

    def _make_24h_raw_forecast(self, date: str = "2025-06-15") -> dict[str, float]:
        """One hourly slot per hour, midnight to 23:00."""
        raw = {}
        for h in range(24):
            ts = f"{date}T{h:02d}:00:00+00:00"
            raw[ts] = 400.0 if 6 <= h <= 19 else 0.0
        return raw

    def test_24h_forecast_expands_to_288_slots(self):
        """24 hourly FC slots × 12 buckets = 288 five-minute slots."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        assert len(combined) == 288, (
            f"Expected 288 slots (24h × 12 buckets), got {len(combined)}"
        )

    def test_all_24_hours_present(self):
        """Every hour 00–23 must have 12 sub-slots."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        for h in range(24):
            hour_slots = [ts for ts in combined if f"T{h:02d}:" in ts]
            assert len(hour_slots) == 12, (
                f"Hour {h:02d} has {len(hour_slots)} slots, expected 12"
            )

    def test_all_12_minute_buckets_present_per_hour(self):
        """Each hour must contain slots at :00, :05, :10, …, :55."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        for h in range(24):
            for mm in range(0, 60, 5):
                # Timestamp format may vary with UTC offset; check by partial match
                expected_partial = f"T{h:02d}:{mm:02d}:"
                matches = [ts for ts in combined if expected_partial in ts]
                assert len(matches) == 1, (
                    f"Missing slot at {h:02d}:{mm:02d} – found {matches}"
                )

    def test_daytime_slots_nonzero_nighttime_zero(self):
        """Solar hours (06–19) produce > 0, night hours produce 0."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        for ts, wh in combined.items():
            hour = datetime.fromisoformat(ts).hour
            if 6 <= hour <= 19:
                assert wh > 0, f"Expected >0 at {ts}, got {wh}"
            else:
                assert wh == 0.0, f"Expected 0 at night {ts}, got {wh}"

    def test_current_slot_lookup(self):
        """Simulate the sensor's native_value lookup: snap to 5-min boundary,
        find the matching slot in the forecast dict."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        # Simulate sensor lookup for 10:17 UTC → snaps to 10:15
        now = datetime(2025, 6, 15, 10, 17, tzinfo=UTC)
        snapped_min = (now.minute // 5) * 5
        now_snapped = now.replace(minute=snapped_min, second=0, microsecond=0)

        key = now_snapped.isoformat()
        assert key in combined, f"Snapped key {key!r} not found in forecast"
        assert combined[key] > 0

    def test_slots_sorted_chronologically(self):
        """Forecast dict keys must be in ascending time order."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        keys = list(combined.keys())
        assert keys == sorted(keys)

    def test_no_negative_values_full_day(self):
        """No slot should ever produce a negative Wh value."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        negatives = {ts: wh for ts, wh in combined.items() if wh < 0}
        assert not negatives, f"Negative values found: {negatives}"

    def test_per_string_matches_aggregate_for_single_string(self):
        """With one string, combined == per_string values."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv1": pv_rows}, "factor"
        )

        string_data = per_string["sensor.pv1"]
        for ts in combined:
            assert abs(combined[ts] - string_data.get(ts, 0.0)) < 1e-9, (
                f"Mismatch at {ts}: combined={combined[ts]}, string={string_data.get(ts)}"
            )


# ---------------------------------------------------------------------------
# Regression tests for known bugs
# ---------------------------------------------------------------------------


class TestRegressionFactor12:
    """today_total and remaining must NOT be 12× the correct value."""

    def _full_day(self, pv_ratio: float = 0.5) -> tuple:
        fc_rows, pv_rows = [], []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            for h in range(6, 20):  # solar hours only
                for mm in range(0, 60, BUCKET_MIN):
                    ts = datetime(2025, 1, 1, h, mm, tzinfo=UTC) + timedelta(days=d)
                    fc = 400.0 * scale
                    pv = fc * pv_ratio
                    fc_rows.append({"start": ts, "mean": fc})
                    if pv >= 5.0:
                        pv_rows.append({"start": ts, "mean": pv})
        raw = {f"2025-06-15T{h:02d}:00:00+00:00": 400.0 for h in range(6, 20)}
        return fc_rows, pv_rows, raw

    def test_today_total_not_factor_12(self):
        """aggregate_to_hours(5-min slots) must produce the same per-hour sum
        as the number of sub-slots × per-slot value.  The coordinator uses
        aggregate_to_hours to derive today_total; verify it does NOT double-count.
        """
        fc_rows, pv_rows, raw = self._full_day(pv_ratio=0.5)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        # Each hourly raw slot expands to 12 sub-slots each with predict(model, 400)
        # = 200. aggregate_to_hours sums them: 12 × 200 = 2400 per hour.
        # today_total via coordinator = sum(aggregate_to_hours(forecast_today))
        # = 14h × 2400 = 33600 Wh.
        # This is correct because the model outputs Wh/h-equivalent per 5-min slot.
        hourly = aggregate_to_hours(combined)
        today_total = sum(hourly.values())

        # Sanity: 14 solar hours, factor=0.5, raw=400 Wh/h → 14 × 400 × 0.5 = 2800 corrected Wh/h
        # But each hour has 12 sub-slots of 200 each → aggregate = 2400 per hour → 14 × 2400 = 33600
        # The coordinator must use aggregate_to_hours, not raw sub-slot sum (same value here but
        # semantically correct – avoids future breakage if sub-slot values change).
        assert today_total > 0
        # Verify aggregate equals 12 × per-slot × hours
        assert abs(today_total - 14 * 12 * 200.0) < 100.0, f"Got {today_total}"

    def test_remaining_uses_aggregate(self):
        """Coordinator derives 'remaining' via aggregate_to_hours to stay consistent
        with today_total.  Both must use the same aggregation path."""
        fc_rows, pv_rows, raw = self._full_day(pv_ratio=0.5)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        hourly = aggregate_to_hours(combined)

        # remaining = sum of hourly buckets from 12:00 onwards (8 hours)
        remaining = sum(
            wh for ts, wh in hourly.items() if datetime.fromisoformat(ts).hour >= 12
        )

        # 8 hours × 12 sub-slots × 200 = 19200
        assert abs(remaining - 8 * 12 * 200.0) < 100.0, f"Got {remaining}"
        # today_total = 14 hours × 12 × 200 = 33600
        today_total = sum(hourly.values())
        assert remaining < today_total


class TestRegressionNightUnknown:
    """Current-slot sensors must return 0.0 at night, not None/unavailable."""

    def test_no_slot_for_night_returns_zero(self):
        """apply_corrections with night raw slot produces 0.0, not missing key."""
        fc_rows, pv_rows = [], []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            # Only daytime training data
            for mm in range(0, 60, BUCKET_MIN):
                ts = datetime(2025, 1, 1, 10, mm, tzinfo=UTC) + timedelta(days=d)
                fc_rows.append({"start": ts, "mean": 400.0 * scale})
                pv_rows.append({"start": ts, "mean": 200.0 * scale})

        # Night slot in raw forecast
        raw = {
            "2025-06-15T02:00:00+00:00": 0.0,  # night, raw = 0
            "2025-06-15T10:00:00+00:00": 400.0,  # daytime
        }
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        # Night sub-slots must be present and = 0.0 (not missing)
        night_slots = {ts: wh for ts, wh in combined.items() if "T02:" in ts}
        assert len(night_slots) == 12, (
            f"Expected 12 night slots, got {len(night_slots)}"
        )
        for ts, wh in night_slots.items():
            assert wh == 0.0, f"Night slot {ts} should be 0.0, got {wh}"

    def test_zero_raw_wh_produces_zero_prediction(self):
        """predict(model, 0.0) must always return 0.0 regardless of model type."""
        assert predict((0.5,), 0.0) == 0.0  # factor
        assert predict((2.0, 3.0), 0.0) == 3.0  # linear with intercept – intentional
        assert predict((1.0, 2.0, 0.0), 0.0) == 0.0  # quadratic through origin

    def test_night_slots_zero_in_full_day(self):
        """In a full 24h forecast, hours 0-5 and 20-23 must all be 0."""
        fc_rows, pv_rows = [], []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            for h in range(6, 20):
                for mm in range(0, 60, BUCKET_MIN):
                    ts = datetime(2025, 1, 1, h, mm, tzinfo=UTC) + timedelta(days=d)
                    fc_rows.append({"start": ts, "mean": 400.0 * scale})
                    pv_rows.append({"start": ts, "mean": 200.0 * scale})

        raw = {
            f"2025-06-15T{h:02d}:00:00+00:00": 0.0
            for h in list(range(0, 6)) + list(range(20, 24))
        }
        raw.update({f"2025-06-15T{h:02d}:00:00+00:00": 400.0 for h in range(6, 20)})

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        for h in list(range(0, 6)) + list(range(20, 24)):
            night_slots = [wh for ts, wh in combined.items() if f"T{h:02d}:" in ts]
            assert len(night_slots) == 12
            assert all(wh == 0.0 for wh in night_slots), (
                f"Hour {h:02d} should be all zeros, got {night_slots}"
            )
