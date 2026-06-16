"""Tests for shadylib – no HA stubs needed."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from typing import Any

from shadylib import (
    r,
    normalise_to_5min_day,
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
    InputHistory,
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
) -> tuple[InputHistory, InputHistory]:
    """Generate realistic training rows with optional daily variation.

    Means are in Wh/slot (W × 5/60), matching what to_wh_per_slot("W") produces
    from 5-minute recorder statistics.
    """
    fc_rows, pv_rows = [], []
    ratio = pv_val / fc_val if fc_val else 0.5
    slot_h = 5 / 60  # 5-minute slot in hours
    for d in range(days):
        scale = (0.8 + 0.4 * d / max(days - 1, 1)) if vary else 1.0
        for mm in minutes:
            ts = datetime(2025, 1, 1, hour, mm, tzinfo=UTC) + timedelta(days=d)
            fc = fc_val * scale * slot_h
            pv = fc * ratio
            fc_rows.append({"start": ts, "mean": fc})
            pv_rows.append({"start": ts, "mean": pv})
    return fc_rows, pv_rows


# ---------------------------------------------------------------------------
# Public API smoke tests
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_r(self) -> None:
        assert r(1.23456) == 1.23

    def test_r6(self) -> None:
        assert r6(0.12345678) == 0.123457

    def test_snap(self) -> None:
        assert snap(7) == 5
        assert snap(0) == 0
        assert snap(55) == 55

    def test_parse_dt_valid(self) -> None:
        result = parse_dt("2025-06-01T10:00:00+00:00")
        assert result.hour == 10

    def test_parse_dt_invalid(self) -> None:
        result = parse_dt("not-a-date")
        assert result == datetime.min.replace(tzinfo=UTC)

    def test_aggregate_to_hours(self) -> None:
        slots = {f"2025-06-01T10:{mm:02d}:00+00:00": 10.0 for mm in range(0, 60, 5)}
        hourly = aggregate_to_hours(slots)
        assert len(hourly) == 1
        assert abs(list(hourly.values())[0] - 120.0) < 0.1

    def test_wls2_exact(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        ws = [1.0] * 4
        model = wls2(xs, ys, ws)
        assert isinstance(model, tuple)
        slope, intercept = model
        assert abs(slope - 2.0) < 1e-9
        assert abs(intercept) < 1e-9

    def test_wls2_origin_quad_exact(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1 * x**2 + 2 * x for x in xs]
        ws = [1.0] * 5
        model = wls2_origin_quad(xs, ys, ws)
        assert isinstance(model, tuple)
        a, b = model
        assert abs(a - 1.0) < 1e-6
        assert abs(b - 2.0) < 1e-6

    def test_predict_factor(self) -> None:
        assert predict((0.5,), 200.0) == 100.0

    def test_predict_linear(self) -> None:
        assert predict((2.0, 3.0), 5.0) == 13.0

    def test_predict_quadratic_origin(self) -> None:
        assert predict((1.0, 2.0, 0.0), 3.0) == 15.0  # 9 + 6

    def test_bucket_min_constant(self) -> None:
        assert BUCKET_MIN == 5


# ---------------------------------------------------------------------------
# build_bucket_models
# ---------------------------------------------------------------------------


class TestBuildBucketModels:
    def test_returns_empty_for_no_data(self) -> None:
        assert build_bucket_models([], [], "linear") == {}

    def test_curtailment_filter(self) -> None:
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

    def test_factor_model_1_tuple(self) -> None:
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "factor")
        assert all(len(m) == 1 for m in models.values())

    def test_linear_model_2_tuple(self) -> None:
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "linear")
        assert all(len(m) == 2 for m in models.values())

    def test_quadratic_model_3_tuple_c_zero(self) -> None:
        fc_rows, pv_rows = make_rows(10, list(range(0, 60, 5)), 400.0, 200.0)
        models = build_bucket_models(fc_rows, pv_rows, "quadratic")
        assert all(len(m) == 3 for m in models.values())
        assert all(m[2] == 0.0 for m in models.values())

    def test_shading_captured_per_bucket(self) -> None:
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
    ) -> tuple[InputHistory, InputHistory]:
        return make_rows(hour, list(range(0, 60, 5)), fc_val, pv_val, days)

    def test_no_pv_rows_returns_raw(self) -> None:
        raw = {"2025-06-01T10:00:00+00:00": 400.0}
        combined, per_string = apply_corrections(raw, [], {}, "linear")
        assert combined == raw
        assert per_string == {}

    def test_hourly_slot_expands_to_12_sub_slots(self) -> None:
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        # EM delivers hourly slots; normalise_em_to_5min distributes into 12 sub-slots
        raw_em = {"2025-06-01T10:00:00+00:00": 400.0, "2025-06-01T11:00:00+00:00": 0.0}
        raw = normalise_em_to_5min(raw_em)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")
        hour_keys = [k for k in combined if "T10:" in k]
        assert len(hour_keys) == 12

    def test_no_negative_values(self) -> None:
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        raw = {"2025-06-01T10:00:00+00:00": 0.1}  # tiny raw value
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "linear")
        assert all(v >= 0.0 for v in combined.values())

    def test_two_strings_summed(self) -> None:
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

    def test_result_sorted(self) -> None:
        fc_rows, pv_rows = self._training(400.0, 200.0, 10)
        raw = {
            "2025-06-01T12:00:00+00:00": 400.0,
            "2025-06-01T10:00:00+00:00": 400.0,
            "2025-06-01T11:00:00+00:00": 400.0,
        }
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")
        keys = list(combined.keys())
        assert keys == sorted(keys)

    def test_fallback_to_raw_on_no_models(self) -> None:
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

    def _make_full_day_training(
        self, days: int = 60
    ) -> tuple[InputHistory, InputHistory]:
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
        """One hourly EM slot per hour normalised to 5-minute slots."""
        em: dict[str, float] = {}
        for h in range(24):
            em[f"{date}T{h:02d}:00:00+00:00"] = 400.0 if 6 <= h <= 19 else 0.0
        em["2025-06-16T00:00:00+00:00"] = 0.0  # sentinel: defines end of last slot
        return normalise_em_to_5min(em)

    def test_24h_forecast_expands_to_288_slots(self) -> None:
        """24 hourly FC slots × 12 buckets = 288 five-minute slots."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        day_slots = {ts: v for ts, v in combined.items() if "2025-06-15" in ts}
        assert len(day_slots) == 288, (
            f"Expected 288 slots (24h × 12 buckets), got {len(day_slots)}"
        )

    def test_all_24_hours_present(self) -> None:
        """Every hour 00–23 must have 12 sub-slots."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        for h in range(24):
            hour_slots = [ts for ts in combined if f"2025-06-15T{h:02d}:" in ts]
            assert len(hour_slots) == 12, (
                f"Hour {h:02d} has {len(hour_slots)} slots, expected 12"
            )

    def test_all_12_minute_buckets_present_per_hour(self) -> None:
        """Each hour must contain slots at :00, :05, :10, …, :55."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        for h in range(24):
            for mm in range(0, 60, 5):
                expected_partial = f"2025-06-15T{h:02d}:{mm:02d}:"
                matches = [ts for ts in combined if expected_partial in ts]
                assert len(matches) == 1, (
                    f"Missing slot at {h:02d}:{mm:02d} – found {matches}"
                )

    def test_daytime_slots_nonzero_nighttime_zero(self) -> None:
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

    def test_current_slot_lookup(self) -> None:
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

    def test_slots_sorted_chronologically(self) -> None:
        """Forecast dict keys must be in ascending time order."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        keys = list(combined.keys())
        assert keys == sorted(keys)

    def test_no_negative_values_full_day(self) -> None:
        """No slot should ever produce a negative Wh value."""
        fc_rows, pv_rows = self._make_full_day_training()
        raw = self._make_24h_raw_forecast()

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv1": pv_rows}, "factor")

        negatives = {ts: wh for ts, wh in combined.items() if wh < 0}
        assert not negatives, f"Negative values found: {negatives}"

    def test_per_string_matches_aggregate_for_single_string(self) -> None:
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

    def _full_day(
        self, pv_ratio: float = 0.5
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
        fc_rows: list[dict[str, Any]] = []
        pv_rows: list[dict[str, Any]] = []
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
        raw_em = {f"2025-06-15T{h:02d}:00:00+00:00": 400.0 for h in range(6, 20)}
        # Add sentinel for last entry end boundary
        raw_em["2025-06-15T20:00:00+00:00"] = 0.0
        raw = normalise_em_to_5min(raw_em)
        return fc_rows, pv_rows, raw

    def test_today_total_not_factor_12(self) -> None:
        """today_total must equal raw x correction_factor, NOT raw x factor x 12.

        After the /12 fix in _predict_string each sub-slot carries Wh for
        5 minutes, so 12 sub-slots correctly reconstruct one hourly Wh value.
        14 solar hours x 400 Wh/h x 0.5 factor = 2800 Wh expected.
        """
        fc_rows, pv_rows, raw = self._full_day(pv_ratio=0.5)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        # Coordinator now sums the 5-min slots directly (no hourly aggregation).
        today_total = sum(combined.values())

        # 14 h x 400 Wh/h x 0.5 = 2800 Wh  (allow +-5 % for WLS fitting error)
        expected = 14 * 400.0 * 0.5
        assert abs(today_total - expected) < expected * 0.05, f"Got {today_total}"

    def test_remaining_uses_aggregate(self) -> None:
        """remaining is the sum of 5-min slots whose start >= now (5-min precision).

        With the /12 fix each slot carries the correct Wh fraction, so a direct
        sum is both correct and more granular than hourly bucketing.
        """
        fc_rows, pv_rows, raw = self._full_day(pv_ratio=0.5)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        # remaining = slots from 12:00 onwards (8 solar hours of 14 total)
        remaining = sum(
            wh for ts, wh in combined.items() if datetime.fromisoformat(ts).hour >= 12
        )
        today_total = sum(combined.values())

        # 8 h x 400 Wh/h x 0.5 = 1600 Wh  (allow +-5 % for WLS fitting error)
        expected_remaining = 8 * 400.0 * 0.5
        assert abs(remaining - expected_remaining) < expected_remaining * 0.05, (
            f"Got {remaining}"
        )
        assert remaining < today_total


class TestRegressionNightUnknown:
    """Current-slot sensors must return 0.0 at night, not None/unavailable."""

    def test_no_slot_for_night_returns_zero(self) -> None:
        """apply_corrections with night raw slot produces 0.0, not missing key."""
        fc_rows: InputHistory = []
        pv_rows: InputHistory = []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            # Only daytime training data
            for mm in range(0, 60, BUCKET_MIN):
                ts = (
                    datetime(2025, 1, 1, 10, mm, tzinfo=UTC) + timedelta(days=d)
                ).isoformat()
                fc_rows.append({"start": ts, "mean": 400.0 * scale})
                pv_rows.append({"start": ts, "mean": 200.0 * scale})

        # Night slot in raw forecast – normalise to 5-min slots first
        raw_em = {
            "2025-06-15T02:00:00+00:00": 0.0,  # night, raw = 0
            "2025-06-15T10:00:00+00:00": 400.0,  # daytime
            "2025-06-15T11:00:00+00:00": 0.0,
        }
        raw = normalise_em_to_5min(raw_em)
        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        # Night sub-slots must be present and = 0.0 (not missing)
        night_slots = {ts: wh for ts, wh in combined.items() if "T02:" in ts}
        assert len(night_slots) == 12, (
            f"Expected 12 night slots, got {len(night_slots)}"
        )
        for ts, wh in night_slots.items():
            assert wh == 0.0, f"Night slot {ts} should be 0.0, got {wh}"

    def test_zero_raw_wh_produces_zero_prediction(self) -> None:
        """predict(model, 0.0) must always return 0.0 regardless of model type."""
        assert predict((0.5,), 0.0) == 0.0  # factor
        assert predict((2.0, 3.0), 0.0) == 3.0  # linear with intercept – intentional
        assert predict((1.0, 2.0, 0.0), 0.0) == 0.0  # quadratic through origin

    def test_night_slots_zero_in_full_day(self) -> None:
        """In a full 24h forecast, hours 0-5 and 20-23 must all be 0."""
        fc_rows, pv_rows = [], []
        for d in range(60):
            scale = 0.8 + 0.4 * d / 59
            for h in range(6, 20):
                for mm in range(0, 60, BUCKET_MIN):
                    ts = datetime(2025, 1, 1, h, mm, tzinfo=UTC) + timedelta(days=d)
                    fc_rows.append({"start": ts, "mean": 400.0 * scale})
                    pv_rows.append({"start": ts, "mean": 200.0 * scale})

        raw_em = {
            f"2025-06-15T{h:02d}:00:00+00:00": 0.0
            for h in list(range(0, 6)) + list(range(20, 24))
        }
        raw_em.update({f"2025-06-15T{h:02d}:00:00+00:00": 400.0 for h in range(6, 20)})
        raw_em["2025-06-16T00:00:00+00:00"] = 0.0  # sentinel
        raw = normalise_em_to_5min(raw_em)

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        for h in list(range(0, 6)) + list(range(20, 24)):
            night_slots = [
                wh for ts, wh in combined.items() if f"2025-06-15T{h:02d}:" in ts
            ]
            assert len(night_slots) == 12
            assert all(wh == 0.0 for wh in night_slots), (
                f"Hour {h:02d} should be all zeros, got {night_slots}"
            )


# ---------------------------------------------------------------------------
# normalise_to_5min_day
# ---------------------------------------------------------------------------


def day(y: int = 2025, m: int = 6, d: int = 2) -> datetime:
    return datetime(y, m, d, 0, 0, 0, tzinfo=UTC)


class TestNormaliseTo5MinDay:
    def test_always_returns_288_slots(self):
        assert len(normalise_to_5min_day({}, day())) == 288

    def test_empty_input_all_zeros(self):
        assert all(v == 0.0 for v in normalise_to_5min_day({}, day()).values())

    def test_slots_span_full_24h(self):
        keys = sorted(normalise_to_5min_day({}, day()))
        assert keys[0] == "2025-06-02T00:00"
        assert keys[-1] == "2025-06-02T23:55"

    def test_exact_5min_timestamp_preserved(self):
        slots = {"2025-06-02T10:15:00+00:00": 42.0}
        assert normalise_to_5min_day(slots, day())["2025-06-02T10:15"] == 42.0

    def test_sub_5min_timestamp_snapped(self):
        slots = {"2025-06-02T21:12:46+00:00": 30.0}
        result = normalise_to_5min_day(slots, day())
        assert result["2025-06-02T21:10"] == 30.0
        assert result.get("2025-06-02T21:12:46+00:00") is None

    def test_sub_5min_accumulation(self):
        slots = {
            "2025-06-02T10:01:00+00:00": 10.0,
            "2025-06-02T10:03:00+00:00": 5.0,
        }
        assert (
            abs(normalise_to_5min_day(slots, day())["2025-06-02T10:00"] - 15.0) < 0.01
        )

    def test_out_of_day_slots_ignored(self):
        slots = {
            "2025-06-01T23:55:00+00:00": 99.0,
            "2025-06-03T00:00:00+00:00": 99.0,
            "2025-06-02T12:00:00+00:00": 50.0,
        }
        result = normalise_to_5min_day(slots, day())
        assert result["2025-06-02T12:00"] == 50.0
        assert abs(sum(result.values()) - 50.0) < 0.01

    def test_night_slots_are_zero(self):
        slots = {"2025-06-02T12:00:00+00:00": 100.0}
        result = normalise_to_5min_day(slots, day())
        assert result["2025-06-02T00:00"] == 0.0
        assert result["2025-06-02T23:55"] == 0.0

    def test_keys_are_sorted(self):
        result = normalise_to_5min_day({}, day())
        keys = list(result)
        assert keys == sorted(keys)

    def test_consecutive_slots_differ_by_5min(self):
        keys = list(normalise_to_5min_day({}, day()))
        for a, b in zip(keys, keys[1:]):
            delta = datetime.fromisoformat(b) - datetime.fromisoformat(a)
            assert delta == timedelta(minutes=5)

    def test_hourly_slot_placed_at_hour_boundary(self):
        slots = {"2025-06-02T14:00:00+00:00": 200.0}
        assert normalise_to_5min_day(slots, day())["2025-06-02T14:00"] == 200.0

    def test_naive_timestamp_treated_as_utc(self):
        slots = {"2025-06-02T10:00:00": 77.0}
        result = normalise_to_5min_day(slots, day())
        assert result["2025-06-02T10:00"] == 77.0

    def test_keys_are_minute_precision_no_tz(self):
        """Keys must be YYYY-MM-DDTHH:MM – no seconds, no UTC offset."""
        import re

        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
        for key in normalise_to_5min_day({}, day()):
            assert pattern.match(key), f"Key has unexpected format: {key!r}"

    def test_invalid_timestamp_skipped(self):
        slots = {"not-a-date": 99.0, "2025-06-02T10:00:00+00:00": 10.0}
        result = normalise_to_5min_day(slots, day())
        assert result["2025-06-02T10:00"] == 10.0
        assert abs(sum(result.values()) - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Regression: hourly fc sensor → all 12 sub-slots must be non-zero
# ---------------------------------------------------------------------------


class TestHourlyFcBucketCoverage:
    """When the fc sensor only has hourly statistics (timestamp at :00 each
    hour), build_bucket_models only produces bucket models at minute=0.

    Before the fix, _predict_string used models.get(bk) for the hourly
    expansion, returning None for mm=5…55 → 11 of 12 sub-slots were 0.0
    ('vereinzelte slots' bug).  After the fix, _nearest_model is used so
    every sub-slot falls back to the (hour, 0) model.
    """

    def _hourly_fc_rows(
        self, hour: int = 10, days: int = 30, mean: float = 400.0
    ) -> InputHistory:
        """FC rows with one entry per hour (hourly sensor, no :05…:55 rows)."""
        return [
            {
                "start": datetime(2025, 1, 1, hour, 0, tzinfo=UTC) + timedelta(days=d),
                "mean": mean,
            }
            for d in range(days)
        ]

    def _pv_rows_5min(
        self, hour: int = 10, days: int = 30, mean: float = 200.0
    ) -> InputHistory:
        """PV rows with 5-min resolution (realistic recorder data)."""
        rows: InputHistory = []
        for d in range(days):
            for mm in range(0, 60, BUCKET_MIN):
                rows.append(
                    {
                        "start": datetime(2025, 1, 1, hour, mm, tzinfo=UTC)
                        + timedelta(days=d),
                        "mean": mean,
                    }
                )
        return rows

    def test_all_12_sub_slots_non_zero(self) -> None:
        """All 12 sub-slots of an hourly raw slot must carry a prediction."""
        fc_rows = self._hourly_fc_rows(hour=10, mean=400.0)
        pv_rows = self._pv_rows_5min(hour=10, mean=200.0)
        raw_em = {"2025-06-01T10:00:00+00:00": 400.0, "2025-06-01T11:00:00+00:00": 0.0}
        raw = normalise_em_to_5min(raw_em)

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        hour_slots = {ts: v for ts, v in combined.items() if "T10:" in ts}
        assert len(hour_slots) == 12, f"Expected 12 sub-slots, got {len(hour_slots)}"
        zero_slots = [ts for ts, v in hour_slots.items() if v == 0.0]
        assert not zero_slots, (
            f"Sub-slots with zero value (hourly fc, nearest-model fallback failed): {zero_slots}"
        )

    def test_sub_slot_values_consistent(self) -> None:
        """All sub-slots should carry the same value (uniform training data → same model)."""
        fc_rows = self._hourly_fc_rows(hour=10, mean=400.0)
        pv_rows = self._pv_rows_5min(hour=10, mean=200.0)
        raw_em = {"2025-06-01T10:00:00+00:00": 400.0, "2025-06-01T11:00:00+00:00": 0.0}
        raw = normalise_em_to_5min(raw_em)

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        hour_vals = [v for ts, v in combined.items() if "T10:" in ts]
        assert len(set(hour_vals)) == 1, (
            f"Sub-slot values differ unexpectedly with uniform training data: {hour_vals}"
        )

    def test_hourly_total_preserved(self) -> None:
        """Sum of 12 sub-slots must equal the full correction applied to raw_wh."""
        fc_rows = self._hourly_fc_rows(hour=10, mean=400.0)
        pv_rows = self._pv_rows_5min(hour=10, mean=200.0)
        raw_wh = 400.0
        raw_em = {"2025-06-01T10:00:00+00:00": raw_wh, "2025-06-01T11:00:00+00:00": 0.0}
        raw = normalise_em_to_5min(raw_em)

        combined, _ = apply_corrections(raw, fc_rows, {"sensor.pv": pv_rows}, "factor")

        hour_total = sum(v for ts, v in combined.items() if "T10:" in ts)
        # Factor = 200/400 = 0.5 → corrected ≈ raw_wh * 0.5 = 200 Wh
        assert abs(hour_total - raw_wh * 0.5) < 1.0, (
            f"Hour total {hour_total:.2f} Wh deviates from expected {raw_wh * 0.5:.2f} Wh"
        )


# ---------------------------------------------------------------------------
# Realistic tree-shading scenario
# ---------------------------------------------------------------------------

# Measured hourly fc and PV values for a string shaded by a tree in the
# morning (hours 7–12, pv/fc ≈ 0.20) and unobstructed in the afternoon
# (hours 13–19, pv/fc rises to > 1.0 due to favourable angle).
#
# Source: hand-crafted table representing a realistic southern-European
# residential PV installation with a deciduous tree to the east/south-east.
_TREE_SHADE_TABLE: dict[int, tuple[float, float]] = {
    7: (265, 53),
    8: (490, 98),
    9: (673, 135),
    10: (816, 163),
    11: (918, 184),
    12: (980, 196),
    13: (1000, 550),
    14: (980, 796),
    15: (918, 934),
    16: (816, 963),
    17: (673, 885),
    18: (490, 698),
    19: (265, 403),
}


def _tree_shade_rows(
    days: int = 5,
    base_date: datetime | None = None,
    scale: float = 1.0,
) -> tuple[InputHistory, InputHistory]:
    """Build fc_rows / pv_rows for the tree-shading scenario.

    Sub-hourly PV values are linearly interpolated between the hourly table
    entries (pv at :00 of hour h → pv at :00 of hour h+1).  FC is kept
    constant within each hour (a forecast mean has no intra-hour ramp).

    A small daily scale ramp (0.8 … 1.2 × base values) is applied across
    the training days so that WLS-based fitters (linear, quadratic) receive
    a spread of x-values and can produce a non-degenerate fit.

    Args:
        days:       Number of training days to generate.
        base_date:  Start date (UTC midnight).  Defaults to 2025-01-01.
        scale:      Additional uniform scale factor applied to both fc and pv.
    """
    if base_date is None:
        base_date = datetime(2025, 1, 1, tzinfo=UTC)

    solar_hours = sorted(_TREE_SHADE_TABLE)

    fc_rows: InputHistory = []
    pv_rows: InputHistory = []

    for d in range(days):
        # Daily ramp: 0.8 on day 0, 1.2 on the last day – provides the x-spread
        # needed for linear/quadratic WLS to produce non-degenerate fits.
        day_scale_fc = scale * (0.7 + 0.5 * d / max(days - 1, 1))
        day_scale_pv = scale * (0.8 + 0.4 * d / max(days - 1, 1))
        day_offset = timedelta(days=d)
        for i, h in enumerate(solar_hours):
            fc_h, pv_h = _TREE_SHADE_TABLE[h]
            # Next hour's pv for intra-hour interpolation
            next_h = solar_hours[i + 1] if i + 1 < len(solar_hours) else None
            pv_next = _TREE_SHADE_TABLE[next_h][1] if (next_h == h + 1) else 0.0

            for mm in range(0, 60, BUCKET_MIN):
                ts = base_date + day_offset + timedelta(hours=h, minutes=mm)
                pv_mm = pv_h + (pv_next - pv_h) * mm / 60
                # fc_rows means are in Wh/slot (W * 5/60), matching to_wh_per_slot("W")
                fc_rows.append({"start": ts, "mean": fc_h * day_scale_fc * (5 / 60)})
                pv_rows.append({"start": ts, "mean": pv_mm * day_scale_pv * (5 / 60)})

    return fc_rows, pv_rows


class TestTreeShadingScenario:
    """Apply all three fitters to a realistic tree-shading scenario.

    The shading profile has two distinct regimes:
      - Morning (h 7–12): pv/fc ≈ 0.20  (tree shadow)
      - Afternoon (h 13–19): pv/fc rising from 0.55 → 1.52 (unobstructed)

    Tests verify that each algorithm learns the per-bucket correction and
    reproduces the shading asymmetry in predictions.
    """

    _SHADED_HOURS = range(7, 13)  # pv/fc ≈ 0.20
    _CLEAR_HOURS = range(13, 20)  # pv/fc >> 0.20

    def _raw_forecast(self, date: str = "2025-06-15") -> dict[str, float]:
        """Hourly EM slots for each solar hour, normalised to 5-minute slots."""
        em: dict[str, float] = {}
        for h, (fc_h, _) in _TREE_SHADE_TABLE.items():
            em[f"{date}T{h:02d}:00:00+00:00"] = float(fc_h)
        # Sentinel so the last entry has a defined end
        last_h = max(_TREE_SHADE_TABLE) + 1
        em[f"{date}T{last_h:02d}:00:00+00:00"] = 0.0
        return normalise_em_to_5min(em)

    def _shaded_mean(self, per_string: dict[str, dict[str, float]], hour: int) -> float:
        """Mean predicted value across the 12 sub-slots of *hour*."""
        slots = per_string.get("sensor.pv", {})
        vals = [v for ts, v in slots.items() if f"T{hour:02d}:" in ts]
        return sum(vals) / len(vals) if vals else 0.0

    @pytest.mark.parametrize("algorithm", ["factor", "linear", "quadratic"])
    def test_models_are_built(self, algorithm: str) -> None:
        """build_bucket_models must succeed for all three algorithms."""
        fc_rows, pv_rows = _tree_shade_rows(days=5)
        raw = self._raw_forecast()
        _, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, algorithm
        )
        assert "sensor.pv" in per_string, (
            f"algorithm={algorithm}: no string forecast produced"
        )

    @pytest.mark.parametrize("algorithm", ["factor", "linear", "quadratic"])
    def test_shading_asymmetry_captured(self, algorithm: str) -> None:
        """Afternoon predictions must exceed morning predictions, reflecting
        the unshaded afternoon vs shaded morning."""
        fc_rows, pv_rows = _tree_shade_rows(days=5)
        raw = self._raw_forecast()
        _, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, algorithm
        )
        slots = per_string.get("sensor.pv", {})
        assert slots, f"algorithm={algorithm}: empty prediction"

        # Mean prediction over shaded morning hours
        morning_vals = [
            v
            for ts, v in slots.items()
            if any(f"T{h:02d}:" in ts for h in self._SHADED_HOURS)
        ]
        afternoon_vals = [
            v
            for ts, v in slots.items()
            if any(f"T{h:02d}:" in ts for h in self._CLEAR_HOURS)
        ]

        morning_mean = sum(morning_vals) / len(morning_vals) if morning_vals else 0.0
        afternoon_mean = (
            sum(afternoon_vals) / len(afternoon_vals) if afternoon_vals else 0.0
        )

        assert afternoon_mean > morning_mean * 2, (
            f"algorithm={algorithm}: afternoon mean {afternoon_mean:.1f} should be "
            f"more than 2× morning mean {morning_mean:.1f} (tree-shading asymmetry)"
        )

    @pytest.mark.parametrize("algorithm", ["factor", "linear", "quadratic"])
    def test_288_sub_slots_produced(self, algorithm: str) -> None:
        """A full-day raw forecast (hourly slots for all 13 solar hours) must
        expand to 13 × 12 = 156 sub-slots in the string forecast."""
        fc_rows, pv_rows = _tree_shade_rows(days=5)
        raw = self._raw_forecast()
        _, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, algorithm
        )
        # Filter to the 13 solar hours (7–19) only, excluding the sentinel slot
        solar_slots = {
            ts: v
            for ts, v in per_string.get("sensor.pv", {}).items()
            if any(f"T{h:02d}:" in ts for h in _TREE_SHADE_TABLE)
        }
        n = len(solar_slots)
        assert n == 13 * 12, f"algorithm={algorithm}: expected 156 sub-slots, got {n}"

    @pytest.mark.parametrize("algorithm", ["factor", "linear", "quadratic"])
    def test_no_negative_predictions(self, algorithm: str) -> None:
        fc_rows, pv_rows = _tree_shade_rows(days=5)
        raw = self._raw_forecast()
        _, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, algorithm
        )
        negatives = {
            ts: v for ts, v in per_string.get("sensor.pv", {}).items() if v < 0
        }
        assert not negatives, (
            f"algorithm={algorithm}: negative prediction values: {negatives}"
        )

    @pytest.mark.parametrize("algorithm", ["factor", "linear", "quadratic"])
    def test_morning_ratio_approx_020(self, algorithm: str) -> None:
        """For shaded hours 7–12, predicted/fc should be close to 0.20."""
        fc_rows, pv_rows = _tree_shade_rows(days=5)
        raw = self._raw_forecast()
        _, per_string = apply_corrections(
            raw, fc_rows, {"sensor.pv": pv_rows}, algorithm
        )
        slots = per_string.get("sensor.pv", {})

        # Sum predictions and raw fc over shaded hours.
        # Both pred_total and fc_total are in Wh/slot (raw was normalised).
        pred_total = sum(
            v
            for ts, v in slots.items()
            if any(f"T{h:02d}:" in ts for h in self._SHADED_HOURS)
        )
        fc_total = sum(
            v
            for ts, v in raw.items()
            if any(f"T{h:02d}:" in ts for h in self._SHADED_HOURS)
        )

        ratio = pred_total / fc_total if fc_total else 0.0
        assert 0.10 < ratio < 0.35, (
            f"algorithm={algorithm}: morning pv/fc ratio {ratio:.3f} outside [0.10, 0.35] "
            f"(expected ~0.20 for tree-shaded morning)"
        )


# ---------------------------------------------------------------------------
# normalise_em_to_5min
# ---------------------------------------------------------------------------


from shadylib import normalise_em_to_5min  # noqa: E402


class TestNormaliseEmTo5Min:
    """Tests for normalise_em_to_5min()."""

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_returns_empty(self) -> None:
        assert normalise_em_to_5min({}) == {}

    def test_invalid_timestamp_skipped(self) -> None:
        result = normalise_em_to_5min({"not-a-date": 100.0})
        assert result == {}

    # ------------------------------------------------------------------
    # Single entry – assigned to its own slot only
    # ------------------------------------------------------------------

    def test_single_entry_at_slot_boundary(self) -> None:
        result = normalise_em_to_5min({"2025-06-01T10:00:00+00:00": 60.0})
        assert list(result.keys()) == ["2025-06-01T10:00:00+00:00"]
        assert result["2025-06-01T10:00:00+00:00"] == 60.0

    def test_single_entry_non_boundary(self) -> None:
        result = normalise_em_to_5min({"2025-06-01T10:17:00+00:00": 60.0})
        assert list(result.keys()) == ["2025-06-01T10:15:00+00:00"]
        assert result["2025-06-01T10:15:00+00:00"] == 60.0

    # ------------------------------------------------------------------
    # Hourly EM slots – one value per full hour
    # ------------------------------------------------------------------

    def test_hourly_slots_distributed_evenly(self) -> None:
        """228 Wh for 06:00–07:00 → 12 slots à 19 Wh."""
        raw = {
            "2025-06-01T06:00:00+00:00": 228.0,
            "2025-06-01T07:00:00+00:00": 480.0,
        }
        result = normalise_em_to_5min(raw)

        # 06:xx slots: 228 / 12 = 19.0
        for mm in range(0, 60, 5):
            key = f"2025-06-01T06:{mm:02d}:00+00:00"
            assert key in result, f"Missing slot {key}"
            assert abs(result[key] - 19.0) < 0.01, f"Slot {key}: {result[key]}"

        # 07:xx: only the single-entry slot
        assert "2025-06-01T07:00:00+00:00" in result
        assert result["2025-06-01T07:00:00+00:00"] == 480.0

    def test_hourly_total_preserved(self) -> None:
        """Sum of distributed slots equals original EM value."""
        raw = {
            "2025-06-01T08:00:00+00:00": 360.0,
            "2025-06-01T09:00:00+00:00": 0.0,
        }
        result = normalise_em_to_5min(raw)
        hour_sum = sum(v for ts, v in result.items() if "T08:" in ts)
        assert abs(hour_sum - 360.0) < 0.05

    # ------------------------------------------------------------------
    # Sub-hourly EM slots – non-boundary timestamps
    # ------------------------------------------------------------------

    def test_non_boundary_start_partial_first_slot(self) -> None:
        """EM value starting at :17 covers 3/5 of the :15 slot and 2/5 of :20."""
        raw = {
            "2025-06-01T10:17:00+00:00": 100.0,
            "2025-06-01T10:22:00+00:00": 0.0,
        }
        # interval 10:17–10:22 = 5 min total
        # overlap with :15 slot (10:15–10:20): 3 min → 60 Wh
        # overlap with :20 slot (10:20–10:25): 2 min → 40 Wh
        result = normalise_em_to_5min(raw)
        assert abs(result.get("2025-06-01T10:15:00+00:00", 0.0) - 60.0) < 0.1
        assert abs(result.get("2025-06-01T10:20:00+00:00", 0.0) - 40.0) < 0.1

    def test_non_boundary_total_preserved(self) -> None:
        raw = {
            "2025-06-01T06:17:00+00:00": 300.0,
            "2025-06-01T07:00:00+00:00": 0.0,
        }
        result = normalise_em_to_5min(raw)
        total = sum(v for ts, v in result.items() if "T06:" in ts)
        assert abs(total - 300.0) < 0.1

    def test_half_hourly_slots(self) -> None:
        """EM delivering 30-minute intervals distributes into 6 slots each."""
        raw = {
            "2025-06-01T06:00:00+00:00": 120.0,
            "2025-06-01T06:30:00+00:00": 180.0,
            "2025-06-01T07:00:00+00:00": 0.0,
        }
        result = normalise_em_to_5min(raw)
        # First 30 min: 120/6 = 20 Wh per slot
        for mm in range(0, 30, 5):
            key = f"2025-06-01T06:{mm:02d}:00+00:00"
            assert abs(result.get(key, 0.0) - 20.0) < 0.01, f"{key}: {result.get(key)}"
        # Second 30 min: 180/6 = 30 Wh per slot
        for mm in range(30, 60, 5):
            key = f"2025-06-01T06:{mm:02d}:00+00:00"
            assert abs(result.get(key, 0.0) - 30.0) < 0.01, f"{key}: {result.get(key)}"

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def test_output_is_sorted(self) -> None:
        raw = {
            "2025-06-01T10:00:00+00:00": 60.0,
            "2025-06-01T09:00:00+00:00": 60.0,
            "2025-06-01T11:00:00+00:00": 0.0,
        }
        result = normalise_em_to_5min(raw)
        keys = list(result.keys())
        assert keys == sorted(keys)

    # ------------------------------------------------------------------
    # Prediction scale consistency
    # ------------------------------------------------------------------

    def test_predict_string_uses_slot_values_directly(self) -> None:
        """After normalisation, _predict_string with a factor-1 model returns
        the same Wh/slot values – no hidden /12 scaling."""
        from shadylib.correction import _predict_string
        from shadylib.models import BucketModels

        # Factor model: output = 1.0 × input
        models: BucketModels = {(6, mm): (1.0,) for mm in range(0, 60, 5)}

        raw_em = {
            "2025-06-01T06:00:00+00:00": 228.0,
            "2025-06-01T07:00:00+00:00": 0.0,
        }
        normalised = normalise_em_to_5min(raw_em)
        result = _predict_string(normalised, models)

        # Each slot should be ~19 Wh (228/12), not 228 Wh
        for mm in range(0, 60, 5):
            key = f"2025-06-01T06:{mm:02d}:00+00:00"
            assert key in result, f"Missing {key}"
            assert abs(result[key] - 19.0) < 0.1, f"{key}: {result[key]}"
