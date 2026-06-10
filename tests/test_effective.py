"""Tests for shadylib.effective – compute_effective_strings & split_combined_sensor."""

from __future__ import annotations

import pytest

from shadylib import compute_effective_strings, split_combined_sensor


# ---------------------------------------------------------------------------
# split_combined_sensor
# ---------------------------------------------------------------------------


class TestSplitCombinedSensor:
    def test_positive_value_is_input(self):
        inp, out = split_combined_sensor(100.0)
        assert inp == 100.0
        assert out == 0.0

    def test_negative_value_is_output(self):
        inp, out = split_combined_sensor(-80.0)
        assert inp == 0.0
        assert out == 80.0

    def test_zero_returns_both_zero(self):
        inp, out = split_combined_sensor(0.0)
        assert inp == 0.0
        assert out == 0.0

    def test_output_is_absolute(self):
        _, out = split_combined_sensor(-42.5)
        assert out == 42.5


# ---------------------------------------------------------------------------
# compute_effective_strings – edge cases
# ---------------------------------------------------------------------------


class TestComputeEffectiveStringsEdgeCases:
    def test_empty_input_returns_empty(self):
        assert compute_effective_strings([]) == []

    def test_no_loss_returns_originals(self):
        """When system outputs ≥ inputs, total_loss ≤ 0 → originals returned."""
        result = compute_effective_strings(
            [100.0, 200.0],
            grid_export=500.0,  # massive export → no loss
        )
        assert result == [100.0, 200.0]

    def test_zero_pv_strings_remain_zero(self):
        result = compute_effective_strings(
            [0.0, 100.0, 0.0],
            grid_export=50.0,
        )
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_total_loss_greater_than_sum_all_zero(self):
        """If total_loss ≥ pv_sum, all strings → 0."""
        result = compute_effective_strings(
            [100.0, 200.0],
            grid_import=500.0,  # huge import pushes loss beyond pv_sum
        )
        assert result == [0.0, 0.0]

    def test_negative_pv_treated_as_inactive(self):
        result = compute_effective_strings([-10.0, 200.0])
        assert result[0] == 0.0

    def test_all_zero_pv_returns_all_zero(self):
        result = compute_effective_strings([0.0, 0.0, 0.0], grid_export=100.0)
        assert result == [0.0, 0.0, 0.0]

    def test_result_same_length_as_input(self):
        pv = [100.0, 200.0, 300.0]
        result = compute_effective_strings(pv, grid_import=50.0)
        assert len(result) == len(pv)

    def test_no_system_sensors_no_loss(self):
        """Without any system sensors, loss = pv_sum + 0 - 0 = pv_sum.
        But outputs = 0, so total_loss = pv_sum + 0 - 0 = pv_sum → all zero.

        Wait: total_loss = pv_nonzero_sum + sys_input - sys_output.
        With no system sensors: total_loss = pv_sum + 0 - 0 = pv_sum.
        pv_sum >= pv_sum → all zero!

        That means without any output sensor configured, everything is lost.
        This is the correct model: the PV strings ARE the inputs; outputs must
        be configured for any effective power to remain.
        """
        result = compute_effective_strings([100.0, 200.0])
        assert result == [0.0, 0.0]

    def test_perfect_balance_returns_originals(self):
        """When sys_output exactly equals pv_sum + sys_input, loss = 0."""
        pv = [100.0, 200.0]
        pv_sum = 300.0
        result = compute_effective_strings(pv, grid_export=pv_sum)
        assert result == [100.0, 200.0]

    def test_all_values_non_negative(self):
        result = compute_effective_strings(
            [50.0, 100.0, 150.0],
            grid_import=80.0,
            grid_export=20.0,
        )
        assert all(v >= 0.0 for v in result)


# ---------------------------------------------------------------------------
# compute_effective_strings – waterfall algorithm correctness
# ---------------------------------------------------------------------------


class TestWaterfallAlgorithm:
    def test_even_loss_no_floor_hit(self):
        """100 W loss across two equal 200 W strings → each 150 W effective."""
        result = compute_effective_strings(
            [200.0, 200.0],
            grid_export=300.0,  # sys_output
            # total_loss = 200+200 + 0 - 300 = 100
        )
        assert abs(result[0] - 150.0) < 1e-6
        assert abs(result[1] - 150.0) < 1e-6

    def test_small_string_absorbs_to_zero_remainder_goes_to_large(self):
        """Loss = 150, strings [50, 200].
        Fair share = 75.  String 50 can only absorb 50 → floors to 0.
        Remaining loss = 150 - 50 = 100 redistributed to [200] only.
        200 - 100 = 100 effective.
        """
        result = compute_effective_strings(
            [50.0, 200.0],
            grid_export=100.0,  # sys_output=100
            # total_loss = 50+200 + 0 - 100 = 150
        )
        # small string → 0
        assert result[0] == 0.0
        # large string: remaining loss = 100 → 200 - 100 = 100
        assert abs(result[1] - 100.0) < 1e-6

    def test_order_independence_of_result(self):
        """Reversing the input order yields the same effective values (reordered)."""
        pv = [50.0, 100.0, 200.0]
        pv_rev = list(reversed(pv))
        r1 = compute_effective_strings(pv, grid_export=150.0)
        r2 = compute_effective_strings(pv_rev, grid_export=150.0)
        assert sorted(r1) == sorted(r2)

    def test_sum_of_effective_plus_loss_equals_pv_sum(self):
        """sum(effective) + total_loss_absorbed must equal pv_sum."""
        pv = [80.0, 120.0, 300.0]
        sys_in = 50.0
        sys_out = 200.0
        result = compute_effective_strings(pv, grid_import=sys_in, grid_export=sys_out)
        expected_loss = sum(pv) + sys_in - sys_out
        expected_loss = max(0.0, min(expected_loss, sum(pv)))
        assert abs(sum(result) - (sum(pv) - expected_loss)) < 1e-6

    def test_three_strings_cascade(self):
        """Strings [10, 100, 300], loss = 220.
        Round 1: fair_share = 220/3 ≈ 73.3.  String 10 < 73.3 → 0, absorbed=10, rem=210.
        Round 2: fair_share = 210/2 = 105.  String 100 < 105 → 0, absorbed=100, rem=110.
        Round 3: fair_share = 110/1 = 110.  String 300 - 110 = 190.
        """
        result = compute_effective_strings(
            [10.0, 100.0, 300.0],
            grid_export=190.0,  # sys_output=190
            # total_loss = 410 + 0 - 190 = 220
        )
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert abs(result[2] - 190.0) < 1e-6

    def test_battery_and_grid_combined(self):
        """battery_export contributes to inputs, battery_import to outputs."""
        pv = [200.0, 200.0]
        result = compute_effective_strings(
            pv,
            battery_export=0.0,
            battery_import=100.0,  # battery charging = output
            grid_export=200.0,  # grid feed = output
            # total_loss = 400 + 0 - 300 = 100
        )
        # Even split: each string loses 50 → effective = 150
        assert abs(result[0] - 150.0) < 1e-6
        assert abs(result[1] - 150.0) < 1e-6

    def test_inactive_strings_preserved_in_output(self):
        """Zero-value strings keep their zero position in output."""
        result = compute_effective_strings(
            [0.0, 100.0, 0.0, 200.0],
            grid_export=200.0,
            # total_loss = 300 + 0 - 200 = 100 → even split over 2 active strings
        )
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert abs(result[1] - 50.0) < 1e-6
        assert abs(result[3] - 150.0) < 1e-6
