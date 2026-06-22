"""Tests for shadylib.effective – compute_effective_strings."""

from __future__ import annotations


from shadylib import compute_effective_strings


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestComputeEffectiveStringsEdgeCases:
    def test_empty_input_returns_empty(self):
        assert compute_effective_strings([]) == []

    def test_no_loss_when_export_exceeds_pv(self):
        """net_export ≥ pv_sum → total_loss ≤ 0 → originals returned."""
        result = compute_effective_strings([100.0, 200.0], net_export=500.0)
        assert result == [100.0, 200.0]

    def test_zero_pv_strings_remain_zero(self):
        result = compute_effective_strings([0.0, 100.0, 0.0], net_export=50.0)
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_total_loss_greater_than_sum_all_zero(self):
        """total_loss ≥ pv_sum → all strings → 0."""
        result = compute_effective_strings([100.0, 200.0], net_import=500.0)
        assert result == [0.0, 0.0]

    def test_negative_pv_treated_as_inactive(self):
        result = compute_effective_strings([-10.0, 200.0])
        assert result[0] == 0.0

    def test_all_zero_pv_returns_all_zero(self):
        result = compute_effective_strings([0.0, 0.0, 0.0], net_export=100.0)
        assert result == [0.0, 0.0, 0.0]

    def test_result_same_length_as_input(self):
        pv = [100.0, 200.0, 300.0]
        result = compute_effective_strings(pv, net_import=50.0)
        assert len(result) == len(pv)

    def test_no_system_sensors_full_loss(self):
        """Without any system sensors: total_loss = pv_sum + 0 - 0 = pv_sum → all zero.

        PV strings are the inputs; without any export configured, everything is lost.
        """
        result = compute_effective_strings([100.0, 200.0])
        assert result == [0.0, 0.0]

    def test_perfect_balance_returns_originals(self):
        """When net_export exactly equals pv_sum, loss = 0."""
        pv = [100.0, 200.0]
        result = compute_effective_strings(pv, net_export=300.0)
        assert result == [100.0, 200.0]

    def test_all_values_non_negative(self):
        result = compute_effective_strings(
            [50.0, 100.0, 150.0],
            net_import=80.0,
            net_export=20.0,
        )
        assert all(v >= 0.0 for v in result)


# ---------------------------------------------------------------------------
# Waterfall algorithm correctness
# ---------------------------------------------------------------------------


class TestWaterfallAlgorithm:
    def test_even_loss_no_floor_hit(self):
        """100 W loss across two equal 200 W strings → each 150 W effective.

        total_loss = 200+200 + 0 - 300 = 100
        """
        result = compute_effective_strings([200.0, 200.0], net_export=300.0)
        assert abs(result[0] - 150.0) < 1e-6
        assert abs(result[1] - 150.0) < 1e-6

    def test_small_string_absorbs_to_zero_remainder_goes_to_large(self):
        """Loss = 150, strings [50, 200].

        total_loss = 50+200 + 0 - 100 = 150
        Fair share = 75.  String 50 < 75 → 0, absorbed=50, rem=100.
        String 200 - 100 = 100 effective.
        """
        result = compute_effective_strings([50.0, 200.0], net_export=100.0)
        assert result[0] == 0.0
        assert abs(result[1] - 100.0) < 1e-6

    def test_order_independence_of_result(self):
        """Reversing input order yields same effective values (reordered)."""
        pv = [50.0, 100.0, 200.0]
        pv_rev = list(reversed(pv))
        r1 = compute_effective_strings(pv, net_export=150.0)
        r2 = compute_effective_strings(pv_rev, net_export=150.0)
        assert sorted(r1) == sorted(r2)

    def test_sum_of_effective_plus_loss_equals_pv_sum(self):
        """sum(effective) + total_loss_absorbed must equal pv_sum."""
        pv = [80.0, 120.0, 300.0]
        net_import = 50.0
        net_export = 200.0
        result = compute_effective_strings(
            pv, net_import=net_import, net_export=net_export
        )
        expected_loss = max(0.0, min(sum(pv) + net_import - net_export, sum(pv)))
        assert abs(sum(result) - (sum(pv) - expected_loss)) < 1e-6

    def test_three_strings_cascade(self):
        """Strings [10, 100, 300], loss = 220.

        total_loss = 410 + 0 - 190 = 220
        Round 1: fair_share = 220/3 ≈ 73.3. String 10 → 0, absorbed=10, rem=210.
        Round 2: fair_share = 210/2 = 105.  String 100 → 0, absorbed=100, rem=110.
        Round 3: fair_share = 110/1.        String 300 - 110 = 190.
        """
        result = compute_effective_strings([10.0, 100.0, 300.0], net_export=190.0)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert abs(result[2] - 190.0) < 1e-6

    def test_import_and_export_combined(self):
        """net_import=50, net_export=300 over pv=[200,200].

        total_loss = 400 + 50 - 300 = 150 → each string loses 75 → effective = 125
        """
        result = compute_effective_strings(
            [200.0, 200.0],
            net_import=50.0,
            net_export=300.0,
        )
        assert abs(result[0] - 125.0) < 1e-6
        assert abs(result[1] - 125.0) < 1e-6

    def test_inactive_strings_preserved_in_output(self):
        """Zero-value strings keep their zero position in output.

        total_loss = 300 + 0 - 200 = 100 → even split over 2 active strings (50 each)
        """
        result = compute_effective_strings(
            [0.0, 100.0, 0.0, 200.0],
            net_export=200.0,
        )
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert abs(result[1] - 50.0) < 1e-6
        assert abs(result[3] - 150.0) < 1e-6
