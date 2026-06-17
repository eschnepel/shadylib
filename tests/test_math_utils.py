"""Tests for math_utils – enforce_monotonic and filter_gap_successors."""

from __future__ import annotations

from datetime import datetime, timezone


from shadylib import enforce_monotonic, filter_gap_successors

UTC = timezone.utc


def _dt(hour: int, minute: int = 0, day: int = 1, second: int = 0) -> datetime:
    return datetime(2025, 6, day, hour, minute, second, tzinfo=UTC)


def _rows(pairs: list[tuple[datetime, float]]) -> list[dict]:
    return [{"start": ts, "mean": v} for ts, v in pairs]


# ---------------------------------------------------------------------------
# enforce_monotonic
# ---------------------------------------------------------------------------


class TestEnforceMonotonic:
    def test_empty_returns_empty(self) -> None:
        assert enforce_monotonic([]) == []

    def test_single_row_returned_unchanged(self) -> None:
        rows = _rows([(_dt(10, 0), 100.0)])
        assert enforce_monotonic(rows) == rows

    def test_strictly_increasing_all_kept(self) -> None:
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 5), 20.0), (_dt(10, 10), 30.0)])
        assert enforce_monotonic(rows) == rows

    def test_flat_equal_values_kept(self) -> None:
        """f(n) == f(n+1) is valid (≤ relation)."""
        rows = _rows([(_dt(10, 0), 50.0), (_dt(10, 5), 50.0), (_dt(10, 10), 60.0)])
        assert enforce_monotonic(rows) == rows

    def test_single_reset_at_end_discarded(self) -> None:
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 5), 20.0), (_dt(10, 10), 5.0)])
        result = enforce_monotonic(rows)
        assert len(result) == 2
        assert result[-1]["mean"] == 20.0

    def test_reset_in_middle_discards_until_recovery(self) -> None:
        """After a counter reset the series must not be re-included until the
        running total exceeds the last valid value again."""
        rows = _rows(
            [
                (_dt(10, 0), 100.0),
                (_dt(10, 5), 110.0),
                (_dt(10, 10), 50.0),  # reset – discard
                (_dt(10, 15), 60.0),  # still below 110 – discard
                (_dt(10, 20), 120.0),  # recovered – keep
            ]
        )
        result = enforce_monotonic(rows)
        means = [r["mean"] for r in result]
        assert means == [100.0, 110.0, 120.0]

    def test_multiple_resets_all_discarded(self) -> None:
        rows = _rows(
            [
                (_dt(10, 0), 10.0),
                (_dt(10, 5), 5.0),  # reset
                (_dt(10, 10), 20.0),  # recovered
                (_dt(10, 15), 10.0),  # reset again
                (_dt(10, 20), 25.0),  # recovered
            ]
        )
        result = enforce_monotonic(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 20.0, 25.0]

    def test_all_decreasing_only_first_kept(self) -> None:
        rows = _rows([(_dt(10, 0), 100.0), (_dt(10, 5), 80.0), (_dt(10, 10), 60.0)])
        result = enforce_monotonic(rows)
        assert len(result) == 1
        assert result[0]["mean"] == 100.0

    def test_input_not_mutated(self) -> None:
        rows = _rows([(_dt(10, 0), 20.0), (_dt(10, 5), 10.0)])
        original_len = len(rows)
        enforce_monotonic(rows)
        assert len(rows) == original_len


# ---------------------------------------------------------------------------
# filter_gap_successors
# ---------------------------------------------------------------------------


class TestFilterGapSuccessors:
    def test_empty_returns_empty(self) -> None:
        assert filter_gap_successors([]) == []

    def test_single_row_returned_unchanged(self) -> None:
        rows = _rows([(_dt(10, 0), 50.0)])
        assert filter_gap_successors(rows) == rows

    def test_no_gap_all_kept(self) -> None:
        """Consecutive 5-minute samples → no row discarded."""
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 5), 20.0), (_dt(10, 10), 30.0)])
        assert filter_gap_successors(rows) == rows

    def test_single_missing_slot_successor_removed(self) -> None:
        """Gap of 10 min (one missing slot) → successor at 10:10 is removed."""
        rows = _rows(
            [
                (_dt(10, 0), 10.0),
                (_dt(10, 10), 99.0),  # gap successor – discard
                (_dt(10, 15), 30.0),
            ]
        )
        result = filter_gap_successors(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 30.0]

    def test_multiple_missing_slots_one_successor_removed(self) -> None:
        """Gap of 30 min (five missing slots) → only the one successor removed."""
        rows = _rows(
            [
                (_dt(10, 0), 10.0),
                (_dt(10, 30), 999.0),  # gap successor – discard
                (_dt(10, 35), 30.0),
            ]
        )
        result = filter_gap_successors(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 30.0]

    def test_multiple_separate_gaps_each_successor_removed(self) -> None:
        rows = _rows(
            [
                (_dt(10, 0), 10.0),
                (_dt(10, 10), 99.0),  # gap successor 1 – discard
                (_dt(10, 15), 20.0),
                (_dt(10, 25), 88.0),  # gap successor 2 – discard
                (_dt(10, 30), 30.0),
            ]
        )
        result = filter_gap_successors(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 20.0, 30.0]

    def test_gap_at_end_last_row_removed(self) -> None:
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 5), 20.0), (_dt(10, 20), 999.0)])
        result = filter_gap_successors(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 20.0]

    def test_exact_5min_gap_not_discarded(self) -> None:
        """Exactly 5 minutes between samples is normal – no discard."""
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 5), 20.0)])
        result = filter_gap_successors(rows)
        assert result == rows

    def test_custom_slot_minutes(self) -> None:
        """With slot_minutes=10 a 10-minute gap is normal, 15-min is a gap."""
        rows = _rows(
            [
                (_dt(10, 0), 10.0),
                (_dt(10, 10), 20.0),  # normal with slot=10
                (_dt(10, 25), 99.0),  # gap of 15 min > 10 → discard
                (_dt(10, 35), 30.0),
            ]
        )
        result = filter_gap_successors(rows, slot_minutes=10)
        means = [r["mean"] for r in result]
        assert means == [10.0, 20.0, 30.0]

    def test_string_timestamps_accepted(self) -> None:
        """Rows with ISO-string start values must also be handled."""
        rows = [
            {"start": "2025-06-01T10:00:00+00:00", "mean": 10.0},
            {"start": "2025-06-01T10:10:00+00:00", "mean": 99.0},  # gap successor
            {"start": "2025-06-01T10:15:00+00:00", "mean": 20.0},
        ]
        result = filter_gap_successors(rows)
        means = [r["mean"] for r in result]
        assert means == [10.0, 20.0]

    def test_input_not_mutated(self) -> None:
        rows = _rows([(_dt(10, 0), 10.0), (_dt(10, 10), 99.0)])
        original_len = len(rows)
        filter_gap_successors(rows)
        assert len(rows) == original_len


# ---------------------------------------------------------------------------
# Combined: enforce_monotonic then filter_gap_successors
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    def test_monotonic_before_gap_filter(self) -> None:
        """After a backup restore a counter reset may appear as a large
        apparent gap.  enforce_monotonic discards the reset rows first;
        filter_gap_successors then handles genuine downtime gaps.

        Running totals are strictly non-decreasing, so the gap-successor value
        (999) must be higher than the last valid pre-gap value (115) and the
        following sample (1005) must be higher still.
        """
        rows = _rows(
            [
                (_dt(10, 0), 100.0),
                (_dt(10, 5), 110.0),
                # backup restore: counter resets to 0
                (_dt(10, 10), 0.0),  # monotonic violation → discarded
                (_dt(10, 15), 5.0),  # still < 110 → discarded
                (_dt(10, 20), 115.0),  # recovered
                # genuine downtime gap of 15 min (10:20 → 10:35)
                (
                    _dt(10, 35),
                    999.0,
                ),  # gap successor (accumulated) → discarded by gap filter
                (_dt(10, 40), 1005.0),  # normal continuation
            ]
        )
        after_mono = enforce_monotonic(rows)
        # enforce_monotonic keeps everything that is ≥ previous valid:
        # 100 → 110 → (0 discarded) → (5 discarded) → 115 → 999 → 1005
        assert [r["mean"] for r in after_mono] == [100.0, 110.0, 115.0, 999.0, 1005.0]

        after_gap = filter_gap_successors(after_mono)
        # after_mono is: 100 (10:00), 110 (10:05), 115 (10:20), 999 (10:35), 1005 (10:40)
        # 115 (10:20) follows 110 (10:05) with gap of 15 min > 5 → discarded (the reset
        #   rows were removed by enforce_monotonic, leaving an apparent gap here too)
        # 999 (10:35) follows 115 (10:20) with gap of 15 min > 5 → discarded
        # 1005 (10:40) follows 999 (10:35) with gap of 5 min → kept
        assert [r["mean"] for r in after_gap] == [100.0, 110.0, 1005.0]

    def test_no_data_loss_for_clean_series(self) -> None:
        rows = _rows([(_dt(10, m), float(m)) for m in range(0, 60, 5)])
        assert enforce_monotonic(rows) == rows
        assert filter_gap_successors(rows) == rows
