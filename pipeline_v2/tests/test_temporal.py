import unittest

import math

from pipeline_v2.temporal import add_months, build_calendar_labels, feature_availability


class CalendarContractTests(unittest.TestCase):
    def test_month_arithmetic_crosses_calendar_year(self):
        self.assertEqual(add_months("2025-12", 1), "2026-01")

    def test_label_does_not_skip_missing_calendar_month(self):
        rows = [
            {"month": "2025-08", "occupation_group": "G", "u": 0.04, "release_date": "2025-10-03"},
            {"month": "2025-09", "occupation_group": "G", "u": 0.05, "release_date": "2025-11-07"},
            {"month": "2025-11", "occupation_group": "G", "u": 0.06, "release_date": "2026-01-09"},
        ]
        label = next(row for row in build_calendar_labels(rows) if row["origin_month"] == "2025-08")
        self.assertEqual([label[f"target_month_{index}"] for index in (1, 2, 3)], [
            "2025-09", "2025-10", "2025-11"
        ])
        self.assertIsNone(label["y"])
        self.assertEqual(label["valid_reason"], "missing_calendar_month:2025-10")

    def test_label_requires_all_three_calendar_months_and_maturity_dates(self):
        rows = [
            {"month": "2024-01", "occupation_group": "G", "u": 0.03, "release_date": "2024-02-02"},
            {"month": "2024-02", "occupation_group": "G", "u": 0.06, "release_date": "2024-03-08"},
            {"month": "2024-03", "occupation_group": "G", "u": 0.09, "release_date": "2024-04-05"},
            {"month": "2024-04", "occupation_group": "G", "u": 0.12, "release_date": "2024-05-03"},
        ]
        label = next(row for row in build_calendar_labels(rows) if row["origin_month"] == "2024-01")
        self.assertAlmostEqual(label["y"], 0.09)
        self.assertEqual(label["label_available_at"], "2024-05-03")

    def test_nan_publication_dates_remain_unknown(self):
        rows = [
            {"month": "2024-01", "occupation_group": "G", "u": 0.03, "release_date": math.nan},
            {"month": "2024-02", "occupation_group": "G", "u": 0.06, "release_date": math.nan},
            {"month": "2024-03", "occupation_group": "G", "u": 0.09, "release_date": math.nan},
            {"month": "2024-04", "occupation_group": "G", "u": 0.12, "release_date": math.nan},
        ]
        label = next(row for row in build_calendar_labels(rows) if row["origin_month"] == "2024-01")
        self.assertIsNone(label["label_available_at"])

    def test_strict_asof_rejects_late_release_and_retrospective_marks_backfill(self):
        args = dict(source_kind="signals", reference_month="2024-07", origin_month="2024-08",
                    release_date="2024-09-15",
                    forecast_date="2024-09-01")
        self.assertEqual(feature_availability(**args, mode="strict_asof"), {
            "available": False, "status": "released_after_forecast"
        })
        self.assertEqual(feature_availability(**args, mode="retrospective", allow_historical_backfill=True), {
            "available": True,
            "status": "historical_backfill_not_realtime",
            "evaluation_label": "non_realtime_backtest",
        })
        self.assertEqual(feature_availability(**args, mode="retrospective"), {
            "available": False, "status": "released_after_forecast"
        })

    def test_acs_late_release_is_never_allowed_as_historical_backfill(self):
        args = dict(source_kind="acs", reference_month="2024-07", origin_month="2024-08",
                    release_date="2024-09-15", forecast_date="2024-09-01",
                    mode="retrospective", allow_historical_backfill=True)
        self.assertEqual(feature_availability(**args), {
            "available": False, "status": "released_after_forecast"
        })

    def test_acs_unknown_release_date_is_never_allowed_as_historical_backfill(self):
        args = dict(source_kind="acs", reference_month="2024-07", origin_month="2024-08",
                    release_date=None, forecast_date="2024-09-01",
                    mode="retrospective", allow_historical_backfill=True)
        self.assertEqual(feature_availability(**args), {
            "available": False, "status": "release_date_unknown"
        })

    def test_future_reference_month_is_rejected_in_every_mode(self):
        result = feature_availability(source_kind="acs", reference_month="2024-09", origin_month="2024-08",
                                      release_date="2024-09-01", forecast_date="2024-09-10",
                                      mode="retrospective")
        self.assertEqual(result, {"available": False, "status": "future_reference_period"})

    def test_unknown_release_date_is_allowed_only_for_explicit_retrospective_backfill(self):
        args = dict(source_kind="signals", reference_month="2024-07", origin_month="2024-08", release_date=None,
                    forecast_date="2024-09-01")
        self.assertEqual(feature_availability(**args, mode="strict_asof", allow_historical_backfill=True), {
            "available": False, "status": "release_date_unknown"
        })
        self.assertEqual(feature_availability(**args, mode="retrospective", allow_historical_backfill=True), {
            "available": True,
            "status": "historical_backfill_release_date_unknown",
            "evaluation_label": "non_realtime_backtest",
        })
        self.assertEqual(feature_availability(**args, mode="retrospective"), {
            "available": False, "status": "release_date_unknown"
        })
        self.assertEqual(feature_availability(**args, mode="retrospective", allow_historical_backfill=True), {
            "available": True,
            "status": "historical_backfill_release_date_unknown",
            "evaluation_label": "non_realtime_backtest",
        })
