import unittest

import numpy as np
import pandas as pd

from pipeline_v2.modeling import (
    L_COLUMNS,
    S_COLUMNS,
    _acs_features_for_origins,
    _assert_same_model_keys,
    _checked_many_to_one_join,
    _design_matrix,
    _fit_pc,
    _raw_month_features,
    _transform_pc,
    _acs_fit_rows_from_training_vintages,
    mature_training_rows,
)


class ModelingContractTests(unittest.TestCase):
    def test_employment_change_does_not_skip_a_missing_month(self):
        panel = pd.DataFrame([
            {"month": "2025-08", "occupation_group": "11-0000", "Emp": 100.0,
             "Unemp": 10.0, "LF": 110.0, "u": 10 / 110, "n_emp": 10, "n_unemp": 2},
            {"month": "2025-10", "occupation_group": "11-0000", "Emp": 121.0,
             "Unemp": 11.0, "LF": 132.0, "u": 11 / 132, "n_emp": 11, "n_unemp": 3},
        ])
        result = _raw_month_features(panel).set_index("month")
        self.assertTrue(np.isnan(result.loc["2025-10", "log_emp_change"]))

    def test_training_labels_must_end_before_test_month(self):
        rows = pd.DataFrame([
            {"origin_month": "2025-06", "target_month_3": "2025-09"},
            {"origin_month": "2025-07", "target_month_3": "2025-10"},
            {"origin_month": "2025-08", "target_month_3": "2025-11"},
        ])
        mature = mature_training_rows(rows, "2025-10")
        self.assertEqual(mature["origin_month"].tolist(), ["2025-06"])

    def test_acs_release_is_gated_at_forecast_date_and_late_data_is_not_backfilled(self):
        acs = pd.DataFrame([{
            "year": 2022, "occupation_group": "11-0000", "release_date": "2023-10-19",
            **{column: 0.1 for column in S_COLUMNS},
        }])
        origins = pd.DataFrame([
            {"occupation_group": "11-0000", "month": "2023-10"},
            {"occupation_group": "11-0000", "month": "2023-11"},
        ])
        selected, audit = _acs_features_for_origins(acs, origins)
        selected = selected.set_index("month")
        self.assertFalse(bool(selected.loc["2023-10", "s_available"]))
        self.assertTrue(bool(selected.loc["2023-11", "s_available"]))
        late = audit.loc[(audit["origin_month"] == "2023-10")].iloc[0]
        self.assertEqual(late["status"], "released_after_forecast")

    def test_acs_pca_rejects_duplicated_occupation_year_observations(self):
        acs = pd.DataFrame([
            {"year": 2022, "occupation_group": "11-0000", "release_date": "2023-10-19",
             **{column: 0.1 for column in S_COLUMNS}},
            {"year": 2022, "occupation_group": "11-0000", "release_date": "2023-10-19",
             **{column: 0.2 for column in S_COLUMNS}},
        ])
        with self.assertRaisesRegex(ValueError, "one row per occupation_group × year"):
            _acs_features_for_origins(acs, pd.DataFrame([
                {"occupation_group": "11-0000", "month": "2023-11"},
            ]))

    def test_acs_pca_uses_unique_vintages_represented_in_training_rows(self):
        acs = pd.DataFrame([
            {"year": 2018, "occupation_group": "11-0000", "release_date": "2019-11-14",
             **{column: 0.1 for column in S_COLUMNS}},
            {"year": 2019, "occupation_group": "11-0000", "release_date": "2020-10-15",
             **{column: 0.2 for column in S_COLUMNS}},
        ])
        train = pd.DataFrame([
            {"occupation_group": "11-0000", "month": "2019-12", "acs_source_year": 2018},
            {"occupation_group": "11-0000", "month": "2020-01", "acs_source_year": 2018},
            {"occupation_group": "11-0000", "month": "2021-01", "acs_source_year": 2019},
        ])
        fit = _acs_fit_rows_from_training_vintages(acs, train)
        self.assertEqual(len(fit), 2)
        self.assertFalse(fit.duplicated(["occupation_group", "year"]).any())

    def test_test_observations_are_transformed_without_refitting_pca(self):
        train = pd.DataFrame({
            "u_t": [0.02, 0.03, 0.04], "log_emp_change": [0.01, 0.02, -0.01],
            "employment_share": [0.2, 0.3, 0.5],
        })
        fitted = _fit_pc(train, L_COLUMNS, "u_t")
        original_loading = np.asarray(fitted["loading"]).copy()
        test = train.iloc[[0]].copy()
        test.loc[:, "u_t"] = 0.99
        transformed = _transform_pc(test, fitted)
        np.testing.assert_array_equal(np.asarray(fitted["loading"]), original_loading)
        self.assertEqual(len(transformed), 1)

    def test_source_join_preserves_rows_and_rejects_missing_values_or_duplicates(self):
        left = pd.DataFrame({"month": ["2025-11", "2025-11"], "occupation_group": ["11", "13"]})
        right = pd.DataFrame({"month": ["2025-11"], "A": [0.4]})
        result = _checked_many_to_one_join(left, right, on=["month"], required_values=["A"])
        self.assertEqual(len(result), len(left))
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            _checked_many_to_one_join(left, pd.concat([right, right]), on=["month"])
        with self.assertRaisesRegex(ValueError, "missing required values"):
            _checked_many_to_one_join(left, pd.DataFrame({"month": ["2025-12"], "A": [0.2]}),
                                      on=["month"], required_values=["A"])

    def test_paired_models_require_identical_month_occupation_keys(self):
        left = pd.DataFrame({"month": ["2025-11", "2025-11"], "occupation_group": ["11", "13"]})
        _assert_same_model_keys(left, left.copy())
        with self.assertRaisesRegex(ValueError, "same group-month"):
            _assert_same_model_keys(left, left.iloc[:1])

    def test_tree_model_feature_names_use_safe_identifiers(self):
        frame = pd.DataFrame({
            "occupation_group": ["11-0000", "13-0000"], "u_t": [0.01, 0.02],
            "L_pc1": [0.1, 0.2], "log_emp_change": [0.0, 0.01],
            "employment_share": [0.3, 0.7], "season_sin": [0.0, 0.5], "season_cos": [1.0, 0.5],
        })
        _matrix, names = _design_matrix(frame, include_s=False)
        self.assertTrue(all("[" not in name and "]" not in name for name in names))
        self.assertIn("occupation_13_0000", names)


if __name__ == "__main__":
    unittest.main()
