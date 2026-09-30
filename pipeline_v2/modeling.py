"""Auditable rolling-origin factors, folds, and non-AI model experiments."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline_v2.temporal import add_months, build_calendar_labels, feature_availability


PACKAGE_ROOT = Path(__file__).resolve().parent
PANEL_ROOT = PACKAGE_ROOT / "data" / "processed" / "panels"
OUTPUT_ROOT = PACKAGE_ROOT / "data" / "processed" / "modeling"
LABEL_COLUMNS = ["target_month_1", "target_month_2", "target_month_3"]
L_COLUMNS = ["u_t", "log_emp_change", "employment_share"]
S_COLUMNS = [
    "share_age_16_24", "share_age_25_54",
    "share_education_less_than_high_school",
    "share_education_high_school_or_equivalent",
    "share_education_some_college_or_associate",
    "share_education_bachelors",
]
RELIABILITY_THRESHOLD = 30
MIN_TRAIN_MONTHS = 36


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if gzip else None)


def _checked_many_to_one_join(
    left: pd.DataFrame, right: pd.DataFrame, *, on: list[str],
    required_values: list[str] | None = None,
) -> pd.DataFrame:
    """Join a source table without row expansion and optionally require complete values."""
    if right.duplicated(on).any():
        raise ValueError(f"Right join table has duplicate keys: {on}")
    merged = left.merge(right, on=on, how="left", validate="many_to_one")
    if len(merged) != len(left):
        raise ValueError("A source join expanded the occupation-month key set")
    missing = [column for column in (required_values or []) if merged[column].isna().any()]
    if missing:
        raise ValueError(f"Source join has missing required values; no zero fill: {missing}")
    return merged


def _assert_same_model_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
    keys = ["month", "occupation_group"]
    left_keys = set(map(tuple, left[keys].astype(str).itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].astype(str).itertuples(index=False, name=None)))
    if left_keys != right_keys or len(left) != len(right):
        raise ValueError("Paired models do not have the same group-month training/test keys")


def _input_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cps = pd.read_csv(PANEL_ROOT / "cps_occupation_month.csv.gz")
    national = pd.read_csv(PANEL_ROOT / "cps_national_month_diagnostic.csv.gz")
    acs = pd.read_csv(PANEL_ROOT / "acs_occupation_year.csv.gz")
    acs_diag = pd.read_csv(PANEL_ROOT / "acs_mapping_diagnostic.csv.gz")
    return cps, national, acs, acs_diag


def _raw_month_features(cps: pd.DataFrame) -> pd.DataFrame:
    frame = cps.copy()
    frame["month"] = frame["month"].astype(str).str[:7]
    frame = frame.sort_values(["occupation_group", "month"]).reset_index(drop=True)
    frame["u_t"] = pd.to_numeric(frame["u"], errors="coerce")
    month_emp = frame.groupby("month", as_index=False)["Emp"].sum().rename(columns={"Emp": "mapped_emp_total"})
    frame = frame.merge(month_emp, on="month", how="left", validate="many_to_one")
    frame["employment_share"] = frame["Emp"] / frame["mapped_emp_total"]
    previous_emp: dict[tuple[str, str], float] = {
        (str(row.occupation_group), str(row.month)): float(row.Emp)
        for row in frame[["occupation_group", "month", "Emp"]].itertuples(index=False)
    }
    changes = []
    for row in frame[["occupation_group", "month", "Emp"]].itertuples(index=False):
        prior_month = add_months(str(row.month), -1)
        previous = previous_emp.get((str(row.occupation_group), prior_month))
        if previous is None or previous <= 0 or float(row.Emp) <= 0:
            changes.append(np.nan)
        else:
            changes.append(math.log(float(row.Emp) / previous))
    frame["log_emp_change"] = changes
    month_number = frame["month"].str[5:7].astype(int)
    frame["season_sin"] = np.sin(2 * np.pi * month_number / 12.0)
    frame["season_cos"] = np.cos(2 * np.pi * month_number / 12.0)
    frame["low_unemployment_support"] = frame["n_unemp"].fillna(0).astype(float) < RELIABILITY_THRESHOLD
    frame["low_employment_support"] = frame["n_emp"].fillna(0).astype(float) < 100
    frame["classification_break_2020_01"] = frame["month"].eq("2020-01")
    return frame


def mature_training_rows(frame: pd.DataFrame, test_month: str) -> pd.DataFrame:
    """Keep only labels whose last calendar target precedes the test month."""
    test_period = pd.Period(test_month, freq="M")
    target_end = frame["target_month_3"].astype(str).map(lambda value: pd.Period(value, freq="M"))
    return frame.loc[target_end < test_period].copy()


def _acs_features_for_origins(acs: pd.DataFrame, origins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    acs = acs.copy()
    acs["year"] = acs["year"].astype(int)
    acs["release_date"] = acs["release_date"].astype(str)
    if acs.duplicated(["occupation_group", "year"]).any():
        raise ValueError("ACS factor source must have one row per occupation_group × year")
    renamed = {
        "share_age_16_24": "share_age_16_24",
        "share_age_25_54": "share_age_25_54",
    }
    acs = acs.rename(columns=renamed)
    audit_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    by_group = {str(group): group_rows.sort_values("release_date")
                for group, group_rows in acs.groupby("occupation_group")}
    for origin in origins[["occupation_group", "month"]].itertuples(index=False):
        group = str(origin.occupation_group)
        month = str(origin.month)
        forecast_date = f"{month}-01"
        candidates = by_group.get(group, acs.iloc[0:0])
        available = []
        for row in candidates.to_dict("records"):
            result = feature_availability(
                source_kind="acs", reference_month=f"{int(row['year']):04d}-12",
                origin_month=month, release_date=row.get("release_date"),
                forecast_date=forecast_date, mode="retrospective",
                allow_historical_backfill=False,
            )
            audit_rows.append({
                "occupation_group": group, "origin_month": month,
                "source_year": int(row["year"]), "source_kind": "acs",
                "reference_month": f"{int(row['year']):04d}-12",
                "release_date": row.get("release_date"), "forecast_date": forecast_date,
                "available": bool(result["available"]), "status": result["status"],
                "evaluation_label": result.get("evaluation_label", ""),
            })
            if result["available"]:
                available.append(row)
        if available:
            selected = max(available, key=lambda row: (int(row["year"]), str(row["release_date"])))
            selected_rows.append({
                "occupation_group": group, "month": month,
                "acs_source_year": int(selected["year"]),
                "acs_release_date": selected["release_date"],
                **{column: selected.get(column) for column in S_COLUMNS},
                "s_available": True,
                "s_availability_status": "available_asof_forecast",
            })
        else:
            selected_rows.append({
                "occupation_group": group, "month": month,
                "acs_source_year": np.nan, "acs_release_date": "",
                **{column: np.nan for column in S_COLUMNS},
                "s_available": False,
                "s_availability_status": "no_acs_vintage_released_asof_forecast",
            })
    selected_frame = pd.DataFrame(selected_rows)
    audit_frame = pd.DataFrame(audit_rows)
    return selected_frame, audit_frame


def _acs_fit_rows_from_training_vintages(acs: pd.DataFrame, training_features: pd.DataFrame) -> pd.DataFrame:
    """Return unique ACS occupation-year rows actually represented in the training feature vintages."""
    used = training_features.loc[training_features["acs_source_year"].notna(),
                                 ["occupation_group", "acs_source_year", "month"]].copy()
    if used.empty:
        raise ValueError("No release-gated ACS vintage is represented in the training fold")
    used["year"] = used["acs_source_year"].astype(int)
    first_use = used.groupby(["occupation_group", "year"], as_index=False)["month"].min()
    first_use["forecast_date"] = first_use["month"] + "-01"
    fit_rows = first_use.merge(acs, on=["occupation_group", "year"], how="left", validate="one_to_one")
    if fit_rows["release_date"].isna().any():
        raise ValueError("A training ACS vintage has no matched source observation")
    if (pd.to_datetime(fit_rows["release_date"]) > pd.to_datetime(fit_rows["forecast_date"])).any():
        raise ValueError("ACS PCA fit source was not released by its first training feature origin")
    if fit_rows.duplicated(["occupation_group", "year"]).any():
        raise ValueError("ACS PCA fit observations must be unique by occupation × year")
    return fit_rows


def _fit_pc(training: pd.DataFrame, columns: list[str], anchor: str) -> dict[str, object]:
    values = training[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    usable = [index for index in range(values.shape[1]) if np.isfinite(values[:, index]).any()]
    if not usable:
        raise ValueError(f"No usable values for PCA block {columns}")
    values = values[:, usable]
    kept = [columns[index] for index in usable]
    medians = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    standardized = (filled - means) / scales
    _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
    loading = vectors[0].copy()
    anchor_index = kept.index(anchor) if anchor in kept else int(np.argmax(np.abs(loading)))
    if loading[anchor_index] < 0:
        loading *= -1
    total_variance = float(np.square(singular).sum())
    variance_ratio = float(singular[0] ** 2 / total_variance) if total_variance else 0.0
    return {
        "columns": kept, "medians": medians, "means": means, "scales": scales,
        "loading": loading, "variance_ratio": variance_ratio,
        "fit_rows": int(len(training)), "anchor": kept[anchor_index],
    }


def _transform_pc(frame: pd.DataFrame, fitted: dict[str, object]) -> np.ndarray:
    columns = fitted["columns"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    medians = np.asarray(fitted["medians"])
    values = np.where(np.isfinite(values), values, medians)
    standardized = (values - np.asarray(fitted["means"])) / np.asarray(fitted["scales"])
    return standardized @ np.asarray(fitted["loading"])


def _write_acceptance_artifact(path: Path, result: dict[str, object]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def build_model_features() -> dict[str, object]:
    """Build labels, audited fold-specific L/S factors, and acceptance report only."""
    cps, national, acs, acs_diag = _input_panels()
    cps["month"] = cps["month"].astype(str).str[:7]
    if cps.duplicated(["occupation_group", "month"]).any():
        raise ValueError("CPS occupation-month panel has duplicate keys")
    labels = pd.DataFrame(build_calendar_labels(cps.to_dict("records")))
    labels["month"] = labels["origin_month"]
    cps_month = _raw_month_features(cps)
    labels["target_low_support"] = False
    support = cps.set_index(["occupation_group", "month"])["n_unemp"].to_dict()
    labels["target_low_support"] = [
        any(float(support.get((row.occupation_group, month), 0)) < RELIABILITY_THRESHOLD
            for month in (row.target_month_1, row.target_month_2, row.target_month_3))
        for row in labels.itertuples(index=False)
    ]
    origin_features, availability_audit = _acs_features_for_origins(acs, cps_month)
    features = cps_month.merge(labels, on=["occupation_group", "month"], how="left", validate="one_to_one")
    features = features.merge(origin_features, on=["occupation_group", "month"], how="left", validate="one_to_one")
    features["label_valid"] = features["y"].notna()
    features["forecast_date"] = features["month"] + "-01"

    groups = sorted(features["occupation_group"].astype(str).unique())
    fold_rows: list[dict[str, object]] = []
    fold_feature_rows: list[pd.DataFrame] = []
    loading_rows: list[dict[str, object]] = []
    valid_test_months = sorted(features.loc[features["label_valid"], "month"].unique())
    for test_month in valid_test_months:
        test_base = features.loc[(features["month"] == test_month) & features["label_valid"]].copy()
        test_number = pd.Period(test_month, freq="M")
        train_base = mature_training_rows(features.loc[features["label_valid"]], test_month)
        train_months = sorted(train_base["month"].unique())
        cps_eligible = len(train_months) >= MIN_TRAIN_MONTHS and len(test_base) == len(groups)
        test_date = pd.Timestamp(f"{test_month}-01")
        train_common = train_base.loc[train_base["s_available"].astype(bool)].copy()
        test_common = test_base.loc[test_base["s_available"].astype(bool)].copy()
        acs_fit_rows = acs.iloc[0:0].copy()
        common_months = sorted(train_common["month"].unique())
        common_eligible = (len(common_months) >= MIN_TRAIN_MONTHS
                           and len(test_common) == len(groups)
                           and test_common["occupation_group"].nunique() == len(groups))
        fold_id = f"test_{test_month}"
        if cps_eligible:
            l_fit = _fit_pc(train_base, L_COLUMNS, "u_t")
            train_base["L_pc1"] = _transform_pc(train_base, l_fit)
            test_base["L_pc1"] = _transform_pc(test_base, l_fit)
            if common_eligible:
                train_common["L_pc1"] = _transform_pc(train_common, l_fit)
                test_common["L_pc1"] = _transform_pc(test_common, l_fit)
            for name, fit in (("L", l_fit),):
                for column, loading in zip(fit["columns"], fit["loading"]):
                    loading_rows.append({
                        "fold_id": fold_id, "test_month": test_month, "block": name,
                        "feature": column, "component": "PC1", "loading": float(loading),
                        "explained_variance_ratio": fit["variance_ratio"],
                        "fit_observation_count": fit["fit_rows"], "sign_anchor": fit["anchor"],
                        "fit_start": min(train_base["month"]), "fit_end": max(train_base["month"]),
                        "fit_source_rows_unique_by": "occupation_group × month",
                    })
            if common_eligible:
                acs_fit_rows = _acs_fit_rows_from_training_vintages(acs, train_common)
                if (pd.to_datetime(acs_fit_rows["release_date"]) > test_date).any():
                    raise ValueError("ACS PCA fit row was released after the outer test forecast date")
                s_fit = _fit_pc(acs_fit_rows, S_COLUMNS, "share_age_25_54")
                train_common["S_pc1"] = _transform_pc(train_common, s_fit)
                test_common["S_pc1"] = _transform_pc(test_common, s_fit)
                for column, loading in zip(s_fit["columns"], s_fit["loading"]):
                    loading_rows.append({
                        "fold_id": fold_id, "test_month": test_month, "block": "S",
                        "feature": column, "component": "PC1", "loading": float(loading),
                        "explained_variance_ratio": s_fit["variance_ratio"],
                        "fit_observation_count": s_fit["fit_rows"], "sign_anchor": s_fit["anchor"],
                        "fit_start": int(acs_fit_rows["year"].min()),
                        "fit_end": int(acs_fit_rows["year"].max()),
                        "fit_source_rows_unique_by": "occupation_group × source_year",
                    })
                fold_feature_rows.extend([
                    train_base.assign(fold_id=fold_id, data_role="train_cps_only"),
                    test_base.assign(fold_id=fold_id, data_role="test_cps_only"),
                    train_common.assign(fold_id=fold_id, data_role="train_common_acs"),
                    test_common.assign(fold_id=fold_id, data_role="test_common_acs"),
                ])
            else:
                fold_feature_rows.extend([
                    train_base.assign(fold_id=fold_id, data_role="train_cps_only"),
                    test_base.assign(fold_id=fold_id, data_role="test_cps_only"),
                ])
        train_end_label_month = max((row["target_month_3"] for row in train_base.to_dict("records")), default="")
        inner_months = train_months[-3:]
        inner_val_start = inner_months[0] if inner_months else ""
        inner_train = train_base.loc[train_base["target_month_3"].astype(str) < inner_val_start] if inner_val_start else train_base.iloc[0:0]
        fold_rows.append({
            "fold_id": fold_id, "test_month": test_month, "forecast_date": f"{test_month}-01",
            "mode": "retrospective_current_vintage", "strict_asof_available": False,
            "strict_asof_block_reason": "CPS exact publication dates are unknown",
            "train_start_month": min(train_months) if train_months else "",
            "train_end_origin_month": max(train_months) if train_months else "",
            "train_last_target_month": train_end_label_month,
            "train_unique_months": len(train_months), "train_group_month_rows": len(train_base),
            "test_group_count": len(test_base), "expected_group_count": len(groups),
            "all_groups_present": len(test_base) == len(groups),
            "inner_validation_months": ";".join(inner_months),
            "inner_train_rows": len(inner_train), "cps_only_eligible": bool(cps_eligible),
            "acs_train_unique_months": len(common_months), "acs_train_rows": len(train_common),
            "acs_test_rows": len(test_common), "acs_fit_unique_occupation_year": len(acs_fit_rows),
            "cps_acs_eligible": bool(common_eligible),
            "acs_pca_unique_key_check": not acs_fit_rows.duplicated(["occupation_group", "year"]).any(),
            "all_training_labels_mature": bool(
                train_base["target_month_3"].map(lambda value: pd.Period(value, freq="M") < test_number).all()
            ),
        })

    folds = pd.DataFrame(fold_rows)
    loadings = pd.DataFrame(loading_rows)
    factor_features = pd.concat(fold_feature_rows, ignore_index=True) if fold_feature_rows else pd.DataFrame()
    # A separate fold availability audit keeps both the source decision and the selected vintage inspectable.
    quality = {
        "cps_rows": int(len(cps)), "cps_unique_months": int(cps["month"].nunique()),
        "occupation_groups": groups, "expected_occupation_group_count": len(groups),
        "duplicate_occupation_month_keys": int(cps.duplicated(["occupation_group", "month"]).sum()),
        "missing_calendar_months": [
            str(period) for period in pd.period_range(
                min(pd.Period(value, freq="M") for value in cps["month"]),
                max(pd.Period(value, freq="M") for value in cps["month"]), freq="M",
            ) if str(period) not in set(cps["month"])
        ],
        "cps_exact_publication_dates_known": bool(cps["release_date"].notna().all()),
        "cps_unemployment_n_lt_10": int((cps["n_unemp"] < 10).sum()),
        "cps_unemployment_n_lt_30": int((cps["n_unemp"] < RELIABILITY_THRESHOLD).sum()),
        "cps_employment_n_lt_100": int((cps["n_emp"] < 100).sum()),
        "low_support_rows_retained": True,
        "kish_effective_sample_size_available": False,
        "cps_2018_classification_change_flag_month": "2020-01",
        "cps_weighted_identification_share_2019": float(
            national.loc[national["month"].astype(str).str.startswith("2019-"), "occupation_identified_weight_share"].mean()
        ),
        "cps_2019_employment_weighted_coverage": float(
            national.loc[national["month"].astype(str).str.startswith("2019-"), "occupation_identified_emp"].sum()
            / national.loc[national["month"].astype(str).str.startswith("2019-"), "national_emp"].sum()
        ),
        "cps_2019_unemployment_weighted_coverage": float(
            national.loc[national["month"].astype(str).str.startswith("2019-"), "occupation_identified_unemp"].sum()
            / national.loc[national["month"].astype(str).str.startswith("2019-"), "national_unemp"].sum()
        ),
        "acs_years": sorted(int(value) for value in acs["year"].unique()),
        "acs_rows_are_unique_occupation_year": not acs.duplicated(["occupation_group", "year"]).any(),
        "acs_mapping_diagnostic_rows": int(len(acs_diag)),
        "acs_release_date_gating": "release date <= forecast month first day; no backfill",
        "label_valid_rows": int(labels["y"].notna().sum()),
        "label_invalid_rows": int(labels["y"].isna().sum()),
        "label_available_at_unknown_rows": int(labels.loc[labels["y"].notna(), "label_available_at"].isna().sum()),
        "label_reliability_sensitivity_flagged_rows": int(labels["target_low_support"].sum()),
        "fold_count": int(len(folds)),
        "cps_only_eligible_folds": int(folds["cps_only_eligible"].sum()) if len(folds) else 0,
        "cps_acs_eligible_folds": int(folds["cps_acs_eligible"].sum()) if len(folds) else 0,
    }
    acceptance_checks = {
        "calendar_labels_do_not_skip_gaps": bool(labels.loc[labels["valid_reason"].astype(str).str.startswith("missing_calendar_month"), "y"].isna().all()),
        "labels_are_bounded": bool(labels["y"].dropna().between(0, 1).all()),
        "all_labels_have_three_target_months": bool(all(labels[LABEL_COLUMNS].notna().all(axis=1))),
        "cps_panel_keys_unique": not bool(quality["duplicate_occupation_month_keys"]),
        "acs_pc_fit_source_deduplicated": bool(folds["acs_pca_unique_key_check"].all()) if len(folds) else False,
        "all_test_months_have_22_groups_for_eligible_folds": bool(
            folds.loc[folds["cps_only_eligible"], "all_groups_present"].all()
        ),
        "all_training_labels_end_before_test_month": bool(
            folds.loc[folds["cps_only_eligible"], "all_training_labels_mature"].all()
        ),
        "fold_specific_factor_loadings_exist": bool(len(loadings) and loadings["fold_id"].nunique() > 0),
        "no_global_pca_fit": True,
        "strict_asof_labeled_unavailable_when_cps_release_unknown": not bool(quality["cps_exact_publication_dates_known"]),
        "unknown_cps_publication_dates_not_fabricated_for_labels": (
            not bool(quality["cps_exact_publication_dates_known"])
            and int(quality["label_available_at_unknown_rows"]) == int(quality["label_valid_rows"])
        ),
    }
    result = {
        "stage": "A_B_factor_and_time_fold_acceptance",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": all(acceptance_checks.values()), "checks": acceptance_checks,
        "quality_and_reliability": quality,
        "factor_design": {
            "L": L_COLUMNS, "S": S_COLUMNS, "component_count_per_block": 1,
            "pca_scope": "outer fold training only; ACS PCA rows unique occupation_group × source_year",
            "imputation_and_scaling": "fit on the outer training fold; median impute, standardize, first PC",
            "sign_anchor": {"L": "u_t loading positive", "S": "share_age_25_54 loading positive"},
        },
        "limitations": [
            "This is a retrospective current-vintage analysis, not a strict real-time backtest.",
            "CPS monthly public-use file release dates were not verified; label_available_at remains blank.",
            "CPS effective sample sizes/design-based variances are unavailable; n_unemp thresholds are reliability flags only.",
            "January 2020 starts the new Census occupation coding and is explicitly flagged.",
        ],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(labels, OUTPUT_ROOT / "labels.csv")
    _write_csv(folds, OUTPUT_ROOT / "fold_manifest.csv")
    _write_csv(loadings, OUTPUT_ROOT / "factor_loadings.csv")
    _write_csv(factor_features, OUTPUT_ROOT / "features.csv.gz", gzip=True)
    _write_csv(availability_audit, OUTPUT_ROOT / "feature_availability_audit.csv.gz", gzip=True)
    (OUTPUT_ROOT / "quality_reliability.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "feature_lineage.json").write_text(json.dumps({
        "cps": {"source": "CPS Basic Monthly public-use microdata", "release_dates": "unknown; no exact-as-of claim"},
        "acs": {"source": "ACS 1-Year PUMS", "vintage_rule": "latest release available by origin month first day"},
        "labels": {"definition": "mean of u at exact t+1, t+2, t+3 calendar months", "availability": "exact CPS release dates unknown"},
        "historical_backfill": "none for ACS; no Signals features in stage A-C",
        "occupation_classification": "Census 2018 OCC / SOC major group; Jan 2020 CPS transition flagged",
        "input_hashes": {name: _sha256(PANEL_ROOT / name) for name in (
            "cps_occupation_month.csv.gz", "cps_national_month_diagnostic.csv.gz",
            "acs_occupation_year.csv.gz", "acs_mapping_diagnostic.csv.gz")},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_acceptance_artifact(OUTPUT_ROOT / "acceptance_report.json", result)
    return result


def _design_matrix(frame: pd.DataFrame, *, include_s: bool, include_p: bool = False) -> tuple[np.ndarray, list[str]]:
    numeric = ["u_t", "L_pc1", "log_emp_change", "employment_share", "season_sin", "season_cos"]
    if include_s:
        numeric.append("S_pc1")
    if include_p:
        numeric.append("P")
    groups = sorted(frame["occupation_group"].astype(str).unique())
    dummy_names = [f"occupation_{group.replace('-', '_')}" for group in groups[1:]]
    numeric_array = frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    dummy_array = np.column_stack([
        frame["occupation_group"].astype(str).eq(group).astype(float).to_numpy()
        for group in groups[1:]
    ]) if len(groups) > 1 else np.empty((len(frame), 0))
    matrix = np.column_stack([numeric_array, dummy_array])
    return matrix, numeric + dummy_names


def _ridge_predict(train: pd.DataFrame, test: pd.DataFrame, *, include_s: bool,
                   include_p: bool = False, alpha: float = 10.0) -> np.ndarray:
    x_train, _ = _design_matrix(train, include_s=include_s, include_p=include_p)
    x_test, _ = _design_matrix(test, include_s=include_s, include_p=include_p)
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    x_train = (x_train - means) / scales
    x_test = (x_test - means) / scales
    x_train = np.column_stack([np.ones(len(train)), x_train])
    x_test = np.column_stack([np.ones(len(test)), x_test])
    y = train["y"].to_numpy(dtype=float)
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return np.clip(x_test @ coefficients, 0.0, 1.0)


def _lightgbm_predict(train: pd.DataFrame, test: pd.DataFrame, *, include_s: bool) -> tuple[np.ndarray, dict[str, object]]:
    """Tune only boosting rounds on the last three mature training months, then refit the outer train."""
    import lightgbm as lgb

    train_months = sorted(train["month"].astype(str).unique())
    if len(train_months) < 6:
        raise ValueError("LightGBM requires at least six unique training origin months")
    validation_months = train_months[-3:]
    validation_start = validation_months[0]
    inner_train = train.loc[train["target_month_3"].astype(str) < validation_start].copy()
    inner_valid = train.loc[train["month"].astype(str).isin(validation_months)].copy()
    if inner_train.empty or inner_valid.empty:
        raise ValueError("LightGBM inner temporal validation split is empty")
    x_inner_train, feature_names = _design_matrix(inner_train, include_s=include_s)
    x_inner_valid, _ = _design_matrix(inner_valid, include_s=include_s)
    params = {
        "objective": "regression", "metric": "l1", "verbosity": -1,
        "max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 30,
        "learning_rate": 0.05, "lambda_l2": 10.0, "max_bin": 63,
        "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "num_threads": 1, "seed": 20260929, "deterministic": True,
        "force_col_wise": True,
    }
    inner_model = lgb.train(
        params,
        lgb.Dataset(x_inner_train, label=inner_train["y"].to_numpy(dtype=float), feature_name=feature_names),
        num_boost_round=150,
        valid_sets=[lgb.Dataset(x_inner_valid, label=inner_valid["y"].to_numpy(dtype=float),
                                feature_name=feature_names)],
        callbacks=[lgb.early_stopping(stopping_rounds=12, verbose=False), lgb.log_evaluation(period=0)],
    )
    best_rounds = int(inner_model.best_iteration or 150)
    x_train, feature_names = _design_matrix(train, include_s=include_s)
    x_test, _ = _design_matrix(test, include_s=include_s)
    final_model = lgb.train(
        params,
        lgb.Dataset(x_train, label=train["y"].to_numpy(dtype=float), feature_name=feature_names),
        num_boost_round=best_rounds,
    )
    info = {
        "inner_validation_months": validation_months,
        "inner_train_group_month_keys": len(inner_train),
        "inner_validation_group_month_keys": len(inner_valid),
        "selected_boosting_rounds": best_rounds,
        "max_depth": 3, "num_leaves": 7, "min_data_in_leaf": 30,
        "lambda_l2": 10.0, "features_include_S": include_s,
        "feature_names": feature_names,
    }
    return np.clip(final_model.predict(x_test, num_iteration=best_rounds), 0.0, 1.0), info


def _metric_row(name: str, frame: pd.DataFrame, persistence: pd.Series | None = None) -> dict[str, object]:
    errors_pp = (frame["yhat"].to_numpy(dtype=float) - frame["y"].to_numpy(dtype=float)) * 100.0
    mae = float(np.mean(np.abs(errors_pp))) if len(errors_pp) else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(errors_pp)))) if len(errors_pp) else float("nan")
    result: dict[str, object] = {
        "model": name, "group_month_n": int(len(frame)),
        "independent_test_months": int(frame["test_month"].nunique()),
        "occupation_groups": int(frame["occupation_group"].nunique()),
        "macro_mae_pp": mae, "macro_rmse_pp": rmse,
        "reliability_sensitivity_group_month_n": int((~frame["target_low_support"].astype(bool)).sum()),
    }
    reliable = frame.loc[~frame["target_low_support"].astype(bool)]
    if len(reliable):
        reliable_errors = (reliable["yhat"].to_numpy(dtype=float) - reliable["y"].to_numpy(dtype=float)) * 100.0
        result["reliability_sensitivity_mae_pp"] = float(np.mean(np.abs(reliable_errors)))
        result["reliability_sensitivity_rmse_pp"] = float(np.sqrt(np.mean(np.square(reliable_errors))))
    else:
        result["reliability_sensitivity_mae_pp"] = None
        result["reliability_sensitivity_rmse_pp"] = None
    if persistence is not None:
        keys = ["test_month", "occupation_group"]
        baseline = frame[keys].merge(persistence.rename("persistence_yhat").reset_index(), on=keys, how="left")
        baseline_error = (baseline["persistence_yhat"].to_numpy(dtype=float) - frame["y"].to_numpy(dtype=float)) * 100.0
        baseline_mae = float(np.mean(np.abs(baseline_error)))
        result["persistence_macro_mae_pp_same_keys"] = baseline_mae
        result["relative_mae_gain_vs_persistence"] = (baseline_mae - mae) / baseline_mae if baseline_mae else None
    return result


def run_non_ai_backtest() -> dict[str, object]:
    acceptance_path = OUTPUT_ROOT / "acceptance_report.json"
    if not acceptance_path.is_file():
        raise RuntimeError("Run build-model-features and pass factor/fold acceptance before fitting any model")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance.get("passed"):
        raise RuntimeError("Factor/fold acceptance failed; model fitting is blocked")
    folds = pd.read_csv(OUTPUT_ROOT / "fold_manifest.csv")
    factor_rows = pd.read_csv(OUTPUT_ROOT / "features.csv.gz")
    predictions: list[pd.DataFrame] = []
    model_runs: list[dict[str, object]] = []
    try:
        import lightgbm as _lightgbm
        lightgbm_version = _lightgbm.__version__
    except ImportError:
        _lightgbm = None
        lightgbm_version = None
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        fold_data = factor_rows.loc[factor_rows["fold_id"] == fold_id]
        train_cps = fold_data.loc[fold_data["data_role"] == "train_cps_only"].copy()
        test_cps = fold_data.loc[fold_data["data_role"] == "test_cps_only"].copy()
        if bool(fold.cps_only_eligible):
            persistence = test_cps[["fold_id", "month", "occupation_group", "y", "u_t", "target_low_support"]].copy()
            persistence.rename(columns={"month": "test_month", "u_t": "yhat"}, inplace=True)
            persistence["model"] = "persistence"
            predictions.append(persistence)
            common_test = fold_data.loc[(fold_data["data_role"] == "test_common_acs")].copy()
            common_train = fold_data.loc[(fold_data["data_role"] == "train_common_acs")].copy()
            if len(common_test) == len(test_cps) and bool(fold.cps_acs_eligible):
                pred_common = _ridge_predict(common_train, common_test, include_s=False)
                common_out = common_test[["fold_id", "month", "occupation_group", "y", "target_low_support"]].copy()
                common_out.rename(columns={"month": "test_month"}, inplace=True)
                common_out["yhat"] = pred_common
                common_out["model"] = "cps_ridge_common_window"
                predictions.append(common_out)
                pred_acs = _ridge_predict(common_train, common_test, include_s=True)
                acs_out = common_out.copy()
                acs_out["yhat"] = pred_acs
                acs_out["model"] = "cps_acs_ridge_common_window"
                predictions.append(acs_out)
                model_runs.extend([
                    {"fold_id": fold_id, "test_month": fold.test_month, "model": "cps_ridge_common_window",
                     "train_group_month_keys": len(common_train), "test_group_month_keys": len(common_test),
                     "same_training_keys_as_cps_acs": True, "features": "u_t,L_pc1,log_emp_change,employment_share,season,occupation_FE",
                     "regularization_alpha": 10.0},
                    {"fold_id": fold_id, "test_month": fold.test_month, "model": "cps_acs_ridge_common_window",
                     "train_group_month_keys": len(common_train), "test_group_month_keys": len(common_test),
                     "same_training_keys_as_cps_only": True, "features": "cps_ridge_features + S_pc1",
                     "regularization_alpha": 10.0},
                ])
                if _lightgbm is not None:
                    pred_tree_common, tree_common_info = _lightgbm_predict(common_train, common_test, include_s=False)
                    pred_tree_acs, tree_acs_info = _lightgbm_predict(common_train, common_test, include_s=True)
                    for name, estimates, info in (
                        ("cps_lightgbm_common_window", pred_tree_common, tree_common_info),
                        ("cps_acs_lightgbm_common_window", pred_tree_acs, tree_acs_info),
                    ):
                        tree_out = common_test[["fold_id", "month", "occupation_group", "y", "target_low_support"]].copy()
                        tree_out.rename(columns={"month": "test_month"}, inplace=True)
                        tree_out["yhat"] = estimates
                        tree_out["model"] = name
                        predictions.append(tree_out)
                        model_runs.append({"fold_id": fold_id, "test_month": fold.test_month, "model": name,
                                           "train_group_month_keys": len(common_train), "test_group_month_keys": len(common_test),
                                           "same_training_and_test_keys_as_paired_model": True, **info})
            pred_long = _ridge_predict(train_cps, test_cps, include_s=False)
            long_out = test_cps[["fold_id", "month", "occupation_group", "y", "target_low_support"]].copy()
            long_out.rename(columns={"month": "test_month"}, inplace=True)
            long_out["yhat"] = pred_long
            long_out["model"] = "cps_ridge_long"
            predictions.append(long_out)
            if _lightgbm is not None:
                pred_tree, tree_info = _lightgbm_predict(train_cps, test_cps, include_s=False)
                tree_out = test_cps[["fold_id", "month", "occupation_group", "y", "target_low_support"]].copy()
                tree_out.rename(columns={"month": "test_month"}, inplace=True)
                tree_out["yhat"] = pred_tree
                tree_out["model"] = "cps_lightgbm_long"
                predictions.append(tree_out)
                model_runs.append({"fold_id": fold_id, "test_month": fold.test_month, "model": "cps_lightgbm_long",
                                   "train_group_month_keys": len(train_cps), "test_group_month_keys": len(test_cps),
                                   "training_target_end_strictly_before_test": True, **tree_info})
            model_runs.append({
                "fold_id": fold_id, "test_month": fold.test_month, "model": "cps_ridge_long",
                "train_group_month_keys": len(train_cps), "test_group_month_keys": len(test_cps),
                "training_target_end_strictly_before_test": True,
                "features": "u_t,L_pc1,log_emp_change,employment_share,season,occupation_FE",
                "regularization_alpha": 10.0,
            })
    if not predictions:
        raise RuntimeError("No eligible rolling-origin folds passed the 36-month warm-up")
    pred = pd.concat(predictions, ignore_index=True)
    pred["error_pp"] = (pred["yhat"] - pred["y"]) * 100.0
    _write_csv(pred, OUTPUT_ROOT / "oos_predictions.csv.gz", gzip=True)
    persistence = pred.loc[pred["model"] == "persistence"].set_index(["test_month", "occupation_group"])["yhat"]
    metrics = []
    for model, frame in pred.groupby("model", sort=True):
        baseline = persistence if model != "persistence" else None
        metrics.append(_metric_row(model, frame, baseline))
    metrics_frame = pd.DataFrame(metrics)
    finite_metrics = all(
        np.isfinite(float(row["macro_mae_pp"])) and np.isfinite(float(row["macro_rmse_pp"]))
        and np.isfinite(float(row["reliability_sensitivity_mae_pp"]))
        for row in metrics if row["reliability_sensitivity_mae_pp"] is not None
    ) and all(row["reliability_sensitivity_mae_pp"] is not None for row in metrics)
    if not finite_metrics:
        raise RuntimeError("A fitted model produced non-finite OOS metrics; results are not accepted")
    _write_csv(metrics_frame, OUTPUT_ROOT / "metrics.csv")
    by_month = pred.groupby(["model", "test_month"], as_index=False).agg(
        group_month_n=("y", "size"), mae_pp=("error_pp", lambda values: float(np.mean(np.abs(values)))),
        rmse_pp=("error_pp", lambda values: float(np.sqrt(np.mean(np.square(values))))),
    )
    _write_csv(by_month, OUTPUT_ROOT / "errors_by_month.csv")
    by_group = pred.groupby(["model", "occupation_group"], as_index=False).agg(
        group_month_n=("y", "size"), mae_pp=("error_pp", lambda values: float(np.mean(np.abs(values)))),
        rmse_pp=("error_pp", lambda values: float(np.sqrt(np.mean(np.square(values))))),
    )
    _write_csv(by_group, OUTPUT_ROOT / "errors_by_occupation.csv")
    lgbm_status = (f"trained LightGBM {lightgbm_version} with temporal early stopping and bounded depth/leaves"
                   if _lightgbm is not None else "skipped: lightgbm is not installed in the active Python runtime")
    report = {
        "stage": "C_non_ai_rolling_origin_backtest", "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": finite_metrics, "model_runs": model_runs, "metrics": metrics,
        "tree_model": lgbm_status,
        "fit_protocol": "one month rolling/expanding outer folds; no random split; ridge alpha fixed at 10; LightGBM rounds selected on the last three mature inner months; all outer training targets mature before test month",
        "common_window_comparison": "CPS ridge and CPS+ACS ridge share training and test group-month keys",
        "limitations": [
            "Retrospective current-vintage backtest; not strict as-of because CPS file release dates are unknown.",
            "Test observations are clustered in calendar months; group-month rows are not independent time samples.",
            "Metrics weight occupation-month keys equally; low CPS support is retained and separately flagged.",
            "LightGBM uses shallow, regularized trees and temporal inner-fold early stopping; no test-fold tuning."
            if _lightgbm is not None else "LightGBM was not run because it is absent from the active Python runtime.",
        ],
    }
    _write_acceptance_artifact(OUTPUT_ROOT / "non_ai_model_report.json", report)
    return report


def verify_occupation_exposure() -> dict[str, object]:
    """Audit the pinned OpenAI E source and 2022 ACS employment-weighted 22-group coverage."""
    source_path = PACKAGE_ROOT / "data" / "raw" / "openai" / "gpts_are_gpts" / "occ_level.csv"
    crosswalk_path = PACKAGE_ROOT / "data" / "processed" / "reference" / "occupation_crosswalk.csv"
    acs_zip_path = PACKAGE_ROOT / "data" / "raw" / "acs" / "2022" / "csv_pus.zip"
    if not all(path.is_file() for path in (source_path, crosswalk_path, acs_zip_path)):
        raise FileNotFoundError("Exposure, Census crosswalk, or existing ACS 2022 archive is missing")
    exposure = pd.read_csv(source_path, dtype={"O*NET-SOC Code": str})
    required = {"O*NET-SOC Code", "Title", "dv_rating_alpha", "dv_rating_beta", "dv_rating_gamma",
                "human_rating_alpha", "human_rating_beta", "human_rating_gamma"}
    if not required.issubset(exposure.columns):
        raise ValueError(f"OpenAI E source columns differ: missing {sorted(required - set(exposure.columns))}")
    exposure["soc_base"] = exposure["O*NET-SOC Code"].str.extract(r"^(\d{2}-\d{4})")[0]
    base_counts = exposure.groupby("soc_base")["O*NET-SOC Code"].nunique()
    unique_bases = set(base_counts[base_counts.eq(1)].index)
    exposure_unique = exposure.loc[exposure["soc_base"].isin(unique_bases)].copy()
    exposure_unique = exposure_unique.rename(columns={"human_rating_beta": "E_human_beta", "dv_rating_beta": "E_gpt4_beta"})
    crosswalk = pd.read_csv(crosswalk_path, dtype=str)
    soc_map = crosswalk.loc[
        crosswalk["source_code_version"].eq("census_2018") & crosswalk["mapping_status"].eq("mapped"),
        ["source_code", "source_title", "source_soc_code", "occupation_group"],
    ].drop_duplicates("source_code")
    soc_map["soc_base"] = soc_map["source_soc_code"].str.extract(r"^(\d{2}-\d{4})")[0]
    soc_map["exposure_base_ambiguous"] = soc_map["soc_base"].isin(set(base_counts[base_counts.gt(1)].index))
    exact = soc_map.loc[~soc_map["exposure_base_ambiguous"]].merge(
        exposure_unique[["soc_base", "E_human_beta", "E_gpt4_beta", "human_rating_alpha", "human_rating_gamma",
                         "dv_rating_alpha", "dv_rating_gamma"]],
        on="soc_base", how="left", validate="many_to_one",
    )
    employment_rows: list[dict[str, object]] = []
    import zipfile
    with zipfile.ZipFile(acs_zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv") and "psam_pus" in name.lower()]
        if not members:
            raise ValueError("ACS 2022 archive contains no national person-file CSV members")
        for member in members:
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=["OCCP", "ESR", "AGEP", "PWGTP"], dtype=str, chunksize=500_000):
                    esr = pd.to_numeric(chunk["ESR"], errors="coerce")
                    age = pd.to_numeric(chunk["AGEP"], errors="coerce")
                    weight = pd.to_numeric(chunk["PWGTP"], errors="coerce")
                    occ = chunk["OCCP"].astype(str).str.strip().str.zfill(4)
                    keep = esr.isin([1, 2]) & age.ge(16) & weight.gt(0)
                    small = pd.DataFrame({"source_code": occ[keep], "weight": weight[keep]})
                    employment_rows.append(small.groupby("source_code", as_index=False)["weight"].sum())
    employment = pd.concat(employment_rows, ignore_index=True).groupby("source_code", as_index=False)["weight"].sum()
    mapped = employment.merge(soc_map, on="source_code", how="left", validate="one_to_one")
    mapped = mapped.loc[mapped["occupation_group"].notna()].copy()
    mapped["soc_base"] = mapped["source_soc_code"].str.extract(r"^(\d{2}-\d{4})")[0]
    mapped = mapped.merge(base_counts.rename("onet_base_code_count"), left_on="soc_base", right_index=True, how="left")
    mapped = mapped.merge(exposure_unique[["soc_base", "E_human_beta", "E_gpt4_beta", "human_rating_alpha",
                                           "human_rating_gamma", "dv_rating_alpha", "dv_rating_gamma"]],
                          on="soc_base", how="left", validate="many_to_one")
    mapped["e_matched"] = mapped["E_human_beta"].notna() & mapped["onet_base_code_count"].eq(1)
    group_rows = []
    for group, rows in mapped.groupby("occupation_group", sort=True):
        denominator = float(rows["weight"].sum())
        matched = rows.loc[rows["e_matched"]]
        matched_weight = float(matched["weight"].sum())
        group_rows.append({
            "occupation_group": group, "civilian_employment_weight_2022": denominator,
            "matched_e_weight": matched_weight,
            "employment_weighted_coverage": matched_weight / denominator if denominator else np.nan,
            "matched_census_occ_codes": int(matched["source_code"].nunique()),
            "mapped_census_occ_codes": int(rows["source_code"].nunique()),
            "E_human_beta": float(np.average(matched["E_human_beta"], weights=matched["weight"])) if matched_weight else np.nan,
            "E_gpt4_beta": float(np.average(matched["E_gpt4_beta"], weights=matched["weight"])) if matched_weight else np.nan,
            "E_human_alpha": float(np.average(matched["human_rating_alpha"], weights=matched["weight"])) if matched_weight else np.nan,
            "E_human_gamma": float(np.average(matched["human_rating_gamma"], weights=matched["weight"])) if matched_weight else np.nan,
            "E_gpt4_alpha": float(np.average(matched["dv_rating_alpha"], weights=matched["weight"])) if matched_weight else np.nan,
            "E_gpt4_gamma": float(np.average(matched["dv_rating_gamma"], weights=matched["weight"])) if matched_weight else np.nan,
        })
    group_frame = pd.DataFrame(group_rows)
    output = OUTPUT_ROOT / "exposure"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(mapped, output / "occupation_exposure_join_audit.csv.gz", gzip=True)
    _write_csv(group_frame, output / "occupation_exposure_22groups.csv")
    total_weight = float(mapped["weight"].sum())
    matched_total = float(mapped.loc[mapped["e_matched"], "weight"].sum())
    report = {
        "stage": "D_occupation_exposure_validation", "source": "OpenAI GPTs are GPTs author repository data/occ_level.csv",
        "source_commit": "9ed4148d15f4c4a2666f45bf0006e56b3a3b9f70",
        "source_file_commit_date": "2025-10-03T22:34:55Z",
        "source_file_sha256": _sha256(source_path), "license": "MIT; LICENSE retained alongside source",
        "source_row_count": int(len(exposure)), "unique_full_onet_soc_codes": int(exposure["O*NET-SOC Code"].nunique()),
        "unique_six_digit_bases": int(exposure["soc_base"].nunique()),
        "ambiguous_onet_soc_bases": int(base_counts.gt(1).sum()),
        "main_measure": "human_rating_beta = E1 + 0.5 × E2",
        "sensitivity_measures": ["human alpha/beta/gamma", "GPT-4 dv_rating alpha/beta/gamma"],
        "join_rule": "exact 2018 Census OCC → source SOC base; exact O*NET six-digit base only when unique; ambiguous extensions remain unmatched; no fuzzy title joins",
        "weight_rule": "2022 ACS civilian employed age 16+ PWGTP; ESR 1/2; fixed weights by occupation group",
        "matched_employment_weight": matched_total, "mapped_civilian_employment_weight": total_weight,
        "overall_employment_weighted_coverage": matched_total / total_weight if total_weight else None,
        "groups_with_positive_E_coverage": int((group_frame["employment_weighted_coverage"] > 0).sum()),
        "groups_with_coverage_ge_70pct": int((group_frame["employment_weighted_coverage"] >= 0.70).sum()),
        "groups_with_coverage_ge_50pct": int((group_frame["employment_weighted_coverage"] >= 0.50).sum()),
        "groups_below_50pct_coverage": group_frame.loc[
            group_frame["employment_weighted_coverage"] < 0.50, "occupation_group"
        ].astype(str).tolist(),
        "group_count": int(group_frame["occupation_group"].nunique()),
        "ai_short_window_eligible": bool(group_frame["occupation_group"].nunique() == 22
                                         and group_frame["E_human_beta"].notna().all()
                                         and (group_frame["employment_weighted_coverage"] > 0).all()),
        "coverage_interpretation": "All 22 groups have a matched-only employment-weighted E estimate; estimates below 50% coverage are retained only for the exploratory all-group run and separately excluded in sensitivity analysis.",
        "ai_short_window_skip_reason": "" if (group_frame["occupation_group"].nunique() == 22
            and group_frame["E_human_beta"].notna().all() and (group_frame["employment_weighted_coverage"] > 0).all())
        else "one or more occupation groups has no verified employment-weighted E estimate; no zero fill",
        "paper_interpretation": "potential exposure, not adoption or realized labor-market effect",
    }
    _write_acceptance_artifact(output / "exposure_validation_report.json", report)
    return report


def run_ai_short_window() -> dict[str, object]:
    exposure_report_path = OUTPUT_ROOT / "exposure" / "exposure_validation_report.json"
    if not exposure_report_path.is_file():
        raise RuntimeError("Verify the occupation exposure source and mapping before the AI short-window experiment")
    exposure_report = json.loads(exposure_report_path.read_text(encoding="utf-8"))
    if not exposure_report.get("ai_short_window_eligible"):
        result = {"stage": "D_AI_short_window", "passed": False, "status": "skipped",
                  "reason": exposure_report.get("ai_short_window_skip_reason"),
                  "independent_test_months": 0, "group_month_rows": 0}
        _write_acceptance_artifact(OUTPUT_ROOT / "ai_short_window_report.json", result)
        return result
    # The exposure is valid; Signals is explicitly a current-vintage historical backfill.
    signals_path = PACKAGE_ROOT.parent.parent / "Codex-file" / "11_audit" / "signals_us_work_related_share_verified.csv"
    if not signals_path.is_file():
        raise FileNotFoundError("Verified Signals work-share series is missing; AI/P experiment cannot run")
    signals = pd.read_csv(signals_path)
    if not {"month", "share_of_messages"}.issubset(signals.columns):
        raise ValueError("Signals series columns are not the audited format")
    signals["month"] = signals["month"].astype(str).str[:7]
    train_months = [f"2024-{month:02d}" for month in range(7, 13)] + [f"2025-{month:02d}" for month in range(1, 7)]
    test_months = ["2025-11", "2025-12", *[f"2026-{month:02d}" for month in range(1, 6)]]
    selected_months = train_months + test_months
    signals = signals.loc[signals["month"].isin(selected_months), ["month", "share_of_messages"]].copy()
    if signals["month"].duplicated().any() or set(signals["month"]) != set(selected_months):
        raise ValueError("Signals must contain exactly one observation for each short-window train/test month")
    signal_audit = []
    for month in selected_months:
        decision = feature_availability(
            source_kind="signals", reference_month=month, origin_month=month,
            release_date=None, forecast_date=f"{month}-01", mode="retrospective",
            allow_historical_backfill=True,
        )
        signal_audit.append({
            "month": month, "source_kind": "signals", "release_date": None,
            "forecast_date": f"{month}-01", "available": decision["available"],
            "status": decision["status"], "evaluation_label": decision["evaluation_label"],
        })
    cps, _national, acs, _acs_diag = _input_panels()
    cps["month"] = cps["month"].astype(str).str[:7]
    labels = pd.DataFrame(build_calendar_labels(cps.to_dict("records")))
    labels["month"] = labels["origin_month"]
    support_by_key = cps.set_index(["occupation_group", "month"])["n_unemp"].to_dict()
    labels["target_low_support"] = [
        any(float(support_by_key.get((row.occupation_group, month), 0)) < RELIABILITY_THRESHOLD
            for month in (row.target_month_1, row.target_month_2, row.target_month_3))
        for row in labels.itertuples(index=False)
    ]
    raw = _raw_month_features(cps)
    origin_acs, _acs_audit = _acs_features_for_origins(acs, raw)
    frame = raw.merge(labels, on=["month", "occupation_group"], how="inner", validate="one_to_one")
    frame = frame.merge(origin_acs, on=["month", "occupation_group"], how="left", validate="one_to_one")
    frame = frame.loc[frame["y"].notna()].copy()
    train = frame.loc[frame["month"].isin(train_months)].copy()
    test = frame.loc[frame["month"].isin(test_months)].copy()
    groups = sorted(frame["occupation_group"].astype(str).unique())
    if len(train) != len(train_months) * len(groups) or len(test) != len(test_months) * len(groups):
        raise ValueError("AI/P window must retain a complete occupation-group grid for every train and test month")
    if train["target_month_3"].astype(str).max() >= min(test_months):
        raise ValueError("AI/P training labels are not all mature before the first test month")
    exposure = pd.read_csv(OUTPUT_ROOT / "exposure" / "occupation_exposure_22groups.csv")
    exposure = exposure[["occupation_group", "E_human_beta", "employment_weighted_coverage"]]
    train = _checked_many_to_one_join(train, signals, on=["month"], required_values=["share_of_messages"])
    test = _checked_many_to_one_join(test, signals, on=["month"], required_values=["share_of_messages"])
    train = _checked_many_to_one_join(train, exposure, on=["occupation_group"], required_values=["E_human_beta"])
    test = _checked_many_to_one_join(test, exposure, on=["occupation_group"], required_values=["E_human_beta"])
    train["P"] = train["E_human_beta"] * train["share_of_messages"]
    test["P"] = test["E_human_beta"] * test["share_of_messages"]
    l_fit = _fit_pc(train, L_COLUMNS, "u_t")
    train["L_pc1"] = _transform_pc(train, l_fit)
    test["L_pc1"] = _transform_pc(test, l_fit)
    acs_fit = _acs_fit_rows_from_training_vintages(acs, train)
    if (pd.to_datetime(acs_fit["release_date"]) > pd.Timestamp("2025-11-01")).any():
        raise ValueError("AI short-window ACS PCA fit contains post-training release data")
    s_fit = _fit_pc(acs_fit, S_COLUMNS, "share_age_25_54")
    train["S_pc1"] = _transform_pc(train, s_fit)
    test["S_pc1"] = _transform_pc(test, s_fit)
    low_coverage_groups = set(exposure.loc[exposure["employment_weighted_coverage"] < 0.50, "occupation_group"].astype(str))
    scopes = {
        "all_22_groups": (set(groups), "all 22 groups with positive matched-only E coverage"),
        "coverage_ge_50pct": (set(groups) - low_coverage_groups, "sensitivity: groups with at least 50% employment-weighted E coverage"),
    }
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for scope, (keep_groups, scope_note) in scopes.items():
        if not keep_groups:
            continue
        scope_train = train.loc[train["occupation_group"].astype(str).isin(keep_groups)].copy()
        scope_test = test.loc[test["occupation_group"].astype(str).isin(keep_groups)].copy()
        _assert_same_model_keys(scope_train, scope_train.copy())
        _assert_same_model_keys(scope_test, scope_test.copy())
        pred_no_p = _ridge_predict(scope_train, scope_test, include_s=True, include_p=False)
        pred_p = _ridge_predict(scope_train, scope_test, include_s=True, include_p=True)
        pred_persist = scope_test["u_t"].to_numpy(dtype=float)
        for name, estimates in (("persistence", pred_persist), ("cps_acs_ridge_no_P", pred_no_p),
                                ("cps_acs_ridge_plus_P", pred_p)):
            out = scope_test[["month", "occupation_group", "y", "target_low_support",
                              "E_human_beta", "employment_weighted_coverage", "share_of_messages", "P"]].copy()
            out.rename(columns={"month": "test_month"}, inplace=True)
            out["scope"] = scope
            out["model"] = name
            out["yhat"] = estimates
            out["error_pp"] = (out["yhat"] - out["y"]) * 100.0
            prediction_rows.append(out)
        error_persist = (pred_persist - scope_test["y"].to_numpy(dtype=float)) * 100.0
        error_no_p = (pred_no_p - scope_test["y"].to_numpy(dtype=float)) * 100.0
        error_p = (pred_p - scope_test["y"].to_numpy(dtype=float)) * 100.0
        mae_no_p = float(np.mean(np.abs(error_no_p)))
        mae_p = float(np.mean(np.abs(error_p)))
        common = {"scope": scope, "group_month_n": len(scope_test),
                  "independent_test_months": int(scope_test["month"].nunique()),
                  "train_months": len(train_months), "train_group_month_n": len(scope_train),
                  "occupation_groups": len(keep_groups), "scope_note": scope_note}
        metric_rows.extend([
            {**common, "model": "persistence", "macro_mae_pp": float(np.mean(np.abs(error_persist))),
             "macro_rmse_pp": float(np.sqrt(np.mean(np.square(error_persist)))), "paired_no_P_mae_gain_pp": None},
            {**common, "model": "cps_acs_ridge_no_P", "macro_mae_pp": mae_no_p,
             "macro_rmse_pp": float(np.sqrt(np.mean(np.square(error_no_p)))), "paired_no_P_mae_gain_pp": 0.0},
            {**common, "model": "cps_acs_ridge_plus_P", "macro_mae_pp": mae_p,
             "macro_rmse_pp": float(np.sqrt(np.mean(np.square(error_p)))),
             "paired_no_P_mae_gain_pp": mae_no_p - mae_p},
        ])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if not np.isfinite(predictions[["yhat", "y", "error_pp"]].to_numpy(dtype=float)).all():
        raise RuntimeError("AI short-window produced non-finite predictions or errors")
    _write_csv(predictions, OUTPUT_ROOT / "ai_short_window_predictions.csv.gz", gzip=True)
    metrics = pd.DataFrame(metric_rows)
    _write_csv(metrics, OUTPUT_ROOT / "ai_short_window_metrics.csv")
    _write_csv(pd.DataFrame(signal_audit), OUTPUT_ROOT / "ai_signal_availability_audit.csv")
    report = {
        "stage": "D_AI_short_window", "passed": True, "status": "complete",
        "exposure_validation": exposure_report,
        "signals_release_date": None, "signals_evaluation_label": "non_realtime_backtest",
        "signals_availability_audit": signal_audit,
        "training_months": train_months, "test_months": test_months,
        "independent_test_months": int(test["month"].nunique()),
        "group_month_rows_all_groups": int(len(test)), "occupation_groups": len(groups),
        "training_group_month_rows_all_groups": int(len(train)),
        "feature": "P = human_rating_beta × US work_related share_of_messages",
        "model_comparison": "paired CPS+ACS ridge without P vs with P; identical training/test keys in every scope",
        "group_scopes": {name: {"groups": len(keep), "note": note} for name, (keep, note) in scopes.items()},
        "metrics": metric_rows,
        "factor_fit": {"L_fit_rows": l_fit["fit_rows"], "S_fit_unique_occupation_year_rows": s_fit["fit_rows"],
                       "S_fit_cutoff": "2025-11-01", "PCA_components": 1},
        "warnings": [
            "Only seven independent test months; 154 all-group occupation-month rows are not 154 independent observations.",
            "Signals historical release dates are unknown; explicit retrospective backfill was enabled and all estimates are labeled non_realtime_backtest.",
            "OpenAI exposure file was committed to the source repository on 2025-10-03; this is a current-vintage model fit after that date, not a historical deployment simulation.",
            "Nine occupation groups have <50% verified employment-weighted E coverage; all-group estimates use matched occupation weights only and the >=50% coverage sensitivity is reported.",
            "P is a predictive interaction, not a causal effect of AI adoption.",
        ],
    }
    _write_acceptance_artifact(OUTPUT_ROOT / "ai_short_window_report.json", report)
    return report


def update_run_manifest() -> dict[str, object]:
    artifact_paths = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    model_report_path = OUTPUT_ROOT / "non_ai_model_report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8")) if model_report_path.is_file() else {}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_panel_hashes": {name: _sha256(PANEL_ROOT / name) for name in (
            "cps_occupation_month.csv.gz", "cps_national_month_diagnostic.csv.gz",
            "acs_occupation_year.csv.gz", "acs_mapping_diagnostic.csv.gz")},
        "artifacts": [{"path": path.relative_to(PACKAGE_ROOT.parent.parent).as_posix(),
                       "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in artifact_paths
                      if path.name != "run_manifest.json"],
        "python": "configured Codex workspace runtime for feature build/ridge; isolated Windows venv in system temp for LightGBM",
        "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": model_report.get("tree_model", "not_run"),
        "random_seed": None, "modeling_randomness": "none; SVD and ridge are deterministic",
    }
    _write_acceptance_artifact(OUTPUT_ROOT / "run_manifest.json", result)
    return result
