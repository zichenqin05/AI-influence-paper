"""Rebuild the Phase 5 LUNA charts from frozen pipeline outputs.

Run from the project root with:
    python -m pipeline_v2.phase5_visualization

By default, PNGs are written to the system temporary directory so the project
does not accumulate generated figures. Use --output-dir to choose another
destination. Source evidence and model metrics are read from the project data.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
MODEL_DIR = PIPELINE_DIR / "data" / "processed" / "modeling"
NCES_WORKBOOK = (
    PIPELINE_DIR
    / "data"
    / "raw"
    / "nces"
    / "cip2020_soc2018"
    / "CIP2020_SOC2018_Crosswalk.xlsx"
)
ONET_OCCUPATIONS = PIPELINE_DIR / "data" / "raw" / "onet" / "31.0" / "occupation_data.csv"

COLORS = {
    "background": "#FFFFFF",
    "ink": "#202B36",
    "muted": "#5D6B78",
    "grid": "#DCE3E8",
    "blue": "#2E6F9E",
    "gold": "#D29428",
    "teal": "#438A79",
    "red": "#B44C43",
    "purple": "#7463A8",
    "grey": "#7A8792",
}

MODEL_LABELS = {
    "persistence": "Persistence",
    "cps_ridge_long": "CPS ridge",
    "cps_lightgbm_long": "CPS LightGBM",
    "cps_ridge_common_window": "CPS ridge",
    "cps_acs_ridge_common_window": "CPS + ACS ridge",
    "cps_lightgbm_common_window": "CPS LightGBM",
    "cps_acs_lightgbm_common_window": "CPS + ACS LightGBM",
    "cps_acs_ridge_no_P": "Ridge, no P",
    "cps_acs_ridge_plus_P": "Ridge, plus P",
}

MODEL_COLORS = {
    "persistence": COLORS["grey"],
    "cps_ridge_long": COLORS["blue"],
    "cps_lightgbm_long": COLORS["red"],
    "cps_ridge_common_window": COLORS["blue"],
    "cps_acs_ridge_common_window": COLORS["teal"],
    "cps_lightgbm_common_window": COLORS["purple"],
    "cps_acs_lightgbm_common_window": COLORS["red"],
    "cps_acs_ridge_no_P": COLORS["blue"],
    "cps_acs_ridge_plus_P": COLORS["gold"],
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
        if bold
        else [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT = _font(16)
FONT_SMALL = _font(13)
FONT_MEDIUM = _font(19)
FONT_TITLE = _font(25, bold=True)


def _canvas(width: int, height: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((42, 26), title, font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((43, 65), subtitle, font=FONT, fill=COLORS["muted"])
    return image, draw


def _footer(draw: ImageDraw.ImageDraw, text: str, y: int, x: int = 43) -> None:
    draw.text((x, y), text, font=FONT_SMALL, fill=COLORS["muted"])


def _month_x(month: str, left: int, right: int) -> float:
    period = pd.Period(str(month)[:7], freq="M")
    first = pd.Period("2016-01", freq="M")
    last = pd.Period("2026-08", freq="M")
    return left + (period.ordinal - first.ordinal) / (last.ordinal - first.ordinal) * (right - left)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    return draw.textbbox((0, 0), text, font=font)[2]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and _text_width(draw, candidate, font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_inputs() -> dict[str, object]:
    inputs: dict[str, object] = {
        "loadings": pd.read_csv(MODEL_DIR / "factor_loadings.csv"),
        "metrics": pd.read_csv(MODEL_DIR / "metrics.csv"),
        "ai_metrics": pd.read_csv(MODEL_DIR / "ai_short_window_metrics.csv"),
        "oos": pd.read_csv(MODEL_DIR / "oos_predictions.csv.gz"),
        "ai": pd.read_csv(MODEL_DIR / "ai_short_window_predictions.csv.gz"),
        "exposure": pd.read_csv(MODEL_DIR / "exposure" / "occupation_exposure_22groups.csv"),
        "errors": pd.read_csv(MODEL_DIR / "errors_by_occupation.csv"),
        "folds": pd.read_csv(MODEL_DIR / "fold_manifest.csv"),
        "availability": pd.read_csv(MODEL_DIR / "feature_availability_audit.csv.gz"),
        "ai_availability": pd.read_csv(MODEL_DIR / "ai_signal_availability_audit.csv"),
        "crosswalk": pd.read_excel(NCES_WORKBOOK, sheet_name="SOC-CIP"),
        "onet": pd.read_csv(ONET_OCCUPATIONS),
    }
    return inputs


def _summarize(inputs: dict[str, object]) -> dict[str, object]:
    loadings: pd.DataFrame = inputs["loadings"]
    metrics: pd.DataFrame = inputs["metrics"]
    ai_metrics: pd.DataFrame = inputs["ai_metrics"]
    oos: pd.DataFrame = inputs["oos"]
    ai: pd.DataFrame = inputs["ai"]
    crosswalk: pd.DataFrame = inputs["crosswalk"]

    long_models = ["persistence", "cps_ridge_long", "cps_lightgbm_long"]
    long: dict[str, dict[str, float | int]] = {}
    for model in long_models:
        rows = oos[oos["model"].eq(model)]
        long[model] = {
            "months": int(rows["test_month"].nunique()),
            "groups": int(rows["occupation_group"].nunique()),
            "mae": float(rows["error_pp"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(rows["error_pp"] ** 2))),
        }

    common_models = [
        "cps_ridge_common_window",
        "cps_acs_ridge_common_window",
        "cps_lightgbm_common_window",
        "cps_acs_lightgbm_common_window",
    ]
    keys = ["test_month", "occupation_group"]
    common = {
        model: oos[oos["model"].eq(model)].set_index(keys).sort_index()
        for model in common_models
    }
    common_keys = common[common_models[0]].index
    assert len(common_keys) == 35 * 22, f"Unexpected common key count: {len(common_keys)}"
    assert all(frame.index.equals(common_keys) for frame in common.values()), "Common-window model keys differ"
    persistence = oos[oos["model"].eq("persistence")].set_index(keys).sort_index().loc[common_keys]
    common_metrics: dict[str, dict[str, float]] = {
        "persistence": {
            "mae": float(persistence["error_pp"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(persistence["error_pp"] ** 2))),
        }
    }
    for model, frame in common.items():
        common_metrics[model] = {
            "mae": float(frame["error_pp"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(frame["error_pp"] ** 2))),
        }

    paired_monthly: dict[str, pd.Series] = {}
    paired_overall: dict[str, float] = {}
    for scope in ["all_22_groups", "coverage_ge_50pct"]:
        rows = ai[
            ai["scope"].eq(scope)
            & ai["model"].isin(["cps_acs_ridge_no_P", "cps_acs_ridge_plus_P"])
        ]
        no_p = rows[rows["model"].eq("cps_acs_ridge_no_P")].set_index(keys).sort_index()
        plus_p = rows[rows["model"].eq("cps_acs_ridge_plus_P")].set_index(keys).sort_index()
        assert no_p.index.equals(plus_p.index), f"AI keys differ in {scope}"
        assert np.allclose(no_p["y"], plus_p["y"]), f"AI labels differ in {scope}"
        paired_monthly[scope] = (plus_p["error_pp"].abs() - no_p["error_pp"].abs()).groupby(level="test_month").mean()
        paired_overall[scope] = float(plus_p["error_pp"].abs().mean() - no_p["error_pp"].abs().mean())
        metric_no = ai_metrics[(ai_metrics["scope"].eq(scope)) & (ai_metrics["model"].eq("cps_acs_ridge_no_P"))].iloc[0]
        metric_plus = ai_metrics[(ai_metrics["scope"].eq(scope)) & (ai_metrics["model"].eq("cps_acs_ridge_plus_P"))].iloc[0]
        assert abs(paired_overall[scope] - (metric_plus["macro_mae_pp"] - metric_no["macro_mae_pp"])) < 1e-10

    # Reconcile row-level results to the recorded long-window metrics.
    for model, values in long.items():
        stored = metrics[metrics["model"].eq(model)].iloc[0]
        assert abs(values["mae"] - stored["macro_mae_pp"]) < 1e-10
        assert abs(values["rmse"] - stored["macro_rmse_pp"]) < 1e-10

    expected_links = {"17-2041": 8, "27-4031": 4, "27-4032": 7, "35-2014": 2, "35-1012": 5, "11-9051": 10}
    for code, expected in expected_links.items():
        count = int(crosswalk["SOC2018Code"].astype(str).eq(code).sum())
        assert count == expected, f"Unexpected NCES row count for {code}: {count}"
    assert not crosswalk["SOC2018Code"].astype(str).eq("27-4030").any()
    for code in ["35-3031", "35-3023", "35-9031"]:
        rows = crosswalk[crosswalk["SOC2018Code"].astype(str).eq(code)]
        assert len(rows) == 1 and str(rows.iloc[0]["CIP2020Code"]) == "99.9999"

    assert loadings.groupby("block")["fold_id"].nunique().to_dict() == {"L": 82, "S": 35}
    assert int((inputs["exposure"]["employment_weighted_coverage"] < 0.5).sum()) == 9
    return {
        "long": long,
        "common": common_metrics,
        "paired_monthly": paired_monthly,
        "paired_overall": paired_overall,
    }


def _figure1(inputs: dict[str, object], output: Path) -> Path:
    oos: pd.DataFrame = inputs["oos"]
    ai: pd.DataFrame = inputs["ai"]
    availability: pd.DataFrame = inputs["availability"]
    ai_availability: pd.DataFrame = inputs["ai_availability"]
    image, draw = _canvas(
        1500,
        700,
        "Figure 1. Data and evaluation timeline",
        "Reference periods and evaluation windows; historical scores use current-vintage retrospective inputs",
    )
    left, right = 255, 1450
    rows = [150, 235, 320, 415, 500, 585]
    labels = [
        "CPS monthly panel",
        "ACS annual source vintages",
        "Signals months in AI window",
        "CPS long OOS tests",
        "CPS+ACS common-window tests",
        "AI/P short window",
    ]
    for y, label in zip(rows, labels):
        draw.text((38, y - 12), label, font=FONT, fill=COLORS["ink"])
        draw.line((left, y + 14, right, y + 14), fill=COLORS["grid"], width=1)

    draw.line((_month_x("2016-01", left, right), rows[0], _month_x("2026-08", left, right), rows[0]), fill=COLORS["blue"], width=14)
    gap_x = _month_x("2025-10", left, right)
    draw.rectangle((gap_x - 5, rows[0] - 12, gap_x + 5, rows[0] + 12), fill="white", outline=COLORS["red"], width=3)
    draw.text((gap_x - 47, rows[0] - 37), "2025-10 gap", font=FONT_SMALL, fill=COLORS["red"])

    releases = (
        availability[availability["source_kind"].eq("acs")]
        .dropna(subset=["release_date"])[["source_year", "release_date"]]
        .drop_duplicates()
        .sort_values("release_date")
    )
    for source_year, release_date in releases.itertuples(index=False, name=None):
        x = _month_x(str(release_date)[:7], left, right)
        draw.ellipse((x - 7, rows[1] - 7, x + 7, rows[1] + 7), fill=COLORS["gold"], outline=COLORS["ink"], width=1)
        draw.text((x - 17, rows[1] - 31), str(source_year), font=FONT_SMALL, fill=COLORS["ink"])

    signal_months = sorted(ai_availability["month"].dropna().astype(str).unique())
    if signal_months:
        draw.line((_month_x(signal_months[0], left, right), rows[2], _month_x(signal_months[-1], left, right), rows[2]), fill=COLORS["purple"], width=12)

    for model, y, color in [
        ("cps_ridge_long", rows[3], COLORS["blue"]),
        ("cps_ridge_common_window", rows[4], COLORS["teal"]),
    ]:
        for month in sorted(oos[oos["model"].eq(model)]["test_month"].astype(str).unique()):
            x = _month_x(month, left, right)
            draw.ellipse((x - 2, y - 5, x + 2, y + 5), fill=color)

    draw.line((_month_x("2024-07", left, right), rows[5], _month_x("2025-06", left, right), rows[5]), fill=COLORS["gold"], width=12)
    ai_test_months = sorted(ai[(ai["scope"].eq("all_22_groups")) & (ai["model"].eq("cps_acs_ridge_no_P"))]["test_month"].astype(str).unique())
    for month in ai_test_months:
        x = _month_x(month, left, right)
        draw.ellipse((x - 6, rows[5] - 6, x + 6, rows[5] + 6), fill=COLORS["red"], outline=COLORS["ink"], width=1)
    draw.text((_month_x("2024-07", left, right), rows[5] - 31), "12 training months", font=FONT_SMALL, fill=COLORS["ink"])
    if ai_test_months:
        draw.text((_month_x(ai_test_months[0], left, right) - 13, rows[5] + 15), "7 test months", font=FONT_SMALL, fill=COLORS["red"])

    for year in range(2016, 2027, 2):
        x = _month_x(f"{year}-01", left, right)
        draw.line((x, 118, x, 615), fill="#EFF2F4", width=1)
        draw.text((x - 19, 635), str(year), font=FONT, fill=COLORS["ink"])
    draw.line((left, 615, right, 615), fill=COLORS["ink"], width=2)
    _footer(draw, "ACS circles mark verified release dates; CPS file release dates remain unknown. Signals historical backfill is non-real-time.", 670)
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _figure2(loadings: pd.DataFrame, output: Path) -> Path:
    image, draw = _canvas(
        1900,
        900,
        "Figure 2. Factor loading stability across outer folds",
        "Median and interquartile range; saved training-fold sign anchors; no AI or skill-factor label is imposed",
    )
    features_by_block = {
        "L": ["u_t", "log_emp_change", "employment_share"],
        "S": [
            "share_age_16_24",
            "share_age_25_54",
            "share_education_less_than_high_school",
            "share_education_high_school_or_equivalent",
            "share_education_some_college_or_associate",
            "share_education_bachelors",
        ],
    }
    for block_index, (block, features) in enumerate(features_by_block.items()):
        top = 135 + block_index * 360
        frame = loadings[loadings["block"].eq(block)]
        color = COLORS["blue"] if block == "L" else COLORS["teal"]
        draw.text((50, top), f"{block} factor — {frame['fold_id'].nunique()} folds", font=FONT_MEDIUM, fill=color)
        x_left, x_right = 490, 1160
        center = (x_left + x_right) / 2
        scale = (x_right - x_left) / 2
        for value in [-1, -0.5, 0, 0.5, 1]:
            x = center + value * scale
            draw.line((x, top + 38, x, top + 320), fill=COLORS["grid"] if value else COLORS["ink"], width=2 if value == 0 else 1)
            draw.text((x - 16, top + 10), f"{value:g}", font=FONT_SMALL, fill=COLORS["muted"])
        for index, feature in enumerate(features):
            rows = frame[frame["feature"].eq(feature)]
            median = float(rows["loading"].median())
            q1 = float(rows["loading"].quantile(0.25))
            q3 = float(rows["loading"].quantile(0.75))
            sign = max(float((rows["loading"] > 0).mean()), float((rows["loading"] < 0).mean()))
            y = top + 73 + index * 39
            draw.text((55, y - 10), feature, font=FONT, fill=COLORS["ink"])
            x1 = center + q1 * scale
            x3 = center + q3 * scale
            xm = center + median * scale
            draw.line((x1, y, x3, y), fill=color, width=5)
            draw.ellipse((xm - 7, y - 7, xm + 7, y + 7), fill=color)
            draw.text((1205, y - 11), f"median {median:+.3f}   IQR [{q1:+.3f}, {q3:+.3f}]   sign {sign:.0%}", font=FONT_SMALL, fill=COLORS["ink"])
        variance = frame["explained_variance_ratio"]
        support = frame["fit_observation_count"]
        draw.text(
            (55, top + 313),
            f"PC1 variance median {variance.median():.3f}; IQR [{variance.quantile(.25):.3f}, {variance.quantile(.75):.3f}]"
            f"   |   fit support {int(support.min())}–{int(support.max())}",
            font=FONT_SMALL,
            fill=COLORS["muted"],
        )
    _footer(draw, "S support counts unique occupation-group × ACS source-year rows. L is labor-market condition; S is age/education composition.", 850)
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _figure3(inputs: dict[str, object], summary: dict[str, object], output: Path) -> Path:
    ai_metrics: pd.DataFrame = inputs["ai_metrics"]
    panels = [
        ("CPS long · 82 months", ["persistence", "cps_ridge_long", "cps_lightgbm_long"], summary["long"]),
        (
            "CPS + ACS common · 35 months",
            ["persistence", "cps_ridge_common_window", "cps_acs_ridge_common_window", "cps_lightgbm_common_window", "cps_acs_lightgbm_common_window"],
            summary["common"],
        ),
        ("AI short · 7 months", ["persistence", "cps_acs_ridge_no_P", "cps_acs_ridge_plus_P"], None),
    ]
    image, draw = _canvas(
        1700,
        880,
        "Figure 3. Out-of-sample MAE by evaluation window",
        "Exact same-key comparisons within each panel; differences across panels are not incremental-effect comparisons",
    )
    for panel_index, (title, models, frame) in enumerate(panels):
        left = 45 + panel_index * 555
        right = left + 520
        top = 140
        draw.rounded_rectangle((left, top, right, 815), radius=8, outline=COLORS["grid"], width=2)
        draw.text((left + 18, top + 15), title, font=FONT_MEDIUM, fill=COLORS["ink"])
        x0, x1 = left + 190, right - 32
        maximum = 1.35 if panel_index == 1 else 1.25
        for tick in np.linspace(0, maximum, 4):
            x = x0 + tick / maximum * (x1 - x0)
            draw.line((x, top + 73, x, top + 560), fill=COLORS["grid"], width=1)
            draw.text((x - 16, top + 45), f"{tick:.1f}", font=FONT_SMALL, fill=COLORS["muted"])
        for index, model in enumerate(models):
            y = top + 112 + index * 74
            if panel_index == 2:
                actual_model = model
                if model == "persistence":
                    actual_model = "persistence"
                record = ai_metrics[(ai_metrics["scope"].eq("all_22_groups")) & (ai_metrics["model"].eq(actual_model))].iloc[0]
                mae, rmse = float(record["macro_mae_pp"]), float(record["macro_rmse_pp"])
            else:
                record = frame[model]
                mae, rmse = float(record["mae"]), float(record["rmse"])
            draw.text((left + 18, y - 11), MODEL_LABELS[model], font=FONT, fill=COLORS["ink"])
            bar_width = mae / maximum * (x1 - x0)
            draw.rectangle((x0, y - 9, x0 + bar_width, y + 12), fill=MODEL_COLORS[model])
            draw.text((x0 + bar_width + 7, y - 11), f"{mae:.3f}", font=FONT, fill=COLORS["ink"])
            draw.text((left + 18, y + 17), f"RMSE {rmse:.3f} pp", font=FONT_SMALL, fill=COLORS["muted"])
        row_note = ["22 groups · 1,804 rows", "22 groups · 770 rows", "22 groups · 154 rows"][panel_index]
        draw.text((left + 18, top + 590), row_note, font=FONT_SMALL, fill=COLORS["muted"])
        draw.text((left + 18, top + 624), "MAE (pp); RMSE below estimate", font=FONT_SMALL, fill=COLORS["muted"])
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _figure4(summary: dict[str, object], output: Path) -> Path:
    monthly: dict[str, pd.Series] = summary["paired_monthly"]
    paired: dict[str, float] = summary["paired_overall"]
    image, draw = _canvas(
        1500,
        720,
        "Figure 4. Monthly paired MAE difference after adding P",
        "Δ = MAE(plus P) − MAE(no P); identical occupation keys within month; no confidence intervals with seven months",
    )
    months = list(monthly["all_22_groups"].index)
    left, right, y_top, y_bottom = 185, 1430, 180, 570
    all_values = np.concatenate([monthly["all_22_groups"].values, monthly["coverage_ge_50pct"].values])
    low, high = min(-0.025, float(all_values.min()) - 0.004), max(0.025, float(all_values.max()) + 0.004)

    def y_pos(value: float) -> float:
        return y_bottom - (value - low) / (high - low) * (y_bottom - y_top)

    for tick in [low, 0, high]:
        y = y_pos(tick)
        draw.text((110, y - 10), f"{tick:+.3f}", font=FONT_SMALL, fill=COLORS["muted"])
        draw.line((left, y, right, y), fill=COLORS["ink"] if tick == 0 else COLORS["grid"], width=2 if tick == 0 else 1)

    series = [
        ("All 22 groups", monthly["all_22_groups"], COLORS["blue"]),
        ("Coverage ≥50% · 13 groups", monthly["coverage_ge_50pct"], COLORS["gold"]),
    ]
    for series_index, (label, values, color) in enumerate(series):
        points: list[tuple[float, float]] = []
        for index, month in enumerate(months):
            x = left + index * (right - left) / (len(months) - 1)
            y = y_pos(float(values.loc[month]))
            points.append((x, y))
            draw.text((x - 24, y_bottom + 15), month[2:], font=FONT_SMALL, fill=COLORS["muted"])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=COLORS["ink"], width=1)
            draw.text((x - 26, y - 27), f"{float(values.loc[month]):+.3f}", font=FONT_SMALL, fill=COLORS["ink"])
        draw.line(points, fill=color, width=4)
        legend_y = 90 + 30 * series_index
        draw.line((920, legend_y + 8, 950, legend_y + 8), fill=color, width=4)
        draw.text((960, legend_y), label, font=FONT, fill=COLORS["ink"])
    _footer(
        draw,
        f"Pooled paired Δ: {paired['all_22_groups']:+.5f} pp (154 rows); coverage ≥50%: {paired['coverage_ge_50pct']:+.5f} pp (91 rows). Positive means worse MAE.",
        665,
    )
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _figure5(exposure: pd.DataFrame, output: Path) -> Path:
    image, draw = _canvas(
        1400,
        1000,
        "Figure 5. 2022 ACS employment-weighted exact-match E coverage",
        "Coverage is matched group employment weight; low coverage is missing mapping, not zero exposure",
    )
    data = exposure.sort_values("employment_weighted_coverage", ascending=True).reset_index(drop=True)
    x0, x1 = 240, 1190
    x50 = x0 + 0.5 * (x1 - x0)
    draw.line((x50, 120, x50, 900), fill=COLORS["gold"], width=3)
    draw.text((x50 - 39, 95), "50% reference", font=FONT_SMALL, fill=COLORS["gold"])
    for index, row in data.iterrows():
        y = 142 + index * 34
        value = float(row["employment_weighted_coverage"])
        color = COLORS["red"] if value < 0.5 else COLORS["blue"]
        draw.text((45, y - 10), str(row["occupation_group"]), font=FONT, fill=COLORS["ink"])
        draw.rectangle((x0, y - 7, x0 + value * (x1 - x0), y + 8), fill=color)
        draw.text((x0 + value * (x1 - x0) + 9, y - 10), f"{value:.1%}", font=FONT_SMALL, fill=COLORS["ink"])
    for tick in [0, 0.25, 0.5, 0.75, 1]:
        x = x0 + tick * (x1 - x0)
        draw.line((x, 125, x, 895), fill=COLORS["grid"], width=1)
        draw.text((x - 15, 910), f"{tick:.0%}", font=FONT, fill=COLORS["ink"])
    overall = float(exposure["matched_e_weight"].sum() / exposure["civilian_employment_weight_2022"].sum())
    below = int((exposure["employment_weighted_coverage"] < 0.5).sum())
    _footer(draw, f"Overall {overall:.2%} matched exactly; {below}/22 groups below 50%. E is potential exposure, not adoption or impact.", 965)
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _appendix_a(inputs: dict[str, object], output: Path) -> Path:
    oos: pd.DataFrame = inputs["oos"]
    errors: pd.DataFrame = inputs["errors"]
    predictions = oos[oos["model"].eq("cps_ridge_long")].copy()
    support = predictions.groupby("occupation_group")["target_low_support"].agg(["sum", "count"])
    frame = errors[errors["model"].eq("cps_ridge_long")].merge(support, left_on="occupation_group", right_index=True)
    assert len(frame) == 22
    frame = frame.sort_values("mae_pp", ascending=True).reset_index(drop=True)
    image, draw = _canvas(
        1400,
        1000,
        "Appendix A. Long-window ridge error by SOC major group",
        "Descriptive 22-group model errors; original case groups highlighted; low-support months shown beside MAE",
    )
    left, right, maximum = 250, 950, 5.2
    highlighted = {"11-0000", "17-0000", "27-0000", "35-0000"}
    for index, row in frame.iterrows():
        y = 138 + index * 34
        color = COLORS["gold"] if row["occupation_group"] in highlighted else COLORS["blue"]
        length = float(row["mae_pp"]) / maximum * (right - left)
        draw.text((45, y - 10), str(row["occupation_group"]), font=FONT, fill=COLORS["ink"])
        draw.rectangle((left, y - 8, left + length, y + 8), fill=color)
        draw.text((left + length + 8, y - 10), f"{row['mae_pp']:.2f}", font=FONT_SMALL, fill=COLORS["ink"])
        draw.text((1025, y - 10), f"low support {int(row['sum'])}/{int(row['count'])}", font=FONT_SMALL, fill=COLORS["muted"])
    for tick in range(6):
        x = left + tick / maximum * (right - left)
        draw.line((x, 120, x, 900), fill=COLORS["grid"], width=1)
        draw.text((x - 8, 910), str(tick), font=FONT_SMALL, fill=COLORS["ink"])
    _footer(draw, "Jan 2020 occupation-code transition and small-sample flags limit comparisons; group errors are not fine-occupation estimates.", 965)
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def _cip_rows(crosswalk: pd.DataFrame, code: str) -> list[tuple[str, str]]:
    rows = crosswalk[crosswalk["SOC2018Code"].astype(str).eq(code)]
    return [(str(row.CIP2020Code), str(row.CIP2020Title)) for row in rows.itertuples(index=False)]


def _appendix_b(inputs: dict[str, object], output: Path) -> Path:
    crosswalk: pd.DataFrame = inputs["crosswalk"]
    image, draw = _canvas(
        2100,
        1600,
        "Appendix B. Occupation → task/skill evidence → related CIP programs",
        "Selected O*NET 31.0 OnLine evidence and exact NCES 2020 CIP–2018 SOC links; crosswalk links are not destination probabilities",
    )
    padding, gap, top, bottom = 34, 24, 120, 1510
    column_width = (2100 - 2 * padding - 2 * gap) // 3
    cases = [
        {
            "title": "Chemical engineering",
            "subtitle": "17-2041.00 → model group 17-0000",
            "body": [
                "Tasks: safety procedures, process troubleshooting, process-data monitoring, and optimization/compliance.",
                "Skills: critical thinking, science, reading comprehension, active learning, and mathematics.",
                "Illustrative curriculum direction: process safety, controls/monitoring, quantitative troubleshooting, and chemical-process modeling.",
            ],
            "socs": ["17-2041"],
            "result": "Group ridge MAE 0.584 pp; low-support flags in 74/82 test months; E coverage 26.37%. This is not an engineer-specific forecast.",
        },
        {
            "title": "Motion and video",
            "subtitle": "27-4030 parent → 27-4031 camera + 27-4032 editor",
            "body": [
                "Camera tasks: compose/frame shots, operate cameras, adjust focus/exposure, coordinate with crew.",
                "Editor tasks: organize raw footage, edit sequences, select shots and shape a coherent story.",
                "Skills include critical thinking, reading, active learning, coordination, problem solving, and time management.",
                "Illustrative direction: camera/lens/exposure practice, digital postproduction, story sequencing, and collaborative production.",
            ],
            "socs": ["27-4031", "27-4032"],
            "result": "Group 27-0000 ridge MAE 1.452 pp; low-support flags in 6/82 months; E coverage 57.11%. Not a camera/editor estimate.",
        },
        {
            "title": "Food and service operations",
            "subtitle": "Representative occupations span groups 35-0000 and 11-0000",
            "body": [
                "Cooks: safe storage/temperatures, sanitation, freshness and preparation.",
                "Supervisors/managers: training, complaints, service standards, sanitation records, and staffing.",
                "Skills: active listening, speaking, monitoring, critical thinking, and coordination.",
                "Illustrative direction: separate food-safety/culinary practice from service communication and operations/staff management.",
            ],
            "socs": ["35-2014", "35-3031", "35-1012", "11-9051"],
            "result": "Group 35-0000 MAE 2.286 pp; E 63.54%. Group 11-0000 MAE 0.540 pp; E 46.56%. Each is a separate broad-group result.",
        },
    ]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"]]
    for case_index, case in enumerate(cases):
        x = padding + case_index * (column_width + gap)
        draw.rounded_rectangle((x, top, x + column_width, bottom), radius=12, outline=COLORS["grid"], width=2)
        draw.text((x + 18, top + 17), case["title"], font=FONT_MEDIUM, fill=colors[case_index])
        draw.text((x + 18, top + 52), case["subtitle"], font=FONT_SMALL, fill=COLORS["muted"])
        y = top + 96
        max_width = column_width - 36
        for paragraph in case["body"]:
            for line in _wrap(draw, paragraph, FONT, max_width):
                draw.text((x + 18, y), line, font=FONT, fill=COLORS["ink"])
                y += 25
            y += 8
        y += 4
        draw.text((x + 18, y), "Exact CIP crosswalk rows", font=FONT_MEDIUM, fill=COLORS["ink"])
        y += 34
        for soc in case["socs"]:
            title_row = crosswalk[crosswalk["SOC2018Code"].astype(str).eq(soc)]
            soc_title = str(title_row.iloc[0]["SOC2018Title"]) if not title_row.empty else soc
            draw.text((x + 18, y), f"SOC {soc}: {soc_title}", font=FONT_SMALL, fill=COLORS["muted"])
            y += 23
            cip_rows = _cip_rows(crosswalk, soc)
            if len(cip_rows) == 1 and cip_rows[0][0] == "99.9999":
                draw.text((x + 30, y), "99.9999 — NO MATCH", font=FONT_SMALL, fill=COLORS["red"])
                y += 22
                continue
            for cip, title in cip_rows:
                for line in _wrap(draw, f"{cip}  {title.rstrip('.')}", FONT_SMALL, max_width - 16):
                    draw.text((x + 30, y), line, font=FONT_SMALL, fill=COLORS["ink"])
                    y += 19
        y = max(y + 8, 1230 if case_index < 2 else y + 8)
        draw.line((x + 18, y, x + column_width - 18, y), fill=COLORS["grid"], width=1)
        y += 14
        for line in _wrap(draw, case["result"], FONT_SMALL, max_width):
            draw.text((x + 18, y), line, font=FONT_SMALL, fill=COLORS["red"])
            y += 20
        assert y < bottom - 15, f"Education map text clipped for {case['title']} (bottom y={y})"
    _footer(
        draw,
        "Selected evidence only; NCES is a many-to-many relevance crosswalk, not graduate outcomes. No course contents or education-effect data were available.",
        1550,
    )
    image.save(output, format="PNG", dpi=(240, 240))
    return output


def build_charts(output_dir: Path | None = None) -> list[Path]:
    inputs = _read_inputs()
    summary = _summarize(inputs)
    destination = output_dir or (Path(tempfile.gettempdir()) / "luna-phase5-inline")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        _figure1(inputs, destination / "figure1_timeline.png"),
        _figure2(inputs["loadings"], destination / "figure2_factor_loadings.png"),
        _figure3(inputs, summary, destination / "figure3_oos_mae.png"),
        _figure4(summary, destination / "figure4_ai_monthly_delta.png"),
        _figure5(inputs["exposure"], destination / "figure5_e_coverage.png"),
        _appendix_a(inputs, destination / "appendixA_group_errors.png"),
        _appendix_b(inputs, destination / "appendixB_education_map.png"),
    ]
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        print(f"{path}\tsha256={_hash(path)}")
    print(f"Rendered {len(paths)} charts from {MODEL_DIR}")
    print(f"Long-window MAE (pp): {summary['long']['cps_ridge_long']['mae']:.6f} ridge; {summary['long']['persistence']['mae']:.6f} persistence")
    print(f"AI paired delta (pp): {summary['paired_overall']['all_22_groups']:+.6f} all groups; {summary['paired_overall']['coverage_ge_50pct']:+.6f} high-coverage")
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG destination; defaults to %%TEMP%%/luna-phase5-inline",
    )
    args = parser.parse_args(argv)
    build_charts(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
