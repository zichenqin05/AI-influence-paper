"""Calendar-time label and feature-availability contracts."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Literal, Mapping


def month_number(value: str) -> int:
    parsed = date.fromisoformat(value if len(value) == 10 else f"{value}-01")
    if parsed.day != 1:
        raise ValueError(f"Expected a calendar month in YYYY-MM or YYYY-MM-01 form: {value}")
    return parsed.year * 12 + parsed.month - 1


def month_string(number: int) -> str:
    year, zero_based_month = divmod(number, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def add_months(value: str, offset: int) -> str:
    return month_string(month_number(value) + offset)


def build_calendar_labels(
    panel_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build exact t+1..t+3 labels; missing calendar months stay invalid."""
    rows = list(panel_rows)
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    origins: set[tuple[str, str]] = set()
    for row in rows:
        month = str(row["month"])[:7]
        group = str(row["occupation_group"])
        key = (group, month)
        if key in by_key:
            raise ValueError(f"Duplicate occupation-month key: {group} {month}")
        by_key[key] = row
        origins.add(key)

    labels: list[dict[str, object]] = []
    for group, origin in sorted(origins, key=lambda item: (item[1], item[0])):
        target_months = [add_months(origin, offset) for offset in (1, 2, 3)]
        targets = [by_key.get((group, month)) for month in target_months]
        missing = [month for month, target in zip(target_months, targets) if target is None]
        values = [target.get("u") if target is not None else None for target in targets]
        invalid_values = [
            month for month, value in zip(target_months, values)
            if value is None or not 0.0 <= float(value) <= 1.0
        ]
        valid = not missing and not invalid_values
        release_dates = []
        for target in targets:
            value = target.get("release_date") if target is not None else None
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in {"nan", "nat", "none", "null"}:
                continue
            release_dates.append(text)
        labels.append({
            "origin_month": origin,
            "occupation_group": group,
            "target_month_1": target_months[0],
            "target_month_2": target_months[1],
            "target_month_3": target_months[2],
            "y": sum(float(value) for value in values) / 3.0 if valid else None,
            "label_available_at": max(release_dates) if valid and len(release_dates) == 3 else None,
            "valid_reason": "valid" if valid else (
                "missing_calendar_month:" + ",".join(missing) if missing
                else "missing_or_invalid_unemployment_rate:" + ",".join(invalid_values)
            ),
        })
    return labels


def feature_availability(
    *,
    source_kind: Literal["acs", "cps", "signals", "other"],
    reference_month: str,
    origin_month: str,
    release_date: str | None,
    forecast_date: str,
    mode: str,
    allow_historical_backfill: bool = False,
) -> dict[str, object]:
    """Apply release-date checks, with an explicit retrospective Signals exception only."""
    if mode not in {"retrospective", "strict_asof"}:
        raise ValueError("mode must be 'retrospective' or 'strict_asof'")
    if source_kind not in {"acs", "cps", "signals", "other"}:
        raise ValueError("source_kind must be 'acs', 'cps', 'signals', or 'other'")
    if month_number(reference_month) > month_number(origin_month):
        return {"available": False, "status": "future_reference_period"}

    signals_backfill = (
        source_kind == "signals" and mode == "retrospective" and allow_historical_backfill
    )
    if release_date is None:
        if signals_backfill:
            return {
                "available": True,
                "status": "historical_backfill_release_date_unknown",
                "evaluation_label": "non_realtime_backtest",
            }
        return {"available": False, "status": "release_date_unknown"}
    parsed_release = date.fromisoformat(release_date)
    parsed_forecast = date.fromisoformat(forecast_date)
    if parsed_release <= parsed_forecast:
        return {
            "available": True,
            "status": "available_asof_forecast",
            "evaluation_label": "asof_available",
        }
    if signals_backfill:
        return {
            "available": True,
            "status": "historical_backfill_not_realtime",
            "evaluation_label": "non_realtime_backtest",
        }
    return {"available": False, "status": "released_after_forecast"}
