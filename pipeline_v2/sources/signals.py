"""Reader for the audited OpenAI Signals work-share series."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path


REQUIRED_COLUMNS = {"month", "country", "work_related", "share_of_messages"}


def read_us_work_share(path: Path) -> list[dict[str, object]]:
    """Return the verified US work-related monthly series without imputing gaps."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"Signals file is missing columns: {sorted(missing)}")

        output: list[dict[str, object]] = []
        seen: set[date] = set()
        for line_no, row in enumerate(reader, start=2):
            if row["country"].strip() != "US" or row["work_related"].strip() != "1":
                continue
            month = date.fromisoformat(row["month"].strip())
            if month.day != 1:
                raise ValueError(f"Signals month must be first-of-month at line {line_no}")
            share = float(row["share_of_messages"])
            if not 0.0 <= share <= 1.0:
                raise ValueError(f"Signals share outside [0, 1] at line {line_no}")
            if month in seen:
                raise ValueError(f"Duplicate US work-related month: {month.isoformat()}")
            seen.add(month)
            output.append({
                "month": month.isoformat(),
                "country": "US",
                "work_related": 1,
                "share_of_messages": share,
            })

    ordered = sorted(output, key=lambda item: str(item["month"]))
    if ordered:
        months = [date.fromisoformat(str(row["month"])) for row in ordered]
        expected_count = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month + 1
        if len(months) != expected_count:
            present = {month for month in months}
            missing = []
            current = months[0]
            while current <= months[-1]:
                if current not in present:
                    missing.append(current.isoformat())
                next_month = current.month % 12 + 1
                next_year = current.year + (current.month == 12)
                current = date(next_year, next_month, 1)
            raise ValueError(f"Signals US work-related series has calendar-month gaps: {missing}")
    return ordered
