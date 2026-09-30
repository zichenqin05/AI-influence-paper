"""Small, dictionary-driven readers for official CPS Basic Monthly files."""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


@dataclass(frozen=True)
class FieldLayout:
    name: str
    start: int  # one-based, inclusive
    end: int  # one-based, inclusive

    @property
    def width(self) -> int:
        return self.end - self.start + 1


def parse_layout(text: str) -> dict[str, FieldLayout]:
    """Extract variable names and fixed-width positions from a Census layout."""
    fields: dict[str, FieldLayout] = {}
    pattern = re.compile(r"^\s*([A-Z][A-Z0-9]{2,})\s+(\d+)\s+.*?\s+(\d+)\s*-\s*(\d+)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, declared_width, start, end = match.groups()
        field = FieldLayout(name=name, start=int(start), end=int(end))
        if field.width != int(declared_width):
            continue
        fields[name] = field
    return fields


def read_fixed_width_fields(
    stream: BinaryIO,
    fields: Iterable[FieldLayout],
    *,
    limit: int | None = None,
) -> tuple[int, Counter[int], list[dict[str, str]]]:
    """Read selected fixed-width columns; return count, row widths, and rows."""
    selected = tuple(fields)
    count = 0
    widths: Counter[int] = Counter()
    rows: list[dict[str, str]] = []
    for raw_line in stream:
        line = raw_line.rstrip(b"\r\n")
        if not line:
            continue
        count += 1
        widths[len(line)] += 1
        rows.append({
            field.name: line[field.start - 1:field.end].decode("ascii", errors="strict").strip()
            for field in selected
        })
        if limit is not None and len(rows) >= limit:
            break
    return count, widths, rows


def inspect_archive(path: Path) -> dict[str, object]:
    """Inspect ZIP members and fixed-width line lengths without extracting files."""
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        candidates = [info for info in members if info.filename.lower().endswith((".dat", ".txt", ".asc"))]
        if not candidates:
            raise ValueError("CPS archive has no .dat, .txt, or .asc record member")
        chosen = max(candidates, key=lambda item: item.file_size)
        count = 0
        widths: Counter[int] = Counter()
        with archive.open(chosen) as stream:
            for raw_line in stream:
                line = raw_line.rstrip(b"\r\n")
                if not line:
                    continue
                count += 1
                widths[len(line)] += 1
    return {
        "archive_bytes": path.stat().st_size,
        "members": [info.filename for info in members],
        "selected_record_member": chosen.filename,
        "selected_member_bytes": chosen.file_size,
        "records": count,
        "record_width_counts": dict(sorted(widths.items())),
    }


def inspect_cps_pilot(archive_path: Path, layout_path: Path) -> dict[str, object]:
    """Confirm selected 2026 fields against its official layout and read a sample."""
    fields = parse_layout(layout_path.read_text(encoding="utf-8", errors="replace"))
    required = ("HRMONTH", "HRYEAR4", "PRPERTYP", "PRTAGE", "PEMLR", "PWSSWGT", "PTIO1OCD")
    missing = [name for name in required if name not in fields]
    archive_summary = inspect_archive(archive_path)
    if missing:
        return {**archive_summary, "layout_fields_found": sorted(fields), "required_fields_missing": missing}

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(str(archive_summary["selected_record_member"])) as stream:
            _, widths, rows = read_fixed_width_fields(stream, (fields[name] for name in required))

    adult_civilians = [row for row in rows if row["PRPERTYP"] == "2"]
    civilian_age_16_plus = [
        row for row in adult_civilians if row["PRTAGE"].isdigit() and int(row["PRTAGE"]) >= 16
    ]
    status_counts: dict[str, int] = {}
    for row in civilian_age_16_plus:
        status_counts[row["PEMLR"]] = status_counts.get(row["PEMLR"], 0) + 1
    weights = [int(row["PWSSWGT"]) for row in civilian_age_16_plus if row["PWSSWGT"].isdigit()]
    valid_occupation_codes = [
        row["PTIO1OCD"] for row in civilian_age_16_plus
        if row["PTIO1OCD"].isdigit() and 0 <= int(row["PTIO1OCD"]) <= 9999
    ]

    return {
        **archive_summary,
        "layout_fields": {name: {"start": fields[name].start, "end": fields[name].end,
                                 "width": fields[name].width} for name in required},
        "parsed_records": len(rows),
        "parsed_record_width_counts": dict(sorted(widths.items())),
        "parsed_month_values": sorted({row["HRMONTH"] for row in rows}),
        "parsed_year_values": sorted({row["HRYEAR4"] for row in rows}),
        "adult_civilian_records": len(adult_civilians),
        "adult_civilian_age_16_plus_records": len(civilian_age_16_plus),
        "civilian_age_16_plus_labor_status_counts": dict(sorted(status_counts.items())),
        "civilian_age_16_plus_status_values_outside_1_to_7": sorted(
            value for value in status_counts if value not in {str(code) for code in range(1, 8)}
        ),
        "civilian_age_16_plus_raw_weight_positive_count": sum(weight > 0 for weight in weights),
        "civilian_age_16_plus_raw_weight_zero_or_invalid_count": len(civilian_age_16_plus) - len(weights)
        + sum(weight == 0 for weight in weights),
        "civilian_age_16_plus_raw_weight_min": min(weights, default=None),
        "civilian_age_16_plus_raw_weight_max": max(weights, default=None),
        "civilian_age_16_plus_valid_occupation_code_count": len(valid_occupation_codes),
        "civilian_age_16_plus_occupation_code_distinct_count": len(set(valid_occupation_codes)),
        "full_archive_width_counts": archive_summary["record_width_counts"],
    }
