"""Streaming, selected-column reader for ACS PUMS person CSV archives."""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from collections import Counter
from pathlib import Path


SELECTED_PERSON_COLUMNS = ("SERIALNO", "SPORDER", "PWGTP", "OCCP", "ESR", "AGEP", "SCHL")


def inspect_and_cache_person_columns(archive_path: Path, cache_path: Path) -> dict[str, object]:
    """Stream official person CSV members to a gzipped selected-column cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".part")
    if temporary_cache.exists():
        raise FileExistsError(f"Refusing to overwrite an incomplete selected-column cache: {temporary_cache}")

    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist()
                   if not info.is_dir() and info.filename.lower().endswith(".csv")]
        if not members:
            raise ValueError("ACS PUMS archive has no CSV member")

        profile: dict[str, object] = {
            "purpose": "schema_and_selected_column_streaming_pilot_only",
            "interpretation": "No occupational estimates, panels, or model results generated",
            "input_archive_bytes": archive_path.stat().st_size,
            "person_csv_members": [],
            "selected_columns": list(SELECTED_PERSON_COLUMNS),
            "records": 0,
            "missing_key_value_counts": {"SERIALNO": 0, "SPORDER": 0},
            # ACS blanks include dictionary-defined N/A-by-universe values.
            "blank_value_counts": {name: 0 for name in SELECTED_PERSON_COLUMNS},
            "invalid_numeric_value_counts": {"PWGTP": 0, "AGEP": 0},
            "esr_value_counts": {},
            "age_min": None,
            "age_max": None,
            "weight_min": None,
            "weight_max": None,
            "weight_zero_count": 0,
            "weight_negative_count": 0,
            "occupation_code_distinct_count": 0,
            "education_code_distinct_count": 0,
        }
        esr_counts: Counter[str] = Counter()
        occupation_codes: set[str] = set()
        education_codes: set[str] = set()
        age_min: int | None = None
        age_max: int | None = None
        weight_min: int | None = None
        weight_max: int | None = None

        try:
            with gzip.open(temporary_cache, "xt", encoding="utf-8", newline="") as cached:
                writer = csv.writer(cached)
                writer.writerow(SELECTED_PERSON_COLUMNS)
                for member in members:
                    with archive.open(member) as binary_stream:
                        text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="")
                        reader = csv.reader(text_stream)
                        try:
                            header = next(reader)
                        except StopIteration as exc:
                            raise ValueError(f"Empty ACS PUMS person member: {member.filename}") from exc
                        missing_columns = [name for name in SELECTED_PERSON_COLUMNS if name not in header]
                        if missing_columns:
                            raise ValueError(
                                f"ACS member {member.filename} is missing required columns: {missing_columns}"
                            )
                        indexes = [header.index(name) for name in SELECTED_PERSON_COLUMNS]
                        member_rows = 0
                        for line_no, row in enumerate(reader, start=2):
                            if len(row) != len(header):
                                raise ValueError(
                                    f"ACS row width differs from its header in {member.filename}:{line_no}"
                                )
                            selected = [row[index] for index in indexes]
                            writer.writerow(selected)
                            member_rows += 1
                            profile["records"] += 1
                            for name, value in zip(SELECTED_PERSON_COLUMNS, selected):
                                if not value:
                                    profile["blank_value_counts"][name] += 1
                            if not selected[0]:
                                profile["missing_key_value_counts"]["SERIALNO"] += 1
                            if not selected[1]:
                                profile["missing_key_value_counts"]["SPORDER"] += 1

                            weight = int(selected[2]) if selected[2].isdigit() else None
                            if weight is None:
                                profile["invalid_numeric_value_counts"]["PWGTP"] += 1
                            else:
                                weight_min = weight if weight_min is None else min(weight_min, weight)
                                weight_max = weight if weight_max is None else max(weight_max, weight)
                                profile["weight_zero_count"] += int(weight == 0)
                                profile["weight_negative_count"] += int(weight < 0)

                            age = int(selected[5]) if selected[5].isdigit() else None
                            if age is None:
                                profile["invalid_numeric_value_counts"]["AGEP"] += 1
                            else:
                                age_min = age if age_min is None else min(age_min, age)
                                age_max = age if age_max is None else max(age_max, age)
                            esr_counts[selected[4]] += 1
                            if selected[3]:
                                occupation_codes.add(selected[3])
                            if selected[6]:
                                education_codes.add(selected[6])

                        profile["person_csv_members"].append({
                            "name": member.filename,
                            "uncompressed_bytes": member.file_size,
                            "records": member_rows,
                            "columns": len(header),
                        })
            temporary_cache.replace(cache_path)
        except Exception:
            if temporary_cache.exists():
                temporary_cache.unlink()
            raise

        profile["esr_value_counts"] = dict(sorted(esr_counts.items()))
        profile["age_min"] = age_min
        profile["age_max"] = age_max
        profile["weight_min"] = weight_min
        profile["weight_max"] = weight_max
        profile["occupation_code_distinct_count"] = len(occupation_codes)
        profile["education_code_distinct_count"] = len(education_codes)
        profile["selected_cache_path"] = str(cache_path)
        profile["selected_cache_bytes"] = cache_path.stat().st_size
        return profile
