"""Streaming builders for survey-weighted CPS and ACS occupation panels."""

from __future__ import annotations

import csv
import gzip
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from pipeline_v2.sources.cps_fixed_width import FieldLayout, parse_layout
from pipeline_v2.sources.occupation import (
    SOC_MAJOR_GROUPS, cps_occupation_code_version, cps_occupation_variable,
)


ACS_RELEASE_DATES = {
    2018: "2019-11-14",
    2019: "2020-10-15",
    2021: "2022-10-20",
    2022: "2023-10-19",
    2023: "2024-10-17",
    2024: "2025-12-04",
}
ACS_YEARS = tuple(ACS_RELEASE_DATES)
CIVILIAN_GROUPS = tuple(
    (f"{number}-0000", title)
    for number, title in SOC_MAJOR_GROUPS.items()
    if number != "55"
)
MONTHS = (("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("may", 5), ("jun", 6),
          ("jul", 7), ("aug", 8), ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12))


def read_crosswalk(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["source_code_version"], row["source_code"])
        if key in mapping:
            raise ValueError(f"Duplicate crosswalk key: {key}")
        mapping[key] = row
    return mapping


def _write_gzip_csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing panel artifact: {path}")
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with gzip.open(temp, "xt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def _cps_layout_fields(path: Path) -> tuple[dict[str, FieldLayout], str]:
    fields = parse_layout(path.read_text(encoding="latin1"))
    occupation_field = cps_occupation_variable(int(path.name[:4])) if path.name[:4].isdigit() else (
        "PTIO1OCD" if "PTIO1OCD" in fields else "PEIO1OCD"
    )
    required = ("HRMONTH", "HRYEAR4", "PRPERTYP", "PRTAGE", "PEMLR", "PWSSWGT", occupation_field)
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"CPS layout {path.name} is missing fields: {missing}")
    if fields["PWSSWGT"].width != 10:
        raise ValueError(f"Unexpected PWSSWGT width in {path.name}: {fields['PWSSWGT'].width}")
    return fields, occupation_field


def _archive_record_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    candidates = [info for info in archive.infolist() if not info.is_dir()
                  and info.filename.lower().endswith((".dat", ".txt", ".asc"))]
    if not candidates:
        raise ValueError("CPS ZIP has no fixed-width .dat, .txt, or .asc member")
    return max(candidates, key=lambda info: info.file_size)


def build_cps_panel(
    *,
    archive_paths: dict[str, Path],
    layout_paths: dict[int, Path],
    crosswalk_path: Path,
    output_path: Path,
    diagnostic_path: Path,
) -> dict[str, object]:
    """Aggregate all available official Basic CPS monthly files.

    CPS PWSSWGT stores four implied decimal places in the published fixed-width
    field. Raw integers are accumulated first, then divided by 10,000 exactly
    once for output. Exact per-file public-use release dates are not recorded;
    the panel keeps those dates blank and marks them unverified.
    """
    mapping = read_crosswalk(crosswalk_path)
    group_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"emp_raw": 0, "unemp_raw": 0, "n_emp": 0, "n_unemp": 0}
    )
    national_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"emp_raw": 0, "unemp_raw": 0, "n_emp": 0, "n_unemp": 0,
                  "matched_emp_raw": 0, "matched_unemp_raw": 0,
                  "matched_n_emp": 0, "matched_n_unemp": 0,
                  "unmapped_occ_raw": 0, "unmapped_occ_n": 0,
                  "invalid_weight_n": 0}
    )
    unmapped_reasons: dict[str, dict[str, list[int]]] = defaultdict(dict)
    month_files: dict[str, str] = {}
    file_profiles: list[dict[str, object]] = []

    for expected_month, archive_path in sorted(archive_paths.items()):
        year = int(expected_month[:4])
        layout_path = layout_paths[year]
        fields, occupation_field = _cps_layout_fields(layout_path)
        version = cps_occupation_code_version(year)
        selected_names = ("HRMONTH", "HRYEAR4", "PRPERTYP", "PRTAGE", "PEMLR", "PWSSWGT", occupation_field)
        selected = tuple(fields[name] for name in selected_names)
        observed_months: set[str] = set()
        record_count = 0
        valid_weight_count = 0
        with zipfile.ZipFile(archive_path) as archive:
            member = _archive_record_member(archive)
            with archive.open(member) as stream:
                for raw_line in stream:
                    line = raw_line.rstrip(b"\r\n")
                    if not line:
                        continue
                    record_count += 1
                    if len(line) < max(field.end for field in selected):
                        raise ValueError(f"Truncated CPS record in {archive_path.name}: {len(line)} bytes")
                    record = {
                        field.name: line[field.start - 1:field.end].decode("ascii", errors="strict").strip()
                        for field in selected
                    }
                    month_num = record["HRMONTH"]
                    year_text = record["HRYEAR4"]
                    if not month_num.isdigit() or not year_text.isdigit():
                        continue
                    actual_month = f"{int(year_text):04d}-{int(month_num):02d}"
                    observed_months.add(actual_month)
                    if actual_month != expected_month:
                        continue
                    if record["PRPERTYP"] != "2":
                        continue
                    age = record["PRTAGE"]
                    status = record["PEMLR"]
                    if not age.isdigit() or int(age) < 16 or status not in {"1", "2", "3", "4"}:
                        continue
                    stats = national_stats[expected_month]
                    weight = record["PWSSWGT"]
                    if not weight.isdigit() or int(weight) <= 0:
                        stats["invalid_weight_n"] += 1
                        continue
                    weight_raw = int(weight)
                    valid_weight_count += 1
                    is_emp = status in {"1", "2"}
                    if is_emp:
                        stats["emp_raw"] += weight_raw
                        stats["n_emp"] += 1
                    else:
                        stats["unemp_raw"] += weight_raw
                        stats["n_unemp"] += 1

                    code = record[occupation_field]
                    code = f"{int(code):04d}" if re.fullmatch(r"\d{1,4}", code) else ""
                    row = mapping.get((version, code)) if code else None
                    group = row.get("occupation_group", "") if row else ""
                    is_civilian_group = row is not None and row.get("civilian_group") == "true"
                    if not group or not is_civilian_group:
                        stats["unmapped_occ_raw"] += weight_raw
                        stats["unmapped_occ_n"] += 1
                        if not code:
                            reason = "blank_or_invalid_occupation_code"
                        elif row is None:
                            reason = "unknown_source_occupation_code"
                        elif row.get("mapping_status") != "mapped":
                            reason = row["mapping_status"]
                        else:
                            reason = "noncivilian_SOC_major_group"
                        reason_total = unmapped_reasons[expected_month].setdefault(reason, [0, 0])
                        reason_total[0] += 1
                        reason_total[1] += weight_raw
                        continue
                    key = (expected_month, group)
                    group_row = group_stats[key]
                    if is_emp:
                        group_row["emp_raw"] += weight_raw
                        group_row["n_emp"] += 1
                        stats["matched_emp_raw"] += weight_raw
                        stats["matched_n_emp"] += 1
                    else:
                        group_row["unemp_raw"] += weight_raw
                        group_row["n_unemp"] += 1
                        stats["matched_unemp_raw"] += weight_raw
                        stats["matched_n_unemp"] += 1

        if observed_months != {expected_month}:
            raise ValueError(f"CPS file {archive_path.name} reported months {sorted(observed_months)}, expected {expected_month}")
        month_files[expected_month] = str(archive_path)
        file_profiles.append({
            "month": expected_month, "archive": archive_path.name,
            "layout": layout_path.name, "occupation_field": occupation_field,
            "occupation_code_version": version, "record_member": member.filename,
            "records": record_count, "records_with_valid_weight_in_universe": valid_weight_count,
        })

    panel_rows: list[dict[str, object]] = []
    for month in sorted(month_files):
        for group, title in CIVILIAN_GROUPS:
            stats = group_stats[(month, group)]
            lf_raw = stats["emp_raw"] + stats["unemp_raw"]
            unemployment_rate = stats["unemp_raw"] / lf_raw if lf_raw else ""
            panel_rows.append({
                "month": month,
                "occupation_group": group,
                "occupation_group_title": title,
                "Emp": stats["emp_raw"] / 10000,
                "Unemp": stats["unemp_raw"] / 10000,
                "LF": lf_raw / 10000,
                "u": unemployment_rate,
                "n_emp": stats["n_emp"],
                "n_unemp": stats["n_unemp"],
                "weight_sum": lf_raw / 10000,
                "release_date": "",
                "release_date_status": "exact_public_use_file_date_not_verified",
                "quality_flag": "valid" if lf_raw else "no_mapped_labor_force_records",
            })
    panel_columns = (
        "month", "occupation_group", "occupation_group_title", "Emp", "Unemp", "LF", "u",
        "n_emp", "n_unemp", "weight_sum", "release_date", "release_date_status", "quality_flag",
    )
    _write_gzip_csv(output_path, panel_columns, panel_rows)

    diagnostic_rows = []
    for month in sorted(month_files):
        stats = national_stats[month]
        lf_raw = stats["emp_raw"] + stats["unemp_raw"]
        matched_raw = stats["matched_emp_raw"] + stats["matched_unemp_raw"]
        diagnostic_rows.append({
            "month": month,
            "national_emp": stats["emp_raw"] / 10000,
            "national_unemp": stats["unemp_raw"] / 10000,
            "national_lf": lf_raw / 10000,
            "national_u": stats["unemp_raw"] / lf_raw if lf_raw else "",
            "national_n_emp": stats["n_emp"],
            "national_n_unemp": stats["n_unemp"],
            "occupation_identified_emp": stats["matched_emp_raw"] / 10000,
            "occupation_identified_unemp": stats["matched_unemp_raw"] / 10000,
            "occupation_identified_lf": matched_raw / 10000,
            "occupation_identified_weight_share": matched_raw / lf_raw if lf_raw else "",
            "unmapped_status_1_to_4_n": stats["unmapped_occ_n"],
            "unmapped_status_1_to_4_weight": stats["unmapped_occ_raw"] / 10000,
            "unmapped_reason_counts_json": json.dumps(
                {reason: values[0] for reason, values in sorted(unmapped_reasons[month].items())},
                ensure_ascii=False,
            ),
            "unmapped_reason_weight_json": json.dumps(
                {reason: values[1] / 10000 for reason, values in sorted(unmapped_reasons[month].items())},
                ensure_ascii=False,
            ),
            "invalid_weight_n": stats["invalid_weight_n"],
            "release_date_status": "exact_public_use_file_date_not_verified",
        })
    diagnostic_columns = tuple(diagnostic_rows[0]) if diagnostic_rows else ("month",)
    _write_gzip_csv(diagnostic_path, diagnostic_columns, diagnostic_rows)
    return {
        "panel_path": str(output_path), "diagnostic_path": str(diagnostic_path),
        "available_months": len(month_files), "first_month": min(month_files, default=None),
        "last_month": max(month_files, default=None), "months": sorted(month_files),
        "missing_months": ["2025-10"], "occupation_groups": len(CIVILIAN_GROUPS),
        "rows": len(panel_rows), "weight_scale_divisor": 10000,
        "occupation_code_versions": {"2016-2019": "2010 Census OCC", "2020 onward": "2018 Census OCC"},
        "cps_public_use_release_dates": "not recorded; exact file availability must not be inferred from reference month",
        "file_profiles": file_profiles,
    }


def _read_person_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        lower = info.filename.lower()
        if info.is_dir() or not lower.endswith(".csv"):
            continue
        if "psam_p" in Path(lower).name:
            members.append(info)
    if not members:
        raise ValueError("ACS ZIP has no national person CSV members (psam_p*.csv)")
    return members


def _schl_band(value: str) -> str | None:
    if not value.isdigit():
        return None
    code = int(value)
    if 1 <= code <= 15:
        return "less_than_high_school"
    if code in (16, 17):
        return "high_school_or_equivalent"
    if 18 <= code <= 20:
        return "some_college_or_associate"
    if code == 21:
        return "bachelors"
    if 22 <= code <= 24:
        return "graduate_or_professional"
    return None


def build_acs_panel(
    *,
    archive_paths: dict[int, Path],
    crosswalk_path: Path,
    output_path: Path,
    diagnostic_path: Path,
) -> dict[str, object]:
    """Build annual employed-civilian structure by 2018 Census SOC major group."""
    mapping = read_crosswalk(crosswalk_path)
    output_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for year, archive_path in sorted(archive_paths.items()):
        if year not in ACS_RELEASE_DATES:
            raise ValueError(f"ACS annual archive year is outside the specified scope: {year}")
        stats: dict[str, dict[str, object]] = defaultdict(lambda: {
            "weight": 0, "n": 0, "age_raw": 0,
            "age_16_24": 0, "age_25_54": 0, "age_55_plus": 0,
            "education_weight": 0,
            "less_than_high_school": 0, "high_school_or_equivalent": 0,
            "some_college_or_associate": 0, "bachelors": 0, "graduate_or_professional": 0,
            "education_missing_n": 0,
        })
        total_adult_employed_weight = 0
        total_adult_employed_n = 0
        matched_weight = 0
        matched_n = 0
        missing_occ_n = 0
        invalid_weight_n = 0
        unmapped_reasons: Counter[str] = Counter()
        unmapped_reason_weights: Counter[str] = Counter()
        records = 0
        member_profiles: list[dict[str, object]] = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in _read_person_members(archive):
                with archive.open(member) as binary_stream:
                    text_stream = __import__("io").TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="")
                    reader = csv.reader(text_stream)
                    header = next(reader, None)
                    if header is None:
                        raise ValueError(f"Empty ACS person member: {member.filename}")
                    required = ("PWGTP", "OCCP", "ESR", "AGEP", "SCHL")
                    missing = [name for name in required if name not in header]
                    if missing:
                        raise ValueError(f"ACS person member {member.filename} missing {missing}")
                    indexes = {name: header.index(name) for name in required}
                    member_rows = 0
                    for line_no, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(f"ACS row width mismatch in {member.filename}:{line_no}")
                        member_rows += 1
                        records += 1
                        esr = row[indexes["ESR"]]
                        age_text = row[indexes["AGEP"]]
                        if esr not in {"1", "2"} or not age_text.isdigit() or int(age_text) < 16:
                            continue
                        weight_text = row[indexes["PWGTP"]]
                        if not weight_text.isdigit() or int(weight_text) <= 0:
                            invalid_weight_n += 1
                            continue
                        weight = int(weight_text)
                        total_adult_employed_weight += weight
                        total_adult_employed_n += 1
                        code_text = row[indexes["OCCP"]]
                        code = f"{int(code_text):04d}" if re.fullmatch(r"\d{1,4}", code_text) else ""
                        source = mapping.get(("census_2018", code)) if code else None
                        group = source.get("occupation_group", "") if source else ""
                        if not group or source.get("civilian_group") != "true":
                            missing_occ_n += 1
                            if not code:
                                reason = "blank_or_invalid_OCCP"
                            elif source is None:
                                reason = "unknown_2018_Census_OCC_code"
                            elif source.get("mapping_status") != "mapped":
                                reason = source["mapping_status"]
                            else:
                                reason = "noncivilian_SOC_major_group"
                            unmapped_reasons[reason] += 1
                            unmapped_reason_weights[reason] += weight
                            continue
                        group_stats = stats[group]
                        matched_n += 1
                        matched_weight += weight
                        group_stats["n"] += 1
                        group_stats["weight"] += weight
                        age = int(age_text)
                        group_stats["age_raw"] += age * weight
                        if age <= 24:
                            group_stats["age_16_24"] += weight
                        elif age <= 54:
                            group_stats["age_25_54"] += weight
                        else:
                            group_stats["age_55_plus"] += weight
                        education = _schl_band(row[indexes["SCHL"]])
                        if education is None:
                            group_stats["education_missing_n"] += 1
                        else:
                            group_stats["education_weight"] += weight
                            group_stats[education] += weight
                    member_profiles.append({"name": member.filename, "rows": member_rows,
                                            "uncompressed_bytes": member.file_size})

        for group, title in CIVILIAN_GROUPS:
            group_stats = stats[group]
            weight = int(group_stats["weight"])
            edu_weight = int(group_stats["education_weight"])
            output_rows.append({
                "year": year, "occupation_group": group, "occupation_group_title": title,
                "n_employed_persons": group_stats["n"], "PWGTP_sum": weight,
                "mean_age": int(group_stats["age_raw"]) / weight if weight else "",
                "share_age_16_24": int(group_stats["age_16_24"]) / weight if weight else "",
                "share_age_25_54": int(group_stats["age_25_54"]) / weight if weight else "",
                "share_age_55_plus": int(group_stats["age_55_plus"]) / weight if weight else "",
                "education_numerator_weight": edu_weight,
                "share_education_less_than_high_school": int(group_stats["less_than_high_school"]) / edu_weight if edu_weight else "",
                "share_education_high_school_or_equivalent": int(group_stats["high_school_or_equivalent"]) / edu_weight if edu_weight else "",
                "share_education_some_college_or_associate": int(group_stats["some_college_or_associate"]) / edu_weight if edu_weight else "",
                "share_education_bachelors": int(group_stats["bachelors"]) / edu_weight if edu_weight else "",
                "share_education_graduate_or_professional": int(group_stats["graduate_or_professional"]) / edu_weight if edu_weight else "",
                "education_missing_n": group_stats["education_missing_n"],
                "release_date": ACS_RELEASE_DATES[year],
                "release_date_status": "official_ACS_1_year_PUMS_release_date",
                "quality_flag": "valid" if weight else "no_mapped_employed_persons",
            })
        diagnostic_rows.append({
            "year": year, "person_records": records,
            "employed_civilian_age_16_plus_n": total_adult_employed_n,
            "employed_civilian_age_16_plus_PWGTP": total_adult_employed_weight,
            "occupation_mapped_n": matched_n, "occupation_mapped_PWGTP": matched_weight,
            "occupation_mapping_n_share": matched_n / total_adult_employed_n if total_adult_employed_n else "",
            "occupation_mapping_weight_share": matched_weight / total_adult_employed_weight if total_adult_employed_weight else "",
            "unmapped_or_blank_occupation_n": missing_occ_n,
            "unmapped_reason_counts_json": json.dumps(dict(sorted(unmapped_reasons.items())), ensure_ascii=False),
            "unmapped_reason_weight_json": json.dumps(dict(sorted(unmapped_reason_weights.items())), ensure_ascii=False),
            "invalid_employed_weight_n": invalid_weight_n,
            "person_csv_members": json.dumps(member_profiles, ensure_ascii=False),
            "release_date": ACS_RELEASE_DATES[year],
        })

    output_columns = tuple(output_rows[0]) if output_rows else ("year",)
    _write_gzip_csv(output_path, output_columns, output_rows)
    diagnostic_columns = tuple(diagnostic_rows[0]) if diagnostic_rows else ("year",)
    _write_gzip_csv(diagnostic_path, diagnostic_columns, diagnostic_rows)
    return {
        "panel_path": str(output_path), "diagnostic_path": str(diagnostic_path),
        "years": [row["year"] for row in diagnostic_rows],
        "rows": len(output_rows), "occupation_groups": len(CIVILIAN_GROUPS),
        "weight": "ACS PWGTP, integer person weight; no scaling",
        "person_universe": "civilian employed age 16+, ESR 1 or 2, known civilian SOC major group",
        "release_dates": ACS_RELEASE_DATES,
        "feature_rule": "annual row is unavailable before its PUMS public release date",
    }


def validate_panel_outputs(
    *, cps_path: Path, cps_diagnostic_path: Path, acs_path: Path, acs_diagnostic_path: Path,
) -> dict[str, object]:
    """Check panel keys, arithmetic, coverage, and release-date metadata."""
    def rows_at(path: Path) -> list[dict[str, str]]:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    failures: list[str] = []
    cps_rows = rows_at(cps_path)
    cps_diag = rows_at(cps_diagnostic_path)
    acs_rows = rows_at(acs_path)
    acs_diag = rows_at(acs_diagnostic_path)

    expected_months = []
    for year in range(2016, 2027):
        final_month = 8 if year == 2026 else 12
        for month in range(1, final_month + 1):
            key = f"{year:04d}-{month:02d}"
            if key != "2025-10":
                expected_months.append(key)
    cps_months = sorted({row["month"] for row in cps_rows})
    cps_keys = [(row["month"], row["occupation_group"]) for row in cps_rows]
    cps_duplicate_count = len(cps_keys) - len(set(cps_keys))
    if cps_duplicate_count:
        failures.append(f"CPS duplicate month/group keys: {cps_duplicate_count}")
    if cps_months != expected_months:
        missing = sorted(set(expected_months) - set(cps_months))
        unexpected = sorted(set(cps_months) - set(expected_months))
        failures.append(f"CPS month coverage differs; missing={missing[:5]}, unexpected={unexpected[:5]}")
    if len(cps_rows) != len(expected_months) * len(CIVILIAN_GROUPS):
        failures.append(f"CPS row count {len(cps_rows)} does not equal expected group-month grid")
    cps_identity_errors = 0
    cps_rate_errors = 0
    cps_negative_weight_rows = 0
    cps_missing_release_rows = 0
    for row in cps_rows:
        emp, unemp, lf = float(row["Emp"]), float(row["Unemp"]), float(row["LF"])
        if abs((emp + unemp) - lf) > 1e-6:
            cps_identity_errors += 1
        rate = float(row["u"]) if row["u"] else float("nan")
        if not (rate == rate and 0 <= rate <= 1):
            cps_rate_errors += 1
        if min(emp, unemp, lf) < 0:
            cps_negative_weight_rows += 1
        if row["release_date"] or row["release_date_status"] != "exact_public_use_file_date_not_verified":
            cps_missing_release_rows += 1
    if cps_identity_errors:
        failures.append(f"CPS Emp + Unemp != LF in {cps_identity_errors} rows")
    if cps_rate_errors:
        failures.append(f"CPS unemployment rate missing/outside [0,1] in {cps_rate_errors} rows")
    if cps_negative_weight_rows:
        failures.append(f"CPS has negative weighted totals in {cps_negative_weight_rows} rows")
    if cps_missing_release_rows:
        failures.append(f"CPS release-date metadata inconsistent in {cps_missing_release_rows} rows")
    if sorted(row["month"] for row in cps_diag) != expected_months:
        failures.append("CPS national diagnostics do not cover the expected month grid")

    acs_keys = [(row["year"], row["occupation_group"]) for row in acs_rows]
    acs_duplicate_count = len(acs_keys) - len(set(acs_keys))
    expected_years = sorted(ACS_RELEASE_DATES)
    acs_years = sorted({int(row["year"]) for row in acs_rows})
    if acs_duplicate_count:
        failures.append(f"ACS duplicate year/group keys: {acs_duplicate_count}")
    if acs_years != expected_years:
        failures.append(f"ACS years differ from expected list: {acs_years}")
    if len(acs_rows) != len(expected_years) * len(CIVILIAN_GROUPS):
        failures.append(f"ACS row count {len(acs_rows)} does not equal expected year/group grid")
    acs_age_share_errors = 0
    acs_education_share_errors = 0
    acs_invalid_ranges = 0
    acs_release_errors = 0
    for row in acs_rows:
        age_shares = [float(row[key]) for key in ("share_age_16_24", "share_age_25_54", "share_age_55_plus")]
        if any(not 0 <= value <= 1 for value in age_shares) or abs(sum(age_shares) - 1.0) > 1e-6:
            acs_age_share_errors += 1
        edu_keys = (
            "share_education_less_than_high_school", "share_education_high_school_or_equivalent",
            "share_education_some_college_or_associate", "share_education_bachelors",
            "share_education_graduate_or_professional",
        )
        edu_shares = [float(row[key]) for key in edu_keys]
        if row["education_numerator_weight"] and abs(sum(edu_shares) - 1.0) > 1e-6:
            acs_education_share_errors += 1
        if any(not 0 <= value <= 1 for value in [*age_shares, *edu_shares]):
            acs_invalid_ranges += 1
        if row["release_date"] != ACS_RELEASE_DATES[int(row["year"])]:
            acs_release_errors += 1
    if acs_age_share_errors:
        failures.append(f"ACS age shares fail their denominator identity in {acs_age_share_errors} rows")
    if acs_education_share_errors:
        failures.append(f"ACS education shares fail their denominator identity in {acs_education_share_errors} rows")
    if acs_invalid_ranges:
        failures.append(f"ACS shares outside [0,1] in {acs_invalid_ranges} rows")
    if acs_release_errors:
        failures.append(f"ACS PUMS release dates incorrect in {acs_release_errors} rows")

    return {
        "passed": not failures, "failures": failures,
        "cps": {
            "rows": len(cps_rows), "months": len(cps_months),
            "first_month": cps_months[0] if cps_months else None,
            "last_month": cps_months[-1] if cps_months else None,
            "missing_reference_month": "2025-10",
            "occupation_groups": len(CIVILIAN_GROUPS),
            "duplicate_keys": cps_duplicate_count,
            "identity_errors": cps_identity_errors,
            "unemployment_rate_errors": cps_rate_errors,
            "negative_weight_rows": cps_negative_weight_rows,
            "release_dates_unverified_as_recorded": len(cps_rows) - cps_missing_release_rows,
        },
        "acs": {
            "rows": len(acs_rows), "years": acs_years,
            "occupation_groups": len(CIVILIAN_GROUPS),
            "duplicate_keys": acs_duplicate_count,
            "age_share_identity_errors": acs_age_share_errors,
            "education_share_identity_errors": acs_education_share_errors,
            "share_range_errors": acs_invalid_ranges,
            "release_date_errors": acs_release_errors,
        },
    }
