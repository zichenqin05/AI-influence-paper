"""Historical source inventory, retrieval, and year-specific schema checks."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_v2.sources.cps_fixed_width import parse_layout
from pipeline_v2.sources.occupation import (
    SOC_MAJOR_GROUPS, cps_occupation_code_version, cps_occupation_variable,
)
from pipeline_v2.sources.panels import ACS_RELEASE_DATES


MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
ACS_RELEASE_SOURCES = {
    2018: "https://www.census.gov/programs-surveys/acs/news/data-releases/2018/release.html",
    2019: "https://www.census.gov/programs-surveys/acs/news/data-releases/2019/release.html",
    2021: "https://www.census.gov/programs-surveys/acs/news/data-releases/2021/release.html",
    2022: "https://www.census.gov/programs-surveys/acs/news/data-releases/2022/release.html",
    2023: "https://www.census.gov/programs-surveys/acs/news/data-releases/2023/release.html",
    2024: "https://www.census.gov/programs-surveys/acs/news/updates.2025.html",
}
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_workspace_path(package_root: Path, path: Path) -> str:
    workspace_root = package_root.parent.parent
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _source_record(
    *, source_id: str, url: str, local_path: str, reference_period: str,
    vintage: str, population: str, frequency: str, definition: str,
    license_text: str = "U.S. Census public-use data/documentation; consult source terms",
    release_date: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id, "url": url, "local_path": local_path,
        "reference_period": reference_period, "release_date": release_date,
        "retrieved_at": None, "vintage": vintage, "sha256": sha256,
        "license": license_text, "population": population, "frequency": frequency,
        "definition": definition,
        "release_date_status": "official_1_year_PUMS_release_date" if release_date else "not_applicable_or_not_verified",
        "release_date_source_url": ACS_RELEASE_SOURCES.get(int(reference_period)) if release_date and reference_period.isdigit() else None,
    }


def reference_source_specs(package_root: Path) -> list[dict[str, Any]]:
    """Describe each unique downloaded layout, dictionary, and official crosswalk."""
    workspace_root = package_root.parent.parent
    ref_root = package_root / "data" / "raw" / "reference"
    specs: list[dict[str, Any]] = []

    layout_specs = [
        ("cps_layout_2016_applied_2015", 2016, "2015", "January_2015_Record_Layout.txt"),
        ("cps_layout_2017_applied_2017_to_2019", "2017-2019", "2017", "2017_Basic_CPS_Public_Use_Record_Layout_plus_IO_Code_list.txt"),
    ]
    for year in range(2020, 2027):
        layout_specs.append((f"cps_layout_{year}", year, str(year),
                             f"{year}_Basic_CPS_Public_Use_Record_Layout_plus_IO_Code_list.txt"))
    for source_id, applied_year, source_year, filename in layout_specs:
        path = ref_root / "cps" / "layouts" / f"{applied_year}_{filename}"
        if not path.is_file():
            # 2017-2019 file copies are named with the applicability year.
            if source_year == "2017":
                path = ref_root / "cps" / "layouts" / "2017_January_2017_Record_Layout.txt"
                url = "https://www2.census.gov/programs-surveys/cps/datasets/2017/basic/January_2017_Record_Layout.txt"
            else:
                raise FileNotFoundError(f"Missing downloaded CPS layout document: {path}")
        else:
            if source_year == "2015":
                url = "https://www2.census.gov/programs-surveys/cps/datasets/2015/basic/January_2015_Record_Layout.txt"
            elif source_year == "2017":
                url = "https://www2.census.gov/programs-surveys/cps/datasets/2017/basic/January_2017_Record_Layout.txt"
            else:
                url = f"https://www2.census.gov/programs-surveys/cps/datasets/{source_year}/basic/{filename}"
        specs.append(_source_record(
            source_id=source_id, url=url, local_path=_relative_workspace_path(package_root, path),
            reference_period=str(applied_year), vintage=f"CPS Basic Monthly record layout source {source_year}; applies to {applied_year}",
            population="CPS Basic Monthly record layout", frequency="annual layout",
            definition="Official fixed-width positions, universes, and I/O code list",
        ))

    for year in (2018, 2019, 2021, 2022, 2023):
        path = ref_root / "acs" / "dictionaries" / f"PUMS_Data_Dictionary_{year}.txt"
        specs.append(_source_record(
            source_id=f"acs_pums_dictionary_{year}",
            url=f"https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_{year}.txt",
            local_path=_relative_workspace_path(package_root, path),
            reference_period=f"{year} ACS 1-Year PUMS",
            vintage=f"ACS {year} PUMS official data dictionary",
            population=f"ACS {year} 1-Year PUMS records", frequency="annual documentation",
            definition="Official variable definitions, values, and universes for year-specific PUMS",
            release_date=ACS_RELEASE_DATES[year],
        ))

    occupation_files = [
        ("census_occupation_crosswalk_2018", "census_2018_occ_to_soc.xlsx",
         "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx",
         "2010 and 2018 Census occupation codes, titles, and equivalent SOC codes"),
        ("census_occupation_codes_2010", "census_2010_occ_to_soc.xls",
         "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2010-occ-codes-with-crosswalk-from-2002-2011.xls",
         "2010 Census occupation codes and SOC crosswalk"),
        ("census_occupation_2010_2018_conversion_rates", "census_2010_to_2018_occ_conversion_rates.xlsx",
         "https://www.census.gov/data/tables/time-series/demo/industry-occupation/acs-tp78.html",
         "Official 2010 to 2018 occupation conversion rates; validation reference only"),
        ("census_occupation_2010_2018_direct_matches", "census_2010_to_2018_occ_direct_matches.xlsx",
         "https://www.census.gov/data/tables/time-series/demo/industry-occupation/acs-tp78.html",
         "Official direct-match occupation table; validation reference only"),
    ]
    for source_id, filename, url, definition in occupation_files:
        path = ref_root / "occupation" / filename
        specs.append(_source_record(
            source_id=source_id, url=url, local_path=_relative_workspace_path(package_root, path),
            reference_period="2010 and 2018 Census Occupation classification",
            vintage="Official Census occupation code/crosswalk publication",
            population="Occupation code classification reference", frequency="static crosswalk",
            definition=definition,
        ))
    return specs


def inspect_reference_schemas(package_root: Path) -> dict[str, Any]:
    """Check year-specific CPS layouts and ACS dictionary definitions."""
    layout_dir = package_root / "data" / "raw" / "reference" / "cps" / "layouts"
    cps_years = []
    for year in range(2016, 2027):
        matches = sorted(layout_dir.glob(f"{year}_*.txt"))
        if not matches:
            raise FileNotFoundError(f"No CPS layout file registered for {year}")
        path = matches[0]
        text = path.read_text(encoding="latin1")
        fields = parse_layout(text)
        # The 2020 classification changes before the field name changes in 2021.
        occ_field = cps_occupation_variable(year)
        required = ("HRMONTH", "HRYEAR4", "PRPERTYP", "PRTAGE", "PEMLR", "PWSSWGT", occ_field)
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"CPS {year} layout missing selected variables {missing}")
        if fields["PWSSWGT"].width != 10:
            raise ValueError(f"CPS {year} PWSSWGT width changed: {fields['PWSSWGT'].width}")
        lines = text.splitlines()
        weight_index = next(i for i, line in enumerate(lines) if line.strip().startswith("PWSSWGT\t"))
        weight_description = " ".join(lines[weight_index:weight_index + 3]).upper()
        if "4 IMPLIED DECIMAL" not in weight_description:
            raise ValueError(f"CPS {year} layout does not document four implied PWSSWGT decimals")
        if fields[occ_field].width != 4:
            raise ValueError(f"CPS {year} occupation field width changed: {fields[occ_field].width}")
        cps_years.append({
            "year": year, "layout_file": path.name,
            "layout_sha256": file_sha256(path),
            "variables": {name: {"start": fields[name].start, "end": fields[name].end,
                                  "width": fields[name].width} for name in required},
            "occupation_variable": occ_field,
            "occupation_code_version": cps_occupation_code_version(year),
            "person_weight": "PWSSWGT with four implied decimal places; analysis weight = field / 10000",
            "weight_universe": "adult civilian CPS person record; restricted to PRPERTYP=2 and age >=16",
        })

    dictionary_years = []
    for year in (2018, 2019, 2021, 2022, 2023, 2024):
        if year == 2024:
            path = package_root / "data" / "raw" / "acs" / "2024" / "PUMS_Data_Dictionary_2024.txt"
        else:
            path = package_root / "data" / "raw" / "reference" / "acs" / "dictionaries" / f"PUMS_Data_Dictionary_{year}.txt"
        text = path.read_text(encoding="latin1")
        expected_definitions = {
            "PWGTP": "Numeric     5",
            "OCCP": "Character   4",
            "ESR": "Character   1",
            "AGEP": "Numeric     2",
            "SCHL": "Character   2",
        }
        variable_blocks: dict[str, str] = {}
        lines = text.splitlines()
        for index, line in enumerate(lines):
            for name in expected_definitions:
                if re.match(rf"^{name}\s+", line):
                    variable_blocks[name] = "\n".join(lines[index:index + 6])
        absent = [name for name in expected_definitions if name not in variable_blocks]
        if absent:
            raise ValueError(f"ACS {year} dictionary missing variables {absent}")
        changed = [name for name, phrase in expected_definitions.items()
                   if phrase not in variable_blocks[name]]
        if changed:
            raise ValueError(f"ACS {year} variable type/width changed: {changed}")
        if "Integer weight of person" not in variable_blocks["PWGTP"]:
            raise ValueError(f"ACS {year} PWGTP is no longer an integer person weight")
        if "2018 OCC codes" not in variable_blocks["OCCP"]:
            raise ValueError(f"ACS {year} OCCP no longer states 2018 Census OCC basis")
        dictionary_years.append({
            "year": year, "dictionary_file": path.name,
            "dictionary_sha256": file_sha256(path),
            "variables": {name: variable_blocks[name].splitlines()[:3]
                          for name in expected_definitions},
            "occupation_code_version": "census_2018",
            "person_weight": "PWGTP integer person weight, no decimal scaling",
            "employment_universe": "ESR 1 or 2: civilian employed at work or with job but not at work",
            "occupation_universe": "OCCP 4-character 2018 Census OCC recode; blank is N/A by documented universe",
            "pums_release_date": ACS_RELEASE_DATES[year],
        })
    return {
        "schema_version": "1.0", "cps_years": cps_years, "acs_years": dictionary_years,
        "occupation_groups": [{"soc_major_code": f"{key}-0000", "title": value,
                               "civilian": key != "55"} for key, value in SOC_MAJOR_GROUPS.items()],
        "cps_classification_break": "2010 Census OCC through 2019; 2018 Census OCC beginning January 2020",
        "weight_comparability": "CPS PWSSWGT is rescaled from 4 implied decimals; ACS PWGTP is an integer person weight and is not rescaled",
    }


def _monthly_specs(package_root: Path) -> list[dict[str, Any]]:
    specs = []
    for year in range(2016, 2027):
        last_month = 8 if year == 2026 else 12
        for month_num, month in enumerate(MONTHS, start=1):
            if month_num > last_month or (year, month_num) == (2025, 10):
                continue
            yy = str(year)[-2:]
            filename = f"{month}{yy}pub.zip"
            url = f"https://www2.census.gov/programs-surveys/cps/datasets/{year}/basic/{filename}"
            local = package_root / "data" / "raw" / "cps" / str(year) / filename
            specs.append({
                "source_id": f"cps_basic_{year}_{month_num:02d}", "url": url,
                "path": local, "period": f"{year}-{month_num:02d}", "filename": filename,
                "year": year,
            })
    return specs


def _acs_specs(package_root: Path) -> list[dict[str, Any]]:
    specs = []
    for year in ACS_RELEASE_DATES:
        local = package_root / "data" / "raw" / "acs" / str(year) / "csv_pus.zip"
        if year == 2024 and local.is_file():
            continue
        specs.append({
            "source_id": f"acs_1year_pums_{year}_person",
            "url": f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_pus.zip",
            "path": local, "period": str(year), "year": year,
        })
    return specs


def _download_zip(spec: dict[str, Any]) -> dict[str, Any]:
    path: Path = spec["path"]
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite pre-existing source archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "occupation-panel-pipeline-v2/0.1"})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response, temp.open("xb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            with zipfile.ZipFile(temp) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError(f"ZIP CRC failure in {bad_member}")
            digest = file_sha256(temp)
            size = temp.stat().st_size
            temp.replace(path)
            return {"source_id": spec["source_id"], "bytes": size, "sha256": digest,
                    "path": path, "url": spec["url"], "period": spec["period"], "year": spec["year"]}
        except Exception as exc:
            last_error = exc
            if temp.exists():
                temp.unlink()
            if attempt < 3:
                time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def fetch_history(package_root: Path, manifest_path: Path, *, workers: int = 4) -> dict[str, Any]:
    """Retrieve the authorized historical CPS/ACS files and update the manifest."""
    workspace_root = package_root.parent.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_by_id = {row["source_id"]: row for row in manifest.get("sources", [])}
    for entry in manifest.get("sources", []):
        if entry.get("source_id") == "acs_1year_pums_2024_person_pilot":
            entry["release_date"] = ACS_RELEASE_DATES[2024]
            entry["release_date_status"] = "official_1_year_PUMS_release_date"
            entry["release_date_source_url"] = ACS_RELEASE_SOURCES[2024]
        elif entry.get("source_id") == "acs_1year_pums_requested_years":
            entry["release_date_status"] = "year_specific_dates_verified_for_downloaded_archives"
    for entry in reference_source_specs(package_root):
        source_id = entry["source_id"]
        if source_id not in existing_by_id:
            path = workspace_root / entry["local_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Registered reference source is missing: {path}")
            entry["sha256"] = file_sha256(path)
            entry["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            manifest["sources"].append(entry)
            existing_by_id[source_id] = entry

    known_gaps = manifest.setdefault("known_gaps", [])
    if not any(item.get("reference_period") == "2025-10" for item in known_gaps):
        known_gaps.append({
            "source_id": "cps_basic_2025_10_unavailable",
            "url": "https://www2.census.gov/programs-surveys/cps/datasets/2025/basic/oct25pub.zip",
            "reference_period": "2025-10", "status": "official_archive_url_returns_404",
            "treatment": "preserve_missing_month; no imputation or adjacent-month substitution",
        })

    specs = _monthly_specs(package_root) + _acs_specs(package_root)
    if "cps_basic_2026_08_pilot" in existing_by_id:
        specs = [spec for spec in specs if spec["source_id"] != "cps_basic_2026_08"]
    if "acs_1year_pums_2024_person_pilot" in existing_by_id:
        specs = [spec for spec in specs if spec["source_id"] != "acs_1year_pums_2024_person"]
    todo = [spec for spec in specs if spec["source_id"] not in existing_by_id]
    completed: list[str] = []
    errors: list[dict[str, str]] = []

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_download_zip, spec): spec for spec in todo}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
                if spec["source_id"].startswith("cps_"):
                    entry = _source_record(
                        source_id=result["source_id"], url=result["url"],
                        local_path=_relative_workspace_path(package_root, result["path"]),
                        reference_period=result["period"], release_date=None,
                        vintage=f"CPS Basic Monthly public-use archive, {result['period']}",
                        population="CPS Basic Monthly person records; research universe applied during aggregation",
                        frequency="monthly", definition="Official monthly fixed-width public-use microdata archive",
                    )
                else:
                    year = int(result["year"])
                    entry = _source_record(
                        source_id=result["source_id"], url=result["url"],
                        local_path=_relative_workspace_path(package_root, result["path"]),
                        reference_period=str(year), release_date=ACS_RELEASE_DATES[year],
                        vintage=f"ACS {year} 1-Year PUMS national person microdata",
                        population=f"ACS {year} 1-Year PUMS United States person records",
                        frequency="annual", definition="Official national person-file ZIP used for occupation structure",
                    )
                entry["sha256"] = result["sha256"]
                entry["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                manifest["sources"].append(entry)
                existing_by_id[entry["source_id"]] = entry
                completed.append(f"{entry['source_id']}:{result['bytes']}")
                save()
            except Exception as exc:
                errors.append({"source_id": spec["source_id"], "error": f"{type(exc).__name__}: {exc}"})

    expected_periods = [spec["source_id"] for spec in specs]
    unavailable_existing = [source_id for source_id in expected_periods if source_id in existing_by_id]
    result = {
        "requested_archives": len(expected_periods), "registered_archives": len(unavailable_existing),
        "newly_downloaded": len(completed), "completed": completed,
        "errors": errors, "known_missing_months": ["2025-10"], "workers": max(1, workers),
    }
    manifest["historical_fetch_status"] = result
    save()
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result
