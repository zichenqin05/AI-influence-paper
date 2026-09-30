"""CLI for the isolated pipeline-v2 research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pipeline_v2.sources.cps_fixed_width import inspect_cps_pilot
from pipeline_v2.sources.acs_pums import inspect_and_cache_person_columns
from pipeline_v2.sources.signals import read_us_work_share
from pipeline_v2.sources.history import fetch_history, inspect_reference_schemas
from pipeline_v2.sources.occupation import build_crosswalk
from pipeline_v2.sources.panels import build_acs_panel, build_cps_panel, validate_panel_outputs
from pipeline_v2.modeling import (
    OUTPUT_ROOT as MODEL_OUTPUT_ROOT,
    build_model_features,
    run_ai_short_window,
    run_non_ai_backtest,
    update_run_manifest,
    verify_occupation_exposure,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
MANIFEST_PATH = PACKAGE_ROOT / "data" / "manifest" / "source_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit(*, manifest_path: Path | None = None, workspace_root: Path | None = None) -> int:
    manifest_path = manifest_path or MANIFEST_PATH
    workspace_root = workspace_root or WORKSPACE_ROOT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("sources", [])
    results = []
    failures = 0
    for entry in entries:
        local_path = entry.get("local_path")
        if not local_path:
            results.append({"source_id": entry["source_id"], "status": "not_downloaded"})
            continue
        path = workspace_root / str(local_path)
        if not path.is_file():
            results.append({"source_id": entry["source_id"], "status": "file_missing", "path": str(path)})
            failures += 1
            continue
        digest = sha256_file(path)
        expected = entry.get("sha256")
        if not expected:
            status = "hash_not_recorded"
            failures += 1
        elif expected == digest:
            status = "verified"
        else:
            status = "sha256_mismatch"
            failures += 1
        result: dict[str, object] = {
            "source_id": entry["source_id"], "status": status,
            "bytes": path.stat().st_size, "sha256": digest,
        }
        if entry["source_id"] == "openai_signals_us_work_share_selected":
            series = read_us_work_share(path)
            months = [str(row["month"]) for row in series]
            result["selected_rows"] = len(series)
            result["first_month"] = months[0] if months else None
            result["last_month"] = months[-1] if months else None
            result["share_range"] = [
                min(float(row["share_of_messages"]) for row in series),
                max(float(row["share_of_messages"]) for row in series),
            ] if series else None
        results.append(result)

    print(json.dumps({
        "manifest": str(manifest_path),
        "sources": results,
        "failure_count": failures,
        "passed": failures == 0,
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def download_once(url: str, destination: Path, expected_sha256: str | None) -> dict[str, object]:
    if destination.exists():
        digest = sha256_file(destination)
        if expected_sha256 and digest == expected_sha256:
            return {"bytes": destination.stat().st_size, "sha256": digest, "status": "already_present"}
        raise FileExistsError(f"Refusing to overwrite unverified existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "occupation-panel-pipeline-v2/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("xb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"bytes": destination.stat().st_size, "sha256": sha256_file(destination), "status": "downloaded"}


def fetch_pilot() -> int:
    manifest = load_manifest()
    by_id = {entry["source_id"]: entry for entry in manifest["sources"]}
    source_ids = ("cps_basic_2026_08_pilot", "cps_basic_2026_layout_pilot")
    fetched: dict[str, object] = {}
    for source_id in source_ids:
        entry = by_id[source_id]
        destination = WORKSPACE_ROOT / entry["local_path"]
        fetched[source_id] = download_once(entry["url"], destination, entry.get("sha256"))
        entry["sha256"] = fetched[source_id]["sha256"]
        entry["retrieved_at"] = entry.get("retrieved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry["release_date_status"] = "unknown; official publication date not recorded in manifest"

    archive_path = WORKSPACE_ROOT / by_id["cps_basic_2026_08_pilot"]["local_path"]
    layout_path = WORKSPACE_ROOT / by_id["cps_basic_2026_layout_pilot"]["local_path"]
    inspection = inspect_cps_pilot(archive_path, layout_path)
    save_manifest(manifest)
    print(json.dumps({"fetched": fetched, "pilot_inspection": inspection}, ensure_ascii=False, indent=2))
    if inspection.get("required_fields_missing"):
        return 2
    return 0


def fetch_acs_pilot() -> int:
    manifest = load_manifest()
    entry = next(source for source in manifest["sources"]
                 if source["source_id"] == "acs_1year_pums_2024_person_pilot")
    archive_path = WORKSPACE_ROOT / entry["local_path"]
    result = download_once(entry["url"], archive_path, entry.get("sha256"))
    entry["sha256"] = result["sha256"]
    entry["retrieved_at"] = entry.get("retrieved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry["release_date_status"] = "unknown; file server timestamp is not treated as publication date"
    save_manifest(manifest)

    cache_path = PACKAGE_ROOT / "data" / "interim" / "acs" / "2024" / "acs_2024_person_selected.csv.gz"
    profile_path = cache_path.with_name("acs_2024_person_pilot_profile.json")
    if cache_path.exists() or profile_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing ACS pilot output: {cache_path.parent}")
    profile = inspect_and_cache_person_columns(archive_path, cache_path)
    profile["source_id"] = entry["source_id"]
    profile["input_sha256"] = entry["sha256"]
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"download": result, "pilot_profile": profile,
                      "profile_path": str(profile_path)}, ensure_ascii=False, indent=2))
    return 0


def fetch_history_command(workers: int) -> int:
    result = fetch_history(PACKAGE_ROOT, MANIFEST_PATH, workers=workers)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _registered_files_by_period(manifest: dict[str, object], family: str) -> dict[str, Path]:
    sources = manifest.get("sources", [])
    found: dict[str, Path] = {}
    for entry in sources:
        source_id = str(entry.get("source_id", ""))
        period = str(entry.get("reference_period", ""))
        local_path = entry.get("local_path")
        if not local_path:
            continue
        if family == "cps" and re.fullmatch(r"\d{4}-\d{2}", period) and source_id.startswith("cps_basic_"):
            found[period] = WORKSPACE_ROOT / str(local_path)
        elif family == "acs" and period.isdigit() and source_id.startswith("acs_1year_pums_"):
            found[period] = WORKSPACE_ROOT / str(local_path)
    return found


def build_panels_command() -> int:
    manifest = load_manifest()
    crosswalk_source = PACKAGE_ROOT / "data" / "raw" / "reference" / "occupation" / "census_2018_occ_to_soc.xlsx"
    conversion_source = PACKAGE_ROOT / "data" / "raw" / "reference" / "occupation" / "census_2010_to_2018_occ_conversion_rates.xlsx"
    direct_match_source = PACKAGE_ROOT / "data" / "raw" / "reference" / "occupation" / "census_2010_to_2018_occ_direct_matches.xlsx"
    crosswalk_path = PACKAGE_ROOT / "data" / "processed" / "reference" / "occupation_crosswalk.csv"
    schema_path = PACKAGE_ROOT / "data" / "processed" / "reference" / "reference_schema_inventory.json"
    if crosswalk_path.exists() or schema_path.exists():
        raise FileExistsError("Reference outputs already exist; move to a new output directory before rebuilding")
    crosswalk_profile = build_crosswalk(
        crosswalk_source, crosswalk_path,
        conversion_rates_path=conversion_source,
        direct_matches_path=direct_match_source,
    )
    schema_profile = inspect_reference_schemas(PACKAGE_ROOT)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cps_archives = _registered_files_by_period(manifest, "cps")
    acs_archive_strings = _registered_files_by_period(manifest, "acs")
    acs_archives = {int(year): path for year, path in acs_archive_strings.items()}
    if not cps_archives or not acs_archives:
        raise RuntimeError("No registered historical CPS or ACS archives; run fetch-history first")
    missing_files = [str(path) for path in [*cps_archives.values(), *acs_archives.values()] if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Registered raw archives are missing: {missing_files[:5]}")
    for entry in manifest.get("sources", []):
        local_path = entry.get("local_path")
        period = str(entry.get("reference_period", ""))
        source_id = str(entry.get("source_id", ""))
        if local_path and ((source_id.startswith("cps_basic_") and re.fullmatch(r"\d{4}-\d{2}", period))
                           or (source_id.startswith("acs_1year_pums_") and period.isdigit())):
            if not entry.get("sha256"):
                raise RuntimeError(f"Source archive is not hash-pinned: {source_id}")

    layout_paths = {}
    layout_dir = PACKAGE_ROOT / "data" / "raw" / "reference" / "cps" / "layouts"
    for year in range(2016, 2027):
        matches = sorted(layout_dir.glob(f"{year}_*.txt"))
        if not matches:
            raise FileNotFoundError(f"Missing year-specific CPS layout for {year}")
        layout_paths[year] = matches[0]

    output_dir = PACKAGE_ROOT / "data" / "processed" / "panels"
    cps_result = build_cps_panel(
        archive_paths=cps_archives, layout_paths=layout_paths, crosswalk_path=crosswalk_path,
        output_path=output_dir / "cps_occupation_month.csv.gz",
        diagnostic_path=output_dir / "cps_national_month_diagnostic.csv.gz",
    )
    acs_result = build_acs_panel(
        archive_paths=acs_archives, crosswalk_path=crosswalk_path,
        output_path=output_dir / "acs_occupation_year.csv.gz",
        diagnostic_path=output_dir / "acs_mapping_diagnostic.csv.gz",
    )
    profile = {"crosswalk": crosswalk_profile, "schema_inventory": str(schema_path),
               "cps": cps_result, "acs": acs_result,
               "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model_fit_status": "no_model_fitted"}
    profile_path = output_dir / "panel_build_profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    artifacts = manifest.setdefault("derived_artifacts", [])
    for path, artifact_type in [
        (crosswalk_path, "occupation_crosswalk"), (schema_path, "yearly_schema_inventory"),
        (output_dir / "cps_occupation_month.csv.gz", "cps_weighted_month_panel"),
        (output_dir / "cps_national_month_diagnostic.csv.gz", "cps_national_reconciliation_diagnostic"),
        (output_dir / "acs_occupation_year.csv.gz", "acs_weighted_year_panel"),
        (output_dir / "acs_mapping_diagnostic.csv.gz", "acs_mapping_coverage_diagnostic"),
        (profile_path, "panel_build_profile"),
    ]:
        artifacts.append({"artifact_type": artifact_type,
                          "local_path": path.resolve().relative_to(WORKSPACE_ROOT).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    save_manifest(manifest)
    print(json.dumps(profile, ensure_ascii=False, indent=2, default=str))
    return 0


def validate_panels_command() -> int:
    output_dir = PACKAGE_ROOT / "data" / "processed" / "panels"
    result = validate_panel_outputs(
        cps_path=output_dir / "cps_occupation_month.csv.gz",
        cps_diagnostic_path=output_dir / "cps_national_month_diagnostic.csv.gz",
        acs_path=output_dir / "acs_occupation_year.csv.gz",
        acs_diagnostic_path=output_dir / "acs_mapping_diagnostic.csv.gz",
    )
    report_path = output_dir / "panel_quality_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = load_manifest()
    artifacts = manifest.setdefault("derived_artifacts", [])
    artifact = {"artifact_type": "panel_quality_report",
                "local_path": report_path.resolve().relative_to(WORKSPACE_ROOT).as_posix(),
                "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)}
    artifacts[:] = [row for row in artifacts if row.get("artifact_type") != "panel_quality_report"]
    artifacts.append(artifact)
    save_manifest(manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def build_model_features_command() -> int:
    result = build_model_features()
    update_run_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["passed"] else 1


def backtest_command() -> int:
    result = run_non_ai_backtest()
    update_run_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["passed"] else 1


def verify_exposure_command() -> int:
    result = verify_occupation_exposure()
    update_run_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ai_short_window_eligible") else 1


def ai_short_window_command() -> int:
    result = run_ai_short_window()
    update_run_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "complete" else (0 if result.get("status") == "skipped" else 1)


def unavailable(stage: str) -> int:
    print(
        f"{stage} is not implemented yet. No model or empirical output was generated; "
        "complete and verify the required source adapters and transformations first.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline_v2.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="check source files, hashes, and the cached Signals series")
    fetch_parser = commands.add_parser("fetch", help="download a bounded official data pilot")
    fetch_group = fetch_parser.add_mutually_exclusive_group(required=True)
    fetch_group.add_argument("--pilot", action="store_true",
                             help="download one CPS month and its layout")
    fetch_group.add_argument("--acs-pilot", action="store_true",
                             help="download and stream the 2024 ACS national person file")
    fetch_group.add_argument("--history", action="store_true",
                             help="download the selected historical CPS months and ACS years")
    fetch_parser.add_argument("--workers", type=int, default=4,
                              help="maximum concurrent official archive downloads (history only)")
    commands.add_parser("build-panel", help="build validated weighted occupation panels")
    commands.add_parser("validate-panel", help="check weighted panel keys, arithmetic, and release metadata")
    commands.add_parser("build-model-features", help="build labels, fold-specific factors, and acceptance artifacts")
    backtest_parser = commands.add_parser("backtest", help="run rolling time-out forecasts after factor/fold acceptance")
    backtest_parser.add_argument("--mode", choices=("retrospective", "strict_asof"), default="retrospective")
    commands.add_parser("verify-exposure", help="verify OpenAI occupational exposure and weighted code coverage")
    commands.add_parser("ai-short-window", help="run the short AI/P experiment after E validation")
    commands.add_parser("report", help="write evidence-backed method and result fragments")
    args = parser.parse_args(argv)
    if args.command == "audit":
        return audit()
    if args.command == "fetch":
        if args.acs_pilot:
            return fetch_acs_pilot()
        if args.history:
            return fetch_history_command(args.workers)
        return fetch_pilot()
    if args.command == "build-panel":
        return build_panels_command()
    if args.command == "validate-panel":
        return validate_panels_command()
    if args.command == "build-model-features":
        return build_model_features_command()
    if args.command == "backtest":
        if args.mode == "strict_asof":
            print(json.dumps({"status": "blocked", "reason": "CPS exact file release dates are unknown"}, ensure_ascii=False))
            return 2
        return backtest_command()
    if args.command == "verify-exposure":
        return verify_exposure_command()
    if args.command == "ai-short-window":
        return ai_short_window_command()
    return unavailable(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
