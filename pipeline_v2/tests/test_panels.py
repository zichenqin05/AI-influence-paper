import csv
import gzip
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pipeline_v2.sources.occupation import CROSSWALK_COLUMNS
from pipeline_v2.sources.panels import build_acs_panel, build_cps_panel


def write_crosswalk(path: Path) -> None:
    row = {
        "source_code_version": "census_2018", "source_code": "0010",
        "source_title": "Chief executives", "source_soc_code": "11-1011",
        "occupation_group": "11-0000", "occupation_group_title": "Management Occupations",
        "civilian_group": "true", "mapping_rule": "test mapping", "mapping_status": "mapped",
    }
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CROSSWALK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


class WeightedPanelTests(unittest.TestCase):
    def test_cps_panel_applies_four_decimal_weight_scale_and_statuses(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            layout = root / "2020_test_layout.txt"
            layout.write_text("\n".join([
                "HRMONTH 2 MONTH 1-2",
                "HRYEAR4 4 YEAR 3-6",
                "PRPERTYP 1 PERSON TYPE 7-7",
                "PRTAGE 2 AGE 8-9",
                "PEMLR 1 LABOR STATUS 10-10",
                "PWSSWGT 10 FINAL WEIGHT 11-20",
                "PEIO1OCD 4 OCCUPATION CODE 21-24",
            ]), encoding="ascii")
            archive = root / "jan20pub.zip"
            rows = [
                "01" + "2020" + "2" + "30" + "1" + "0000010000" + "0010",
                "01" + "2020" + "2" + "30" + "4" + "0000002500" + "0010",
            ]
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("jan20pub.dat", "\n".join(rows) + "\n")
            crosswalk = root / "crosswalk.csv"
            write_crosswalk(crosswalk)
            output = root / "cps.csv.gz"
            diagnostics = root / "cps_diag.csv.gz"

            profile = build_cps_panel(
                archive_paths={"2020-01": archive}, layout_paths={2020: layout},
                crosswalk_path=crosswalk, output_path=output, diagnostic_path=diagnostics,
            )
            with gzip.open(output, "rt", encoding="utf-8", newline="") as stream:
                rows_by_group = {row["occupation_group"]: row for row in csv.DictReader(stream)}
            chief = rows_by_group["11-0000"]
            self.assertEqual(profile["rows"], 22)
            self.assertAlmostEqual(float(chief["Emp"]), 1.0)
            self.assertAlmostEqual(float(chief["Unemp"]), 0.25)
            self.assertAlmostEqual(float(chief["LF"]), 1.25)
            self.assertAlmostEqual(float(chief["u"]), 0.2)
            self.assertEqual(chief["n_emp"], "1")
            self.assertEqual(chief["n_unemp"], "1")

    def test_acs_panel_uses_integer_person_weight_and_employed_civilian_universe(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "acs.zip"
            contents = io.StringIO(newline="")
            writer = csv.writer(contents)
            writer.writerow(["PWGTP", "OCCP", "ESR", "AGEP", "SCHL"])
            writer.writerow(["2", "0010", "1", "20", "21"])
            writer.writerow(["3", "0010", "2", "55", "22"])
            writer.writerow(["100", "0010", "3", "35", "24"])
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("psam_pusa.csv", contents.getvalue())
            crosswalk = root / "crosswalk.csv"
            write_crosswalk(crosswalk)
            output = root / "acs.csv.gz"
            diagnostics = root / "acs_diag.csv.gz"

            build_acs_panel(
                archive_paths={2018: archive}, crosswalk_path=crosswalk,
                output_path=output, diagnostic_path=diagnostics,
            )
            with gzip.open(output, "rt", encoding="utf-8", newline="") as stream:
                rows_by_group = {row["occupation_group"]: row for row in csv.DictReader(stream)}
            chief = rows_by_group["11-0000"]
            self.assertEqual(chief["n_employed_persons"], "2")
            self.assertEqual(chief["PWGTP_sum"], "5")
            self.assertAlmostEqual(float(chief["mean_age"]), 41.0)
            self.assertAlmostEqual(float(chief["share_age_16_24"]), 0.4)
            self.assertAlmostEqual(float(chief["share_age_55_plus"]), 0.6)
            self.assertEqual(chief["release_date"], "2019-11-14")


if __name__ == "__main__":
    unittest.main()
