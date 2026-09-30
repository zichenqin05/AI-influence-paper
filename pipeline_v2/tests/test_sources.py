import unittest
import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pipeline_v2.sources.acs_pums import inspect_and_cache_person_columns
from pipeline_v2.sources.cps_fixed_width import FieldLayout, parse_layout, read_fixed_width_fields
from pipeline_v2.sources.signals import read_us_work_share


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class SourceAdapterTests(unittest.TestCase):
    def test_acs_person_reader_selects_and_preserves_requested_columns(self):
        with TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "person.zip"
            cache_path = Path(temp_dir) / "selected.csv.gz"
            headers = ["SERIALNO", "SPORDER", "PWGTP", "OCCP", "ESR", "AGEP", "SCHL", "EXTRA"]
            with ZipFile(archive_path, "w") as archive:
                import io
                content = io.StringIO(newline="")
                writer = csv.writer(content)
                writer.writerow(headers)
                writer.writerow(["0001", "01", "12", "1230", "1", "30", "20", "x"])
                writer.writerow(["0002", "02", "8", "0000", "6", "67", "16", "y"])
                archive.writestr("psam_pusa.csv", content.getvalue())

            profile = inspect_and_cache_person_columns(archive_path, cache_path)
            self.assertEqual(profile["records"], 2)
            self.assertEqual(profile["age_min"], 30)
            self.assertEqual(profile["age_max"], 67)
            self.assertEqual(profile["esr_value_counts"], {"1": 1, "6": 1})
            self.assertEqual(profile["blank_value_counts"]["OCCP"], 0)
            with gzip.open(cache_path, "rt", encoding="utf-8", newline="") as cached:
                records = list(csv.DictReader(cached))
            self.assertEqual(records[0]["SERIALNO"], "0001")
            self.assertEqual(records[1]["SCHL"], "16")
            self.assertNotIn("EXTRA", records[0])

    def test_signals_reader_filters_us_work_related_rows(self):
        rows = read_us_work_share(FIXTURE_DIR / "signals_minimal.csv")
        self.assertEqual([(row["month"], row["share_of_messages"]) for row in rows], [
            ("2024-07-01", 0.509), ("2024-08-01", 0.516)
        ])

    def test_layout_parser_uses_dictionary_positions_and_width(self):
        layout = "\n".join([
            "HRMONTH     2    MONTH OF INTERVIEW                                  16-17",
            "PRTAGE      2    PERSONS AGE                                         122 - 123",
            "IGNORED     2    MISMATCH DECLARED WIDTH                              7 - 9",
        ])
        fields = parse_layout(layout)
        self.assertEqual(fields["HRMONTH"], FieldLayout("HRMONTH", 16, 17))
        self.assertEqual(fields["PRTAGE"].width, 2)
        self.assertNotIn("IGNORED", fields)

    def test_fixed_width_selected_columns_are_one_based_and_inclusive(self):
        import io

        count, widths, rows = read_fixed_width_fields(
            io.BytesIO(b"0123456789\nabcdefghij\n"),
            [FieldLayout("middle", 4, 7)],
        )
        self.assertEqual(count, 2)
        self.assertEqual(dict(widths), {10: 2})
        self.assertEqual(rows, [{"middle": "3456"}, {"middle": "defg"}])
