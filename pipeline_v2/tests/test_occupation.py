import unittest

from pipeline_v2.sources.occupation import (
    cps_occupation_code_version,
    cps_occupation_variable,
    normalize_census_code,
    resolve_target_major_groups,
    soc_major_group,
)


class OccupationVersionTests(unittest.TestCase):
    def test_numeric_code_preserves_four_digit_census_code(self):
        self.assertEqual(normalize_census_code(1005), "1005")
        self.assertEqual(normalize_census_code("0010"), "0010")
        self.assertIsNone(normalize_census_code("001"))

    def test_soc_major_group_accepts_official_x_ending_codes(self):
        self.assertEqual(soc_major_group("15-121X"), "15")
        self.assertEqual(soc_major_group("11-0000"), "11")
        self.assertIsNone(soc_major_group("not a SOC code"))

    def test_cps_classification_break_and_field_name_change_are_distinct(self):
        self.assertEqual(cps_occupation_code_version(2019), "census_2010")
        self.assertEqual(cps_occupation_code_version(2020), "census_2018")
        self.assertEqual(cps_occupation_variable(2020), "PEIO1OCD")
        self.assertEqual(cps_occupation_variable(2021), "PTIO1OCD")

    def test_2010_bridge_maps_only_when_official_destinations_share_a_major_group(self):
        majors = {"0051": "11", "0052": "11", "0705": "13", "1108": "15"}
        self.assertEqual(resolve_target_major_groups({"0051", "0052"}, majors), ("11", "mapped"))
        self.assertEqual(resolve_target_major_groups({"0705", "1108"}, majors),
                         (None, "unmapped_ambiguous_cross_year_major_group"))
        self.assertEqual(resolve_target_major_groups(set(), majors),
                         (None, "unmapped_no_verified_2018_bridge"))


if __name__ == "__main__":
    unittest.main()
