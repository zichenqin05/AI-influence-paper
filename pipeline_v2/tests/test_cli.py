import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_v2.cli import audit


class SourceAuditTests(unittest.TestCase):
    def test_missing_registered_file_fails_audit_and_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = root / "present.bin"
            payload.write_bytes(b"known source")
            expected = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"sources": [
                {"source_id": "present", "local_path": "present.bin", "sha256": expected},
                {"source_id": "missing", "local_path": "missing.bin", "sha256": expected},
                {"source_id": "planned", "local_path": None, "sha256": None},
            ]}), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = audit(manifest_path=manifest_path, workspace_root=root)

            report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual({row["source_id"]: row["status"] for row in report["sources"]}, {
                "present": "verified",
                "missing": "file_missing",
                "planned": "not_downloaded",
            })

    def test_existing_file_without_pinned_hash_is_unverified_and_fails_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "unpinned.bin").write_bytes(b"present but unpinned")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"sources": [
                {"source_id": "unpinned", "local_path": "unpinned.bin", "sha256": None},
            ]}), encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = audit(manifest_path=manifest_path, workspace_root=root)

            report = json.loads(output.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(report["sources"][0]["status"], "hash_not_recorded")


if __name__ == "__main__":
    unittest.main()
