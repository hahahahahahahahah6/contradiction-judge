"""Smoke tests for contradiction-judge (stdlib unittest). Run:
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from contradiction_judge import check_memories, set_judge
from contradiction_judge.cli import main


class TestRuleDetectors(unittest.TestCase):
    def test_age_contradiction_detected(self):
        report = check_memories([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I am 21 years old."},
        ])
        self.assertEqual(len(report["contradictions"]), 1)
        c = report["contradictions"][0]
        self.assertEqual(c["kind"], "numeric")
        self.assertEqual(c["engine"], "rules")
        self.assertEqual(c["entry_a"], "a")
        self.assertEqual(c["entry_b"], "b")

    def test_same_age_no_contradiction(self):
        report = check_memories([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "My age is 20."},
        ])
        self.assertEqual(report["contradictions"], [])

    def test_age_unit_normalization(self):
        # 240 months == 20 years -> no contradiction
        report = check_memories([
            {"id": "a", "text": "I'm 240 months old."},
            {"id": "b", "text": "I am 20 years old."},
        ])
        self.assertEqual(report["contradictions"], [])

    def test_birthday_conflict(self):
        report = check_memories([
            {"id": "a", "text": "My birthday is May 5, 2006."},
            {"id": "b", "text": "My birthday is 2006-06-01."},
        ])
        self.assertEqual(len(report["contradictions"]), 1)
        self.assertEqual(report["contradictions"][0]["kind"], "date")

    def test_preference_polarity_conflict(self):
        report = check_memories([
            {"id": "a", "text": "I love pizza."},
            {"id": "b", "text": "I hate pizza."},
        ])
        kinds = [c["kind"] for c in report["contradictions"]]
        self.assertIn("polarity", kinds)

    def test_location_conflict(self):
        report = check_memories([
            {"id": "a", "text": "I live in Pasadena."},
            {"id": "b", "text": "I live in Arcadia."},
        ])
        kinds = [c["kind"] for c in report["contradictions"]]
        self.assertIn("text", kinds)

    def test_unrelated_facts_no_contradiction(self):
        report = check_memories([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I love pizza."},
        ])
        self.assertEqual(report["contradictions"], [])


class TestArbitration(unittest.TestCase):
    def test_recency_wins_when_timestamped(self):
        report = check_memories([
            {"id": "old", "text": "I live in Pasadena.",
             "created_at": "2025-01-01T00:00:00"},
            {"id": "new", "text": "I live in Arcadia.",
             "created_at": "2026-06-01T00:00:00"},
        ])
        self.assertEqual(len(report["contradictions"]), 1)
        arb = report["contradictions"][0]["arbitration"]
        self.assertEqual(arb["strategy"], "recency")
        self.assertIn("new", arb["recommendation"])

    def test_manual_when_no_timestamps(self):
        report = check_memories([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I am 21 years old."},
        ])
        arb = report["contradictions"][0]["arbitration"]
        self.assertEqual(arb["strategy"], "manual")
        self.assertTrue(arb["reason"])


class TestLLMHook(unittest.TestCase):
    def tearDown(self):
        set_judge(None)

    def test_hook_reports_semantic_contradiction_rules_miss(self):
        calls = []

        def fake_judge(a, b):
            calls.append((a, b))
            return {"contradiction": True,
                    "explanation": "one says parents' house, other says own apartment"}

        set_judge(fake_judge)
        report = check_memories([
            {"id": "a", "text": "I live with my parents."},
            {"id": "b", "text": "My apartment lease renews in June."},
        ])
        self.assertTrue(calls, "judge was never called")
        self.assertEqual(len(report["contradictions"]), 1)
        c = report["contradictions"][0]
        self.assertEqual(c["kind"], "semantic")
        self.assertEqual(c["engine"], "llm")

    def test_hook_not_called_when_rules_already_flag(self):
        calls = []

        def fake_judge(a, b):
            calls.append((a, b))
            return None

        set_judge(fake_judge)
        check_memories([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I am 21 years old."},
        ])
        self.assertEqual(calls, [])


class TestCLI(unittest.TestCase):
    def tearDown(self):
        set_judge(None)

    def _write_memory(self, entries):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh)
        self.addCleanup(os.unlink, path)
        return path

    def test_cli_check_finds_contradiction_and_exits_zero(self):
        path = self._write_memory([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I am 21 years old."},
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["check", path])
        report = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(len(report["contradictions"]), 1)

    def test_cli_check_clean_file(self):
        path = self._write_memory([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I love pizza."},
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["check", path])
        report = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["contradictions"], [])

    def test_cli_fail_flag_exits_1_on_contradiction(self):
        path = self._write_memory([
            {"id": "a", "text": "I am 20 years old."},
            {"id": "b", "text": "I am 21 years old."},
        ])
        with redirect_stdout(io.StringIO()):
            code = main(["check", path, "--fail"])
        self.assertEqual(code, 1)

    def test_cli_missing_file_exits_2(self):
        with redirect_stdout(io.StringIO()):
            code = main(["check", "/nonexistent/memory.json"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
