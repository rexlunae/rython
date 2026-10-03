"""Sweep accounting tests use Cargo protocol fixtures, not Python semantics claims."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import rank_causes
import run_sweep
import summarize


def diagnostic(crate, code=None, message="unsupported construct", level="error"):
    return json.dumps({
        "reason": "compiler-message",
        "target": {"src_path": str(crate / "src/lib.rs")},
        "message": {"level": level, "code": {"code": code} if code else None,
                    "message": message, "rendered": f"{level}: {message}\n"},
    }) + "\n"


def finished(success):
    return json.dumps({"reason": "build-finished", "success": success}) + "\n"


def parse(lines, status=101):
    return run_sweep.parse_build(lines, Path("/project"), status, io.StringIO())


class DiagnosticsTests(unittest.TestCase):
    def test_coded_uncoded_and_summary_are_distinct(self):
        result = parse([
            diagnostic(Path("/project"), "E0308", "expected `i64`, found `String`"),
            diagnostic(Path("/project")),
            diagnostic(Path("/project"), message="unused field", level="warning"),
            "error: could not compile `probe` (lib) due to 2 previous errors\n",
            finished(False),
        ])
        self.assertEqual(result["status"], "build-failed")
        self.assertEqual((result["total"], result["uncoded_total"], result["diagnostic_total"]), (1, 1, 2))
        self.assertEqual(result["e0308_pairs"], {"i64 | String": 1})

    def test_compile_error_is_a_measured_failure_with_zero_coded_errors(self):
        result = parse([diagnostic(Path("/project")), finished(False)])
        self.assertEqual(result["status"], "build-failed")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["uncoded_total"], 1)

    def test_denied_lint_code_is_not_a_historical_e_code(self):
        result = parse([diagnostic(Path("/project"), "dependency_on_unit_never_type_fallback"), finished(False)])
        self.assertEqual(result["status"], "build-failed")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["uncoded_total"], 1)
        self.assertEqual(result["histogram"], {})

    def test_cargo_failure_is_not_zero_errors(self):
        for log in (["error: failed to get dependency\n"], [finished(False)], []):
            with self.subTest(log=log):
                result = parse(log)
                self.assertEqual(result["status"], "unmeasured")
                self.assertIsNone(result["total"])

    def test_dependency_failure_does_not_measure_the_generated_crate(self):
        result = parse([diagnostic(Path("/dependency"), "E0308"), finished(False)])
        self.assertEqual(result["status"], "unmeasured")
        self.assertEqual(result["dependency_errors"], 1)
        self.assertIsNone(result["total"])

    def test_interrupted_diagnostics_are_not_a_complete_measurement(self):
        result = parse([diagnostic(Path("/project"), "E0308")], -9)
        self.assertEqual(result["status"], "unmeasured")
        self.assertIsNone(result["total"])

    def test_success_requires_a_success_event_and_no_errors(self):
        self.assertEqual(parse([finished(True)], 0)["status"], "built")
        self.assertEqual(parse([], 0)["status"], "unmeasured")
        self.assertEqual(parse([diagnostic(Path("/project")), finished(True)], 0)["status"], "unmeasured")

    def test_canonical_paths_and_non_diagnostic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crate = root / "real"
            crate.mkdir()
            alias = root / "alias"
            alias.symlink_to(crate, target_is_directory=True)
            out = io.StringIO()
            result = run_sweep.parse_build([
                diagnostic(alias, "E0308"), "[]\n", '{"reason":"compiler-artifact"}\n', finished(False),
            ], crate, 101, out)
            self.assertEqual(result["total"], 1)
            self.assertIn("unsupported construct", out.getvalue())


class ProcessTests(unittest.TestCase):
    def test_timeout_preserves_log_and_terminates_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_sweep.run_logged(
                    [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                    log, timeout=0.3,
                )
            self.assertIn("started", log.read_text())

    def test_conversion_failure_never_runs_cargo(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_sweep, "run_logged", return_value=1) as call:
            result = run_sweep.sweep_one({"name": "demo", "requirement": "demo==1"}, Path("/rypip"), Path(tmp))
            self.assertEqual(result["status"], "convert-failed")
            self.assertIsNone(result["total"])
            self.assertEqual(call.call_count, 1)

    def test_successful_conversion_logs_cargo_and_scrubs_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def execute(cmd, log, **kwargs):
                if cmd[0] == "/rypip":
                    log.write_text("conversion warning retained\n")
                    return 0
                self.assertEqual(cmd, ["cargo", "build", "--message-format=json"])
                for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_DIR"):
                    self.assertNotIn(key, kwargs["env"])
                log.write_text(diagnostic(root / "crate-demo") + finished(False))
                return 101
            with patch.dict(os.environ, {"RUSTFLAGS": "-Dwarnings", "CARGO_TARGET_DIR": "/shared", "CARGO_ENCODED_RUSTFLAGS": "-Dwarnings"}), patch.object(run_sweep, "run_logged", side_effect=execute):
                result = run_sweep.sweep_one({"name": "demo", "requirement": "demo==1"}, Path("/rypip"), root)
            self.assertEqual(result["status"], "build-failed")
            self.assertIn("warning retained", (root / "demo-convert.log").read_text())
            self.assertIn("unsupported construct", (root / "demo-build.log").read_text())

    def test_timeout_and_missing_cargo_are_unmeasured(self):
        for exc in (FileNotFoundError("cargo missing"), subprocess.TimeoutExpired("cargo", 1)):
            with self.subTest(exc=exc), tempfile.TemporaryDirectory() as tmp, patch.object(run_sweep, "run_logged", side_effect=[0, exc]):
                result = run_sweep.sweep_one({"name": "demo", "requirement": "demo==1"}, Path("/rypip"), Path(tmp))
                self.assertEqual(result["status"], "unmeasured")
                self.assertEqual(result["phase"], "build")
                self.assertIsNone(result["total"])


class ComparisonTests(unittest.TestCase):
    def record(self, total=2, uncoded=0, **changes):
        return {"status": "build-failed", "total": total, "uncoded_total": uncoded,
                "convert_status": 0, "build_status": 101, "requirement": "demo==1", **changes}

    def compare(self, b, c):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            valid = summarize.summarize(b, c)
        return valid, out.getvalue()

    def test_coded_error_replaced_by_refusal_is_not_build_success(self):
        valid, text = self.compare({"packages": {"demo": self.record(1)}},
                                   {"packages": {"demo": self.record(0, 1)}})
        self.assertTrue(valid)
        self.assertIn("build-failed -> build-failed", text)
        self.assertIn("all diagnostics): +0", text)

    def test_missing_conversion_failed_and_unmeasured_candidates_have_no_grand_delta(self):
        for candidate in ({}, {"status": "convert-failed", "total": None},
                          self.record(total=None, status="unmeasured")):
            with self.subTest(candidate=candidate):
                valid, text = self.compare({"packages": {"demo": self.record()}}, {"packages": {"demo": candidate}})
                self.assertFalse(valid)
                self.assertIn("grand total delta: unavailable", text)

    def test_different_pin_is_not_comparable(self):
        valid, text = self.compare({"packages": {"demo": self.record()}},
                                   {"packages": {"demo": self.record(requirement="demo==2")}})
        self.assertFalse(valid)
        self.assertIn("pin changed", text)

    def test_legacy_zero_error_failure_is_ambiguous(self):
        legacy = {"total": 0, "convert_status": 0, "build_status": 101}
        self.assertIsNotNone(summarize.measurement_problem(legacy))
        legacy["total"] = 2
        self.assertIsNone(summarize.measurement_problem(legacy))

    def test_legacy_uncoded_count_is_unknown_not_zero(self):
        legacy = self.record()
        del legacy["status"], legacy["uncoded_total"]
        valid, text = self.compare({"packages": {"demo": legacy}}, {"packages": {"demo": self.record(1)}})
        self.assertTrue(valid)
        self.assertIn("unknown (legacy)", text)
        self.assertIn("all diagnostics): unavailable", text)

    def test_changed_converter_invalidates_comparison(self):
        valid, text = self.compare({"packages": {"demo": self.record()}},
                                   {"packages": {"demo": self.record()}, "measurement_error": "sources changed"})
        self.assertFalse(valid)
        self.assertIn("UNMEASURED", text)


class ProvenanceTests(unittest.TestCase):
    def test_binary_inputs_follow_cargo_depfile_and_reject_stale_or_foreign_builds(self):
        with tempfile.TemporaryDirectory(prefix="sweep inputs ") as tmp:
            root = Path(tmp)
            source = root / "crates/rypip/src/main.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn main() {}")
            binary = root / "rypip"
            binary.write_bytes(b"test binary")
            stamp = source.stat().st_mtime_ns + 1_000_000_000
            os.utime(binary, ns=(stamp, stamp))
            depfile = binary.with_suffix(".d")
            depfile.write_text(f"{binary}: " + str(source).replace(" ", "\\ ") + "\n")
            run_sweep.check_binary_inputs(binary, root)
            runtime = root / "crates/stdpython/src/lib.rs"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("runtime compiled by generated crate, not rypip")
            os.utime(runtime, ns=(stamp + 1, stamp + 1))
            run_sweep.check_binary_inputs(binary, root)
            os.utime(source, ns=(stamp + 1, stamp + 1))
            with self.assertRaisesRegex(ValueError, "stale rypip"):
                run_sweep.check_binary_inputs(binary, root)
            depfile.write_text(f"{binary}: /another/checkout/crates/rypip/src/main.rs\n")
            with self.assertRaisesRegex(ValueError, "not built from this checkout"):
                run_sweep.check_binary_inputs(binary, root)
            depfile.unlink()
            with self.assertRaises(ValueError):
                run_sweep.check_binary_inputs(binary, root)

    def test_source_snapshot_tracks_untracked_deleted_and_dirty_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
            git("init", "-q")
            source = root / "crates/rypip/src/main.rs"
            source.parent.mkdir(parents=True)
            source.write_text("fn main() {}")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "source")
            before = run_sweep.source_snapshot(root)
            self.assertFalse(before["source_dirty"])
            (root / "README.md").write_text("documentation only")
            git("add", "README.md")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "docs")
            self.assertEqual(before, run_sweep.source_snapshot(root))
            extra = source.with_name("new.rs")
            extra.write_text("new source")
            added = run_sweep.source_snapshot(root)
            self.assertTrue(added["source_dirty"])
            self.assertNotEqual(before["source_sha256"], added["source_sha256"])
            extra.unlink()
            source.unlink()
            deleted = run_sweep.source_snapshot(root)
            self.assertTrue(deleted["source_dirty"])
            self.assertNotEqual(before["source_sha256"], deleted["source_sha256"])


class EntryPointTests(unittest.TestCase):
    def invoke(self, result, snapshots=None, with_idioms=False):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            snapshot = {"source_commit": "source-revision", "source_sha256": "hash",
                        "source_dirty": False}
            argv = ["run_sweep.py", "--rypip", sys.executable, "--workdir", tmp,
                    "--out", str(out), "--package", "certifi"]
            if with_idioms:
                argv.append("--with-idioms")
            with patch.object(sys, "argv", argv), patch.object(run_sweep, "check_binary_inputs"), patch.object(run_sweep, "source_snapshot", side_effect=snapshots or [snapshot, snapshot]), patch.object(run_sweep, "run", return_value=subprocess.CompletedProcess([], 0, "version", "")), patch.object(run_sweep, "sweep_one", return_value=result), patch.object(run_sweep.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = run_sweep.main()
            return status, json.loads(out.read_text())

    def test_incomplete_run_writes_result_and_exits_nonzero(self):
        code, data = self.invoke({"status": "unmeasured", "total": None, "error": "missing dependency"})
        self.assertEqual(code, 1)
        self.assertFalse(data["measurement_complete"])
        self.assertIsNone(data["packages"]["certifi"]["total"])

    def test_measured_compile_failure_is_a_completed_sweep_not_a_successful_build(self):
        code, data = self.invoke({"status": "build-failed", "total": 0, "uncoded_total": 1})
        self.assertEqual(code, 0)
        self.assertTrue(data["measurement_complete"])
        self.assertEqual(data["packages"]["certifi"]["status"], "build-failed")
        self.assertEqual(data["rypip_commit"], "source-revision")

    def test_missing_binary_at_final_check_preserves_results_as_incomplete(self):
        with patch.object(run_sweep, "binary_digest", side_effect=["initial", FileNotFoundError("binary removed")]):
            code, data = self.invoke({"status": "built", "total": 0, "uncoded_total": 0})
        self.assertEqual(code, 1)
        self.assertIn("measurement_error", data)

    def test_requested_idiom_failure_marks_sweep_incomplete(self):
        code, data = self.invoke({"status": "built", "total": 0, "uncoded_total": 0}, with_idioms=True)
        self.assertEqual(code, 1)
        self.assertIn("error", data["idioms"])

    def test_summary_cli_returns_nonzero_for_incomplete_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            cand = Path(tmp) / "cand.json"
            base.write_text(json.dumps({"packages": {"demo": ComparisonTests().record()}}))
            cand.write_text(json.dumps({"packages": {"demo": {"total": None, "status": "unmeasured"}}}))
            p = subprocess.run([sys.executable, str(Path(summarize.__file__)), str(base), str(cand)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn("grand total delta: unavailable", p.stdout)

class RankCausesTests(unittest.TestCase):
    """The ranking must count the boundary it names, or it misleads the next round."""

    METHOD_ON_PYVALUE = ("no method named `close` found for enum `stdpython::PyValue` "
                         "in the current scope")
    METHOD_ON_OTHER = ("no method named `push` found for enum `PyList` "
                       "in the current scope")
    FIELD_ON_PYVALUE = "no field `timeout` on type `stdpython::PyValue`"

    def message(self, code, text, file_name="src/demo.rs", crate=None):
        return {"level": "error", "code": {"code": code}, "message": text,
                "spans": [{"is_primary": True, "file_name": file_name}]}

    def event(self, message, crate):
        return {"reason": "compiler-message",
                "target": {"src_path": f"{crate}/src/demo.rs"}, "message": message}

    def write_workdir(self, tmp, messages, package="demo", extra=None):
        workdir = Path(tmp)
        crate = workdir / f"crate-{package}"
        (crate / "src").mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self.event(m, crate)) for m in messages]
        lines += [json.dumps(e) for e in (extra or [])]
        (workdir / f"{package}-cargo.jsonl").write_text("\n".join(lines) + "\n")
        return workdir

    def record(self, tmp, histogram, package="demo"):
        path = Path(tmp) / "run.json"
        path.write_text(json.dumps({
            "rypip_commit": "abc1234",
            "packages": {package: {"status": "build-failed", "histogram": histogram}}}))
        return path

    def run_rank(self, histogram, messages):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record(tmp, histogram)
            workdir = self.write_workdir(tmp, messages)
            argv = ["rank_causes.py", str(record), "--workdir", str(workdir)]
            with patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                code = rank_causes.main()
            return code, out.getvalue()

    def test_pyvalue_boundary_is_counted_apart_from_other_receivers(self):
        code, text = self.run_rank(
            {"E0599": 2, "E0609": 1},
            [self.message("E0599", self.METHOD_ON_PYVALUE),
             self.message("E0599", self.METHOD_ON_OTHER),
             self.message("E0609", self.FIELD_ON_PYVALUE)])
        self.assertEqual(code, 0)
        # 2 of 3 sites are the boundary; the other-receiver site is separate.
        self.assertIn("the PyValue attribute boundary: 2 sites", text)
        self.assertIn("E0599 no method on PyValue", text)
        self.assertIn("E0599 no method (other receiver)", text)
        self.assertIn("E0609 no field on PyValue", text)

    def test_percentages_use_the_record_histogram_not_the_event_count(self):
        # The shape share comes from the corpus total (the record's
        # histogram), so it stays meaningful when only a sample of the
        # events for a code is present in the workdir.
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record(tmp, {"E0599": 99, "E0609": 1})
            workdir = self.write_workdir(tmp, [], package="demo")
            argv = ["rank_causes.py", str(record), "--workdir", str(workdir)]
            with patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                # Reconciled counts differ, so the log set is refused rather
                # than ranked against a histogram it does not match.
                self.assertEqual(rank_causes.main(), 1)
            self.assertIn("do not match what was measured", out.getvalue())

    def test_missing_workdir_is_an_explicit_failure_not_an_empty_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record(tmp, {"E0599": 1})
            argv = ["rank_causes.py", str(record),
                    "--workdir", str(Path(tmp) / "absent")]
            with patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(rank_causes.main(), 1)
            self.assertIn("NOT COMPARABLE", out.getvalue())

    def test_non_record_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"packages": "not-a-dict"}))
            with self.assertRaises(SystemExit):
                rank_causes.load_record(bad)

    def test_dependency_diagnostics_are_not_ranked_as_generated_code(self):
        # parse_build excludes dependency diagnostics from the histogram, so
        # counting them here would inflate shapes against a denominator that
        # never included them.
        with tempfile.TemporaryDirectory() as tmp:
            crate = Path(tmp) / "crate-demo"
            (crate / "src").mkdir(parents=True, exist_ok=True)
            dependency = {"reason": "compiler-message",
                          "target": {"src_path": "/registry/src/hashdep/src/dep.rs"},
                          "message": self.message("E0599", self.METHOD_ON_PYVALUE)}
            workdir = self.write_workdir(tmp, [], extra=[dependency])
            record = self.record(tmp, {"E0308": 2})
            argv = ["rank_causes.py", str(record), "--workdir", str(workdir)]
            with patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(rank_causes.main(), 1)
            self.assertNotIn("PyValue attribute boundary: 1", out.getvalue())

    def test_wrapper_receivers_are_not_counted_as_the_pyvalue_boundary(self):
        # Option<PyValue>/Vec<PyValue> fail at the wrapper, which is a
        # different defect; reference sugar is still a PyValue receiver.
        code, text = self.run_rank(
            {"E0599": 3},
            [self.message("E0599", "no method named `close` found for enum "
                                   "`Option<stdpython::PyValue>` in the current scope"),
             self.message("E0599", "no field `x` on type `(stdpython::PyValue,)`"),
             self.message("E0599", "no method named `close` found for mutable "
                                   "reference `&mut stdpython::PyValue` in the current scope")])
        self.assertEqual(code, 0)
        self.assertIn("the PyValue attribute boundary: 1 sites", text)
        self.assertIn("E0599 no method (other receiver)", text)
        self.assertTrue(rank_causes.is_plain_pyvalue("&mut stdpython::PyValue"))
        self.assertFalse(rank_causes.is_plain_pyvalue("Option<stdpython::PyValue>"))
        self.assertFalse(rank_causes.is_plain_pyvalue("Vec<stdpython::PyValue>"))
        self.assertFalse(rank_causes.is_plain_pyvalue("PyRef<HTTPHeaderDict>"))

    def test_a_missing_package_log_refuses_the_whole_ranking(self):
        # A partial log set must not be ranked against the full denominator.
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "run.json"
            record.write_text(json.dumps({"rypip_commit": "abc1234", "packages": {
                "present": {"status": "build-failed", "histogram": {"E0599": 1}},
                "gone": {"status": "build-failed", "histogram": {"E0599": 100}}}}))
            workdir = self.write_workdir(tmp, [self.message("E0599", self.METHOD_ON_PYVALUE)],
                                         package="present")
            argv = ["rank_causes.py", str(record), "--workdir", str(workdir)]
            with patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(rank_causes.main(), 1)
            text = out.getvalue()
            self.assertIn("NOT COMPARABLE", text)
            self.assertIn("gone", text)

    def test_shapes_separate_unmatched_attribute_errors_from_other_codes(self):
        # An E0609 whose message matches no shape is still E0609, so it must
        # not be reported as "another code".
        code, text = self.run_rank(
            {"E0599": 1, "E0609": 1, "E0308": 2},
            [self.message("E0599", self.METHOD_ON_PYVALUE),
             self.message("E0609", "some future rustc wording for a missing field"),
             self.message("E0308", "mismatched types"),
             self.message("E0308", "mismatched types")])
        self.assertEqual(code, 0)
        self.assertIn("E0599/E0609 not matching a shape above", text)
        self.assertIn("(other codes", text)
        # E0308's 2 errors are the "other codes" row; the unmatched E0609 is not.
        self.assertIn("     2  (other codes", text)

    def test_a_zero_error_corpus_is_an_explicit_refusal_not_an_empty_ranking(self):
        # A clean build is not a frontier: ranking must say so, not divide by zero.
        code, text = self.run_rank({}, [])
        self.assertEqual(code, 1)
        self.assertIn("Nothing to rank", text)


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
