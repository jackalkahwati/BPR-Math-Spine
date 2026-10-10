"""Tests for scripts/check_openai_math_pin.py on a synthetic fixture.

Oracles: a known-clean synthetic tree (tests/fixtures/openai_math_pin/upstream, invented
content, hashes recorded in index.json) must report no change; each single edit of a kind
upstream actually makes (withdrawal, new edition, changed comparator, changed axioms, new
NOTICE, ...) must be reported under its own kind for the right key and for nothing else; the
script must refuse a clone inside the BPR tree and refuse to run without ``python -I``; and a
planted module in the clone must never be imported.

The checker is run on a copy in a temporary directory, because the fixture itself lives inside
the BPR tree. No real upstream file is used.
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_openai_math_pin.py"
FIXTURE = ROOT / "tests" / "fixtures" / "openai_math_pin"
FIXTURE_INDEX = FIXTURE / "index.json"
ONE = "synth:901:seven-is-odd"
TWO = "synth:902:nine-is-odd"
P1 = "preprints/Synthetic-Paper-One-September-1-2026"
P2 = "preprints/Synthetic-Paper-Two-September-2-2026"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_openai_math_pin", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.fixture
def clone(tmp_path):
    target = tmp_path / "openai-math-clone"
    shutil.copytree(FIXTURE / "upstream", target)
    return target


@pytest.fixture(scope="module")
def index():
    return json.loads(FIXTURE_INDEX.read_text(encoding="utf-8"))


def _run(args, cwd=None, isolated=True):
    cmd = [sys.executable] + (["-I"] if isolated else []) + [str(SCRIPT)] + [str(a) for a in args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)


def _kinds(report, key=None):
    return {f["kind"] for f in report["findings"] if f["severity"] == "change" and (key is None or f["key"] == key)}


def _edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, (path, old)
    path.write_text(text.replace(old, new), encoding="utf-8")


# --------------------------------------------------------------------------- end to end


def test_clean_copy_reports_no_change_and_disclaims_validity(clone):
    result = _run([clone, "--index", FIXTURE_INDEX])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No changes for indexed entries." in result.stdout
    assert "Not a validity check" in result.stdout


def test_json_output_is_parseable_and_reports_changes(clone):
    (clone / "NOTICE").write_text("Synthetic notice.\n", encoding="utf-8")
    result = _run([clone, "--index", FIXTURE_INDEX, "--json"])
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["changes"] == 1 and _kinds(report) == {"notice_added"}
    assert "not a validity check" in report["disclaimer"].lower()


def test_refuses_a_clone_inside_the_bpr_tree():
    result = _run([FIXTURE / "upstream", "--index", FIXTURE_INDEX])
    assert result.returncode == 2
    assert "inside the BPR tree" in result.stderr
    with pytest.raises(checker.Refused):
        checker.check_location(ROOT)


def test_refuses_a_directory_containing_the_bpr_tree(tmp_path):
    with pytest.raises(checker.Refused):
        checker.check_location(tmp_path, bpr_root=tmp_path / "BPR-Math-Spine")


def test_refuses_without_isolated_mode(clone):
    result = _run([clone, "--index", FIXTURE_INDEX], isolated=False)
    assert result.returncode == 2
    assert "python3 -I" in result.stderr


def test_refuses_something_that_is_not_a_clone(tmp_path):
    result = _run([tmp_path, "--index", FIXTURE_INDEX])
    assert result.returncode == 2 and "does not look like" in result.stderr


def test_planted_modules_in_the_clone_are_never_imported(clone, tmp_path):
    marker = tmp_path / "IMPORTED"
    payload = f"open({str(marker)!r}, 'w').write('x')\n"
    for name in ("json.py", "hashlib.py", "re.py", "argparse.py", "sitecustomize.py", "usercustomize.py"):
        (clone / name).write_text(payload, encoding="utf-8")
        (clone / "lean" / name).write_text(payload, encoding="utf-8")
    (clone / "lean" / "lakefile.lean").write_text("run_cmd do IO.println \"never run\"\n", encoding="utf-8")
    result = _run([clone, "--index", FIXTURE_INDEX], cwd=clone)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_symlinked_files_in_the_clone_are_not_followed(clone, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("outside the clone\n", encoding="utf-8")
    src = clone / P2 / "src" / "intro.tex"
    src.unlink()
    src.symlink_to(secret)
    report = checker.check(checker.check_location(clone), json.loads(FIXTURE_INDEX.read_text()))
    assert _kinds(report, TWO) == {"source_missing"}


def test_script_contains_no_execution_primitives():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile", "__import__"}, node.func.id
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"system", "popen", "spawn", "exec_module", "import_module", "write_text",
                                     "write_bytes"}, node.attr
    assert imported <= {"argparse", "hashlib", "json", "re", "sys", "pathlib"}, imported


# --------------------------------------------------------------------------- one edit, one kind


def _mutate_withdraw(c):
    (c / P1 / "README.md").write_text(
        "# [Withdrawal notice: Synthetic paper one](paper.pdf)\n\n**Withdrawn on October 1, 2026.**\n", encoding="utf-8")


def _mutate_contents(c):
    _edit(c / "CONTENTS.md", "&emsp;[Synthetic paper two](preprints/Synthetic-Paper-Two-September-2-2026/paper.pdf)\n", "")


def _mutate_family_gone(c):
    _edit(c / "CONTENTS.md", "**902. Synthetic family two.**", "Family two removed.")


def _mutate_new_edition(c):
    newer = c / "preprints" / "Synthetic-Paper-Two-October-3-2026"
    newer.mkdir()
    (newer / "README.md").write_text(
        "# [Synthetic paper two](paper.pdf)\n\n## Version note\n\nThis version repairs a proof; see the "
        "[previous version](../Synthetic-Paper-Two-September-2-2026/paper.pdf).\n", encoding="utf-8")


def _mutate_version_note_on_pinned(c):
    with open(c / P2 / "README.md", "a", encoding="utf-8") as handle:
        handle.write("\n## Version note\n\nRevised edition.\n")


def _mutate_statement(c):
    with open(c / "lean/ComparatorChallenges/SyntheticOne.lean", "a", encoding="utf-8") as handle:
        handle.write("-- changed\n")


def _mutate_axioms(c):
    _edit(c / "lean/ComparatorChallenges/SyntheticOne.json", '"propext"', '"propext",\n    "sorryAx"')


def _mutate_declarations(c):
    _edit(c / "lean/ComparatorChallenges/SyntheticOne.json", '"Synthetic.one"', '"Synthetic.renamed"')


def _mutate_nanoda(c):
    _edit(c / "lean/ComparatorChallenges/SyntheticOne.json", '"enable_nanoda": false', '"enable_nanoda": true')


def _mutate_comparator_gone(c):
    (c / "lean/ComparatorChallenges/SyntheticOne.lean").unlink()


def _mutate_main_results(c):
    _edit(c / "lean/formalization.yaml",
          "    - comparator_config: ComparatorChallenges/SyntheticOne.json\n"
          "      declaration: Synthetic.one\n      file: Synthetic/One.lean\n", "")


def _mutate_doc_scope(c):
    _edit(c / "lean/docs/901.md", "the number seven is odd.", "the number seven is odd and prime.")


def _mutate_doc_unlinks(c):
    _edit(c / "lean/docs/901.md", "[SyntheticOne.lean](../ComparatorChallenges/SyntheticOne.lean)", "removed")


def _mutate_doc_gone(c):
    (c / "lean/docs/901.md").unlink()


def _mutate_lean_added(c):
    (c / "lean/docs/902.md").write_text(
        "# Synthetic family two\n\n- [Synthetic paper two](../../preprints/Synthetic-Paper-Two-September-2-2026/paper.pdf)\n"
        "\n## Scope\n\nInvented.\n", encoding="utf-8")


def _mutate_yaml_source(c):
    _edit(c / "lean/formalization.yaml", "\nstatus:",
          "  - title: \"Synthetic paper two\"\n    id: ../preprints/Synthetic-Paper-Two-September-2-2026/paper.pdf\n"
          "    type: article\n\nstatus:")


def _mutate_source(c):
    _edit(c / P2 / "src/intro.tex", "Nine is odd.", "Nine is odd and composite.")


def _mutate_source_gone(c):
    (c / P2 / "src/intro.tex").unlink()


def _mutate_readme_gone(c):
    shutil.rmtree(c / P2)


def _mutate_history(c):
    with open(c / "history.md", "a", encoding="utf-8") as handle:
        handle.write("\n## October 9, 2026\n\nWe revised Synthetic paper two.\n")


def _mutate_history_unrelated(c):
    with open(c / "history.md", "a", encoding="utf-8") as handle:
        handle.write("\n## October 9, 2026\n\nWe revised an unindexed paper.\n")


def _mutate_notice(c):
    (c / "NOTICE").write_text("Synthetic notice.\n", encoding="utf-8")


def _mutate_license(c):
    _edit(c / "LICENSE", "Synthetic licence", "Changed synthetic licence")


MUTATIONS = [
    (_mutate_withdraw, ONE, {"withdrawn"}),
    (_mutate_contents, TWO, {"contents_unlisted"}),
    (_mutate_family_gone, TWO, {"family_missing"}),
    (_mutate_new_edition, TWO, {"new_edition"}),
    (_mutate_version_note_on_pinned, TWO, {"version_note"}),
    (_mutate_statement, ONE, {"statement_changed"}),
    (_mutate_axioms, ONE, {"permitted_axioms_changed"}),
    (_mutate_declarations, ONE, {"declarations_missing"}),
    (_mutate_nanoda, ONE, {"enable_nanoda_changed"}),
    (_mutate_comparator_gone, ONE, {"comparator_missing"}),
    (_mutate_main_results, ONE, {"main_results_changed"}),
    (_mutate_doc_scope, ONE, {"lean_doc_changed"}),
    (_mutate_doc_unlinks, ONE, {"lean_doc_changed", "lean_removed"}),
    (_mutate_doc_gone, ONE, {"lean_doc_removed"}),
    (_mutate_lean_added, TWO, {"lean_doc_added", "lean_added"}),
    (_mutate_yaml_source, TWO, {"lean_added"}),
    (_mutate_source, TWO, {"source_changed"}),
    (_mutate_source_gone, TWO, {"source_missing"}),
    (_mutate_readme_gone, TWO, {"manuscript_missing", "source_missing", "contents_unlisted"}),
    (_mutate_history, TWO, {"history_mention"}),
]


@pytest.mark.parametrize("mutate, key, expected", MUTATIONS, ids=[m[0].__name__[8:] for m in MUTATIONS])
def test_each_upstream_change_is_reported_under_its_kind(clone, index, mutate, key, expected):
    if mutate is _mutate_readme_gone:
        _edit(clone / "CONTENTS.md", "preprints/Synthetic-Paper-Two-September-2-2026/paper.pdf", "preprints/gone/paper.pdf")
    mutate(clone)
    report = checker.check(checker.check_location(clone), index)
    assert _kinds(report, key) == expected
    other = ONE if key == TWO else TWO
    assert _kinds(report, other) == set()


@pytest.mark.parametrize("mutate, expected", [
    (_mutate_notice, {"notice_added"}),
    (_mutate_license, {"license_changed"}),
    (_mutate_history_unrelated, {"history_new_section"}),
])
def test_global_changes_are_reported_once(clone, index, mutate, expected):
    mutate(clone)
    report = checker.check(checker.check_location(clone), index)
    assert _kinds(report, "<global>") == expected
    assert _kinds(report, ONE) == _kinds(report, TWO) == set()


def test_changes_to_unindexed_manuscripts_are_ignored(clone, index):
    extra = clone / "preprints" / "Synthetic-Paper-Three-September-3-2026"
    extra.mkdir()
    (extra / "README.md").write_text("# [Withdrawal notice: Synthetic paper three](paper.pdf)\n", encoding="utf-8")
    with open(clone / "CONTENTS.md", "a", encoding="utf-8") as handle:
        handle.write("\n**903. Synthetic family three.** Invented.\n")
    (clone / "lean/docs/903.md").write_text("# Synthetic family three\n", encoding="utf-8")
    report = checker.check(checker.check_location(clone), index)
    assert report["changes"] == 0


def test_clone_commit_is_read_as_text(clone, index):
    git = clone / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text(index["upstream"]["commit"] + "\n", encoding="utf-8")
    report = checker.check(checker.check_location(clone), index)
    assert report["clone_commit"] == index["upstream"]["commit"]
    assert report["findings"] == []


# --------------------------------------------------------------------------- the real index


def test_real_index_carries_every_field_the_checker_reads():
    data = json.loads((ROOT / "doc/external/openai_math_index.json").read_text(encoding="utf-8"))
    up = data["upstream"]
    assert {"commit", "notice_file", "license_sha256", "history_sections"} <= set(up)
    for e in data["entries"]:
        assert {"key", "family_id", "manuscript_dir", "title", "source_file", "source_sha256",
                "upstream_status", "lean", "lean_doc", "lean_doc_sha256"} <= set(e), e["key"]
        for rec in ([e["lean"]] + e["lean"]["additional_comparators"]) if e["lean"] else []:
            assert {"comparator_config", "challenge_file", "declarations", "permitted_axioms",
                    "enable_nanoda", "statement_sha256", "in_main_results"} <= set(rec), e["key"]


@pytest.mark.skipif(not os.environ.get("OPENAI_MATH_CLONE"), reason="opt-in: set OPENAI_MATH_CLONE to a clone outside the repo")
def test_real_clone_at_the_pin_reports_no_change():
    """Opt-in, not default CI: reads a separately fetched clone as text and expects no drift at the pin."""
    result = _run([os.environ["OPENAI_MATH_CLONE"]])
    assert result.returncode == 0, result.stdout + result.stderr
