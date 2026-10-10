"""Report upstream drift for the openai/math entries BPR has indexed. Text only.

Usage (always isolated mode, always a clone kept outside this repository):

    python3 -I scripts/check_openai_math_pin.py /path/to/openai-math-clone [--index PATH] [--json]

The clone is untrusted data. This script reads its files as bytes or text and
never imports, builds or executes anything from it (no Lake, no Lean, no
``import`` of clone paths). It refuses to run on a clone inside the BPR tree and
refuses to run without ``python -I``. Symlinks and oversized files in the clone
are skipped.

What it compares, for indexed entries only (plus three global items):

* ``history.md``: dated sections not recorded in the index, and indexed titles
  mentioned in them;
* each indexed manuscript's ``README.md``: withdrawal notices and version notes,
  plus version notes elsewhere that point back to an indexed manuscript (a new
  edition);
* the family blocks of ``CONTENTS.md``, which silently drop withdrawn papers;
* ``lean/formalization.yaml``: ``main_results`` membership and ``sources``;
* ``lean/docs/<family>.md``: added, removed or changed scope pages;
* ``lean/ComparatorChallenges/*.json``: declarations, ``permitted_axioms``,
  ``enable_nanoda``;
* sha256 of every challenge ``.lean`` and source ``.tex`` file the index names;
* globally: a ``NOTICE`` file appearing, the ``LICENSE`` hash, new history sections.

It detects text changes, not mathematical errors. It is not a validity check of
any upstream result, and a clean run upgrades nothing.

Exit status: 0 no changes for indexed entries, 1 changes found, 2 refused or usage error.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "doc" / "external" / "openai_math_index.json"
MAX_BYTES = 64 * 1024 * 1024
DISCLAIMER = "Text-only drift report; detects text changes, not mathematical errors. Not a validity check."


class Refused(Exception):
    """The run was refused (clone location, isolation, or not a clone)."""


# --------------------------------------------------------------------------- safe reading


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_location(clone, bpr_root=ROOT) -> Path:
    """Resolve the clone path and refuse anything inside (or containing) the BPR tree."""
    clone = Path(clone).resolve()
    root = Path(bpr_root).resolve()
    if clone == root or _inside(clone, root):
        raise Refused(f"refusing: {clone} is inside the BPR tree; keep the clone outside the repository")
    if _inside(root, clone):
        raise Refused(f"refusing: the BPR tree lies inside {clone}")
    if not (_regular(clone / "CONTENTS.md") and _regular(clone / "lean" / "formalization.yaml")):
        raise Refused(f"refusing: {clone} does not look like an openai/math clone")
    return clone


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def read_bytes(path: Path):
    """Bytes of a regular, non-symlinked file below the size cap; otherwise None."""
    if not _regular(path) or path.stat().st_size > MAX_BYTES:
        return None
    with open(path, "rb") as handle:
        return handle.read(MAX_BYTES + 1)[:MAX_BYTES]


def read_text(path: Path):
    data = read_bytes(path)
    return None if data is None else data.decode("utf-8", errors="replace")


def sha256(path: Path):
    data = read_bytes(path)
    return None if data is None else hashlib.sha256(data).hexdigest()


def clone_commit(clone: Path):
    """Commit the clone has checked out, read from .git as text (None if unknown)."""
    git = clone / ".git"
    if _regular(git):  # worktree or submodule: "gitdir: <path>"
        text = read_text(git) or ""
        m = re.match(r"gitdir:\s*(\S+)", text)
        if not m:
            return None
        git = (clone / m.group(1)).resolve()
    head = read_text(git / "HEAD")
    if head is None:
        return None
    head = head.strip()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    m = re.fullmatch(r"ref:\s*(\S+)", head)
    if not m:
        return None
    ref = read_text(git / m.group(1))
    if ref and re.fullmatch(r"[0-9a-f]{40}", ref.strip()):
        return ref.strip()
    packed = read_text(git / "packed-refs") or ""
    for line in packed.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == m.group(1) and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
            return parts[0]
    return None


# --------------------------------------------------------------------------- parsers


def history_sections(text: str) -> dict:
    """Map each '## <date>' heading of history.md to its body text."""
    heads = list(re.finditer(r"^## +(.+?)\s*$", text, re.M))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[m.group(1)] = text[m.end():end]
    return out


def contents_families(text: str) -> dict:
    """Map family id -> manuscript directories listed in its CONTENTS.md block."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"\*\*(\d{3})\. ", text)]
    out = {}
    for i, (pos, fid) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        out[fid] = re.findall(r"\]\(preprints/([^/)\s]+)/", text[pos:end])
    return out


def formalization_catalogue(text: str):
    """(main_results comparator configs relative to the clone root, source manuscript dirs)."""
    sources = set(re.findall(r"id:\s*\.\./preprints/([^/\s]+)/", text))
    configs = set()
    m = re.search(r"^\s*main_results:\s*$", text, re.M)
    if m:
        block = text[m.end():]
        stop = re.search(r"^\S", block, re.M)  # next top-level key
        block = block[:stop.start()] if stop else block
        configs = {"lean/" + c for c in re.findall(r"comparator_config:\s*(\S+)", block)}
    return configs, sources


def lean_doc_links(text: str):
    """(manuscript dirs named by a lean/docs page, comparator .lean files it links)."""
    papers = re.findall(r"preprints/([^/)\s]+)/", text)
    lean = ["lean/ComparatorChallenges/" + f for f in re.findall(r"\(\.\./ComparatorChallenges/([^)\s]+\.lean)\)", text)]
    return papers, lean


def readme_flags(text: str) -> dict:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    return {
        "withdrawn": "withdrawal notice" in first.lower() or bool(re.search(r"\*\*Withdrawn on", text)),
        "version_note": bool(re.search(r"^#+\s*Version note", text, re.M | re.I)),
    }


def version_note_successors(clone: Path) -> dict:
    """Map an older manuscript dir -> newer dirs whose README version note links to it."""
    out = {}
    pre = clone / "preprints"
    if not pre.is_dir():
        return out
    for readme in sorted(pre.glob("*/README.md")):
        text = read_text(readme)
        if not text or not readme_flags(text)["version_note"]:
            continue
        note = re.split(r"^#+\s*Version note.*$", text, maxsplit=1, flags=re.M | re.I)[-1]
        for old in set(re.findall(r"\]\(\.\./([^/)\s]+)/", note)):
            if old != readme.parent.name:
                out.setdefault(old, []).append(readme.parent.name)
    return out


# --------------------------------------------------------------------------- comparison


def _add(findings, key, kind, detail, severity="change"):
    findings.append({"key": key, "kind": kind, "detail": detail, "severity": severity})


def _check_comparator(findings, clone, key, rec, label):
    cfg_path = clone / rec["comparator_config"]
    cfg_text = read_text(cfg_path)
    if cfg_text is None:
        _add(findings, key, "comparator_missing", f"{label}: {rec['comparator_config']} is missing")
    else:
        try:
            cfg = json.loads(cfg_text)
        except ValueError:
            cfg = {}
            _add(findings, key, "comparator_unparsable", f"{label}: {rec['comparator_config']} is not JSON")
        names = cfg.get("theorem_names") or []
        missing = [d for d in rec.get("declarations", []) if d not in names]
        if missing:
            _add(findings, key, "declarations_missing", f"{label}: no longer listed: {', '.join(missing)}")
        if "permitted_axioms" in rec and cfg.get("permitted_axioms") != rec["permitted_axioms"]:
            _add(findings, key, "permitted_axioms_changed",
                 f"{label}: {rec['permitted_axioms']} -> {cfg.get('permitted_axioms')}")
        if "enable_nanoda" in rec and cfg.get("enable_nanoda") != rec["enable_nanoda"]:
            _add(findings, key, "enable_nanoda_changed", f"{label}: {rec['enable_nanoda']} -> {cfg.get('enable_nanoda')}")
    now = sha256(clone / rec["challenge_file"])
    if now is None:
        _add(findings, key, "comparator_missing", f"{label}: {rec['challenge_file']} is missing")
    elif now != rec.get("statement_sha256"):
        _add(findings, key, "statement_changed",
             f"{label}: {rec['challenge_file']} sha256 {str(rec.get('statement_sha256'))[:12]} -> {now[:12]}")


def check(clone, index: dict) -> dict:
    """Compare a clone (already location-checked) with an index; return a report dict."""
    clone = Path(clone)
    up = index.get("upstream", {})
    entries = index.get("entries", [])
    findings = []

    commit = clone_commit(clone)
    if commit != up.get("commit"):
        _add(findings, "<global>", "clone_commit",
             f"pinned {str(up.get('commit'))[:12]}, clone at {commit[:12] if commit else 'unknown'}", "info")

    if up.get("notice_file") is None:
        notices = [p.name for p in sorted(clone.glob("NOTICE*")) if _regular(p)]
        if notices:
            _add(findings, "<global>", "notice_added", f"upstream now has {', '.join(notices)} (Apache-2.0 section 4(d))")
    lic = sha256(clone / "LICENSE")
    if up.get("license_sha256") and lic != up.get("license_sha256"):
        _add(findings, "<global>", "license_changed",
             f"LICENSE sha256 {str(up.get('license_sha256'))[:12]} -> {lic[:12] if lic else 'missing'}")

    history = history_sections(read_text(clone / "history.md") or "")
    new_sections = [h for h in history if h not in set(up.get("history_sections", []))]
    for h in new_sections:
        _add(findings, "<global>", "history_new_section", f"history.md has a new section: {h}")

    contents = contents_families(read_text(clone / "CONTENTS.md") or "")
    main_configs, yaml_sources = formalization_catalogue(read_text(clone / "lean" / "formalization.yaml") or "")
    successors = version_note_successors(clone)

    for e in entries:
        key = e.get("key", "?")
        mdir = str(e.get("manuscript_dir", ""))
        name = mdir.split("/", 1)[-1]
        title = str(e.get("title", ""))

        for h in new_sections:
            body = history[h].casefold()
            if title and (title.casefold() in body or name.casefold() in body):
                _add(findings, key, "history_mention", f"named in history.md section {h}")

        readme = read_text(clone / mdir / "README.md")
        if readme is None:
            _add(findings, key, "manuscript_missing", f"{mdir}/README.md is missing")
        else:
            flags = readme_flags(readme)
            if flags["withdrawn"] and e.get("upstream_status") != "withdrawn":
                _add(findings, key, "withdrawn", f"{mdir}/README.md carries a withdrawal notice")
            if flags["version_note"] and e.get("upstream_status") == "active":
                _add(findings, key, "version_note", f"{mdir}/README.md carries a version note")
        for newer in successors.get(name, []):
            _add(findings, key, "new_edition", f"preprints/{newer} is a newer edition (version note links back)")

        fid = e.get("family_id")
        if fid not in contents:
            _add(findings, key, "family_missing", f"family {fid} no longer appears in CONTENTS.md")
        elif name not in contents[fid]:
            _add(findings, key, "contents_unlisted", f"{name} is no longer listed under family {fid} in CONTENTS.md")

        src = sha256(clone / str(e.get("source_file", "")))
        if src is None:
            _add(findings, key, "source_missing", f"{e.get('source_file')} is missing")
        elif src != e.get("source_sha256"):
            _add(findings, key, "source_changed",
                 f"{e.get('source_file')} sha256 {str(e.get('source_sha256'))[:12]} -> {src[:12]}")

        doc_rel = f"lean/docs/{fid}.md"
        doc_text = read_text(clone / doc_rel)
        recorded_doc = e.get("lean_doc")
        if recorded_doc is None and doc_text is not None:
            _add(findings, key, "lean_doc_added", f"{doc_rel} now exists")
        elif recorded_doc is not None and doc_text is None:
            _add(findings, key, "lean_doc_removed", f"{recorded_doc} no longer exists")
        elif recorded_doc is not None and sha256(clone / recorded_doc) != e.get("lean_doc_sha256"):
            _add(findings, key, "lean_doc_changed", f"{recorded_doc} changed; re-read the scope against lean_scope_excerpt")
        doc_papers, doc_lean = lean_doc_links(doc_text or "")

        lean = e.get("lean")
        if lean:
            records = [("primary", lean)] + [(f"additional {i + 1}", r) for i, r in
                                             enumerate(lean.get("additional_comparators") or [])]
            for label, rec in records:
                _check_comparator(findings, clone, key, rec, label)
                if doc_text is not None and rec["challenge_file"] not in doc_lean:
                    _add(findings, key, "lean_removed", f"{label}: {rec['challenge_file']} is no longer linked from {doc_rel}")
                in_main = rec["comparator_config"] in main_configs
                if "in_main_results" in rec and in_main != rec["in_main_results"]:
                    _add(findings, key, "main_results_changed",
                         f"{label}: in formalization.yaml main_results {rec['in_main_results']} -> {in_main}")
        else:
            if name in doc_papers:
                _add(findings, key, "lean_added", f"{doc_rel} now scopes this manuscript")
            if name in yaml_sources:
                _add(findings, key, "lean_added", "lean/formalization.yaml now lists this manuscript as a source")

    changes = [f for f in findings if f["severity"] == "change"]
    return {
        "disclaimer": DISCLAIMER,
        "pinned_commit": up.get("commit"),
        "clone_commit": commit,
        "entries_checked": len(entries),
        "changes": len(changes),
        "findings": findings,
    }


def format_report(report: dict, index_path) -> str:
    lines = [
        "openai/math pin check",
        DISCLAIMER,
        f"pinned commit: {report['pinned_commit']}; clone commit: {report['clone_commit'] or 'unknown'}",
        f"index: {index_path} ({report['entries_checked']} entries)",
    ]
    for f in report["findings"]:
        tag = "info" if f["severity"] == "info" else f["kind"]
        lines.append(f"[{f['key']}] {tag}: {f['detail']}")
    if report["changes"]:
        lines.append(f"{report['changes']} change(s) for indexed entries: review each entry before moving the pin.")
    else:
        lines.append("No changes for indexed entries.")
    return "\n".join(lines)


def _load_index(path: Path) -> dict:
    def no_duplicates(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key: " + k)
            out[k] = v
        return out

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clone", help="path to a separately fetched openai/math clone, outside this repository")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="index JSON (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        print("refusing: run with python3 -I (isolated mode) so nothing from the clone can be imported",
              file=sys.stderr)
        return 2
    try:
        clone = check_location(args.clone)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    index_path = Path(args.index)
    report = check(clone, _load_index(index_path))
    print(json.dumps(report, indent=1) if args.json else format_report(report, index_path))
    return 1 if report["changes"] else 0


if __name__ == "__main__":
    sys.exit(main())
