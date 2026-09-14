"""Check documentary structure, never mathematical or physical validity.

No scientific modules are imported. Source locators and review labels are
provenance supplied by the author, not assertions authenticated by this tool.
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = ROOT / "doc/derivations/foundation_compatibility_2026-09-13.json"
SUCCESS = "structurally valid record; mathematical and physical validity not checked."
FIELDS = {
    "root": "schema_version validation_scope physics_validated empirical_validation sources propositions arguments routes".split(),
    "sources": "id path locator".split(),
    "propositions": "id kind statement domain quantifiers sources status".split(),
    "arguments": "id kind premises conclusion proof review_status minimality".split(),
    "routes": "id changed_premises affected_arguments obligations assessment explanation".split(),
    "proof": "source locator".split(),
}
KINDS = (
    "microscopic_assumption", "empirical_input", "physical_target",
    "bridge_assumption", "conditional_result", "diagnostic", "obligation",
)
STATUSES = ("assumed", "supplied_input", "reviewed_result", "open")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number: " + value)
    return number


def load_record(path):
    """Load strict JSON; loading does not validate its documentary schema."""
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream, object_pairs_hook=_object,
                           parse_constant=_constant, parse_float=_float)
    if type(record) is not dict:
        raise ValueError("record must be a JSON object")
    return record


def _fields(value, schema, label):
    if type(value) is not dict:
        raise ValueError(label + " must be a built-in dict")
    if set(value) != set(FIELDS[schema]):
        raise ValueError(label + " must have exactly fields: " + ", ".join(FIELDS[schema]))


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")


def _enum(value, choices, label):
    _string(value, label)
    if value not in choices:
        raise ValueError(label + " has an unsupported value")


def _list(value, label, nonempty=True):
    if type(value) is not list or (nonempty and not value):
        raise ValueError(label + " must be a " + ("nonempty " if nonempty else "") + "built-in list")


def _reference(value, table, label):
    _string(value, label)
    if value not in table:
        raise ValueError(label + " has an unresolved reference: " + value)


def _references(values, table, label):
    _list(values, label)
    seen = set()
    for value in values:
        _reference(value, table, label)
        if value in seen:
            raise ValueError(label + " has a duplicate reference: " + value)
        seen.add(value)


def _source_path(value, root, label):
    _string(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(label + " must be a contained POSIX relative file path")
    try:
        resolved = (root / value).resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise ValueError("not a regular file")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(label + " must resolve to an existing file inside root") from error


def validate_record(record, root):
    """Validate exact fields, references and acyclicity without changing input."""
    _fields(record, "root", "record")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    _enum(record["validation_scope"], ("record_structure_only",), "validation_scope")
    for name in ("physics_validated", "empirical_validation"):
        if record[name] is not False:
            raise ValueError(name + " must be boolean false")
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("not a directory")
    except (TypeError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("root must resolve to an existing directory") from error

    tables = {}
    all_ids = set()
    for section in ("sources", "propositions", "arguments", "routes"):
        values = record[section]
        _list(values, section, nonempty=section != "routes")
        table = {}
        for index, value in enumerate(values):
            label = section + "[" + str(index) + "]"
            _fields(value, section, label)
            identifier = value["id"]
            _string(identifier, label + ".id")
            if identifier in all_ids:
                raise ValueError("globally duplicate id: " + identifier)
            all_ids.add(identifier)
            table[identifier] = value
        tables[section] = table

    sources = tables["sources"]
    propositions = tables["propositions"]
    arguments = tables["arguments"]
    for identifier, source in sources.items():
        _source_path(source["path"], root, identifier + ".path")
        _string(source["locator"], identifier + ".locator")
    for identifier, proposition in propositions.items():
        _enum(proposition["kind"], KINDS, identifier + ".kind")
        _enum(proposition["status"], STATUSES, identifier + ".status")
        for field in ("statement", "domain", "quantifiers"):
            _string(proposition[field], identifier + "." + field)
        _references(proposition["sources"], sources, identifier + ".sources")

    successors = {identifier: set() for identifier in propositions}
    indegrees = dict.fromkeys(propositions, 0)
    for identifier, argument in arguments.items():
        _enum(argument["kind"], ("deduction", "claimed_incompatibility"), identifier + ".kind")
        _enum(argument["review_status"], ("unreviewed", "independently_reviewed"), identifier + ".review_status")
        _enum(argument["minimality"], ("not_claimed",), identifier + ".minimality")
        _references(argument["premises"], propositions, identifier + ".premises")
        conclusion = argument["conclusion"]
        _reference(conclusion, propositions, identifier + ".conclusion")
        proof = argument["proof"]
        _fields(proof, "proof", identifier + ".proof")
        _reference(proof["source"], sources, identifier + ".proof.source")
        _string(proof["locator"], identifier + ".proof.locator")
        for premise in argument["premises"]:
            if conclusion not in successors[premise]:
                successors[premise].add(conclusion)
                indegrees[conclusion] += 1

    # Iterative traversal avoids recursion limits on long documentary chains.
    ready = deque(identifier for identifier in propositions if indegrees[identifier] == 0)
    visited = 0
    while ready:
        identifier = ready.popleft()
        visited += 1
        for child in sorted(successors[identifier]):
            indegrees[child] -= 1
            if indegrees[child] == 0:
                ready.append(child)
    if visited != len(propositions):
        raise ValueError("premise-to-conclusion graph contains a cycle")

    for identifier, route in tables["routes"].items():
        _references(route["changed_premises"], propositions, identifier + ".changed_premises")
        _references(route["affected_arguments"], arguments, identifier + ".affected_arguments")
        _references(route["obligations"], propositions, identifier + ".obligations")
        for obligation in route["obligations"]:
            if propositions[obligation]["kind"] != "obligation":
                raise ValueError(identifier + ".obligations must reference obligation propositions")
        _enum(route["assessment"], ("conditional_escape_obligations", "unassessed"), identifier + ".assessment")
        _string(route["explanation"], identifier + ".explanation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", default=str(DEFAULT_RECORD))
    args = parser.parse_args(argv)
    try:
        validate_record(load_record(args.record), ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        print("invalid record: " + str(error))
        return 1
    print(SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
