"""Independent structural oracles from section 7 of the frozen contract.

This suite neither checks the mathematics nor imports bpr/science. The fixture,
field sets, enums and expected outcomes were designed from the contract before
reading the checker. No checker constants or actual-record content define them.
"""

from collections import UserDict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/check_foundation_compatibility.py"
ACTUAL_RECORD = REPOSITORY / "doc/derivations/foundation_compatibility_2026-09-13.json"
SUCCESS = "structurally valid record; mathematical and physical validity not checked.\n"

# Literal contract oracles, deliberately not copied from checker constants.
FIELDS = {
    (): ("schema_version", "validation_scope", "physics_validated",
         "empirical_validation", "sources", "propositions", "arguments", "routes"),
    ("sources", 0): ("id", "path", "locator"),
    ("propositions", 0): ("id", "kind", "statement", "domain", "quantifiers",
                          "sources", "status"),
    ("arguments", 0): ("id", "kind", "premises", "conclusion", "proof",
                       "review_status", "minimality"),
    ("arguments", 0, "proof"): ("source", "locator"),
    ("routes", 0): ("id", "changed_premises", "affected_arguments", "obligations",
                    "assessment", "explanation"),
}
ENUMS = {
    ("validation_scope",): ("record_structure_only",),
    ("propositions", 0, "kind"): (
        "microscopic_assumption", "empirical_input", "physical_target",
        "bridge_assumption", "conditional_result", "diagnostic", "obligation"),
    ("propositions", 0, "status"): (
        "assumed", "supplied_input", "reviewed_result", "open"),
    ("arguments", 0, "kind"): ("deduction", "claimed_incompatibility"),
    ("arguments", 0, "review_status"): ("unreviewed", "independently_reviewed"),
    ("arguments", 0, "minimality"): ("not_claimed",),
    ("routes", 0, "assessment"): ("conditional_escape_obligations", "unassessed"),
}
REFERENCE_LISTS = {
    ("propositions", 0, "sources"): ("sources", "s_one"),
    ("arguments", 0, "premises"): ("propositions", "p_seed"),
    ("routes", 0, "changed_premises"): ("propositions", "p_seed"),
    ("routes", 0, "affected_arguments"): ("arguments", "a_first"),
    ("routes", 0, "obligations"): ("propositions", "p_obligation"),
}
REFERENCE_SCALARS = {
    ("arguments", 0, "conclusion"): "propositions",
    ("arguments", 0, "proof", "source"): "sources",
}
STRING_FIELDS = (
    ("validation_scope",),
    *(("sources", 0, field) for field in ("id", "path", "locator")),
    *(("propositions", 0, field) for field in
      ("id", "kind", "statement", "domain", "quantifiers", "status")),
    *(("arguments", 0, field) for field in
      ("id", "kind", "conclusion", "review_status", "minimality")),
    ("arguments", 0, "proof", "source"),
    ("arguments", 0, "proof", "locator"),
    *(("routes", 0, field) for field in ("id", "assessment", "explanation")),
)


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class IntSubclass(int):
    pass


class StringSubclass(str):
    pass


def at(record, path):
    for component in path:
        record = record[component]
    return record


def replace(record, path, value):
    if not path:
        return value
    at(record, path[:-1])[path[-1]] = value
    return record


def fixture_record():
    """Small documentary graph, unrelated to the shipped scientific record."""
    return {
        "schema_version": 1,
        "validation_scope": "record_structure_only",
        "physics_validated": False,
        "empirical_validation": False,
        "sources": [
            {"id": "s_one", "path": "evidence/source.txt", "locator": "unverified locator"},
            {"id": "s_two", "path": "evidence/source.txt", "locator": "another locator"},
        ],
        "propositions": [
            {"id": "p_seed", "kind": "microscopic_assumption", "statement": "Suppose A.",
             "domain": "An unspecified set.", "quantifiers": "For every member.",
             "sources": ["s_one", "s_two"], "status": "assumed"},
            {"id": "p_middle", "kind": "conditional_result", "statement": "Claim B.",
             "domain": "The same set.", "quantifiers": "For every member.",
             "sources": ["s_one"], "status": "open"},
            {"id": "p_end", "kind": "physical_target", "statement": "Require C.",
             "domain": "A proposed construction.", "quantifiers": "There exists one.",
             "sources": ["s_two"], "status": "open"},
            {"id": "p_obligation", "kind": "obligation", "statement": "Supply a construction.",
             "domain": "Not yet specified.", "quantifiers": "There exists one.",
             "sources": ["s_one"], "status": "open"},
        ],
        "arguments": [
            {"id": "a_first", "kind": "deduction", "premises": ["p_seed"],
             "conclusion": "p_middle", "proof": {"source": "s_one", "locator": "Claim A"},
             "review_status": "unreviewed", "minimality": "not_claimed"},
            {"id": "a_second", "kind": "claimed_incompatibility", "premises": ["p_middle"],
             "conclusion": "p_end", "proof": {"source": "s_two", "locator": "Claim B"},
             "review_status": "unreviewed", "minimality": "not_claimed"},
        ],
        "routes": [
            {"id": "r_one", "changed_premises": ["p_seed"],
             "affected_arguments": ["a_first", "a_second"],
             "obligations": ["p_obligation"], "assessment": "unassessed",
             "explanation": "A proposal, not a construction."},
        ],
    }


def rename_id(record, section, old, new):
    """Rename an ID and its typed references without introducing dangling edges."""
    for item in record[section]:
        if item["id"] == old:
            item["id"] = new
    for proposition in record["propositions"]:
        if section == "sources":
            proposition["sources"] = [new if value == old else value
                                      for value in proposition["sources"]]
    for argument in record["arguments"]:
        if section == "sources" and argument["proof"]["source"] == old:
            argument["proof"]["source"] = new
        if section == "propositions":
            argument["premises"] = [new if value == old else value
                                    for value in argument["premises"]]
            if argument["conclusion"] == old:
                argument["conclusion"] = new
    for route in record["routes"]:
        fields = {"propositions": ("changed_premises", "obligations"),
                  "arguments": ("affected_arguments",)}.get(section, ())
        for field in fields:
            route[field] = [new if value == old else value for value in route[field]]


CHECKER = None


def setUpModule():
    global CHECKER
    spec = importlib.util.spec_from_file_location("foundation_compatibility_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the checker script")
    CHECKER = importlib.util.module_from_spec(spec)
    # Authoring and test runs must not leave an import cache in the repository.
    with mock.patch.object(sys, "dont_write_bytecode", True):
        spec.loader.exec_module(CHECKER)


class FoundationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="foundation-compatibility-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "root"
        (self.root / "evidence").mkdir(parents=True)
        self.source = self.root / "evidence/source.txt"
        self.source.write_text("This file provides no mathematical proof.\n", encoding="utf-8")
        self.record = fixture_record()

    def assert_valid(self, record=None, root=None):
        record = self.record if record is None else record
        before = deepcopy(record)
        self.assertIsNone(CHECKER.validate_record(record, self.root if root is None else root))
        self.assertEqual(record, before, "validation must not mutate its input")

    def assert_invalid(self, record, root=None):
        before = deepcopy(record)
        messages = []
        for _ in range(2):
            # TypeError/KeyError and all other incidental errors fail this assertion.
            with self.assertRaises(ValueError) as caught:
                CHECKER.validate_record(record, self.root if root is None else root)
            messages.append(str(caught.exception))
            self.assertEqual(record, before, "failed validation must not mutate its input")
        self.assertEqual(messages[0], messages[1], "diagnostics must be deterministic")

    def mutated(self, path, value):
        return replace(deepcopy(self.record), path, value)

    def raw_json(self, text):
        path = self.temp / "input.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_callable_api_and_independent_valid_fixture(self):
        for name in ("load_record", "validate_record", "main"):
            self.assertTrue(callable(getattr(CHECKER, name, None)), name)
        self.assert_valid()
        path = self.raw_json(json.dumps(self.record))
        loaded = CHECKER.load_record(path)
        self.assertIs(type(loaded), dict)
        self.assertEqual(loaded, self.record)
        self.assert_valid(loaded, str(self.root))

    def test_actual_record_is_structurally_valid_without_frozen_review_labels(self):
        record = CHECKER.load_record(ACTUAL_RECORD)
        before = deepcopy(record)
        self.assertIsNone(CHECKER.validate_record(record, REPOSITORY))
        self.assertEqual(record, before)
        self.assertEqual(record["validation_scope"], "record_structure_only")
        self.assertIs(record["physics_validated"], False)
        self.assertIs(record["empirical_validation"], False)

    def test_loader_is_not_a_schema_validator(self):
        self.assertEqual(CHECKER.load_record(self.raw_json("{}")), {})
        loaded = CHECKER.load_record(self.raw_json(
            '{"finite": [1e308, -1e308, 1e-9999, 123456789012345678901234567890],'
            '"nested": {"same": 1}, "other": {"same": 2}}'))
        self.assertEqual(loaded["finite"], [1e308, -1e308, 0.0, 123456789012345678901234567890])
        self.assertEqual(loaded["nested"], {"same": 1})

    def test_loader_rejects_duplicate_keys_at_every_nesting_depth(self):
        cases = (
            '{"x": 1, "x": 2}',
            '{"sources": [{"id": "a", "id": "b"}]}',
            '{"arguments": [{"proof": {"source": "a", "source": "b"}}]}',
            '{"nested": [{"deeper": [{"k": 1, "k": 1}]}]}',
            '{"id": 1, "\\u0069d": 2}',
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                CHECKER.load_record(self.raw_json(text))

    def test_loader_rejects_nonfinite_constants_and_finite_literal_overflow(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999", "9.99e308"):
            for text in ('{"x": ' + token + '}', '{"x": [{"y": ' + token + '}]}'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    CHECKER.load_record(self.raw_json(text))

    def test_loader_rejects_nonobject_roots_and_malformed_json(self):
        cases = ("[]", "null", "true", "false", "1", "1.5", '"text"', "",
                 "{", '{"x": 1,}', '{"x": 1} trailing', '{"x": 1}{"y": 2}',
                 '{"x": /*comment*/ 1}', '{"x": "unescaped\nnewline"}')
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                CHECKER.load_record(self.raw_json(text))

    def test_loader_file_failures_remain_io_errors(self):
        for path in (self.temp / "absent.json", self.root):
            with self.subTest(path=path), self.assertRaises(OSError):
                CHECKER.load_record(path)

    def test_every_object_has_exact_fields(self):
        for path, fields in FIELDS.items():
            self.assertEqual(set(at(self.record, path)), set(fields))
            for field in fields:
                record = deepcopy(self.record)
                del at(record, path)[field]
                with self.subTest(path=path, missing=field):
                    self.assert_invalid(record)
            for extra in ("unknown", 17, None, ("tuple",)):
                record = deepcopy(self.record)
                at(record, path)[extra] = "extra"
                with self.subTest(path=path, extra=extra):
                    self.assert_invalid(record)

    def test_every_mapping_requires_an_exact_builtin_dict(self):
        for path in FIELDS:
            original = at(self.record, path)
            for invalid in (None, False, 0, "dict", [], (), DictSubclass(original), UserDict(original)):
                with self.subTest(path=path, invalid=repr(invalid)):
                    self.assert_invalid(self.mutated(path, invalid))

    def test_every_list_requires_an_exact_builtin_list(self):
        paths = [(section,) for section in ("sources", "propositions", "arguments", "routes")]
        paths.extend(REFERENCE_LISTS)
        for path in paths:
            original = at(self.record, path)
            for invalid in (None, False, 0, "list", {}, tuple(original), ListSubclass(original)):
                with self.subTest(path=path, invalid=repr(invalid)):
                    self.assert_invalid(self.mutated(path, invalid))
            if path != ("routes",):
                with self.subTest(path=path, invalid="empty list"):
                    self.assert_invalid(self.mutated(path, []))
        self.assert_valid(self.mutated(("routes",), []))

    def test_root_version_and_flags_have_exact_types_and_values(self):
        for invalid in (True, False, 0, 2, -1, 1.0, "1", None, [], {}, IntSubclass(1)):
            with self.subTest(field="schema_version", invalid=repr(invalid)):
                self.assert_invalid(self.mutated(("schema_version",), invalid))
        for field in ("physics_validated", "empirical_validation"):
            for invalid in (True, 0, 1, 0.0, "false", None, [], {}):
                with self.subTest(field=field, invalid=repr(invalid)):
                    self.assert_invalid(self.mutated((field,), invalid))

    def test_all_string_fields_reject_blank_and_wrong_types(self):
        for path in STRING_FIELDS:
            for invalid in ("", " \t\n", "\u2003", None, False, 1, 1.5, [], {}, b"text"):
                with self.subTest(path=path, invalid=repr(invalid)):
                    self.assert_invalid(self.mutated(path, invalid))

    def test_all_enums_accept_exact_members_without_semantic_pairing_rules(self):
        for path, choices in ENUMS.items():
            for choice in choices:
                with self.subTest(path=path, choice=choice):
                    self.assert_valid(self.mutated(path, choice))
            for invalid in ("unsupported", choices[0].upper(), " " + choices[0], choices[0] + " "):
                with self.subTest(path=path, invalid=invalid):
                    self.assert_invalid(self.mutated(path, invalid))
        for kind, status in itertools.product(ENUMS[("propositions", 0, "kind")],
                                               ENUMS[("propositions", 0, "status")]):
            record = deepcopy(self.record)
            record["propositions"][0].update(kind=kind, status=status)
            with self.subTest(kind=kind, status=status):
                self.assert_valid(record)

    def test_string_subclasses_are_strings_under_the_contract(self):
        # Exact built-in constraints are stated for dict/list/int/bool, not str.
        record = deepcopy(self.record)
        record["propositions"][0]["statement"] = StringSubclass("A string assertion.")
        self.assert_valid(record)

    def test_all_reference_fields_reject_dangling_and_wrong_section_ids(self):
        representatives = {"sources": "s_one", "propositions": "p_seed",
                           "arguments": "a_first", "routes": "r_one"}
        for path, (section, _) in REFERENCE_LISTS.items():
            for invalid in ("missing", "", " \t", None, False, 7, [], {}):
                with self.subTest(path=path, invalid=repr(invalid)):
                    self.assert_invalid(self.mutated(path, [invalid]))
            for other, identifier in representatives.items():
                if other != section:
                    with self.subTest(path=path, wrong_section=other):
                        self.assert_invalid(self.mutated(path, [identifier]))
        for path, section in REFERENCE_SCALARS.items():
            for other, identifier in {**representatives, "missing": "missing"}.items():
                if other != section:
                    with self.subTest(path=path, wrong_section=other):
                        self.assert_invalid(self.mutated(path, identifier))

    def test_each_reference_list_rejects_duplicates(self):
        for path, (_, identifier) in REFERENCE_LISTS.items():
            with self.subTest(path=path):
                self.assert_invalid(self.mutated(path, [identifier, identifier]))

    def test_ids_are_globally_unique_within_and_across_every_section(self):
        sections = ("sources", "propositions", "arguments", "routes")
        for section in sections:
            record = deepcopy(self.record)
            record[section].append(deepcopy(record[section][0]))
            with self.subTest(section=section):
                self.assert_invalid(record)
        for first, second in itertools.combinations(sections, 2):
            record = deepcopy(self.record)
            rename_id(record, second, record[second][0]["id"], record[first][0]["id"])
            with self.subTest(first=first, second=second):
                self.assert_invalid(record)

    def test_route_obligations_require_obligation_kind_only(self):
        for kind in ENUMS[("propositions", 0, "kind")]:
            record = deepcopy(self.record)
            record["propositions"][3]["kind"] = kind
            with self.subTest(kind=kind):
                if kind == "obligation":
                    self.assert_valid(record)
                else:
                    self.assert_invalid(record)
        for status in ENUMS[("propositions", 0, "status")]:
            record = deepcopy(self.record)
            record["propositions"][3]["status"] = status
            self.assert_valid(record)

    def test_strings_and_ids_are_not_trimmed_or_rewritten(self):
        record = deepcopy(self.record)
        for section in ("sources", "propositions", "arguments", "routes"):
            identifier = record[section][0]["id"]
            rename_id(record, section, identifier, " " + identifier + " ")
        for path in (("sources", 0, "locator"), ("propositions", 0, "statement"),
                     ("propositions", 0, "domain"), ("propositions", 0, "quantifiers"),
                     ("arguments", 0, "proof", "locator"), ("routes", 0, "explanation")):
            replace(record, path, " \tDocumentary text.\n")
        self.assert_valid(record)
        self.assert_invalid(self.mutated(("propositions", 0, "sources"), [" s_one "]))
        self.assert_invalid(self.mutated(("arguments", 0, "conclusion"), " p_middle "))
        padded_file = self.root / "evidence/ padded source.txt "
        padded_file.write_text("No proof.\n", encoding="utf-8")
        self.assert_valid(self.mutated(("sources", 0, "path"), "evidence/ padded source.txt "))

    def test_unsupported_and_false_mathematics_still_pass_structure(self):
        record = deepcopy(self.record)
        record["propositions"][1].update(
            statement="Every integer equals zero, therefore 1 = 0.",
            domain="The ordinary integers.", quantifiers="For every integer.",
            status="reviewed_result")
        record["arguments"][0]["review_status"] = "independently_reviewed"
        record["arguments"][0]["proof"]["locator"] = "A nonexistent theorem in a nonproof file."
        self.assert_valid(record)

    def test_graph_rejects_self_two_node_and_longer_cycles(self):
        for argument_index, conclusion in ((0, "p_seed"), (1, "p_seed")):
            record = deepcopy(self.record)
            record["arguments"][argument_index]["conclusion"] = conclusion
            with self.subTest(argument=argument_index):
                self.assert_invalid(record)
        record = deepcopy(self.record)
        closing = deepcopy(record["arguments"][0])
        closing.update(id="a_close", premises=["p_end"], conclusion="p_seed")
        record["arguments"].append(closing)
        self.assert_invalid(record)

    def test_graph_detects_disconnected_cycle(self):
        record = deepcopy(self.record)
        for identifier in ("p_x", "p_y"):
            proposition = deepcopy(record["propositions"][0])
            proposition["id"] = identifier
            record["propositions"].append(proposition)
        for identifier, premise, conclusion in (("a_x", "p_x", "p_y"), ("a_y", "p_y", "p_x")):
            argument = deepcopy(record["arguments"][0])
            argument.update(id=identifier, premises=[premise], conclusion=conclusion)
            record["arguments"].append(argument)
        self.assert_invalid(record)

    def test_graph_accepts_shared_edges_diamond_and_non_topological_order(self):
        record = deepcopy(self.record)
        duplicate_edge = deepcopy(record["arguments"][0])
        duplicate_edge["id"] = "a_same_edge"
        record["arguments"].append(duplicate_edge)
        branch = deepcopy(record["arguments"][0])
        branch.update(id="a_branch", conclusion="p_obligation")
        record["arguments"].append(branch)
        record["arguments"][1]["premises"] = ["p_middle", "p_obligation", "p_seed"]
        self.assert_valid(record)
        for section in ("sources", "propositions", "arguments", "routes"):
            record[section].reverse()
        record["arguments"][-2]["premises"].reverse()
        self.assert_valid(record)

    def test_source_paths_reject_escape_invalid_and_nonfile_targets(self):
        outside = self.temp / "outside.txt"
        outside.write_text("Outside root.\n", encoding="utf-8")
        paths = ("../outside.txt", "evidence/../evidence/source.txt", "evidence/../../outside.txt",
                 str(self.source), str(outside), "/does-not-exist", "//server/share",
                 "evidence\\source.txt", "..\\outside.txt", "C:\\outside.txt",
                 "evidence", ".", "missing.txt", "evidence/missing.txt", "bad\x00name")
        for path in paths:
            with self.subTest(path=repr(path)):
                self.assert_invalid(self.mutated(("sources", 0, "path"), path))
        for path in ("evidence/source.txt", "./evidence/source.txt", "evidence//source.txt"):
            with self.subTest(valid=path):
                self.assert_valid(self.mutated(("sources", 0, "path"), path))

    def test_symlinks_allow_contained_files_but_not_escapes(self):
        outside = self.temp / "root-sibling"
        outside.mkdir()
        (outside / "source.txt").write_text("Outside root.\n", encoding="utf-8")
        (self.root / "inside-link").symlink_to(self.source)
        (self.root / "relative-link").symlink_to("evidence/source.txt")
        (self.root / "inside-directory-link").symlink_to(self.root / "evidence", target_is_directory=True)
        (self.root / "outside-link").symlink_to(outside / "source.txt")
        (self.root / "outside-directory-link").symlink_to(outside, target_is_directory=True)
        (self.root / "broken-link").symlink_to(self.temp / "absent")
        (self.root / "loop-link").symlink_to("loop-link")
        for path in ("inside-link", "relative-link", "inside-directory-link/source.txt"):
            with self.subTest(valid=path):
                self.assert_valid(self.mutated(("sources", 0, "path"), path))
        for path in ("outside-link", "outside-directory-link/source.txt", "broken-link", "loop-link"):
            with self.subTest(invalid=path):
                self.assert_invalid(self.mutated(("sources", 0, "path"), path))
        root_alias = self.temp / "root-alias"
        root_alias.symlink_to(self.root, target_is_directory=True)
        self.assert_valid(root=root_alias)

    def test_invalid_roots_raise_value_error(self):
        # None is deliberately passed directly because the assertion helper uses
        # None to select its default fixture root.
        for root in (None, 17, False, [], {}, self.source, self.temp / "absent", "bad\x00root"):
            with self.subTest(root=repr(root)), self.assertRaises(ValueError):
                CHECKER.validate_record(deepcopy(self.record), root)

    def test_diagnostics_do_not_depend_on_mapping_insertion_order(self):
        first = deepcopy(self.record)
        first.update(z_extra=True, a_extra=False)
        second = dict(reversed(list(first.items())))
        messages = []
        for record in (first, second):
            with self.assertRaises(ValueError) as caught:
                CHECKER.validate_record(record, self.root)
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])

    def invoke_main(self, argv, loader_error=None, validator_error=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(CHECKER, "load_record", return_value=self.record,
                               side_effect=loader_error) as loader, \
             mock.patch.object(CHECKER, "validate_record", side_effect=validator_error) as validator, \
             redirect_stdout(out), redirect_stderr(err):
            result = CHECKER.main(argv)
        return result, out.getvalue(), err.getvalue(), loader, validator

    def test_main_default_and_explicit_paths_with_mocked_boundaries(self):
        result, out, err, loader, validator = self.invoke_main([])
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        self.assertEqual(out, SUCCESS)
        self.assertEqual(err, "")
        self.assertEqual(Path(loader.call_args.args[0]), ACTUAL_RECORD)
        self.assertEqual(loader.call_count, 1)
        validator.assert_called_once_with(self.record, REPOSITORY)
        for path in (str(self.temp / "chosen.json"), "relative-record.json"):
            result, out, err, loader, validator = self.invoke_main([path])
            self.assertEqual((result, out, err), (0, SUCCESS, ""))
            loader.assert_called_once_with(path)
            validator.assert_called_once_with(self.record, REPOSITORY)
        with mock.patch.object(sys, "argv", [str(SCRIPT), "argv-record.json"]):
            result, out, err, loader, _ = self.invoke_main(None)
        self.assertEqual((result, out, err), (0, SUCCESS, ""))
        loader.assert_called_once_with("argv-record.json")

    def test_main_loading_and_validation_failures_use_stdout_and_exit_one(self):
        errors = (ValueError("duplicate nested key"),
                  json.JSONDecodeError("malformed", "{", 1),
                  FileNotFoundError("missing record"), PermissionError("unreadable record"),
                  IsADirectoryError("record is a directory"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                result, out, err, loader, validator = self.invoke_main([], loader_error=error)
                self.assertIs(type(result), int)
                self.assertEqual(result, 1)
                self.assertTrue(out.strip())
                self.assertNotIn(SUCCESS.strip(), out)
                self.assertEqual(err, "")
                self.assertEqual(loader.call_count, 1)
                validator.assert_not_called()
        result, out, err, _, validator = self.invoke_main([], validator_error=ValueError("bad structure"))
        self.assertEqual(result, 1)
        self.assertIn("bad structure", out)
        self.assertEqual(err, "")
        self.assertNotIn(SUCCESS.strip(), out)
        self.assertEqual(validator.call_count, 1)

    def test_main_unknown_argument_uses_normal_argparse_failure(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(CHECKER, "load_record") as loader, \
             redirect_stdout(out), redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            CHECKER.main(["--not-a-supported-option"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(err.getvalue().strip())
        loader.assert_not_called()

    def test_three_cli_processes_from_empty_working_directories(self):
        # The only subprocesses in this suite. Each is bounded to ten seconds.
        # Explicit fixture sources point at an ordinary known repository file,
        # not at content or schema copied from the actual scientific record.
        record = deepcopy(self.record)
        for source in record["sources"]:
            source["path"] = "scripts/check_foundation_compatibility.py"
        record["propositions"][1]["statement"] = "Unsupported assertion: 1 = 0."
        chosen = self.temp / "chosen.json"
        chosen.write_text(json.dumps(record), encoding="utf-8")
        cases = (
            ([], 0),
            ([str(chosen)], 0),
            (["doc/derivations/foundation_compatibility_2026-09-13.json"], 1),
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
            environment[variable] = "1"
        for index, (arguments, expected) in enumerate(cases):
            cwd = self.temp / ("empty-cwd-" + str(index))
            cwd.mkdir()
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "-B", "-W", "error", str(SCRIPT), *arguments],
                    cwd=cwd, env=environment, capture_output=True, text=True,
                    timeout=10, check=False)
                self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
                self.assertEqual(completed.stderr, "")
                if expected == 0:
                    self.assertEqual(completed.stdout, SUCCESS)
                else:
                    self.assertTrue(completed.stdout.strip())
                    self.assertNotIn(SUCCESS.strip(), completed.stdout)
                self.assertEqual(list(cwd.iterdir()), [], "CLI must not write in its working directory")


if __name__ == "__main__":
    unittest.main()
