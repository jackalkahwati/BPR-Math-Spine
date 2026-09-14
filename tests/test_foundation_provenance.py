"""Static provenance guards for the foundation-prerequisites metadata patch.

Oracle: git commit d9bccda2dfa64f8a17e094ecc420b6e64a80245f, inspected before
reading the implementation. The three hashes below were independently computed
from that baseline, not from the corrected working tree. Expected status/value
fields are handwritten from the approved contract and baseline observations.

These tests parse source; they never import bpr.experimental_data,
bpr.first_principles, or any other project module. Runtime needs neither git
nor an external snapshot. No scientific calculation or benchmark is performed.
Comments/docstrings and approved metadata prose may change without invalidating
the executable AST oracle. This is a preservation check, not physical validation
of the legacy numbers, exactness flags, comparison counts, or verdict labels.
"""

import ast
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "bpr" / "experimental_data.py"
FLAVOR_PATH = ROOT / "doc" / "BPR_FLAVOR_SECTOR.md"
INDEX_PATH = ROOT / "doc" / "experiments" / "README.md"

BASELINE_REGISTRY_ENTRIES = 87
BASELINE_REGISTRY_SHA256 = (
    "6c277bb14675e53d25193f90b40a494936515a4eb47e09e15f41ccf8bc745968"
)
BASELINE_MODULE_SHA256 = (
    "e10fd99cb86b40e5841cddfae49bad9887bd3abb4b12ba8cfd9a8dbd174509ab"
)
BASELINE_LEGACY_ROWS_SHA256 = (
    "905a1273bcfb3085866e00f44953afd125000874e1080f39ce07d98b0fab0a45"
)

# Only these exact IDs may have name/source/status string text changed.
# No blanket stripping of all registry statuses or all string constants.
APPROVED_STATUSES = {
    "P5.5_theta12_deg": "FRAMEWORK",
    "P5.6_theta23_deg": "FRAMEWORK",
    "P5.7_theta13_deg": "CONJECTURAL",
    "P5.10_number_of_generations": "FRAMEWORK",
}
APPROVED_METADATA_FIELDS = frozenset({"name", "source", "bpr_status"})
GENERATION_ID = "P5.10_number_of_generations"

# Values, types, and explicitly supplied flags are frozen independently of the
# metadata author. The AST hashes also freeze omitted flag defaults and every
# field of all other measurements, including nonliteral formula expressions.
EXPECTED_MEASUREMENTS = {
    "P5.5_theta12_deg": {
        "value": 33.41, "uncertainty": 0.8, "unit": "deg", "year": 2024,
    },
    "P5.6_theta23_deg": {
        "value": 49.0, "uncertainty": 1.3, "unit": "deg", "year": 2024,
    },
    "P5.7_theta13_deg": {
        "value": 8.54, "uncertainty": 0.15, "unit": "deg", "year": 2024,
    },
    GENERATION_ID: {
        "value": 3, "uncertainty": 0.0, "unit": "", "year": 2006,
        "is_exact": True,
    },
}


def _literal_string(node):
    if not isinstance(node, ast.Constant) or type(node.value) is not str:
        raise AssertionError("Registry ID/metadata must be a literal string")
    return node.value


def _extract_registry(tree):
    """Find the actual top-level registry, rejecting ambiguous/dynamic entries."""
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_REGISTRY"
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.List):
        raise AssertionError("Expected one literal-list _REGISTRY declaration")
    registry = assignments[0].value
    entries = {}
    for call in registry.elts:
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Measurement"
            or call.args
        ):
            raise AssertionError("Expected keyword-only Measurement entries")
        fields = {}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in fields:
                raise AssertionError("Unpacked/duplicate registry keyword")
            fields[keyword.arg] = keyword.value
        if "prediction_id" not in fields:
            raise AssertionError("Registry entry lacks prediction_id")
        prediction_id = _literal_string(fields["prediction_id"])
        if prediction_id in entries:
            raise AssertionError(f"Duplicate registry ID: {prediction_id}")
        entries[prediction_id] = (call, fields)
    return registry, entries


def _canonical_ast(value):
    """Stable, location-free AST representation; never evaluates expressions.

    Remove only genuine leading docstrings. Normalize Python 3.8's slice
    wrappers and Python 3.12's empty type_params field across AST versions.
    Keep all other fields, their order, typed constants, and expression trees.
    """
    if isinstance(value, ast.AST) and type(value).__name__ == "Index":
        return _canonical_ast(value.value)
    if isinstance(value, ast.AST) and type(value).__name__ == "ExtSlice":
        # Python 3.9+ represents multidimensional slices as a loaded Tuple.
        return ["Tuple", [
            ["elts", _canonical_ast(value.dims)],
            ["ctx", ["Load", []]],
        ]]
    if isinstance(value, ast.AST):
        fields = []
        for name, item in ast.iter_fields(value):
            if name == "type_params" and not item:
                continue
            if name == "body" and isinstance(
                value, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if (
                    item
                    and isinstance(item[0], ast.Expr)
                    and isinstance(item[0].value, ast.Constant)
                    and isinstance(item[0].value.value, str)
                ):
                    item = item[1:]
            fields.append([name, _canonical_ast(item)])
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    return value


def _digest(value):
    serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _masked_trees(source):
    # Parse afresh rather than modifying the tree used by explicit field tests.
    tree = ast.parse(source, filename=str(REGISTRY_PATH))
    registry, entries = _extract_registry(tree)
    for prediction_id in APPROVED_STATUSES:
        if prediction_id not in entries:
            raise AssertionError(f"Missing approved registry ID: {prediction_id}")
        fields = entries[prediction_id][1]
        for field in APPROVED_METADATA_FIELDS:
            if field not in fields:
                raise AssertionError(f"Missing {prediction_id}.{field}")
            _literal_string(fields[field])
            fields[field].value = "<approved-metadata>"
    return tree, registry


def _plain(text):
    """Ignore Markdown emphasis, whitespace, and typographic hyphen choices."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"[-‐-―]", " ", text)
    return " ".join(text.split())


def _paragraphs(text):
    return [_plain(part) for part in re.split(r"\n\s*\n", text) if part.strip()]


def _denies_dirac_confirmation(text):
    """Require a negated positive-evidence verb, not e.g. 'disprove'."""
    negated_evidence = re.compile(
        r"\b(?:does not|do not|cannot|can not|can't|doesn't|not|neither)\s+"
        r"(?:(?:by itself|by themselves|on its own|alone|in itself|therefore|necessarily)\s+)*"
        r"(?:establish(?:es|ed)?|confirm(?:s|ed)?|prove(?:s|d)?|"
        r"demonstrat(?:e|es|ed)|impl(?:y|ies|ied)|validat(?:e|es|ed))\b"
        r"[^.!?;,:]{0,100}\bdirac\b"
    )
    # Do not borrow a negation from a different clause or contrast statement.
    clauses = re.split(r"[.!?;,:]|\b(?:but|however|whereas)\b", _plain(text))
    return any(negated_evidence.search(clause) for clause in clauses)


def _qualifies_nonindependence(text):
    """Accept an explicit limitation, never the substring in 'independent'."""
    limitation = re.compile(
        r"\bnon\s*independent\b"
        r"|\bnot\s+(?:all\s+|necessarily\s+|statistically\s+)?independent\b"
        r"|\b(?:do not|does not|cannot|can't|must not|should not)\s+"
        r"(?:constitute|represent|provide|count as|imply|establish|"
        r"be (?:read|treated|interpreted|counted) as)\s+"
        r"(?:statistically\s+)?independent\b"
    )
    clauses = re.split(r"[.!?;,:]|\b(?:but|however|whereas)\b", _plain(text))
    return any(limitation.search(clause) for clause in clauses)


class TestProvenanceProseGuards(unittest.TestCase):
    """Pure-string examples protect the guards from reversed meanings."""

    def test_dirac_guard_accepts_explicit_limits(self):
        for text in (
            "A null 0νββ result does not establish Dirac neutrinos.",
            "A null result cannot by itself prove Dirac neutrinos.",
            "Non-observation does not confirm the Dirac hypothesis.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_denies_dirac_confirmation(text))

    def test_dirac_guard_rejects_reversed_or_cross_clause_claims(self):
        for text in (
            "A null result does not disprove Dirac neutrinos.",
            "A null result proves Dirac neutrinos.",
            "It does not establish Majorana neutrinos; it proves Dirac neutrinos.",
            "It does not establish Majorana neutrinos, but it confirms Dirac neutrinos.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_denies_dirac_confirmation(text))

    def test_nonindependence_guard_accepts_explicit_limits(self):
        for text in (
            "These are non-independent comparisons.",
            "These tests are not independent confirmations.",
            "Counts must not be read as independent confirmations.",
            "Comparisons do not constitute statistically independent evidence.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_qualifies_nonindependence(text))

    def test_nonindependence_guard_rejects_substrings_and_positive_claims(self):
        for text in (
            "All comparisons are independent confirmations.",
            "Independent and dependent variables are listed.",
            "These tests do not fail; they are independent confirmations.",
            "These counts do not deny that tests are independent.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_qualifies_nonindependence(text))


class TestFoundationRegistryProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = REGISTRY_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(REGISTRY_PATH))
        cls.registry, cls.entries = _extract_registry(cls.tree)

    def test_only_four_authorized_statuses_are_reclassified(self):
        self.assertEqual(len(self.entries), BASELINE_REGISTRY_ENTRIES)
        for prediction_id, expected in APPROVED_STATUSES.items():
            with self.subTest(prediction_id=prediction_id):
                fields = self.entries[prediction_id][1]
                self.assertEqual(_literal_string(fields["bpr_status"]), expected)

    def test_explicit_measurements_and_generation_legacy_flag_are_unchanged(self):
        for prediction_id, expected_fields in EXPECTED_MEASUREMENTS.items():
            fields = self.entries[prediction_id][1]
            with self.subTest(prediction_id=prediction_id):
                self.assertEqual(
                    set(fields),
                    {"prediction_id", "name", "source", "bpr_status"}
                    | set(expected_fields),
                )
                for name, expected in expected_fields.items():
                    with self.subTest(field=name):
                        self.assertIsInstance(fields[name], ast.Constant)
                        self.assertIs(type(fields[name].value), type(expected))
                        self.assertEqual(fields[name].value, expected)

    def test_all_other_registry_fields_match_independent_baseline(self):
        _, registry = _masked_trees(self.source)
        self.assertEqual(
            _digest(_canonical_ast(registry)), BASELINE_REGISTRY_SHA256,
            "Registry numbers, expressions, IDs, units, years, flags or "
            "unauthorized metadata/statuses changed from the frozen baseline",
        )

    def test_registry_defaults_lookup_and_executable_formulas_are_unchanged(self):
        tree, _ = _masked_trees(self.source)
        self.assertEqual(
            _digest(_canonical_ast(tree)), BASELINE_MODULE_SHA256,
            "Executable source outside the four metadata string edits changed; "
            "comments and docstrings are deliberately excluded",
        )

    def test_generation_metadata_describes_empirical_family_input_and_lep_scope(self):
        fields = self.entries[GENERATION_ID][1]
        name = _plain(_literal_string(fields["name"]))
        source = _plain(_literal_string(fields["source"]))
        metadata = name + " " + source
        self.assertRegex(name, r"famil(?:y|ies)")
        self.assertRegex(metadata, r"empirical|observed|measured")
        self.assertRegex(metadata, r"input|three|\b3\b")
        self.assertIn("lep", source)
        self.assertIn("light", source)
        self.assertIn("active", source)
        self.assertIn("neutrino", source)

    def test_generation_exact_flag_is_qualified_as_historical_grading(self):
        call = self.entries[GENERATION_ID][0]
        position = self.registry.elts.index(call)
        previous_end = self.registry.elts[position - 1].end_lineno if position else 0
        # Include the entry's preceding comment, inline comments, and source.
        context = _plain("\n".join(
            self.source.splitlines()[previous_end:call.end_lineno]
        ))
        self.assertIn("is_exact", context)
        self.assertRegex(context, r"historical|legacy")
        self.assertRegex(context, r"grad(?:e|es|ing)|scor(?:e|es|ing)|classification")
        self.assertRegex(context, r"physical|physically|measurement")
        self.assertRegex(context, r"exactness|exact|uncertainty")
        self.assertRegex(context, r"\bnot\b|\bno\b|doesn't|isn't|never")


class TestFoundationDocumentProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flavor = FLAVOR_PATH.read_text(encoding="utf-8")
        cls.index = INDEX_PATH.read_text(encoding="utf-8")

    def test_flavor_does_not_claim_parameter_free_derivation(self):
        paragraphs = _paragraphs(self.flavor)
        # Accept different ways of expressing the limitation, not a fixed
        # replacement sentence. Require negation close to the specific claim.
        limitation = re.compile(
            r"(?:\bnot\b|\bno\b|\black\w*\b|\bwithout\b)"
            r"[^.!?]{0,100}parameter\s*free[^.!?]{0,100}deriv"
            r"|parameter\s*free[^.!?]{0,100}deriv[^.!?]{0,100}"
            r"(?:\bnot\b|\bno\b|unestablished|unproven)"
        )
        self.assertTrue(
            any(limitation.search(part) for part in paragraphs),
            "Flavor claims must explicitly disclaim a parameter-free derivation",
        )
        self.assertNotRegex(
            _plain(self.flavor),
            r"no free continuous parameters in the chain from substrate to predictions",
        )

    def test_generation_count_is_an_empirical_flavor_input(self):
        candidates = [
            part for part in _paragraphs(self.flavor)
            if "n_gen" in part or re.search(r"generation|famil(?:y|ies)", part)
        ]
        self.assertTrue(any(
            re.search(r"empirical|observed|measured", part)
            and re.search(r"input|assum(?:ed|ption)", part)
            and re.search(r"three|\b3\b", part)
            for part in candidates
        ), "Three families must be acknowledged as empirical input")
        self.assertNotRegex(_plain(self.flavor), r"n_gen\s*=\s*3\s*\(generations from topology\)")

    def test_pmns_coefficients_remain_explicitly_fitted(self):
        paragraphs = _paragraphs(self.flavor)
        for coefficient in ("1.35", "3.5"):
            with self.subTest(coefficient=coefficient):
                self.assertTrue(any(
                    coefficient in part
                    and re.search(r"fitt?\w*|phenomenolog\w*|calibrat\w*", part)
                    for part in paragraphs
                ), "PMNS coefficient provenance must remain explicit")

    def test_null_double_beta_result_does_not_establish_dirac_neutrinos(self):
        candidates = [
            part for part in _paragraphs(self.flavor)
            if "dirac" in part
            and re.search(r"0νββ|0nubb|neutrinoless|double beta", part)
        ]
        self.assertTrue(any(
            re.search(r"null|non observation|nonobservation|non detection|limit", part)
            and _denies_dirac_confirmation(part)
            for part in candidates
        ), "A null 0νββ result is not proof of Dirac neutrinos")
        self.assertNotRegex(
            " ".join(candidates), r"vindicated as predicted|confirmed by legend",
        )

    def test_historical_verdict_rows_are_preserved_not_recomputed(self):
        rows = [
            tuple(cell.strip().replace("**", "")
                  for cell in line.strip().strip("|").split("|"))
            for line in self.index.splitlines() if line.lstrip().startswith("|")
        ]
        legacy = [row for row in rows if row[0] in {"CONFIRM", "FALSIFY", "INCONCLUSIVE"}]
        self.assertEqual(len(legacy), 3)
        self.assertEqual(
            [(row[0], row[1]) for row in legacy],
            [("CONFIRM", "115"), ("FALSIFY", "0"), ("INCONCLUSIVE", "11")],
        )
        # Preserve legacy table contents, not cosmetic cell padding/emphasis.
        self.assertEqual(_digest(legacy), BASELINE_LEGACY_ROWS_SHA256)
        index = _plain(self.index)
        self.assertRegex(index, r"\b250\s*\+")
        self.assertRegex(index, r"\b129\s+tests\b")

    def test_experimental_index_qualifies_grading_and_nonindependence(self):
        paragraphs = _paragraphs(self.index)
        self.assertTrue(any(
            "heuristic" in part
            and re.search(r"grad\w*|scor\w*|verdict\w*|label\w*", part)
            for part in paragraphs
        ), "Legacy verdict labels require a heuristic-grading caveat")
        self.assertTrue(any(
            _qualifies_nonindependence(part)
            and re.search(r"compar\w*|test\w*|count\w*|confirm\w*|evidence", part)
            for part in paragraphs
        ), "Legacy comparison counts must not imply independent confirmations")


if __name__ == "__main__":
    unittest.main()
