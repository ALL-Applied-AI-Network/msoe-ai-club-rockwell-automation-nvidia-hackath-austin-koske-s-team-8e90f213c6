import unittest

from replay.explain import SECTIONS, contradictions, parse, rule_explain

ALLOWED = set(SECTIONS) | {"image:fill_end_frame"}


def packet(pattern, cos=0.9):
    return {"context": {"seq": 4, "intended_recipe": "6", "previous_recipe": "5", "next_recipe": "6"},
            "measurement": {"endpoint_lab": [20, -4, 7], "qa_valid_samples": 20, "samples": 60,
                            "final_liquid_px": 7000, "qa_flags": {}},
            "comparison": {"de_intended": 24.0, "de_previous": 8.0, "direction_cos": cos, "verdict": "OUT_OF_SPEC",
                           "pattern": pattern}}


class ExplainTest(unittest.TestCase):
    def test_valid_output_with_reasoning_block(self):
        text = ('<think>colour moved toward 5</think>{"cause": "carryover", "confidence": 0.8, '
                '"evidence_ids": ["comparison", "image:fill_end_frame"], "explanation": "Shifted toward 5.", '
                '"next_check": "Flush."}')
        explanation, problems = parse(text, ALLOWED)
        assert problems == [] and explanation["cause"] == "carryover"

    def test_rejects_unknown_cause_and_ungrounded_ids(self):
        explanation, problems = parse('{"cause": "gremlins", "confidence": 2, "evidence_ids": ["image:x"], '
                                      '"explanation": "x", "next_check": "y"}', ALLOWED)
        assert explanation["cause"] == "unknown" and explanation["confidence"] == 1.0
        assert {"cause_not_in_taxonomy", "ungrounded_evidence"} <= set(problems)

    def test_not_json(self):
        explanation, problems = parse("I think it is carryover.", ALLOWED)
        assert problems == ["not_json"] and explanation["cause"] == "unknown"

    def test_rule_baseline_is_schema_valid(self):
        for pattern in ("carryover", "deviation", "persistent_deviation", "no_reference", "unmeasured"):
            explanation, problems = parse(rule_explain(packet(pattern)), ALLOWED)
            assert problems == [], (pattern, problems)
        assert rule_explain(packet("carryover"))["cause"] == "carryover"


    def test_carryover_needs_a_previous_recipe_and_direction(self):
        first = packet("deviation", cos=None)
        first["context"]["previous_recipe"] = None
        self.assertEqual(contradictions(first, {"cause": "carryover"}), ["carryover_without_previous_recipe"])
        self.assertEqual(contradictions(packet("deviation", cos=0.1), {"cause": "carryover"}),
                         ["carryover_but_colour_not_toward_previous"])
        self.assertEqual(contradictions(packet("carryover"), {"cause": "carryover"}), [])


if __name__ == "__main__":
    unittest.main()
