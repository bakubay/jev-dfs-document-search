from __future__ import annotations

import unittest
from types import SimpleNamespace

from openai_grounded_answer import (
    CitationValidationError,
    EvidenceCitation,
    GroundedAnswer,
    generate_grounded_answer,
    validate_citations,
)


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="resp_test",
            output_parsed=self.parsed,
            usage=SimpleNamespace(input_tokens=100, output_tokens=25),
        )


class FakeOpenAI:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


class GroundedAnswerTests(unittest.TestCase):
    def test_accepts_only_verbatim_quotes_on_allowed_pages(self):
        parsed = GroundedAnswer(
            answer="The functions are GOVERN, MAP, MEASURE, and MANAGE.",
            fully_answered=True,
            limitations="",
            citations=[
                EvidenceCitation(
                    physical_pdf_page=25,
                    quote="GOVERN, MAP, MEASURE, and MANAGE",
                )
            ],
        )
        evidence = {25: "The four functions are GOVERN, MAP, MEASURE, and MANAGE."}
        client = FakeOpenAI(parsed)
        result = generate_grounded_answer(
            api_key="test",
            model="test-model",
            question="What are the functions?",
            evidence_pages=evidence,
            client=client,
        )
        self.assertEqual(result.citations[0]["physical_pdf_page"], 25)
        self.assertFalse(client.responses.request["store"])
        self.assertIs(client.responses.request["text_format"], GroundedAnswer)

    def test_rejects_unavailable_page(self):
        parsed = GroundedAnswer(
            answer="Unsupported",
            fully_answered=True,
            limitations="",
            citations=[EvidenceCitation(physical_pdf_page=99, quote="text")],
        )
        with self.assertRaises(CitationValidationError):
            validate_citations(parsed, {25: "text"})

    def test_rejects_quote_not_on_page(self):
        parsed = GroundedAnswer(
            answer="Unsupported",
            fully_answered=True,
            limitations="",
            citations=[EvidenceCitation(physical_pdf_page=25, quote="invented quote")],
        )
        with self.assertRaises(CitationValidationError):
            validate_citations(parsed, {25: "real source text"})


if __name__ == "__main__":
    unittest.main()
