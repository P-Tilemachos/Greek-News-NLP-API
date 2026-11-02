"""Phase 18 RAG guard regressions for real observed malformed model outputs.

Uses a fake retriever and generator; no network or actual LLM weights required.
"""

from greek_nlp.rag.grounded import GroundedRAG

QUESTION = "Σε ποιους επιρρίπτει πειθαρχικές ευθύνες το πόρισμα για τις πλημμύρες στη Μάνδρα;"
ARTICLE = (
    "Πειθαρχικές ευθύνες επιρρίπτει το πόρισμα για τις πλημμύρες στη Μάνδρα. "
    "Σε σειρά διαφορετικών φορέων του δημοσίου εντοπίζει ευθύνες."
)


class Retriever:
    def __init__(self, article=ARTICLE):
        self.docs_by_id = {"mandra": {"text": article, "label": "news"}}

    def search(self, _question, k=3):
        return [{"id": "mandra", "score": 0.9}][:k]


class Generator:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, prompt, max_new_tokens=96):
        self.calls.append((prompt, max_new_tokens))
        ix = min(len(self.calls) - 1, len(self.outputs) - 1)
        return {"model": "fake/model", "generated_text": self.outputs[ix]}


def test_prompt_echo_with_evidence_section_abstains():
    echoed = QUESTION + "\n\nΤεκμήρια:\n\n[1] Πειθαρχικές ευθύνες επιρρίπτει"
    generator = Generator([echoed, echoed])
    result = GroundedRAG(Retriever(), lambda: generator).ask(QUESTION)
    assert result["abstained"] is True
    assert result["answer"] is None
    assert result["reason"] == "invalid_generated_answer"
    assert result["faithfulness_verified"] is False
    assert len(generator.calls) == 2


def test_invalid_answer_may_recover_if_verbatim_evidence_is_returned():
    article = (
        "Το πόρισμα για τις πλημμύρες στη Μάνδρα επιρρίπτει πειθαρχικές "
        "ευθύνες σε σειρά διαφορετικών φορέων του δημοσίου."
    )
    generator = Generator(
        [
            QUESTION + "\n\nΤεκμήρια: [1] " + article[:40],
            "σε σειρά διαφορετικών φορέων του δημοσίου [1]",
        ]
    )
    result = GroundedRAG(Retriever(article), lambda: generator).ask(QUESTION)
    assert result["abstained"] is False
    assert result["answer"] == "σε σειρά διαφορετικών φορέων του δημοσίου [1]"
    assert result["sources"][0]["doc_id"] == "mandra"
    assert result["faithfulness_verified"] is False
    assert len(generator.calls) == 2


def test_answer_without_valid_citation_abstains_when_retry_fails():
    generator = Generator(["Στη Μάνδρα, το πόρισμα είναι άνθρωποι."])
    result = GroundedRAG(Retriever(), lambda: generator).ask(QUESTION)
    assert result["abstained"] is True
    assert result["reason"] == "invalid_generated_answer"
    assert result["answer"] is None
    assert len(generator.calls) == 2
