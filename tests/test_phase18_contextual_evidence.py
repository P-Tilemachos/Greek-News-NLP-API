"""Regression tests for adjacent-sentence evidence selection in grounded RAG.

All tests are offline: no model weights, network calls or HTTP server.
"""

from greek_nlp.rag.grounded import (
    MAX_EXCERPT_CHARS,
    GroundedRAG,
    _candidate_excerpts,
)

QUESTION = "Σε ποιους επιρρίπτει πειθαρχικές ευθύνες το πόρισμα για τις πλημμύρες στη Μάνδρα;"
ARTICLE = (
    "Πειθαρχικές ευθύνες επιρρίπτει το πόρισμα για τις πλημμύρες στη Μάνδρα. "
    "Σε σειρά διαφορετικών φορέων του δημοσίου εντοπίζει ευθύνες."
)


class Retriever:
    docs_by_id = {"mandra": {"text": ARTICLE, "label": "news"}}

    def search(self, query, k=3):
        return [{"id": "mandra", "score": 0.9}][:k]


class BadThenVerbatimGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens=96):
        self.calls.append(prompt)
        if len(self.calls) == 1:
            generated_text = QUESTION + "\nΤεκμήρια: [1]"
        else:
            generated_text = "σε σειρά διαφορετικών φορέων του δημοσίου [1]"
        return {"model": "fake-model", "generated_text": generated_text}


def test_adjacent_related_sentences_form_one_contiguous_evidence_span():
    hits = [{"id": "mandra", "score": 0.9}]
    sources = _candidate_excerpts(QUESTION, hits, Retriever.docs_by_id)
    assert len(sources) == 1
    evidence = sources[0]["excerpt"]
    assert evidence == ARTICLE
    assert len(evidence) <= MAX_EXCERPT_CHARS
    assert evidence in Retriever.docs_by_id["mandra"]["text"]


def test_verbatim_retry_can_use_answer_phrase_in_adjacent_sentence():
    generator = BadThenVerbatimGenerator()
    response = GroundedRAG(Retriever(), lambda: generator).ask(QUESTION, k=1)
    assert response["abstained"] is False
    assert response["answer"] == "σε σειρά διαφορετικών φορέων του δημοσίου [1]"
    assert response["faithfulness_verified"] is False
    assert len(generator.calls) == 2
    assert "Σε σειρά διαφορετικών φορέων του δημοσίου" in generator.calls[0]


def test_unrelated_adjacent_sentence_is_not_attached():
    article = (
        "Πειθαρχικές ευθύνες επιρρίπτει το πόρισμα για τις πλημμύρες στη Μάνδρα. "
        "Η ομάδα κέρδισε στον τελικό του κυπέλλου."
    )
    docs = {"mandra": {"text": article, "label": "news"}}
    sources = _candidate_excerpts(QUESTION, [{"id": "mandra", "score": 1.0}], docs)
    assert len(sources) == 1
    assert "ομάδα" not in sources[0]["excerpt"]


def test_two_query_words_far_apart_do_not_bypass_evidence_gate():
    question = "Τι ανακοινώθηκε για τις πλημμύρες;"
    article = (
        "Ανακοινώθηκε ένα σχέδιο για τις μεταφορές και τις αστικές συγκοινωνίες. "
        "Η ομάδα συζήτησε ένα νέο πρόγραμμα για την κατάρτιση. "
        "Οι πλημμύρες έπληξαν τις παραποτάμιες περιοχές και προκάλεσαν προβλήματα."
    )
    docs = {"x": {"text": article, "label": "news"}}
    sources = _candidate_excerpts(question, [{"id": "x", "score": 0.8}], docs)
    assert sources == []


def test_related_sentence_pair_over_limit_is_never_merged():
    first = "Πειθαρχικές ευθύνες επιρρίπτει το πόρισμα για τις πλημμύρες στη Μάνδρα."
    second = (
        "Σε σειρά διαφορετικών φορέων του δημοσίου εντοπίζει ευθύνες "
        "και σε επιπλέον πολλές διοικητικές υπηρεσίες που συμμετείχαν "
        "σε μια μεγάλη και μακρόχρονη διαδικασία ελέγχων."
    )
    docs = {"x": {"text": f"{first} {second}", "label": "news"}}
    sources = _candidate_excerpts(QUESTION, [{"id": "x", "score": 0.8}], docs)
    assert len(sources) == 1
    assert sources[0]["excerpt"] != f"{first} {second}"
    assert len(sources[0]["excerpt"]) <= MAX_EXCERPT_CHARS


def test_who_retry_requests_exact_evidence_phrase_not_a_paraphrase():
    from greek_nlp.rag.grounded import _extractive_retry_prompt

    prompt = _extractive_retry_prompt(
        QUESTION,
        {"excerpt": ARTICLE, "doc_id": "mandra"},
    )
    assert "αντέγραψέ την ΑΥΤΟΥΣΙΑ" in prompt
    assert "Μην επαναλάβεις την ερώτηση" in prompt
    assert "Σε σειρά διαφορετικών φορέων του δημοσίου" in prompt
    assert len(prompt) <= 720


def test_unsupported_retry_cannot_be_returned_as_an_answer():
    class UnsupportedGenerator:
        calls = 0

        def generate(self, prompt, max_new_tokens=96):
            self.calls += 1
            text = QUESTION if self.calls == 1 else "σε άγνωστους ιδιώτες [1]"
            return {"model": "fake", "generated_text": text}

    generator = UnsupportedGenerator()
    response = GroundedRAG(Retriever(), lambda: generator).ask(QUESTION)
    assert response["abstained"] is True
    assert response["reason"] == "invalid_generated_answer"
    assert response["answer"] is None
    assert generator.calls == 2
