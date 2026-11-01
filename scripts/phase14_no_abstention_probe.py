"""Phase 14 diagnostic: one-source extraction without an abstention marker.

From the repository root in the greek-nlp environment:
    python scripts/phase14_no_abstention_probe.py

This is a diagnostic only. It does not alter the production RAG policy or
claim that any generated text is verified. One generation, no benchmarks.
"""

import os

from greek_nlp.api.main import get_grounded_rag
from greek_nlp.rag.grounded import _candidate_excerpts, _safe_context

QUESTION = "Πότε χτυπήθηκαν οι καλλιέργειες στον Τύρναβο από ακραία καιρικά φαινόμενα;"
DOCUMENT_ID = "macocu.el.6726937"


def main() -> None:
    print("MODEL:", os.environ.get("LLM_MODEL_ID", "<default>"), flush=True)
    print("DEVICE:", os.environ.get("LLM_DEVICE", "auto"), flush=True)

    rag = get_grounded_rag()
    hits = rag.retriever.search(QUESTION, k=3)
    candidates = _candidate_excerpts(QUESTION, hits, rag.retriever.docs_by_id)
    source = next((item for item in candidates if item["doc_id"] == DOCUMENT_ID), None)
    if source is None:
        print("STOP: The expected genuine source was not retrieved. No model call.", flush=True)
        return

    excerpt = _safe_context(source["excerpt"])
    print("SOURCE_DOCUMENT_ID:", DOCUMENT_ID, flush=True)
    print("SOURCE_EXCERPT:", repr(excerpt), flush=True)

    # Unlike the preceding probe, do not mention any abstention marker.
    # English extraction instruction helps isolate the model's Greek-language
    # instruction-following ability. The factual answer is not in the prompt
    # instructions; it must come from the genuine article excerpt.
    prompt = (
        "Extract the time expression from the Greek evidence that answers the question. "
        "Copy the time expression as written in the evidence and then add [1]. "
        "Do not provide any explanation.\n"
        f"Evidence [1]: {excerpt}\n"
        f"Question: {_safe_context(QUESTION)}\n"
        "Answer:"
    )

    print("STARTING ONE INFERENCE...", flush=True)
    output = rag.generator_loader().generate(prompt, max_new_tokens=48)
    raw = str(output.get("generated_text") or "")
    print("\n=== RAW OUTPUT WITHOUT ABSTENTION INSTRUCTION ===", flush=True)
    print(repr(raw), flush=True)
    print("INPUT_TOKENS:", output.get("input_tokens"), flush=True)
    print("OUTPUT_TOKENS:", output.get("output_tokens"), flush=True)
    print("IS_EMPTY:", not bool(raw.strip()), flush=True)
    print("NOTE: This is not a RAG faithfulness validation or a production change.")


if __name__ == "__main__":
    main()
