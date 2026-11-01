"""One-shot local diagnostic of Phase 14 RAG model abstention.

Run at the repository root with: python scripts/phase14_diagnose_model.py
This script does not edit project files, change model configuration, or save output.
It loads the same retriever and local generator as POST /rag/ask and makes
one inference for the question shown below. No benchmarks are rerun.
"""

import os
import sys

from greek_nlp.api.main import get_grounded_rag

QUESTION = "Πότε χτυπήθηκαν οι καλλιέργειες στον Τύρναβο από ακραία καιρικά φαινόμενα;"


class InspectGenerator:
    def __init__(self, actual_generator):
        self.actual = actual_generator
        # GroundedRAG uses this tokenizer to measure the complete chat prompt.
        self.tokenizer = actual_generator.tokenizer

    def generate(self, prompt, max_new_tokens=96):
        result = self.actual.generate(prompt, max_new_tokens)
        raw_text = str(result.get("generated_text") or "")
        print("\n=== RAW MODEL OUTPUT (repr) ===", flush=True)
        print(repr(raw_text), flush=True)
        print("=== END RAW MODEL OUTPUT ===", flush=True)
        print("MODEL_ID:", result.get("model"), flush=True)
        print("INPUT_TOKENS:", result.get("input_tokens"), flush=True)
        print("OUTPUT_TOKENS:", result.get("output_tokens"), flush=True)
        print("IS_EMPTY:", not raw_text.strip(), flush=True)
        print("CONTAINS_ABSTENTION_MARKER:", "ΑΝΕΠΑΡΚΗ ΣΤΟΙΧΕΙΑ" in raw_text.upper(), flush=True)
        return result


def main():
    print("LLM_MODEL_ID:", os.environ.get("LLM_MODEL_ID", "<not set>"), flush=True)
    print("LLM_DEVICE:", os.environ.get("LLM_DEVICE", "auto"), flush=True)
    if not os.environ.get("LLM_MODEL_ID"):
        print("Set LLM_MODEL_ID to the model you tested in Swagger before running.")
        return 2
    rag = get_grounded_rag()
    actual_loader = rag.generator_loader
    rag.generator_loader = lambda: InspectGenerator(actual_loader())
    result = rag.ask(QUESTION, k=3, max_new_tokens=96)
    print("\n=== RAG DECISION ===", flush=True)
    print("ABSTAINED:", result.get("abstained"), flush=True)
    print("REASON:", result.get("reason"), flush=True)
    print("ANSWER:", repr(result.get("answer")), flush=True)
    print("SOURCE_IDS:", [source.get("doc_id") for source in result.get("sources", [])], flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("DIAGNOSTIC_ERROR:", type(exc).__name__, str(exc)[:240], flush=True)
        raise
