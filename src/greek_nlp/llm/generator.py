"""Lazy, bounded local text generation using a public Hugging Face model.

Phase 13 only: no retrieval, grounding, citations, QLoRA adapter, or evaluation.
Dependencies and weights are loaded ONLY on the first /generate request.
"""

import os
from threading import Lock

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "Είσαι ένας βοηθός που απαντά σε φυσικά, γραμματικά σωστά νέα ελληνικά. "
    "Απάντησε απευθείας στην ερώτηση με απλές, σύντομες και σαφείς προτάσεις. "
    "Μην επαναλαμβάνεις την ερώτηση και μην επινοείς πληροφορίες. "
    "Αν δεν γνωρίζεις την απάντηση, πες ότι δεν είσαι σίγουρος."
)
MAX_INPUT_TOKENS = 1024  # bounded context for grounded RAG plus chat-template overhead


class LocalGenerator:
    """Load a local Hugging Face causal LLM once; serialize generation calls."""

    def __init__(self) -> None:
        # Imports inside the initializer keep normal API endpoints light-weight.
        import torch

        requested_device = os.environ.get("LLM_DEVICE", "auto").strip().lower()
        if requested_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("LLM_DEVICE must be cpu, cuda or auto")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = (
            "cuda"
            if requested_device == "cuda"
            or (requested_device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        self.model_id = os.environ.get("LLM_MODEL_ID", DEFAULT_MODEL_ID).strip()
        if not self.model_id:
            raise ValueError("LLM_MODEL_ID must not be empty")
        self.revision = os.environ.get("LLM_MODEL_REVISION") or None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._lock = Lock()
        # No trust_remote_code; never allow API requests to select model IDs.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        """Greedy decode of *new tokens only*; result is not fact-checked."""
        if not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be within 1..128")
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        with self._lock:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # The chat template already contains Qwen's role/control tokens.
            # Re-adding special tokens can corrupt the intended conversation format.
            inputs = self.tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            n_input = int(inputs["input_ids"].shape[-1])
            # Do not pass a prompt whose trailing assistant header may have been
            # truncated, since the model could then continue the user's message.
            if n_input >= MAX_INPUT_TOKENS:
                raise ValueError("Prompt exceeds the safe input-token budget")
            with self._torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output[0][n_input:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return {
                "mode": "local_llm",
                "model": self.model_id,
                "device": self.device,
                "generated_text": text,
                "input_tokens": n_input,
                "output_tokens": len(generated_ids),
            }
