"""Phase 13 API tests: no model files or outbound Hugging Face requests."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main
from greek_nlp.api.schemas import GenerateRequest


@pytest.fixture
def fake_torch(monkeypatch):
    """Lightweight torch substitute: CI tests should not require ML dependencies."""

    class FakeTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            if self.data and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def to(self, device):
            assert device == "cpu"
            return self

        def __getitem__(self, index):
            return FakeTensor(self.data[index])

        def __len__(self):
            return len(self.data)

        def tolist(self):
            return self.data

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.tensor = FakeTensor
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens=64):
        self.calls.append((prompt, max_new_tokens))
        return {
            "mode": "local_llm",
            "model": "test/model",
            "device": "cpu",
            "generated_text": "Ενδεικτική δοκιμαστική απάντηση.",
            "input_tokens": 12,
            "output_tokens": 6,
        }


def test_generate_returns_bounded_structured_result(monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(main, "get_generator", lambda: fake)
    response = TestClient(main.app).post(
        "/generate", json={"prompt": "Γράψε μια μικρή περίληψη στα ελληνικά.", "max_new_tokens": 24}
    )
    assert response.status_code == 200
    assert response.json() == {
        "mode": "local_llm",
        "model": "test/model",
        "device": "cpu",
        "generated_text": "Ενδεικτική δοκιμαστική απάντηση.",
        "input_tokens": 12,
        "output_tokens": 6,
    }
    assert fake.calls == [("Γράψε μια μικρή περίληψη στα ελληνικά.", 24)]


def test_generate_has_safe_default_token_limit(monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(main, "get_generator", lambda: fake)
    result = TestClient(main.app).post("/generate", json={"prompt": "Πες μου μια είδηση."})
    assert result.status_code == 200
    assert fake.calls[0][1] == 64


def test_invalid_input_does_not_load_model(monkeypatch):
    def never_load():
        raise AssertionError("Invalid input must be validated before model loading")

    monkeypatch.setattr(main, "get_generator", never_load)
    client = TestClient(main.app)
    for payload in (
        {"prompt": "ab"},
        {"prompt": "a" * 1501},
        {"prompt": "ελληνικά", "max_new_tokens": 0},
        {"prompt": "ελληνικά", "max_new_tokens": 129},
    ):
        assert client.post("/generate", json=payload).status_code == 422


def test_dependency_missing_returns_503_without_internal_details(monkeypatch):
    def missing():
        raise OSError("cache/private/path/access_token")

    monkeypatch.setattr(main, "get_generator", missing)
    response = TestClient(main.app).post("/generate", json={"prompt": "Πες μου μια είδηση"})
    assert response.status_code == 503
    assert "private" not in response.text
    assert "Local LLM unavailable" in response.text


def test_runtime_memory_error_returns_503(monkeypatch):
    class BrokenGenerator:
        def generate(self, prompt, max_new_tokens=64):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(main, "get_generator", BrokenGenerator)
    response = TestClient(main.app).post("/generate", json={"prompt": "Γράψε ελληνικά"})
    assert response.status_code == 503


def test_health_does_not_load_llm(monkeypatch):
    def never_load():
        raise AssertionError("Normal endpoints must not load the LLM")

    monkeypatch.setattr(main, "get_generator", never_load)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_schema_token_bounds():
    req = GenerateRequest(prompt="Τι συνέβη στην Κρήτη;")
    assert req.max_new_tokens == 64


def test_local_generator_only_decodes_new_tokens(monkeypatch, fake_torch):
    """Exercise the real LocalGenerator logic with tiny in-memory fakes."""
    from greek_nlp.llm.generator import LocalGenerator

    calls = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages[1]["content"] == "Μια μικρή δοκιμή"
            assert tokenize is False and add_generation_prompt is True
            return "FAKE CHAT PREFIX"

        def __call__(self, rendered, **kwargs):
            assert rendered == "FAKE CHAT PREFIX"
            assert kwargs["truncation"] and kwargs["max_length"] == 1024
            assert kwargs["add_special_tokens"] is False
            return {
                "input_ids": fake_torch.tensor([[10, 20, 30]]),
                "attention_mask": fake_torch.tensor([[1, 1, 1]]),
            }

        def decode(self, generated, skip_special_tokens):
            calls["decoded"] = generated.tolist()
            assert skip_special_tokens
            return "  απάντηση  "

    class FakeModel:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 2
            assert kwargs["do_sample"] is False
            assert kwargs["input_ids"].tolist() == [[10, 20, 30]]
            return fake_torch.tensor([[10, 20, 30, 101, 102]])

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert name == "Qwen/Qwen2.5-0.5B-Instruct"
            assert kwargs["trust_remote_code"] is False
            return FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert kwargs["torch_dtype"] == fake_torch.float32
            assert kwargs["trust_remote_code"] is False
            return FakeModel()

    monkeypatch.setenv("LLM_DEVICE", "cpu")
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)
    monkeypatch.delenv("LLM_MODEL_REVISION", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeTokenizerFactory,
            AutoModelForCausalLM=FakeModelFactory,
        ),
    )
    generator = LocalGenerator()
    result = generator.generate("Μια μικρή δοκιμή", max_new_tokens=2)
    assert result == {
        "mode": "local_llm",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cpu",
        "generated_text": "απάντηση",
        "input_tokens": 3,
        "output_tokens": 2,
    }
    assert calls == {"eval": True, "decoded": [101, 102]}


def test_cuda_request_does_not_silently_fall_back(monkeypatch, fake_torch):
    from greek_nlp.llm.generator import LocalGenerator

    monkeypatch.setenv("LLM_DEVICE", "cuda")
    monkeypatch.setattr(fake_torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        LocalGenerator()


def test_local_generator_context_limit_supports_grounded_rag():
    """RAG's allowed token budget must stay below LocalGenerator's input bound."""
    from greek_nlp.llm.generator import MAX_INPUT_TOKENS
    from greek_nlp.rag.grounded import MAX_PROMPT_TOKENS

    assert MAX_INPUT_TOKENS == 1024
    assert 512 < MAX_PROMPT_TOKENS < MAX_INPUT_TOKENS
