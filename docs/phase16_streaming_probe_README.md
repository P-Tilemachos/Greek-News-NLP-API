# Phase 16.3 — GGUF streaming probe (development-only)

This is a **local experimental benchmark**, not a FastAPI production streaming endpoint.
It uses exactly the existing `dev_generate_greeting` user prompt, `max_new_tokens=48`,
and `DEFAULT_SYSTEM_PROMPT` from `greek_nlp.llm.generator`.

1. Stop the PyTorch Uvicorn server if unnecessary, so it does not compete for RAM.
2. In one Anaconda Prompt launch llama.cpp on port 9931:

   ```bat
   llama serve -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --host 127.0.0.1 --port 9931 -t 8 -c 2048 -ngl 0
   ```

3. In a second prompt, from the project root:

   ```bat
   python scripts\phase16_streaming_probe.py
   python scripts\phase16_streaming_probe.py --warmup 1 --repeat 3 --execute
   ```

4. Inspect `results/phase16_local/phase16_gguf_streaming_DEV_ONLY.json`.
   It is deliberately not tracked by Git.

The script records time to first **visible content**, not guaranteed time to first
internal model token. No output text is saved. SSE chunks are not individual tokens.
It does not measure server peak RAM or quality. Do not compare its first-content
latency to PyTorch until equivalent streaming has been implemented on that route.

If rerunning, **back up or rename** the prior result before repeating; the script
refuses to overwrite benchmark records.
