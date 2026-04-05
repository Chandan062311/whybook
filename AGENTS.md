# AGENTS.md

Operating guide for agentic coding assistants in this repository.

## 1) Repository Snapshot
- Purpose: execute the WhyBook hackathon workflow from specs.
- State: documentation-first repo; no packaged Python app, CI, or formal test suite yet.
- Core docs: `idea-spec.md`, `synthetic-data-pipeline.md`, `finetuning-spec.md`, `ORCHESTRATOR.md`.

## 2) Required Read Order
Before changing code, notebooks, or docs for a phase, read:
1. `idea-spec.md`
2. Relevant phase spec: `synthetic-data-pipeline.md` or `finetuning-spec.md`
3. `ORCHESTRATOR.md`

If instructions conflict, prefer the phase spec plus `ORCHESTRATOR.md` over assumptions.

## 3) Cursor/Copilot Rules Audit
Checked for additional IDE agent rules:
- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

Implication: this file and the four spec files above are the only local agent policy sources.

## 4) Execution Model
- Treat the repo as Kaggle-notebook-first.
- Follow orchestrator phase gates strictly.
- Do not jump to a later phase until the current gate passes.
- Do not redesign prompts, schema keys, or hyperparameters unless the user explicitly asks.
- If implementation reality forces a change, document the deviation and follow `ORCHESTRATOR.md`.

## 5) Build Commands
There is no compiled build target. In practice, “build” means producing the required phase artifacts.

### Environment setup
Primary install:
```bash
pip install unsloth trl datasets pdfplumber openai llama-cpp-python gradio -q
```

Fallback if `unsloth` install fails:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
```

Minimal fine-tuning install:
```bash
pip install unsloth trl datasets -q
```

### Practical build milestones
1. Generate `data/ncert_chemistry_whybook.jsonl`
2. Train LoRA adapters
3. Save merged model
4. Export GGUF at `models/gguf/unsloth.Q4_K_M.gguf`

## 6) Lint And Format Commands
No formatter or linter is configured in-repo today.

If you add Python files, use Ruff unless the user specifies otherwise:
```bash
pip install ruff
python -m ruff check .
python -m ruff format .
```

If only docs change, no lint step is required.

## 7) Test And Validation Commands
There is no `pytest` suite yet. Use the spec-defined validation and smoke checks instead.

### Phase 0 checks
- Verify GPU is a T4 and VRAM prints correctly.
- Verify OpenRouter connectivity returns `connected`.
- Verify directories exist: `data/`, `models/lora/`, `models/merged/`, `models/gguf/`.

### Phase 1 checks
- Validate every generated record with `validate_record(record)`.
- Confirm the dataset file exists:
```python
import os
print(os.path.exists("data/ncert_chemistry_whybook.jsonl"))
```
- Confirm gate metrics: valid record count `>= 90`, rejection rate `< 15%`, manual review failures `<= 2/10`.

### Phase 2 checks
- Confirm the split is roughly `Train: 99 | Eval: 11` for the MVP dataset.
- Confirm best `eval_loss < 1.5`.
- Run inference before GGUF export.
- Confirm the GGUF exists and is roughly 2-3 GB.

## 8) Single-Test Guidance
Because there is no formal test runner, a “single test” means one narrow validation or smoke check.

Preferred single checks:
1. Validate one generated record:
```python
ok, reason = validate_record(one_record)
print(ok, reason)
```

2. Check one gate condition:
```python
import os
print(os.path.exists("data/ncert_chemistry_whybook.jsonl"))
```

3. Run one inference probe before export:
```python
FastLanguageModel.for_inference(model)
# run one prompt such as NaCl and inspect the what/why/real_world sections
```

4. Verify one artifact directly from shell:
```bash
ls models/gguf
du -sh models/gguf/*.gguf
```

If a real test suite is added later, update this section with exact single-test commands such as `pytest path/to/test_file.py::test_name`.

## 9) Code Style Guidelines
### General
- Implement the spec, not a broader redesign.
- Keep changes minimal, phase-scoped, and easy to trace.
- Prefer explicit code over clever abstractions.
- Preserve documented prompts, schema keys, paths, and hyperparameters unless intentionally changing them.

### Python version and types
- Target Python 3.10+ style typing.
- Use type hints for public functions and reusable helpers.
- Prefer concrete return types like `dict[str, str]`, `list[dict]`, `tuple[bool, str]`.
- Use `None` explicitly for nullable outcomes and make callers handle it.

### Imports
- Order imports as standard library, third-party, then local imports.
- Keep imports at file top unless notebook-cell constraints require local imports.
- Remove unused imports immediately.

### Naming
- Use `snake_case` for variables, functions, and modules.
- Use `UPPER_SNAKE_CASE` for constants.
- Prefer descriptive domain names such as `chapter_text`, `valid_records`, `class_num`.
- Avoid one-letter names except small loop indices.

### Formatting
- Follow PEP 8 defaults.
- Keep functions small and single-purpose where practical.
- Split long strings and f-strings for readability.
- Keep quote style consistent within a file.

### Data and schema rules
- Preserve these JSON keys exactly: `concept`, `chapter`, `class`, `what`, `why`, `real_world`.
- Do not rename fields without updating all downstream formatting and inference steps.
- Validate each record before saving.
- Save incrementally during generation so Kaggle session loss does not destroy progress.

### Prompting and model I/O
- Copy required prompts exactly from the phase spec when the spec says “do not change”.
- Keep generation temperature low as documented, typically `0.2` to `0.3`.
- Enforce structured outputs where specified with `response_format={"type": "json_object"}`.
- Do not silently switch JSON outputs to another format, or vice versa.

### Error handling
- Wrap API and filesystem operations that can fail in `try/except`.
- Return structured failure states like `None` or `(False, reason)` instead of crashing long batch jobs.
- Log concise, actionable errors including the concept or artifact that failed.
- Use retry/backoff for rate limits and transient API failures.

### Reproducibility and security
- Preserve documented seeds such as `42`.
- Do not change LoRA rank, alpha, LR, dropout, batch size, or sequence length casually.
- If you must change a training parameter, document why and the expected effect.
- Never hardcode API keys.
- Use Kaggle Secrets for `OPENROUTER_API_KEY`.
- Do not commit tokens, notebook outputs with secrets, or local credentials.

## 10) Completion Checklist
Before finishing work:
1. Re-read the applicable spec section.
2. Confirm no prompt, schema, path, or hyperparameter drift.
3. Run the smallest relevant validation or smoke check.
4. Verify expected artifact paths and names.
5. Summarize what changed and any remaining risk.
