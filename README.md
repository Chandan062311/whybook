# WhyBook

WhyBook is a chemistry learning assistant built for students who study from physical NCERT books with limited internet access.

It focuses on one learning gap: textbooks tell students *what* a concept is, but students often miss *why* it matters and *where* it appears in real life.

WhyBook answers in a structured format:
- `What`
- `Why`
- `Real World`

## Project Objective

- Build a practical, low-cost, notebook-first pipeline for a domain-specific tutor.
- Generate high-quality synthetic data from NCERT chemistry content.
- Fine-tune a compact Gemma-based model using QLoRA on Kaggle T4.
- Export edge-friendly artifacts (LoRA + GGUF).
- Publish a public demo Space for prompt-based student interaction.

## Core Pipeline

### 1) Synthetic Data (Phase 1)

Records use this schema exactly:
- `concept`
- `chapter`
- `class`
- `what`
- `why`
- `real_world`

Validation checks enforce:
- required fields and minimum quality,
- word-count bounds,
- removal of refusal/boilerplate language,
- non-duplicative sections.

### 2) Fine-Tuning (Phase 2)

- Base model: `google/gemma-4-E2B-it`
- Training approach: Unsloth + QLoRA (Kaggle GPU workflow)
- LoRA configuration:
  - `r=16`
  - `alpha=32`
  - `dropout=0.05`
  - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### 3) Export and Hosting

- Save LoRA adapters.
- Merge and quantize to GGUF (`Q4_K_M`) for local inference.
- Publish artifacts and demo on Hugging Face.

## Hugging Face Assets

- Dataset: `Stinger2311/whybook-chemistry-dataset`
- LoRA: `Stinger2311/whybook-gemma4-e2b-lora`
- GGUF: `Stinger2311/whybook-gemma4-e2b-gguf`
- Demo Space: `Stinger2311/whybook-gradio-demo`

## Repository Structure

- `idea-spec.md` - concept and hackathon framing
- `synthetic-data-pipeline.md` - Phase 1 generation and validation
- `finetuning-spec.md` - Phase 2 training and export spec
- `ORCHESTRATOR.md` - phase gates and run order
- `create_synthetic_data.py` - dataset generation script
- `whybook_phase2_finetune_kaggle.ipynb` - main Kaggle fine-tuning notebook
- `notebook3db17cc3c5.ipynb` - updated fine-tuning notebook variant
- `index.html` - public project showcase page
- `.stitch/DESIGN.md` - design system used for the public UI

## Run Notes

- This is a notebook-first project, not a packaged Python app.
- For local agent work, follow `AGENTS.md` and phase specs before editing.
- Keep prompts, schema keys, artifact paths, and key hyperparameters stable unless intentionally changing them.

## Status

- End-to-end workflow implemented from data to public demo.
- Demo UI is prompt-first, styled for public presentation, and deployed on HF Space.
