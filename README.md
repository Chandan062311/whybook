# WhyBook: A Domain-Specific Tutor for NCERT Chemistry

WhyBook is a practical, low-cost AI tutor built to help students bridge the gap between *what* a concept is and *why* it matters. It reads NCERT chemistry chapters and generates structured, bite-sized explanations that connect theory to the real world.

This project was developed for the E2B Data-Centric AI Competition, demonstrating a complete notebook-first workflow from synthetic data generation to a publicly deployed model.

## Core Idea

Textbooks are great at explaining the "what," but students often struggle with the "why" and "where." WhyBook tackles this by providing answers in a simple, three-part format:

- **What:** A concise definition.
- **Why:** The core reason the concept is important.
- **Real World:** A tangible example of where the concept appears in everyday life.

## Key Achievements

- **High-Quality Synthetic Data:** Generated a custom dataset of 100+ high-quality, validated `(What, Why, Real World)` examples directly from NCERT chemistry PDFs.
- **Efficient Fine-Tuning:** Fine-tuned the `google/gemma-2b-it` model using QLoRA on a single Kaggle T4 GPU, achieving strong performance with minimal resources.
- **Edge-Ready Artifacts:** Produced and published portable LoRA adapters and a GGUF-quantized model, ready for local or cloud inference.
- **Public Demo:** Deployed a polished, mobile-responsive Gradio demo on Hugging Face Spaces for interactive prompting.

## Technical Workflow

The project is executed in two main phases, managed by specification documents.

### 1. Synthetic Data Generation

- **Source:** NCERT Class 11 & 12 Chemistry PDFs.
- **Schema:** Each record strictly follows `{concept, chapter, class, what, why, real_world}`.
- **Validation:** A rigorous pipeline enforces quality gates, including word-count bounds, removal of boilerplate language, and checks for semantic duplication.

### 2. Fine-Tuning and Export

- **Base Model:** `google/gemma-2b-it`
- **Technique:** QLoRA via the Unsloth library for memory-efficient training.
- **LoRA Config:** `r=16`, `alpha=32`, targeting all major projection layers.
- **Export:** LoRA adapters were merged and the final model was quantized to `Q4_K_M` GGUF format for broad compatibility.

## Hugging Face Assets

- **Dataset:** [Stinger2311/whybook-chemistry-dataset](https://huggingface.co/datasets/Stinger2311/whybook-chemistry-dataset)
- **LoRA Adapters:** [Stinger2311/whybook-gemma-2b-it-lora](https://huggingface.co/Stinger2311/whybook-gemma-2b-it-lora)
- **GGUF Model:** [Stinger2311/whybook-gemma-2b-it-gguf](https-huggingface.co/Stinger2311/whybook-gemma-2b-it-gguf)
- **Gradio Demo:** [Stinger2311/whybook-gradio-demo](https://huggingface.co/spaces/Stinger2311/whybook-gradio-demo)

## Repository Structure

This is a notebook-first project. The core logic and run order are defined in the specification documents and Python/notebook files.

- `idea-spec.md`: The initial concept and project goals.
- `synthetic-data-pipeline.md`: The spec for Phase 1 data generation.
- `finetuning-spec.md`: The spec for Phase 2 training and export.
- `ORCHESTRATOR.md`: Defines the phase gates and execution order.
- `create_synthetic_data.py`: The script for generating the dataset.
- `whybook_phase2_finetune_kaggle.ipynb`: The main Kaggle notebook for fine-tuning.
- `index.html`: A simple, public-facing project showcase page.
- `whybook-gradio-demo-space/`: The source code for the Gradio demo.

For local agent-based development, please adhere to the guidelines in `AGENTS.md`.
