<div align="center">
  
# 📚 WhyBook
*A Next-Generation Domain-Specific AI Tutor for NCERT Chemistry*

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/Stinger2311/whybook-chemistry-dataset)
[![Model](https://img.shields.io/badge/Model-GGUF-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/Stinger2311/whybook-gemma4-e2b-gguf)
[![Demo](https://img.shields.io/badge/Demo-Gradio_Space-green?style=for-the-badge&logo=gradio)](https://huggingface.co/spaces/Stinger2311/whybook-gradio-demo)

</div>

<br/>

> **"Textbooks tell you the *What*. WhyBook tells you the *Why* and *Where*."**

WhyBook is a practical, low-cost AI tutor built to help students bridge the conceptual gap in traditional learning. Designed specifically for students who study from physical NCERT books, WhyBook reads complex chemistry chapters and generates structured, bite-sized explanations that connect abstract theory directly to the real world.

---

## 🌟 The WhyBook Edge

Standard textbooks are excellent at explaining definitions, but students often struggle to understand the underlying significance and real-world applications of what they are learning. WhyBook tackles this exact problem by answering queries in a strict, high-impact three-part format:

- 📖 **What:** A concise, clear definition of the concept.
- 🧠 **Why:** The fundamental scientific reason the concept is important.
- 🌍 **Real World:** A tangible, everyday example of where this concept appears in life.

---

## 🚀 Powered by `gemma-4-e2b`

WhyBook is built on top of the absolute best lightweight model for educational domains: **`gemma-4-e2b`**. 

We chose `gemma-4-e2b` because it delivers state-of-the-art reasoning and instructional capabilities while remaining incredibly efficient. Through rigorous fine-tuning, this model has been adapted to become a highly accurate, domain-specific chemistry tutor.

### ⚙️ Fine-Tuning Excellence
- **Base Model:** `gemma-4-e2b` – The pinnacle of compact, high-performance educational LLMs.
- **Optimization:** We used **QLoRA** via the **Unsloth** library for blazing-fast, memory-efficient training on a single Kaggle T4 GPU.
- **LoRA Configuration:** `r=16`, `alpha=32`, meticulously targeting all major projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) to extract maximum performance.
- **Edge-Ready Export:** The LoRA adapters were seamlessly merged, and the final model was quantized into the `Q4_K_M` GGUF format for ultra-fast, local inference on consumer hardware.

---

## 📊 High-Quality Data Generation

Before fine-tuning, we engineered a custom, fully validated synthetic dataset.

- **Source Material:** NCERT Class 11 & 12 Chemistry PDFs.
- **Strict Schema:** Every single record strictly adheres to `{concept, chapter, class, what, why, real_world}`.
- **Validation Gates:** A rigorous automated pipeline enforces word-count bounds, strips out AI boilerplate/refusals, and guarantees zero semantic duplication across sections.

*(Note: The correct and final dataset used for this project is `data.jsonl`, which is included in this repository.)*

---

## 🔗 Hugging Face Assets

Explore the full suite of WhyBook's published artifacts:

| Asset | Link |
|-------|------|
| 📝 **Dataset** | [Stinger2311/whybook-chemistry-dataset](https://huggingface.co/datasets/Stinger2311/whybook-chemistry-dataset) |
| 🧩 **LoRA Adapters** | [Stinger2311/whybook-gemma4-e2b-lora](https://huggingface.co/Stinger2311/whybook-gemma4-e2b-lora) |
| ⚡ **GGUF Model** | [Stinger2311/whybook-gemma4-e2b-gguf](https://huggingface.co/Stinger2311/whybook-gemma4-e2b-gguf) |
| 🎨 **Interactive Demo** | [Stinger2311/whybook-gradio-demo](https://huggingface.co/spaces/Stinger2311/whybook-gradio-demo) |

---

## 📁 Repository Structure

This project was developed with a notebook-first workflow. The core logic, generation pipelines, and execution order are cleanly defined in our specification documents and scripts:

```text
├── data/
│   └── data.jsonl                         # The final, validated dataset
├── idea-spec.md                           # Initial concept and project goals
├── synthetic-data-pipeline.md             # Spec for Phase 1 data generation
├── finetuning-spec.md                     # Spec for Phase 2 training and export
├── ORCHESTRATOR.md                        # Defines the phase gates and execution order
├── create_synthetic_data.py               # Dataset generation script
├── whybook_phase2_finetune_kaggle.ipynb   # Main Kaggle fine-tuning notebook
├── index.html                             # Public-facing project showcase page
└── whybook-gradio-demo-space/             # Source code for the interactive Gradio demo
```

*For local agent-based development, please adhere to the guidelines in `AGENTS.md`.*
