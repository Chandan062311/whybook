# ORCHESTRATOR — WhyBook Implementation Master
## Gemma 4 Good Hackathon | Deadline: May 18, 2026

---

## Engineer Persona

You are a Staff ML Engineer executing a pre-designed project. Every decision has already been made. Your job is **execution with precision**, not design.

Rules you operate by:
- Read the relevant spec file before touching any code
- Validate every gate before moving to the next phase
- If something breaks, diagnose and fix — do not skip or workaround
- Do not add features not in the spec
- Do not change hyperparameters without documenting why
- One phase at a time. Never run Phase N+1 until Phase N gates pass

---

## Project Map

```
WhyBook/
├── docs/gemma4-hackathon/
│   ├── ORCHESTRATOR.md          ← you are here — master execution guide
│   ├── idea-spec.md             ← what and why (read first, always)
│   ├── synthetic-data-pipeline.md  ← Phase 1 spec
│   └── finetuning-spec.md       ← Phase 2 spec
└── data/
    └── ncert_chemistry_whybook.jsonl  ← generated in Phase 1, consumed in Phase 2
```

**Read order before implementing any phase:**
1. `idea-spec.md` — understand what you are building
2. The phase-specific spec file
3. This orchestrator — for gates and sequencing

---

## Project State Machine

Track current state here. Update before closing any session.

```
CURRENT PHASE : [ ] 0-Setup  [ ] 1-Data  [ ] 2-Finetune  [ ] 3-Demo  [ ] 4-Submit
CURRENT STEP  : ___
LAST GATE     : PASSED / FAILED / NOT RUN
BLOCKER       : none / describe issue
NEXT ACTION   : ___
```

---

## Phase 0 — Environment Setup
**Where:** Kaggle Notebook (New notebook → GPU T4 x2 → Internet ON)
**Spec file:** none — follow steps below exactly

### Steps

**0.1 — Verify GPU**
```python
import torch
print(torch.cuda.get_device_name(0))   # must print T4
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected: T4, 15.7 GB
```

**0.2 — Install Libraries**
```bash
!pip install unsloth trl datasets pdfplumber openai llama-cpp-python gradio -q
```

Install order matters. If unsloth install fails, run:
```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
```

**0.3 — Set OpenRouter API Key**
Add to Kaggle Secrets (not in code — never hardcode keys):
- Key name: `OPENROUTER_API_KEY`
- Value: your OpenRouter free key

```python
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient()
OPENROUTER_KEY = secret.get_secret("OPENROUTER_API_KEY")
```

**0.4 — Verify OpenRouter Connection**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

test = client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct:free",
    messages=[{"role": "user", "content": "Reply with the word: connected"}],
    max_tokens=10,
)
print(test.choices[0].message.content)   # must print: connected
```

**0.5 — Create Directory Structure**
```python
import os
os.makedirs("data", exist_ok=True)
os.makedirs("models/lora", exist_ok=True)
os.makedirs("models/merged", exist_ok=True)
os.makedirs("models/gguf", exist_ok=True)
```

### Phase 0 Gate
All 5 steps must print expected output.
**DO NOT proceed to Phase 1 until this gate passes.**

---

## Phase 1 — Synthetic Data Generation
**Spec file:** `synthetic-data-pipeline.md` — read fully before starting
**Input:** NCERT Chemistry PDFs (Class 9 + 10)
**Output:** `data/ncert_chemistry_whybook.jsonl`
**Target:** ≥ 90 valid records

### Steps

**1.1 — Download NCERT PDFs**

NCERT textbooks are public domain. Download from the official source:
```python
import urllib.request

NCERT_PDFS = {
    "class9":  "https://ncert.nic.in/textbook/pdf/iesc1dd.zip",   # Science Class 9
    "class10": "https://ncert.nic.in/textbook/pdf/jesc1dd.zip",   # Science Class 10
}

# Download and extract — check ncert.nic.in for correct URLs if these change
# Chemistry chapters are embedded in Science books for Class 9-10
```

If direct download fails, search `site:ncert.nic.in chemistry class 9 pdf` — the files are always public.

**1.2 — Extract Text by Chapter**
```python
import pdfplumber

def extract_chapters(pdf_path: str) -> dict[str, str]:
    """Returns {chapter_name: text} for chemistry chapters only."""
    chemistry_keywords = [
        "matter", "atom", "molecule", "acid", "base", "salt",
        "metal", "carbon", "chemical", "reaction", "element"
    ]
    chapters = {}
    with pdfplumber.open(pdf_path) as pdf:
        current_chapter = None
        current_text = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            # detect chapter headings — NCERT uses "Chapter N" pattern
            if text.strip().startswith("Chapter"):
                if current_chapter:
                    full_text = " ".join(current_text)
                    if any(kw in full_text.lower() for kw in chemistry_keywords):
                        chapters[current_chapter] = full_text
                current_chapter = text.strip().split("\n")[0]
                current_text = [text]
            else:
                current_text.append(text)
    return chapters
```

**1.3 — Extract Concept List Per Chapter**

Do not use an LLM for this step. Use a curated list — it is faster and more reliable:

```python
CONCEPTS = {
    # CLASS 9
    "Matter in Our Surroundings": [
        "evaporation", "sublimation", "diffusion", "Brownian motion",
        "latent heat", "condensation", "states of matter",
    ],
    "Is Matter Around Us Pure": [
        "mixture", "solution", "colloid", "suspension", "distillation",
        "chromatography", "compound", "element",
    ],
    "Atoms and Molecules": [
        "atom", "molecule", "atomic mass", "molecular mass", "mole",
        "Avogadro number", "chemical formula",
    ],
    "Structure of the Atom": [
        "electron", "proton", "neutron", "atomic number", "mass number",
        "isotope", "isobar", "valence shell",
    ],
    # CLASS 10
    "Chemical Reactions and Equations": [
        "oxidation", "reduction", "redox", "corrosion", "rancidity",
        "decomposition reaction", "displacement reaction",
    ],
    "Acids Bases and Salts": [
        "pH scale", "indicator", "neutralization", "NaCl", "H2SO4",
        "HCl", "NaOH", "baking soda", "bleaching powder",
    ],
    "Metals and Non-metals": [
        "reactivity series", "corrosion", "alloy", "ionic bond",
        "electrolytic refining", "thermite reaction",
    ],
    "Carbon and its Compounds": [
        "CH4 methane", "ethanol", "ethanoic acid", "soap", "detergent",
        "covalent bond", "isomer", "functional group", "hydrocarbon",
    ],
    "Periodic Classification": [
        "period", "group", "atomic radius trend", "Dobereiner triads",
        "Newlands octaves", "Mendeleev periodic law",
    ],
}
```

**1.4 — Generate Q&A Records**

Full system prompt and user prompt template are in `synthetic-data-pipeline.md`.
Copy them exactly — do not rephrase.

```python
import json, time

SYSTEM_PROMPT = """[COPY EXACTLY FROM synthetic-data-pipeline.md — Section: System Prompt]"""

def generate_record(client, chapter: str, concept: str,
                    class_num: str, chapter_text: str) -> dict | None:
    # Find the relevant chunk (300 tokens around the concept mention)
    idx = chapter_text.lower().find(concept.split()[0].lower())
    chunk = chapter_text[max(0, idx-500) : idx+1500] if idx > -1 else chapter_text[:2000]

    user_prompt = (
        f"TEXTBOOK PASSAGE:\n---\n{chunk}\n---\n\n"
        f"CONCEPT TO EXPLAIN: {concept}\n"
        f"CHAPTER: {chapter}\n"
        f"CLASS: {class_num}\n\n"
        f"Generate the JSON explanation now."
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        record = json.loads(raw)
        record["concept"] = concept
        record["chapter"] = chapter
        record["class"]   = class_num
        return record
    except Exception as e:
        print(f"  ERROR: {concept} — {e}")
        return None
```

**1.5 — Validate and Save**

Validation function is in `synthetic-data-pipeline.md` — Section: Quality Validation Rules.
Copy it exactly. Do not weaken any rule.

```python
OUTPUT_PATH = "data/ncert_chemistry_whybook.jsonl"
valid_records = []
rejected = []

for chapter, concepts in CONCEPTS.items():
    class_num = "9" if chapter in CLASS9_CHAPTERS else "10"
    chapter_text = extracted_chapters.get(chapter, "")

    for concept in concepts:
        print(f"Generating: {concept}...")
        record = generate_record(client, chapter, concept, class_num, chapter_text)

        if record is None:
            rejected.append({"concept": concept, "reason": "api_error"})
            continue

        is_valid, reason = validate_record(record)
        if is_valid:
            valid_records.append(record)
            # Save incrementally — do not lose progress on session drop
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        else:
            rejected.append({"concept": concept, "reason": reason})
            print(f"  REJECTED: {concept} — {reason}")

        time.sleep(6)  # rate limiting — do not remove

print(f"\nValid: {len(valid_records)} | Rejected: {len(rejected)}")
print(f"Rejection rate: {len(rejected)/(len(valid_records)+len(rejected))*100:.1f}%")
```

**1.6 — Manual Review (Required)**

Randomly sample 10 records. Read each one. Check:
- [ ] real_world mentions something a student would recognise (kitchen, bike, farm, market)
- [ ] what field does not copy textbook word-for-word
- [ ] why field explains pedagogy, not just repeats the definition
- [ ] language is simple (no "furthermore", "heretofore", "elucidates")

If more than 2 of 10 fail: re-run those concepts with temperature=0.2.

### Phase 1 Gate

| Check | Required | How to verify |
|---|---|---|
| Record count | ≥ 90 | `len(valid_records)` |
| Rejection rate | < 15% | `len(rejected) / total` |
| File exists | yes | `os.path.exists("data/ncert_chemistry_whybook.jsonl")` |
| Manual review | ≤ 2/10 fail | read 10 random records |

**DO NOT proceed to Phase 2 until all 4 checks pass.**

---

## Phase 2 — Fine-Tuning
**Spec file:** `finetuning-spec.md` — read fully before starting
**Input:** `data/ncert_chemistry_whybook.jsonl`
**Output:** `models/gguf/unsloth.Q4_K_M.gguf`

### Steps

Execute the code blocks in `finetuning-spec.md` in this exact order:

| Step | Code Section in finetuning-spec.md | Expected Output |
|---|---|---|
| 2.1 | Section 5 → Block 0 (Install) | no errors |
| 2.2 | Section 5 → Block 1 (Load Model) | model loaded, VRAM printed |
| 2.3 | Section 5 → Block 2 (LoRA) | trainable params < 1M |
| 2.4 | Section 5 → Block 3 (Dataset) | Train: ~99, Eval: ~11 |
| 2.5 | Section 5 → Blocks 4+5 (Train) | loss logs every 5 steps |
| 2.6 | Section 6 (Loss monitoring) | compare against expected trajectory |
| 2.7 | Section 7 (Inference test) | output matches good output example |
| 2.8 | Section 5 → Block 7 (Save LoRA) | `models/lora/` populated |
| 2.9 | Section 5 → Block 8 (GGUF) | `models/gguf/*.gguf` file exists |

**Do not skip step 2.7.** Inference test is the only way to know if fine-tuning worked before moving to the demo.

### Phase 2 Gate

| Check | Required | How to verify |
|---|---|---|
| Eval loss at best epoch | < 1.5 | training logs |
| Inference output format | follows what/why/real_world | visual inspection |
| Inference quality | real_world is specific, not abstract | read 3 test outputs |
| GGUF file exists | yes | `ls models/gguf/` |
| GGUF file size | 2–3 GB | `du -sh models/gguf/*.gguf` |

**DO NOT proceed to Phase 3 until all 5 checks pass.**

If eval loss > 1.5: consult Section 9 (Troubleshooting) in `finetuning-spec.md`.

---

## Phase 3 — Demo Notebook
**Spec file:** `idea-spec.md` → Section: 3-min Demo Script
**Input:** `models/gguf/unsloth.Q4_K_M.gguf`
**Output:** A complete, runnable Kaggle notebook that IS the submission

### The Demo Flow (Implement Exactly This — No Additions)

**Cell 1 — Title + Setup**
```python
# WhyBook — Chemistry Context Companion for Offline Learners
# Gemma 4 Good Hackathon | Kaggle × Google DeepMind
# Fine-tuned on NCERT Chemistry using QLoRA 4-bit | Zero paid compute

!pip install llama-cpp-python gradio -q
from llama_cpp import Llama
```

**Cell 2 — Load Model**
```python
llm = Llama(
    model_path="models/gguf/unsloth.Q4_K_M.gguf",
    n_ctx=512,
    n_gpu_layers=-1,
    verbose=False,
)
print("Model loaded. VRAM used: fine-tuned Gemma 4 E4B at 4-bit.")
print("This model runs on a ₹8,000 Android phone. No internet after download.")
```

**Cell 3 — Demo: Base Model vs Fine-Tuned (The Contrast)**
```python
# Same concept. Same model architecture. Different training.
# Judges see the gap. This is the core of the submission.

concept = "CH4 (Methane)"
chapter = "Carbon and its Compounds"
class_num = "10"

prompt_base    = build_prompt(concept, chapter, class_num)
prompt_tuned   = build_prompt(concept, chapter, class_num)  # same prompt, different model

# Load base Gemma 4 E4B (no fine-tune) — show output
# Load fine-tuned GGUF — show output
# Print both side by side
```

**Cell 4 — Language Switch**
```python
# Same concept. Output in Marathi.
# Show: no retraining. No code change. Just language instruction.

prompt_marathi = build_prompt(concept, chapter, class_num, language="Marathi")
response = llm(prompt_marathi, max_tokens=300, temperature=0.3)
print(response["choices"][0]["text"])
```

**Cell 5 — Gradio UI (Interactive Demo)**
```python
import gradio as gr

LANGUAGES = ["English", "Hindi", "Marathi", "Bengali", "Tamil"]

def whybook_explain(concept_input: str, language: str) -> str:
    prompt = build_prompt(concept_input, language=language)
    response = llm(prompt, max_tokens=300, temperature=0.3,
                   stop=["<end_of_turn>"])
    return response["choices"][0]["text"]

demo = gr.Interface(
    fn=whybook_explain,
    inputs=[
        gr.Textbox(label="Enter concept or formula",
                   placeholder="e.g. H2SO4, photosynthesis, Newton's law"),
        gr.Dropdown(choices=LANGUAGES, value="Hindi", label="Language"),
    ],
    outputs=gr.Textbox(label="WhyBook Explanation"),
    title="WhyBook — Chemistry Context Companion",
    description=(
        "Powered by Gemma 4 E4B fine-tuned on NCERT Chemistry. "
        "Runs offline. No internet needed. "
        "Built for 250M students studying without a tutor."
    ),
    examples=[
        ["CH4 (Methane)", "Hindi"],
        ["H2SO4 (Sulphuric Acid)", "Marathi"],
        ["NaCl (Sodium Chloride)", "Bengali"],
    ],
)

demo.launch(share=True)   # share=True gives a public Gradio URL for the video
```

**Cell 6 — Personal Close**
```python
print("""
I built this because I was this student.

2018. Tier-3 city. 2G internet. NCERT Chemistry textbook.
I memorized CH4 without ever knowing it was in my kitchen.

WhyBook is the tutor I didn't have.

Fine-tuned on Kaggle free tier. Zero paid compute.
Runs on a ₹8,000 Android phone with no internet.

Education should be free and accessible to everyone.
""")
```

### Phase 3 Gate

| Check | Required |
|---|---|
| All cells run top to bottom without error | yes |
| Gradio UI launches and responds | yes |
| Hindi output is readable and correct | yes |
| Notebook is clean (no debug prints, no failed cells) | yes |

**DO NOT record the video until all 4 checks pass.**

---

## Phase 4 — Submission
**Spec file:** `idea-spec.md` → Sections: 3-min Demo Script + Impact Narrative
**Deadline: May 18, 2026 — do not cut this close**

### Steps

**4.1 — Make Notebook Public**
Kaggle → Notebook settings → Visibility: Public
Verify the public URL loads correctly in a private browser window.

**4.2 — Record the Video (3 minutes max)**

Follow the exact script from `idea-spec.md` → Section: 3-min Demo Script.

| Time | Action |
|---|---|
| 0:00–0:20 | Show notebook, say the "₹8,000 phone" line |
| 0:20–0:50 | Run Cell 3 — show CH4 explained in Hindi |
| 0:50–1:30 | Run Cell 3 base vs fine-tuned contrast |
| 1:30–2:15 | Run Cell 4 — switch to Marathi |
| 2:15–3:00 | Read Cell 6 personal close — naturally, not scripted |

Upload to YouTube (unlisted or public). Copy the URL.

**4.3 — Write Kaggle Submission Writeup**

Title: `WhyBook — NCERT Chemistry Explained for Offline Learners`
Subtitle: `Fine-tuned Gemma 4 E4B on Kaggle free tier. Runs on any Android phone. No internet needed.`

Writeup structure (max 1,500 words):
1. The problem (150 words) — use the personal story
2. Why Gemma 4 specifically (200 words) — use the "Why Now" section from idea-spec.md
3. Technical approach (400 words) — data pipeline → fine-tuning → GGUF → Gradio
4. Results (300 words) — show base vs fine-tuned comparison outputs
5. Impact and next steps (200 words) — use the Impact Narrative from idea-spec.md
6. Reproducibility note (150 words) — all code in notebook, all compute free, all data public domain

**4.4 — Submit**

Kaggle submission form:
- Notebook URL: [your public notebook]
- YouTube URL: [your video]
- Writeup: paste from 4.3

Submit at least 24 hours before May 18 deadline. Kaggle submission queues can be slow.

### Phase 4 Gate

| Check | Required |
|---|---|
| Notebook public URL works | yes |
| Video is under 3 minutes | yes |
| Submission confirmed on Kaggle | yes |
| Submitted before May 17 EOD | yes (buffer) |

---

## Non-Negotiable Rules

These cannot be changed without updating the relevant spec file and documenting the reason:

1. **No paid compute.** Every step runs on Kaggle free tier or Colab free tier.
2. **No paid APIs.** OpenRouter free key only. If rate limited, wait — do not upgrade.
3. **No features beyond the spec.** WhyBook explains what/why/real_world. Nothing else.
4. **No skipping gates.** A failing gate is information. Fix it before proceeding.
5. **No hardcoded API keys.** Use Kaggle Secrets. Always.
6. **No model other than Gemma 4 E4B.** The hackathon story depends on it.
7. **NCERT data only.** Public domain. No proprietary datasets. No scraping.

---

## Deviation Protocol

If you MUST deviate from the spec:

1. Stop. Do not implement the deviation.
2. Document: what the spec says, what reality requires, why they differ.
3. Update the relevant spec file with the change and the reason.
4. Confirm the deviation does not break any gate.
5. Then implement.

Example: if Gemma 4 E4B is not yet available on HuggingFace when you start, the deviation is: use `google/gemma-3-4b-it` temporarily and re-run with E4B once available. Document this in `finetuning-spec.md`.

---

## Timeline Recommendation

| Week | Dates | Work |
|---|---|---|
| Week 1 | Apr 3–10 | Phase 0 + Phase 1 (data generation) |
| Week 2 | Apr 11–18 | Phase 2 (fine-tuning + iterate) |
| Week 3 | Apr 19–30 | Phase 3 (demo notebook) |
| Week 4 | May 1–10 | Polish + Phase 4 prep |
| Buffer | May 11–17 | Fix unexpected issues |
| **Deadline** | **May 18** | **Submit** |

Do not let Phase 2 slip past April 25. Fine-tuning often needs 2-3 iterations to get right. Give yourself time.
