# WhyBook — Fine-Tuning Spec
## Gemma 4 E2B × Unsloth × QLoRA × Kaggle 2×T4

> **DEVIATION (2026-04-04):** Changed from E4B → E2B per user request.
> - E2B: ~5.1B total params, ~2.3B effective (smaller than E4B's ~8B total / ~4.5B effective)
> - E2B fits more comfortably on a single T4 with even more VRAM headroom
> - All LoRA hyperparameters (r=16, α=32, dropout=0.05) remain unchanged
> - HuggingFace model ID: `google/gemma-4-E2B-it`

---

## What This File Is

Every hyperparameter decision in this file is derived from the LoRA/QLoRA reference truth table. The reasoning chain is shown for each decision so you can adjust confidently when the training behaves unexpectedly.

---

## 1. Memory Budget (Know This Before Writing a Single Line of Code)

**Hardware:** Kaggle 2×T4 = 32GB VRAM total

| Component | Precision | Memory |
|---|---|---|
| Gemma 4 E2B base weights (~5.1B total params) | 4-bit | ~1.5 GB |
| LoRA adapters A+B (r=16, 7 modules) | FP16 | ~0.04 GB |
| Optimizer states (AdamW, LoRA params only) | FP32 | ~0.15 GB |
| Activations (batch=2, seq=512) | FP16 | ~0.4 GB |
| Gradient checkpointing savings | — | −60% activation memory |
| **Total (one T4)** | | **~2.1 GB** |

**You are using ~13% of one T4.** You have enormous headroom. If training is slow, increase `per_device_train_batch_size` to 4 or 8.

**Why not use both T4s?** Unsloth on Kaggle runs on a single GPU by default. Multi-GPU requires `accelerate` + DDP config. For 110 records, one T4 is fast enough — don't add complexity.

---

## 2. Hyperparameter Decisions

Every value below is justified against your specific constraints: **~110 records, 3 output fields, medium task complexity, Kaggle T4.**

### Rank (r = 16)

From the decision tree:
```
Dataset size: 110 records → borderline < 100
Task: medium complexity (reasoning about why concepts exist)
→ r = 16
```

Why not r=8: 110 records is a small dataset. r=8 gives too few parameters to capture the "what/why/real_world" pattern reliably — the model will underfit.

Why not r=32: risk of overfitting on 110 examples. The validation loss will likely diverge from training loss.

**Trainable parameters at r=16:**
```
Per module: (4096×16) + (16×4096) = 131,072 params
7 modules:   131,072 × 7          = 917,504 params
Total:       917,504 / 4,000,000,000 = 0.023% of model
```

### Alpha (α = 32)

From the reference:
```
Small dataset (< 200 examples) → α = 2r → α = 32
```

Why: small datasets need stronger LoRA updates to learn the pattern in fewer steps. α=32 with r=16 gives a scale factor of 2.0 — double the default. If the model memorizes training data (loss → 0 but outputs repeat exactly), drop to α=16.

### Learning Rate (2e-4)

Standard LoRA learning rate. The update formula:
```
Final update = LR × (α/r) × (B × A)
             = 2e-4 × (32/16) × (B × A)
             = 2e-4 × 2.0 × (B × A)
```

Effective update speed is 2× the learning rate due to α scaling. If loss diverges (goes up), drop LR to 1e-4.

### Epochs (3)

```
Dataset: 110 records → small → epochs = 3
Steps per epoch: 110 / (batch=2 × accum=4) = ~14 steps
Total steps: 14 × 3 = ~42 steps
```

42 steps is short. The model needs 3 passes to learn the pattern. If validation loss starts rising after epoch 2, stop at 2 (use `load_best_model_at_end=True`).

### Batch Size + Gradient Accumulation

```
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
→ effective batch size = 2 × 4 = 8
```

Effective batch of 8 gives stable gradient estimates. You have headroom to raise `per_device_train_batch_size` to 4 if training is slow — then drop `gradient_accumulation_steps` to 2 to keep effective batch = 8.

### Sequence Length (512)

Your Q&A pairs are short:
```
Instruction prompt:  ~60 tokens
JSON output:         ~200-250 tokens
Total per sample:    ~310 tokens
```

512 is safe headroom. Do NOT set this to 2048 — it wastes memory and slows training for no reason on this dataset.

### Dropout (0.05)

Reference says 0.0 for small datasets. Using 0.05 here as light regularization because:
- 110 records is very small
- 3 epochs × r=16 × α=32 is a fairly aggressive config
- 0.05 adds noise to LoRA outputs during training, reducing memorization risk

If loss seems noisy or converges slowly, drop to 0.0.

### Warmup Steps (5)

```
Total steps: ~42
Warmup: 5 steps ≈ 12% of training
```

Warmup prevents large gradient updates at the start when weights are mismatched. 5 steps is conservative and safe.

---

## 3. Target Modules — All 7

Apply LoRA to all 7 weight matrices in each transformer layer:

```python
target_modules = [
    "q_proj",    # Q = X × W_q  — what am I looking for?
    "k_proj",    # K = X × W_k  — what do I contain?
    "v_proj",    # V = X × W_v  — what do I give out?
    "o_proj",    # O = concat(heads) → projection
    "gate_proj", # SwiGLU gate — controls neuron activation
    "up_proj",   # expands to higher dimension
    "down_proj", # compresses back to model dimension
]
```

Do not remove any of these. For a domain-shift task (teaching the model a new output format + chemistry context), all 7 modules need to adapt.

---

## 4. Data Format for Unsloth

Unsloth's SFTTrainer expects **chat-formatted data** using the model's native chat template.

### Gemma 4 Chat Template

```
<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

### Convert Your JSONL to This Format

```python
def format_record(record: dict) -> dict:
    instruction = (
        f"You are WhyBook, an NCERT Chemistry tutor for Indian students.\n"
        f"A student wants to understand this concept: "
        f"{record['concept']} from {record['chapter']} (Class {record['class']}).\n"
        f"Explain it clearly using the WhyBook format — "
        f"what it is, why it is in the textbook, and where the student will see it in real life."
    )

    output = (
        f"What it is:\n{record['what']}\n\n"
        f"Why it is in your textbook:\n{record['why']}\n\n"
        f"Where you will see it in real life:\n{record['real_world']}"
    )

    return {
        "messages": [
            {"role": "user",    "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }
```

**Why not raw JSON output?** JSON output format requires the model to learn both the content AND the syntax. Plain structured text is easier to learn on 110 examples and equally readable for the demo. Switch to JSON if you want programmatic parsing later.

---

## 5. Full Unsloth Training Code

```python
# ─── 0. Install ─────────────────────────────────────────────────────────────
# !pip install unsloth trl datasets -q

# ─── 1. Load Model ──────────────────────────────────────────────────────────
from unsloth import FastLanguageModel

MODEL_ID      = "google/gemma-4-e4b-it"   # confirm on HuggingFace after release
MAX_SEQ_LEN   = 512
LOAD_IN_4BIT  = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,          # auto-detect: bf16 on A100, fp16 on T4
    load_in_4bit   = LOAD_IN_4BIT,
)

# ─── 2. Attach LoRA Adapters ────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = 32,
    lora_dropout        = 0.05,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth's optimized version — DO NOT change
    random_state        = 42,
    use_rslora          = False,
    loftq_config        = None,
)

# ─── 3. Load + Format Dataset ───────────────────────────────────────────────
import json
from datasets import Dataset

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def format_record(record: dict) -> dict:
    instruction = (
        f"You are WhyBook, an NCERT Chemistry tutor for Indian students.\n"
        f"A student wants to understand: {record['concept']} "
        f"from {record['chapter']} (Class {record['class']}).\n"
        f"Explain clearly — what it is, why it is taught, and where the student will see it in real life."
    )
    output = (
        f"What it is:\n{record['what']}\n\n"
        f"Why it is in your textbook:\n{record['why']}\n\n"
        f"Where you will see it in real life:\n{record['real_world']}"
    )
    return {"messages": [
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": output},
    ]}

raw_data  = load_jsonl("data/ncert_chemistry_whybook.jsonl")
formatted = [format_record(r) for r in raw_data]

# 90/10 train-eval split
split     = int(0.9 * len(formatted))
train_ds  = Dataset.from_list(formatted[:split])
eval_ds   = Dataset.from_list(formatted[split:])

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
# Expected: Train: 99 | Eval: 11

# ─── 4. Training Arguments ──────────────────────────────────────────────────
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir                  = "whybook-e4b-checkpoints",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 4,        # effective batch = 8
    warmup_steps                = 5,
    learning_rate               = 2e-4,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 5,        # log every 5 steps (you have ~42 total)
    evaluation_strategy         = "epoch",  # evaluate after each epoch
    save_strategy               = "epoch",
    load_best_model_at_end      = True,     # auto-select best checkpoint
    metric_for_best_model       = "eval_loss",
    optim                       = "adamw_8bit",   # Unsloth's memory-efficient optimizer
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    seed                        = 42,
    report_to                   = "none",   # disable wandb/tensorboard — not needed
)

# ─── 5. Trainer ─────────────────────────────────────────────────────────────
import torch

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    max_seq_length  = MAX_SEQ_LEN,
    dataset_num_proc = 2,
    args            = training_args,
)

# ─── 6. Train ───────────────────────────────────────────────────────────────
trainer_stats = trainer.train()
print(f"Training complete. Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ─── 7. Save + Merge ────────────────────────────────────────────────────────
# Save LoRA adapters only (lightweight — for resuming or sharing)
model.save_pretrained("whybook-e4b-lora")
tokenizer.save_pretrained("whybook-e4b-lora")

# Merge + save full model in FP16 (needed before GGUF export)
model.save_pretrained_merged("whybook-e4b-merged", tokenizer, save_method="merged_16bit")

# ─── 8. Quantize to GGUF ────────────────────────────────────────────────────
# Q4_K_M = 4-bit quantization with medium quality. Best balance for on-device.
model.save_pretrained_gguf("whybook-e4b-gguf", tokenizer, quantization_method="q4_k_m")
# Output: whybook-e4b-gguf/unsloth.Q4_K_M.gguf — use this with llama.cpp
```

---

## 6. What to Watch During Training

You have ~42 total steps. Print logs every 5 steps. Here's what each pattern means:

| Pattern | What it means | Action |
|---|---|---|
| Loss drops steadily each epoch | Training correctly | Do nothing |
| Train loss ↓ but eval loss ↑ after epoch 2 | Overfitting | Stop at epoch 2 (checkpoint already saved) |
| Loss stuck > 1.5 after epoch 1 | Underfitting | Increase α to 64 or LR to 3e-4 |
| Loss → 0.0 on train, high on eval | Memorizing | Reduce epochs to 2, increase dropout to 0.1 |
| Loss NaN or explodes | LR too high | Restart with LR = 1e-4 |

**Expected loss trajectory for this config:**
```
Epoch 1 end:  train ~1.2–1.5  eval ~1.4–1.8
Epoch 2 end:  train ~0.6–0.9  eval ~0.9–1.3
Epoch 3 end:  train ~0.3–0.6  eval ~0.8–1.2
```

If eval loss at epoch 3 is higher than epoch 2, the best checkpoint (epoch 2) is auto-loaded due to `load_best_model_at_end=True`.

---

## 7. Inference Test (Run This Before GGUF Export)

Always test the merged model before exporting:

```python
FastLanguageModel.for_inference(model)

test_prompt = """You are WhyBook, an NCERT Chemistry tutor for Indian students.
A student wants to understand: NaCl (Sodium Chloride) from Acids, Bases and Salts (Class 10).
Explain clearly — what it is, why it is taught, and where the student will see it in real life."""

inputs = tokenizer(
    [test_prompt],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.3,       # low — keep answers consistent
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**What good output looks like:**
```
What it is:
Common salt — sodium bonded to chlorine. White crystals. Dissolves in water.
Tastes salty.

Why it is in your textbook:
NaCl is the simplest example of an ionic compound — one metal and one
non-metal sharing electrons completely. Understanding it is the foundation
for understanding all salts formed in neutralization reactions.

Where you will see it in real life:
The salt in your kitchen is NaCl. It is also used to preserve food like
pickles and dried fish. Your body needs it to regulate fluids — that is
why you feel weak when you sweat heavily and do not drink enough water.
```

**Red flags in output:**
- Repeating the instruction back verbatim → prompt format issue, fix the template
- Output in JSON when you didn't ask → chat template applied wrong
- Gibberish or repeating tokens → LR was too high, retrain with 1e-4
- Generic "it is used in many industries" → fine-tuning didn't take, check data quality

---

## 8. GGUF File — What It Is and How to Use It

After `save_pretrained_gguf`, you get:
```
whybook-e4b-gguf/
└── unsloth.Q4_K_M.gguf    (~2.5 GB)
```

**Q4_K_M explained:**
- Q4 = 4-bit quantization (same as training)
- K_M = "K-quant medium" — uses mixed precision internally, better quality than plain Q4

**Use in Kaggle notebook demo:**
```python
# Install llama-cpp-python
# !pip install llama-cpp-python -q

from llama_cpp import Llama

llm = Llama(
    model_path="whybook-e4b-gguf/unsloth.Q4_K_M.gguf",
    n_ctx=512,
    n_gpu_layers=-1,     # use GPU fully
    verbose=False,
)

response = llm(
    prompt=your_formatted_prompt,
    max_tokens=300,
    temperature=0.3,
    stop=["<end_of_turn>"],
)
print(response["choices"][0]["text"])
```

---

## 9. Troubleshooting Cheatsheet

| Problem | Likely Cause | Fix |
|---|---|---|
| CUDA OOM | Sequence length too high | Lower `max_seq_length` to 256 |
| Loss not decreasing | LR too low or α too low | Raise LR to 3e-4 or α to 64 |
| Eval loss rises from epoch 1 | Overfitting immediately | Drop to 1-2 epochs, add dropout 0.1 |
| Output ignores format | Chat template not applied | Check `format_record()` output manually |
| Training very slow | Batch too small | Raise `per_device_train_batch_size` to 4 |
| GGUF export fails | Unsloth version | `!pip install unsloth --upgrade` |
| Model ID not found | Gemma 4 not yet on HF | Check `google/gemma-4-e4b-it` on huggingface.co |

---

## 10. Complete Hyperparameter Summary

| Hyperparameter | Value | Justification |
|---|---|---|
| r (rank) | 16 | ~110 records, medium task complexity |
| α (alpha) | 32 | Small dataset → α = 2r for stronger updates |
| lora_dropout | 0.05 | Light regularization for small dataset |
| learning_rate | 2e-4 | Standard LoRA LR; effective = 2e-4 × 2.0 = 4e-4 |
| per_device_batch | 2 | Conservative; raise to 4 if memory allows |
| grad_accumulation | 4 | Effective batch = 8 |
| epochs | 3 | Small dataset; stop at 2 if eval loss rises |
| warmup_steps | 5 | ~12% of total steps |
| max_seq_length | 512 | Data is ~310 tokens; 512 gives safe headroom |
| weight_decay | 0.01 | Light L2 regularization |
| optimizer | adamw_8bit | Unsloth's memory-efficient optimizer |
| quantization | Q4_K_M | Best quality 4-bit for on-device deployment |
| target_modules | all 7 | Domain-shift task needs full adapter coverage |
