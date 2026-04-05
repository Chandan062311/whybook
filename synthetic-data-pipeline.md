# Synthetic Data Generation Pipeline
## WhyBook — NCERT Chemistry Q&A Dataset

---

## Purpose

Generate high-quality, structured Q&A pairs from NCERT Chemistry textbooks using an OpenRouter LLM. These pairs will be used to fine-tune Gemma 4 E4B with QLoRA 4-bit on Kaggle.

**Build in two phases — MVP first, expand only if time allows:**

| Phase | Classes | Concepts | API Calls | API Time | Purpose |
|---|---|---|---|---|---|
| **MVP** | Class 9 + 10 | ~110 | ~110 | ~20 min | Fine-tune + demo |
| **Full** | Class 9 + 10 + 11 + 12 | ~380 | ~380 | ~2 hrs | Complete submission |

Start with MVP. Fine-tune E4B. Validate quality. Then expand to Class 11-12 if time permits before May 18, 2026.

**Why skip 11-12 initially:** Class 11-12 concepts (thermodynamics, electrochemistry, equilibrium) are harder to ground in tangible everyday examples for the target user (Class 10 student). Class 9-10 concepts — CH₄, H₂SO₄, metals, acids — have the clearest real-world stories and make the strongest demo.

Each pair teaches the model to answer three questions about any chemistry concept:
1. **What** — simple explanation, no jargon
2. **Why** — why this concept is in the textbook (pedagogical purpose)
3. **Real World** — where an Indian student will encounter this in daily life

---

## Model Recommendation

**Use: `meta-llama/llama-3.3-70b-instruct:free`**

Reasons:
- Free on OpenRouter, no credit card needed
- Reliably follows structured JSON output schemas
- Strong factual accuracy for chemistry/science content
- Well-tested for instruction-following — least likely to deviate

Fallback (if rate limited): `openai/gpt-oss-20b:free` — also supports structured outputs, faster.

**OpenRouter config:**
```python
model = "meta-llama/llama-3.3-70b-instruct:free"
temperature = 0.3        # LOW — keeps answers consistent, reduces hallucination
max_tokens = 600         # Enough for all 3 fields, not enough to ramble
response_format = {"type": "json_object"}   # Enforce JSON output
```

---

## Output JSON Schema

Every generated record must match this exact structure:

```json
{
  "concept": "string — the formula, compound, or term extracted from the page",
  "chapter": "string — chapter name from NCERT",
  "class": "string — '9', '10', '11', or '12'",
  "what": "string — max 80 words. Simple explanation. No textbook language.",
  "why": "string — max 80 words. Why this concept is taught. Pedagogical purpose.",
  "real_world": "string — max 120 words. Specific real-world example. Must be India-relevant."
}
```

---

## System Prompt (Send This to OpenRouter — Do Not Change)

```
You are an expert chemistry teacher creating educational explanations for Indian students in Class 9-12.

You will be given a passage from an NCERT Chemistry textbook and a specific concept to explain.

Your job is to generate a JSON response with exactly three fields:
- "what": Explain the concept in simple language. Max 80 words. No jargon. Write as if explaining to a 14-year-old who has never heard this word before.
- "why": Explain WHY this concept is in the textbook. What is the student supposed to understand or be able to do after learning this? Max 80 words. Do NOT just repeat the definition.
- "real_world": Give ONE specific, concrete example of where this concept appears in an Indian student's daily life. Mention a real object, place, or situation they would recognise — like their kitchen, school, market, farm, or home. Max 120 words.

STRICT RULES:
1. Use ONLY information from the provided textbook passage. Do not add facts from outside it.
2. Output JSON only. No markdown. No explanation before or after the JSON.
3. The "real_world" field MUST mention something specific and tangible — not "it is used in industry" or "it has many applications". Name the actual thing.
4. Write in simple English. The student reading this struggled with English — be clear, not clever.
5. Never start any field with "In conclusion", "To summarize", "As mentioned", or "As an AI".
6. If the textbook passage does not contain enough information about the concept, return: {"error": "insufficient_context"}

OUTPUT FORMAT — return only this JSON, nothing else:
{
  "concept": "<concept name>",
  "chapter": "<chapter name>",
  "class": "<class number>",
  "what": "<your explanation>",
  "why": "<your explanation>",
  "real_world": "<your example>"
}
```

---

## User Prompt Template (Per Concept)

```
TEXTBOOK PASSAGE:
---
{extracted_text_chunk}
---

CONCEPT TO EXPLAIN: {concept_name}
CHAPTER: {chapter_name}
CLASS: {class_number}

Generate the JSON explanation now.
```

---

## Few-Shot Examples (Embed These in First API Call for Calibration)

Use these as the first batch to verify the model is generating correctly before running the full dataset.

### Example 1
**Input concept:** CH₄ (Methane), Chapter: Carbon and its Compounds, Class: 10

**Expected output:**
```json
{
  "concept": "CH₄ (Methane)",
  "chapter": "Carbon and its Compounds",
  "class": "10",
  "what": "Methane is the simplest compound made of carbon and hydrogen. One carbon atom bonds with four hydrogen atoms. It is a colourless gas with no smell at normal conditions.",
  "why": "Methane is taught because it is the simplest example of how carbon atoms bond with hydrogen. Understanding methane helps students learn the pattern of carbon chemistry, which is the basis for all living things and most fuels.",
  "real_world": "The gas in your kitchen LPG cylinder is mostly methane and butane. When you light the stove, methane burns and releases heat that cooks your food. The same gas is produced naturally when organic waste decomposes — you may have seen gas flames on landfill sites or heard of biogas plants in villages that use cow dung to produce cooking gas."
}
```

### Example 2
**Input concept:** H₂SO₄ (Sulphuric Acid), Chapter: Acids, Bases and Salts, Class: 10

**Expected output:**
```json
{
  "concept": "H₂SO₄ (Sulphuric Acid)",
  "chapter": "Acids, Bases and Salts",
  "class": "10",
  "what": "Sulphuric acid is a strong acid made of hydrogen, sulphur, and oxygen. It is a thick, oily liquid that is colourless. It is very corrosive — meaning it can dissolve and destroy many materials including metals and skin.",
  "why": "Sulphuric acid is taught because it is one of the most important and widely used chemicals in the world. Understanding it helps students learn what makes an acid strong, how acids react with metals and bases, and why handling chemicals safely matters.",
  "real_world": "The battery in your father's motorcycle or a car contains sulphuric acid dissolved in water. This acid reacts inside the battery to produce electricity that starts the engine. When a battery is old and leaks, the acid can corrode the metal parts around it — that white or blue-green crust you sometimes see on battery terminals is caused by this reaction."
}
```

---

## Quality Validation Rules

Run these checks on every generated record before adding it to the dataset. **Reject and regenerate if any rule fails.**

```python
def validate_record(record: dict) -> tuple[bool, str]:
    # Rule 1: All required fields present
    required = ["concept", "chapter", "class", "what", "why", "real_world"]
    for field in required:
        if field not in record or not record[field].strip():
            return False, f"Missing or empty field: {field}"

    # Rule 2: Word count limits
    if len(record["what"].split()) > 100:
        return False, "what field exceeds 100 words"
    if len(record["why"].split()) > 100:
        return False, "why field exceeds 100 words"
    if len(record["real_world"].split()) > 150:
        return False, "real_world field exceeds 150 words"

    # Rule 3: No AI refusal phrases
    refusal_phrases = ["i cannot", "i don't know", "as an ai", "i'm not able",
                       "it is used in industry", "it has many applications",
                       "in conclusion", "to summarize"]
    for field in ["what", "why", "real_world"]:
        text = record[field].lower()
        for phrase in refusal_phrases:
            if phrase in text:
                return False, f"Refusal/generic phrase in {field}: '{phrase}'"

    # Rule 4: real_world must not be abstract
    abstract_phrases = ["various industries", "many applications", "widely used",
                        "plays an important role", "has many uses"]
    text = record["real_world"].lower()
    for phrase in abstract_phrases:
        if phrase in text:
            return False, f"Abstract phrase in real_world: '{phrase}'"

    # Rule 5: Fields must not be identical to each other
    if record["what"].strip() == record["why"].strip():
        return False, "what and why are identical"
    if record["why"].strip() == record["real_world"].strip():
        return False, "why and real_world are identical"

    # Rule 6: Minimum length (too short = lazy answer)
    if len(record["what"].split()) < 20:
        return False, "what field too short (< 20 words)"
    if len(record["why"].split()) < 20:
        return False, "why field too short (< 20 words)"
    if len(record["real_world"].split()) < 30:
        return False, "real_world field too short (< 30 words)"

    return True, "ok"
```

---

## Rate Limiting Strategy (Free Tier)

OpenRouter free tier has rate limits. To avoid hitting them:

```python
import time

REQUESTS_PER_MINUTE = 10       # Conservative — adjust based on your key
DELAY_BETWEEN_REQUESTS = 6     # seconds (60 / 10 = 6)
MAX_RETRIES = 3
RETRY_DELAY = 30               # seconds — wait longer on rate limit error

def call_with_backoff(client, prompt, retries=0):
    try:
        response = client.chat.completions.create(...)
        time.sleep(DELAY_BETWEEN_REQUESTS)
        return response
    except RateLimitError:
        if retries < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retries + 1))
            return call_with_backoff(client, prompt, retries + 1)
        raise
```

**Save progress every 50 records to a `.jsonl` file.** Free tier sessions can drop — don't lose work.

---

## Dataset Size Target

### Phase 1 — MVP (Start Here)

| Class | Chapters | Concepts | API Calls | Est. API Time |
|---|---|---|---|---|
| Class 9 | 4 chemistry chapters | ~42 | ~42 | ~8 min |
| Class 10 | 5 chemistry chapters | ~68 | ~68 | ~12 min |
| **Phase 1 Total** | | **~110** | **~110** | **~20 min** |

**Token estimate per call:** ~750 input + ~350 output = ~1,100 tokens
**Total tokens Phase 1:** ~121,000 tokens — well within free tier limits

110 records is enough to fine-tune E4B and produce a working demo. Quality over quantity.

### Phase 2 — Full Dataset (Only if time allows before May 18)

| Class | Chapters | Concepts | API Calls | Est. API Time |
|---|---|---|---|---|
| Class 11 | 14 chapters | ~130 | ~130 | ~25 min |
| Class 12 | 16 chapters | ~140 | ~140 | ~28 min |
| **Phase 2 Addition** | | **~270** | **~270** | **~53 min** |
| **Grand Total** | | **~380** | **~380** | **~1.5 hrs** |

**Do not block the demo on Phase 2.** Get Phase 1 fine-tuned and working first.

---

## Output File Format

Save generated records as `.jsonl` (one JSON per line):

```
{"concept": "CH₄ (Methane)", "chapter": "Carbon and its Compounds", "class": "10", "what": "...", "why": "...", "real_world": "..."}
{"concept": "H₂SO₄ (Sulphuric Acid)", "chapter": "Acids, Bases and Salts", "class": "10", "what": "...", "why": "...", "real_world": "..."}
```

Final file: `data/ncert_chemistry_whybook.jsonl`

---

## Anti-Deviation Checklist (Run Before Fine-Tuning)

Before sending this dataset to fine-tuning, manually review 10% of records (random sample):

- [ ] Does every `real_world` mention something a student in India would recognise?
- [ ] Does any `what` field copy directly from the textbook word-for-word? (Bad — reject)
- [ ] Does any `why` field just say "because it is important"? (Bad — reject)
- [ ] Are concepts from all 4 classes represented?
- [ ] Is the language consistently simple (no "furthermore", "consequently", "heretofore")?
- [ ] Do the examples feel local — kitchen, farm, market, motorcycle — not "laboratory settings"?

If more than 5% of your sample fails — regenerate that batch with a lower temperature (try 0.2).

---

## Sources

- [OpenRouter Free Models](https://openrouter.ai/collections/free-models)
- [OpenRouter Structured Outputs Guide](https://openrouter.ai/docs/guides/features/structured-outputs)
- NCERT Chemistry Textbooks: [ncert.nic.in](https://ncert.nic.in) — Class 9, 10, 11, 12 (public domain)
