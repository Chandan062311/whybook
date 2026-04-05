from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pdfplumber
from openai import OpenAI

MODEL_NAMES = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
]
TEMPERATURE = 0.3
MAX_TOKENS = 600
REQUESTS_PER_MINUTE = 10
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
MAX_RETRIES = 3
RETRY_DELAY = 30

SYSTEM_PROMPT = """You are an expert chemistry teacher creating educational explanations for Indian students in Class 9-12.

You will be given a passage from an NCERT Chemistry textbook and a specific concept to explain.

Your job is to generate a JSON response with exactly three fields:
- "what": Explain the concept in simple language. Max 80 words. No jargon. Write as if explaining to a 14-year-old who has never heard this word before.
- "why": Explain WHY this concept is in the textbook. What is the student supposed to understand or be able to do after learning this? Max 80 words. Do NOT just repeat the definition.
- "real_world": Give ONE specific, concrete example of where this concept appears in an Indian student's daily life. Mention a real object, place, or situation they would recognise - like their kitchen, school, market, farm, or home. Max 120 words.

STRICT RULES:
1. Use ONLY information from the provided textbook passage. Do not add facts from outside it.
2. Output JSON only. No markdown. No explanation before or after the JSON.
3. The "real_world" field MUST mention something specific and tangible - not "it is used in industry" or "it has many applications". Name the actual thing.
4. Write in simple English. The student reading this struggled with English - be clear, not clever.
5. Never start any field with "In conclusion", "To summarize", "As mentioned", or "As an AI".
6. If the textbook passage does not contain enough information about the concept, return: {"error": "insufficient_context"}

OUTPUT FORMAT - return only this JSON, nothing else:
{
  "concept": "<concept name>",
  "chapter": "<chapter name>",
  "class": "<class number>",
  "what": "<your explanation>",
  "why": "<your explanation>",
  "real_world": "<your example>"
}"""

CLASS9_CHAPTERS = {
    "Matter in Our Surroundings",
    "Is Matter Around Us Pure",
    "Atoms and Molecules",
    "Structure of the Atom",
}

CONCEPTS = {
    "Matter in Our Surroundings": [
        "evaporation",
        "sublimation",
        "diffusion",
        "Brownian motion",
        "latent heat",
        "condensation",
        "states of matter",
    ],
    "Is Matter Around Us Pure": [
        "mixture",
        "solution",
        "colloid",
        "suspension",
        "distillation",
        "chromatography",
        "compound",
        "element",
    ],
    "Atoms and Molecules": [
        "atom",
        "molecule",
        "atomic mass",
        "molecular mass",
        "mole",
        "Avogadro number",
        "chemical formula",
    ],
    "Structure of the Atom": [
        "electron",
        "proton",
        "neutron",
        "atomic number",
        "mass number",
        "isotope",
        "isobar",
        "valence shell",
    ],
    "Chemical Reactions and Equations": [
        "oxidation",
        "reduction",
        "redox",
        "corrosion",
        "rancidity",
        "decomposition reaction",
        "displacement reaction",
    ],
    "Acids Bases and Salts": [
        "pH scale",
        "indicator",
        "neutralization",
        "NaCl",
        "H2SO4",
        "HCl",
        "NaOH",
        "baking soda",
        "bleaching powder",
    ],
    "Metals and Non-metals": [
        "reactivity series",
        "corrosion",
        "alloy",
        "ionic bond",
        "electrolytic refining",
        "thermite reaction",
    ],
    "Carbon and its Compounds": [
        "CH4 methane",
        "ethanol",
        "ethanoic acid",
        "soap",
        "detergent",
        "covalent bond",
        "isomer",
        "functional group",
        "hydrocarbon",
    ],
    "Periodic Classification": [
        "period",
        "group",
        "atomic radius trend",
        "Dobereiner triads",
        "Newlands octaves",
        "Mendeleev periodic law",
    ],
}


def extract_chapters(pdf_path: str) -> dict[str, str]:
    """Return chemistry chapters from an NCERT PDF."""
    chemistry_keywords = [
        "matter",
        "atom",
        "molecule",
        "acid",
        "base",
        "salt",
        "metal",
        "carbon",
        "chemical",
        "reaction",
        "element",
    ]
    chapters: dict[str, str] = {}
    with pdfplumber.open(pdf_path) as pdf:
        current_chapter: str | None = None
        current_text: list[str] = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            chapter_line_index = next(
                (
                    index
                    for index, line in enumerate(lines[:5])
                    if line.upper().startswith("CHAPTER")
                ),
                None,
            )

            if chapter_line_index is not None:
                if current_chapter:
                    full_text = " ".join(current_text)
                    if any(kw in full_text.lower() for kw in chemistry_keywords):
                        chapters[current_chapter] = full_text
                title_lines: list[str] = []
                for line in lines[chapter_line_index + 1 : chapter_line_index + 5]:
                    if len(line.split()) > 5:
                        break
                    title_lines.append(line)
                current_chapter = " ".join(title_lines) if title_lines else lines[0]
                current_text = [text]
            else:
                current_text.append(text)
        if current_chapter:
            full_text = " ".join(current_text)
            if any(kw in full_text.lower() for kw in chemistry_keywords):
                chapters[current_chapter] = full_text
    return chapters


def validate_record(record: dict) -> tuple[bool, str]:
    required = ["concept", "chapter", "class", "what", "why", "real_world"]
    for field in required:
        if field not in record or not str(record[field]).strip():
            return False, f"Missing or empty field: {field}"

    if len(record["what"].split()) > 100:
        return False, "what field exceeds 100 words"
    if len(record["why"].split()) > 100:
        return False, "why field exceeds 100 words"
    if len(record["real_world"].split()) > 150:
        return False, "real_world field exceeds 150 words"

    refusal_phrases = [
        "i cannot",
        "i don't know",
        "as an ai",
        "i'm not able",
        "it is used in industry",
        "it has many applications",
        "in conclusion",
        "to summarize",
    ]
    for field in ["what", "why", "real_world"]:
        text = record[field].lower()
        for phrase in refusal_phrases:
            if phrase in text:
                return False, f"Refusal/generic phrase in {field}: '{phrase}'"

    abstract_phrases = [
        "various industries",
        "many applications",
        "widely used",
        "plays an important role",
        "has many uses",
    ]
    text = record["real_world"].lower()
    for phrase in abstract_phrases:
        if phrase in text:
            return False, f"Abstract phrase in real_world: '{phrase}'"

    if record["what"].strip() == record["why"].strip():
        return False, "what and why are identical"
    if record["why"].strip() == record["real_world"].strip():
        return False, "why and real_world are identical"

    if len(record["what"].split()) < 20:
        return False, "what field too short (< 20 words)"
    if len(record["why"].split()) < 20:
        return False, "why field too short (< 20 words)"
    if len(record["real_world"].split()) < 30:
        return False, "real_world field too short (< 30 words)"

    return True, "ok"


def build_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def normalize_chapter_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def get_chunk(chapter_text: str, concept: str) -> str:
    idx = chapter_text.lower().find(concept.split()[0].lower())
    if idx > -1:
        return chapter_text[max(0, idx - 500) : idx + 1500]
    return chapter_text[:2000]


def call_with_backoff(client: OpenAI, messages: list[dict[str, str]], retries: int = 0):
    last_error: Exception | None = None
    for model_name in MODEL_NAMES:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return response
        except Exception as exc:
            last_error = exc

    if retries < MAX_RETRIES:
        time.sleep(RETRY_DELAY * (retries + 1))
        return call_with_backoff(client, messages, retries + 1)
    raise last_error if last_error is not None else RuntimeError("Model call failed")


def generate_record(
    client: OpenAI,
    chapter: str,
    concept: str,
    class_num: str,
    chapter_text: str,
) -> dict | None:
    user_prompt = (
        f"TEXTBOOK PASSAGE:\n---\n{get_chunk(chapter_text, concept)}\n---\n\n"
        f"CONCEPT TO EXPLAIN: {concept}\n"
        f"CHAPTER: {chapter}\n"
        f"CLASS: {class_num}\n\n"
        "Generate the JSON explanation now."
    )
    try:
        response = call_with_backoff(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
        if raw is None:
            raise ValueError("Model returned no message content")
        record = json.loads(raw)
        if "error" in record:
            return None
        record["concept"] = concept
        record["chapter"] = chapter
        record["class"] = class_num
        return record
    except Exception as exc:
        print(f"ERROR: {concept} - {exc}")
        return None


def collect_chapters(pdf_paths: list[str]) -> dict[str, str]:
    chapters: dict[str, str] = {}
    for pdf_path in pdf_paths:
        chapters.update(extract_chapters(pdf_path))
    return chapters


def generate_dataset(
    pdf_paths: list[str],
    output_path: str,
    api_key: str,
    chapter_filter: set[str] | None = None,
    concept_filter: set[str] | None = None,
    max_concepts: int | None = None,
) -> None:
    client = build_client(api_key)
    extracted_chapters = collect_chapters(pdf_paths)
    normalized_chapters = {
        normalize_chapter_name(chapter): text
        for chapter, text in extracted_chapters.items()
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    valid_records: list[dict] = []
    rejected: list[dict[str, str]] = []
    processed_concepts = 0

    for chapter, concepts in CONCEPTS.items():
        normalized_chapter = normalize_chapter_name(chapter)
        if chapter_filter and normalized_chapter not in chapter_filter:
            continue
        class_num = "9" if chapter in CLASS9_CHAPTERS else "10"
        chapter_text = normalized_chapters.get(normalized_chapter, "")
        if not chapter_text:
            print(f"SKIP: missing extracted text for chapter '{chapter}'")
            continue

        for concept in concepts:
            if concept_filter and concept.lower() not in concept_filter:
                continue
            if max_concepts is not None and processed_concepts >= max_concepts:
                total = len(valid_records) + len(rejected)
                rejection_rate = (len(rejected) / total * 100) if total else 0.0
                print(f"Valid: {len(valid_records)} | Rejected: {len(rejected)}")
                print(f"Rejection rate: {rejection_rate:.1f}%")
                return
            print(f"Generating: {concept}...")
            record = generate_record(client, chapter, concept, class_num, chapter_text)
            processed_concepts += 1

            if record is None:
                rejected.append(
                    {"concept": concept, "reason": "api_error_or_insufficient_context"}
                )
                continue

            is_valid, reason = validate_record(record)
            if is_valid:
                valid_records.append(record)
                with output_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            else:
                rejected.append({"concept": concept, "reason": reason})
                print(f"REJECTED: {concept} - {reason}")

    total = len(valid_records) + len(rejected)
    rejection_rate = (len(rejected) / total * 100) if total else 0.0
    print(f"Valid: {len(valid_records)} | Rejected: {len(rejected)}")
    print(f"Rejection rate: {rejection_rate:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WhyBook synthetic chemistry data."
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        action="append",
        required=True,
        help="Path to an NCERT PDF. Pass once per PDF.",
    )
    parser.add_argument(
        "--output",
        default="data/ncert_chemistry_whybook.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--chapter",
        dest="chapters",
        action="append",
        help="Restrict generation to a named chapter. Pass once per chapter.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        help="Maximum number of concepts to generate before stopping.",
    )
    parser.add_argument(
        "--concept",
        dest="concepts",
        action="append",
        help="Restrict generation to a specific concept. Pass once per concept.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable: {args.api_key_env}")
    chapter_filter = (
        {normalize_chapter_name(chapter) for chapter in args.chapters}
        if args.chapters
        else None
    )
    concept_filter = (
        {concept.lower() for concept in args.concepts} if args.concepts else None
    )
    generate_dataset(
        args.pdf_paths,
        args.output,
        api_key,
        chapter_filter=chapter_filter,
        concept_filter=concept_filter,
        max_concepts=args.max_concepts,
    )


if __name__ == "__main__":
    main()
