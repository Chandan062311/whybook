import os
from functools import lru_cache
from typing import Any

import gradio as gr
from huggingface_hub import InferenceClient, hf_hub_download


REPO_ID = os.environ.get("WHYBOOK_MODEL_REPO", "Stinger2311/whybook-gemma4-e2b-gguf")
FILENAME = os.environ.get("WHYBOOK_MODEL_FILE", "gemma-4-E2B-it.Q4_K_M.gguf")
MAX_CONTEXT = int(os.environ.get("WHYBOOK_N_CTX", "2048"))
MAX_TOKENS = int(os.environ.get("WHYBOOK_MAX_NEW_TOKENS", "320"))
TEMPERATURE = float(os.environ.get("WHYBOOK_TEMPERATURE", "0.25"))
MODEL_DOWNLOAD_TIMEOUT = int(os.environ.get("WHYBOOK_MODEL_TIMEOUT", "1800"))

ENABLE_GGUF = os.environ.get("WHYBOOK_ENABLE_GGUF", "0") == "1"
ENABLE_REMOTE_LLM = os.environ.get("WHYBOOK_ENABLE_REMOTE_LLM", "1") == "1"
REMOTE_MODEL = os.environ.get("WHYBOOK_REMOTE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
REMOTE_MAX_TOKENS = int(os.environ.get("WHYBOOK_REMOTE_MAX_TOKENS", "520"))


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Comic+Neue:wght@400;700&display=swap');

body {
  background:
    radial-gradient(circle at 20% 20%, rgba(250, 204, 21, 0.08), transparent 24%),
    radial-gradient(circle at 78% 18%, rgba(239, 68, 68, 0.1), transparent 26%),
    radial-gradient(circle at 82% 80%, rgba(59, 130, 246, 0.1), transparent 26%),
    #f8fafc;
}

#wb-hero {
  background:
    radial-gradient(circle at 12% 18%, rgba(250, 204, 21, 0.35), transparent 34%),
    radial-gradient(circle at 90% 86%, rgba(239, 68, 68, 0.28), transparent 30%),
    linear-gradient(145deg, #0f172a 0%, #1d4ed8 58%, #ef4444 100%);
  border: 3px solid #0f172a;
  border-radius: 16px;
  padding: 22px;
  color: #f8fafc;
  margin-bottom: 12px;
  box-shadow: 8px 8px 0 #0f172a;
}
#wb-hero h1 {
  margin: 0 0 8px 0;
  font-size: 34px;
  line-height: 1.08;
  font-family: "Bangers", "Comic Neue", sans-serif;
  letter-spacing: 1px;
}
#wb-hero p {
  margin: 0;
  color: #eff6ff;
  font-family: "Comic Neue", sans-serif;
  font-weight: 700;
}
#wb-meta {
  font-size: 14px;
  color: #fef08a;
  margin-top: 8px;
  font-family: "Comic Neue", sans-serif;
  font-weight: 700;
}
#wb-card {
  border: 3px solid #0f172a;
  border-radius: 14px;
  padding: 14px;
  background: #ffffff;
  box-shadow: 6px 6px 0 #0f172a;
}
#wb-output {
  border: 3px solid #0f172a;
  border-radius: 14px;
  background: #ffffff;
  padding: 10px;
  box-shadow: 6px 6px 0 #0f172a;
}
#wb-note {
  background: #fef3c7;
  border: 3px solid #0f172a;
  border-radius: 12px;
  padding: 10px;
  font-family: "Comic Neue", sans-serif;
  font-weight: 700;
  box-shadow: 4px 4px 0 #0f172a;
}
#wb-tips {
  background: #dbeafe;
  border: 3px solid #0f172a;
  border-radius: 12px;
  padding: 10px;
  font-family: "Comic Neue", sans-serif;
  font-weight: 700;
  box-shadow: 4px 4px 0 #0f172a;
}
.wb-footer {
  color: #0f172a;
  font-size: 13px;
  font-family: "Comic Neue", sans-serif;
  font-weight: 700;
}

button {
  border: 3px solid #0f172a !important;
  box-shadow: 3px 3px 0 #0f172a !important;
  font-family: "Bangers", "Comic Neue", sans-serif !important;
  letter-spacing: 0.8px;
}

button:hover {
  transform: translate(-1px, -1px);
}

textarea, input, .gradio-dropdown {
  border: 2px solid #0f172a !important;
  font-family: "Comic Neue", sans-serif !important;
}

label, .gradio-markdown, .prose {
  font-family: "Comic Neue", sans-serif !important;
}

@media (max-width: 768px) {
  body {
    background: #f8fafc;
  }

  #wb-hero {
    background: linear-gradient(155deg, #0f172a 0%, #1e3a8a 60%, #7f1d1d 100%);
    border: 2px solid #0f172a;
    box-shadow: 3px 3px 0 #0f172a;
    padding: 14px;
  }

  #wb-hero h1 {
    font-size: 26px;
    line-height: 1.1;
    letter-spacing: 0.4px;
  }

  #wb-hero p {
    color: #f8fafc;
    font-size: 15px;
    line-height: 1.45;
  }

  #wb-meta {
    color: #dbeafe;
    font-size: 12px;
  }

  #wb-card,
  #wb-output,
  #wb-note,
  #wb-tips {
    box-shadow: 2px 2px 0 #0f172a;
    border-width: 2px;
  }

  #wb-note {
    background: #fef9c3;
  }

  #wb-tips {
    background: #eff6ff;
  }

  button {
    box-shadow: 2px 2px 0 #0f172a !important;
    border-width: 2px !important;
    font-size: 14px !important;
  }

  textarea,
  input,
  .gradio-dropdown {
    font-size: 15px !important;
    line-height: 1.45 !important;
  }
}

/* Dark comic theme overrides */
body,
.gradio-container {
  color-scheme: dark;
  background:
    radial-gradient(circle at 12% 18%, rgba(59, 130, 246, 0.1), transparent 24%),
    radial-gradient(circle at 86% 16%, rgba(239, 68, 68, 0.1), transparent 24%),
    radial-gradient(circle at 84% 82%, rgba(250, 204, 21, 0.07), transparent 22%),
    #0b1220 !important;
  color: #f1f5f9 !important;
}

#wb-card,
#wb-output,
#wb-note,
#wb-tips {
  background: #111827 !important;
  color: #f1f5f9 !important;
  border-color: #334155 !important;
  box-shadow: 4px 4px 0 #020617 !important;
}

#wb-note {
  background: #1f2937 !important;
  color: #f8fafc !important;
}

#wb-tips {
  background: #0f172a !important;
  color: #f8fafc !important;
}

.gradio-markdown,
.gradio-markdown p,
.gradio-markdown li,
.gradio-markdown span,
label,
.prose,
.prose p,
.prose li,
.prose strong,
.prose h1,
.prose h2,
.prose h3 {
  color: #f1f5f9 !important;
}

#wb-output .prose,
#wb-output .prose p,
#wb-output .prose li,
#wb-output .prose span,
#wb-output .prose strong,
#wb-output .prose h1,
#wb-output .prose h2,
#wb-output .prose h3,
#wb-output .prose h4 {
  color: #f8fafc !important;
}

#wb-output .prose h2,
#wb-output .prose h3 {
  color: #fde68a !important;
}

#wb-output .prose code {
  background: #1e293b !important;
  color: #facc15 !important;
  border: 1px solid #334155;
  padding: 2px 6px;
  border-radius: 6px;
}

#wb-output .prose a {
  color: #7dd3fc !important;
}

textarea,
input,
select {
  background: #0b1220 !important;
  color: #f8fafc !important;
  border-color: #334155 !important;
}

textarea::placeholder,
input::placeholder {
  color: #94a3b8 !important;
  opacity: 1;
}

#wb-card label,
#wb-card .gradio-markdown,
#wb-card .gradio-markdown p,
#wb-card .gradio-markdown li {
  color: #e2e8f0 !important;
}

button {
  background: #1d4ed8 !important;
  color: #f8fafc !important;
  border-color: #0f172a !important;
}

button.secondary {
  background: #334155 !important;
}

.wb-footer {
  color: #cbd5e1 !important;
}
"""


def mode_badge() -> str:
    if ENABLE_GGUF and ENABLE_REMOTE_LLM:
        return "<div id='wb-note'><b>Mode:</b> Local GGUF + Remote fallback (best resilience).</div>"
    if ENABLE_GGUF:
        return "<div id='wb-note'><b>Mode:</b> Local GGUF inference only.</div>"
    if ENABLE_REMOTE_LLM:
        return "<div id='wb-note'><b>Mode:</b> Remote model generation (broad input coverage).</div>"
    return "<div id='wb-note'><b>Mode:</b> Offline fallback only.</div>"


def build_system_prompt(length_mode: str, language_mode: str) -> str:
    length_instructions = {
        "Short": "Keep each section concise: 2-4 lines.",
        "Exam": "Use exam-ready wording: define clearly, mention key terms, and add one memory cue.",
        "Detailed": "Give fuller explanation with gentle depth and at least 2 practical examples.",
    }
    language_instructions = {
        "English": "Write in clear English.",
        "Hinglish": "Write in student-friendly Hinglish (Roman script), but keep chemistry terms accurate.",
        "Bilingual": "Write primarily in English and add brief Hinglish support lines where useful.",
    }
    return (
        "You are WhyBook, an NCERT chemistry tutor for school students. "
        "Return markdown with exactly these headings: What, Why, Real World. "
        "Be accurate, practical, and beginner-friendly. "
        f"{length_instructions.get(length_mode, length_instructions['Exam'])} "
        f"{language_instructions.get(language_mode, language_instructions['English'])}"
    )


def fallback_answer(user_prompt: str, length_mode: str, language_mode: str) -> str:
    style_note = f"Mode: {length_mode} | Language: {language_mode}"
    return (
        f"## Prompt\n{user_prompt}\n\n"
        "### What\n"
        "Define the core chemistry idea, name the key particles/substances, and state the main rule or process in simple terms.\n\n"
        "### Why\n"
        "Explain why this topic matters for understanding reactions, solving exam questions, and building scientific reasoning.\n\n"
        "### Real World\n"
        "Connect it to everyday contexts like food, cleaning, health, rusting, batteries, farming, or water treatment.\n\n"
        "---\n"
        f"_Fallback response ({style_note}) because model backend is unavailable right now._"
    )


@lru_cache(maxsize=1)
def get_llm() -> Any:
    if not ENABLE_GGUF:
        raise RuntimeError("GGUF mode disabled by configuration")
    try:
        from llama_cpp import Llama
    except Exception as exc:
        raise RuntimeError(
            "llama_cpp import failed. Native runtime is incompatible on this Space."
        ) from exc

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        resume_download=True,
        etag_timeout=MODEL_DOWNLOAD_TIMEOUT,
    )
    return Llama(
        model_path=model_path,
        n_ctx=MAX_CONTEXT,
        n_gpu_layers=0,
        verbose=False,
    )


def remote_generate_answer(
    user_prompt: str, length_mode: str, language_mode: str
) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Missing HF token in Space secrets. Add HF_TOKEN.")
    client = InferenceClient(api_key=token)
    completion = client.chat.completions.create(
        model=REMOTE_MODEL,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(length_mode, language_mode),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=REMOTE_MAX_TOKENS,
    )
    text = completion.choices[0].message.content.strip()
    if not text:
        raise RuntimeError("Remote model returned empty response")
    return f"## Prompt\n{user_prompt}\n\n{text}"


def local_generate_answer(
    user_prompt: str, length_mode: str, language_mode: str
) -> str:
    prompt = (
        build_system_prompt(length_mode, language_mode)
        + "\nStudent prompt: "
        + user_prompt
    )
    llm = get_llm()
    response = llm(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["<end_of_turn>", "<turn|>", "<eos>"],
    )
    text = response["choices"][0]["text"].strip()
    if not text:
        raise RuntimeError("Local GGUF returned empty response")
    return f"## Prompt\n{user_prompt}\n\n{text}"


def generate_answer(
    user_prompt: str, length_mode: str, language_mode: str
) -> tuple[str, str]:
    user_prompt = user_prompt.strip()
    if not user_prompt:
        return (
            "Please enter your chemistry prompt first.",
            "No response generated yet.",
        )

    backend_used = "Fallback"

    if ENABLE_GGUF:
        try:
            answer = local_generate_answer(user_prompt, length_mode, language_mode)
            backend_used = "Local GGUF"
            return answer, answer
        except Exception as local_exc:
            if ENABLE_REMOTE_LLM:
                try:
                    answer = remote_generate_answer(
                        user_prompt, length_mode, language_mode
                    )
                    backend_used = f"Remote ({REMOTE_MODEL}) after local fail"
                    answer = answer + f"\n\n---\n_Backend used: {backend_used}_"
                    return answer, answer
                except Exception as remote_exc:
                    fallback = fallback_answer(user_prompt, length_mode, language_mode)
                    note = (
                        "### Backend Note\n"
                        f"Local GGUF failed: `{local_exc}`\n"
                        f"Remote failed: `{remote_exc}`\n\n"
                    )
                    answer = note + fallback
                    return answer, answer
            fallback = fallback_answer(user_prompt, length_mode, language_mode)
            answer = (
                "### Backend Note\n"
                f"GGUF backend unavailable: `{local_exc}`\n\n" + fallback
            )
            return answer, answer

    if ENABLE_REMOTE_LLM:
        try:
            answer = remote_generate_answer(user_prompt, length_mode, language_mode)
            backend_used = f"Remote ({REMOTE_MODEL})"
            answer = answer + f"\n\n---\n_Backend used: {backend_used}_"
            return answer, answer
        except Exception as remote_exc:
            fallback = fallback_answer(user_prompt, length_mode, language_mode)
            answer = (
                "### Backend Note\n"
                f"Remote generation unavailable: `{remote_exc}`\n\n" + fallback
            )
            return answer, answer

    answer = fallback_answer(user_prompt, length_mode, language_mode)
    return answer, answer


EXAMPLES = [
    [
        "What is the chemical formula of common salt and why is it important?",
        "Exam",
        "English",
    ],
    [
        "Why is baking soda basic and where is it used in daily life?",
        "Detailed",
        "English",
    ],
    ["Electrolysis kya hota hai? Simple language me batao.", "Short", "Hinglish"],
    ["Explain pH and its importance in health and environment.", "Exam", "Bilingual"],
]


with gr.Blocks(
    title="WhyBook - Chemistry Tutor",
    theme=gr.themes.Soft(
        primary_hue="blue", secondary_hue="emerald", neutral_hue="slate"
    ),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
<div id="wb-hero">
  <h1>WhyBook Comic Lab</h1>
  <p>Bam! Zap! Learn chemistry with bold, structured answers in <b>What</b>, <b>Why</b>, and <b>Real World</b> format.</p>
  <div id="wb-meta">Comic-style public demo: high energy visuals + reliable learning output.</div>
</div>
"""
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=6, elem_id="wb-card"):
            prompt_box = gr.Textbox(
                label="Drop Your Chemistry Question",
                placeholder="e.g. What is the chemical formula of common salt and why is it important?",
                lines=4,
            )
            with gr.Row():
                length_mode = gr.Dropdown(
                    ["Short", "Exam", "Detailed"],
                    value="Exam",
                    label="Answer Style",
                )
                language_mode = gr.Dropdown(
                    ["English", "Hinglish", "Bilingual"],
                    value="English",
                    label="Language",
                )
            with gr.Row():
                submit = gr.Button("Generate Answer", variant="primary")
                clear = gr.Button("Reset Panel", variant="secondary")

        with gr.Column(scale=4):
            gr.HTML(mode_badge())
            gr.Markdown(
                """
<div id="wb-tips">
<b>Comic Mission Tips</b><br>
- Ask one clear question per prompt.<br>
- Mention class level if needed.<br>
- Use Detailed mode for deeper understanding.<br>
- Try Hinglish for friendlier revision sessions.
</div>
"""
            )

    output = gr.Markdown(label="Answer Panel", elem_id="wb-output")
    export_md = gr.Textbox(
        label="Copy / Export Script",
        lines=8,
        show_copy_button=True,
        placeholder="Generated answer markdown appears here for easy copy...",
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_box, length_mode, language_mode],
        outputs=[output, export_md],
        fn=generate_answer,
    )

    submit.click(
        generate_answer,
        inputs=[prompt_box, length_mode, language_mode],
        outputs=[output, export_md],
    )
    clear.click(
        lambda: ("", "Exam", "English", "", ""),
        outputs=[prompt_box, length_mode, language_mode, output, export_md],
    )

    gr.Markdown(
        "<div class='wb-footer'>WhyBook Comic Lab by Stinger2311 | Learn with clarity, speed, and style.</div>"
    )


if __name__ == "__main__":
    demo.launch()
