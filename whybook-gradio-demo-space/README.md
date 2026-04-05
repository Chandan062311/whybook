---
title: WhyBook Demo
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
python_version: 3.13
app_file: app.py
pinned: false
license: apache-2.0
---

# WhyBook Demo

Gradio demo for the WhyBook chemistry tutor model.

This Space can run in two modes:

1) Lightweight fallback tutor mode (default in constrained runtime)
2) Local GGUF inference mode when `llama-cpp-python` is available

For GGUF mode, it downloads the published model from:
- `Stinger2311/whybook-gemma4-e2b-gguf`

Then it runs local inference with `llama-cpp-python` and exposes a simple tutoring UI.

## Notes

- First startup may take time because the GGUF file must be downloaded.
- CPU Spaces can run this, but responses may be slow.
- Upgrading to better Space hardware will improve latency.
