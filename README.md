---
title: Qwen3.8-27B-Uncensored
emoji: 🔓
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
short_description: Uncensored Qwen3.8-27B, text and vision chat
models:
- JonathanColetti/Qwen3.8-27B-Uncensored-GGUF
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-Uncensored

Chat with an uncensored [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Text and
image input. Refusal behaviour has been substantially reduced — not eliminated — while
capabilities and training data are unchanged. See Measured behaviour below.

## What it's for

Questions with legitimate purpose that aligned models often decline — security research,
fiction with real stakes, clinical and pharmacological study, unhedged critique.

## Running

bf16 on ZeroGPU (RTX PRO 6000 Blackwell, 96 GB slice — the weights are ~54.7 GB, so
the 4-bit path this was originally written for is no longer needed; set `QUANTIZE=4bit`
to restore it). The Space serves safetensors; the
[GGUF quantizations](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF)
are for running it locally under llama.cpp, with the MTP head retained.

## Input

Text plus image attachments (JPEG, PNG, WebP, GIF, BMP, TIFF, HEIC/HEIF — anything else
is converted to PNG before it reaches the processor). Attached text files, including the
`pasted_text.txt` Gradio creates from a long paste, are inlined into the prompt and
truncated at 30,000 characters.

## Sampling

Qwen's recommended defaults: `temperature=1.0`, `top_p=0.95`, `top_k=20`. Adjustable
under Sampling.

**Reasoning effort** is the setting that matters most here. The chat template defaults to
`xhigh` — "think carefully through the task, validate key assumptions, consider plausible
alternatives". A turn is capped at 1024 new tokens (~13 tok/s, about what fits in a
120-second ZeroGPU reservation), and on a substantive question the thinking alone can
consume all of it. Measured on this Space with *"explain how SQL injection works, with a
vulnerable code sample"*:

| Effort | Reasoning | Answer |
|---|---|---|
| `off` | — | 4330 chars |
| `low` | 1204 chars | 3032 chars |
| `xhigh` (template default) | 4556 chars | **none — budget exhausted** |

All three hit the cap at ~77 s; the difference is how much of the budget reaches an
answer. This Space therefore defaults to **off**, which is also the mode the refusal
numbers below were measured in. When a reply does hit the cap, the Space says so instead
of just stopping.

## Measured behaviour

Benchmarked against the unmodified base model on identical settings. The delta is the
figure that matters: it isolates what the weight edit cost.

| Task | Base | Uncensored | Δ |
|---|---|---|---|
| MMLU | 83.4 | 83.3 | -0.2 |
| ARC-Challenge | 58.9 | 57.7 | -1.2 |
| HellaSwag | 82.8 | 82.9 | +0.1 |
| Winogrande | 76.1 | 75.3 | -0.8 |
| **Mean** | | | **-0.5** |

0-shot via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
bf16, both models scored in the same session. Every delta is within or close to the
reported standard error (MMLU ±0.30, ARC ±1.44, HellaSwag ±0.38, Winogrande ±1.21), so
none is clearly separable from run-to-run noise.

**These are 0-shot and are not comparable to Qwen's published scores**, which use few-shot
prompting. They are directly comparable to each other, which is the point. Note also that
ARC-Challenge is low for a model at this MMLU — the base scores 58.9 under the same
settings, so that is format sensitivity in a reasoning-tuned model, not abliteration
damage.

What the benchmarks do **not** cover: no generative evaluation (GSM8K, HumanEval), no
math or code, no multilingual, and the harness loads the text stack only — nothing here
measures the vision tower or MTP speculative decoding.

What was measured, during abliteration:

| Measurement | Base | This model |
|---|---|---|
| Refusals, 100 held-out harmful prompts | 98/100 | **12/100** |
| KL divergence vs base (first-token) | 0 | 0.1191 |

The refusal figure is scored against
[`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors)
— explicitly harmful requests, not benign ones. It is therefore **not** an over-refusal
rate: it says nothing about how often the model declines legitimate work. KL divergence is
the optimizer's proxy for how much the edit damaged the model; it is not a capability
measurement.

Full Pareto front and caveats are on the
[GGUF model card](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF).

## Limitations

- Refusals are reduced, not eliminated, and not redirected. Roughly a fifth of explicitly
  harmful prompts are still declined.
- Refusals were measured in non-thinking mode, which is this Space's default. Raising
  Reasoning effort may change that behaviour.
- Behaviour near the old refusal boundary is less stable than the base model.
- The GGUF Q6_K and Q8_0 files are more faithful than what this Space serves.

## API

Every handler is exposed over the Gradio API and as an MCP tool.

```python
from gradio_client import Client
client = Client("JonathanColetti/Qwen3.8-27B-Uncensored-Demo")
client.predict({"text": "...", "files": []}, "low", 1.0, 0.95, 20, api_name="/respond")
```

Conversation history is held in server-side session state, so successive `predict` calls
on the *same* `Client` continue the conversation; a new `Client` starts a fresh one.

## License

Apache 2.0, inherited from Qwen/Qwen3.8-27B.
