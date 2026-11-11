import os
import tempfile
import spaces

import torch
import gradio as gr
from PIL import Image
from threading import Thread
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                          BitsAndBytesConfig, TextIteratorStreamer)

MODEL_ID = os.environ.get("MODEL_ID", "JonathanColetti/Qwen3.8-27B-Uncensored")
# ~10 tok/s measured on the ZeroGPU Blackwell slice, so 1024 tokens is ~100s -- close to
# the 120s reservation below. Raising this without raising `duration` just moves the
# truncation from the token cap to a mid-stream GPU abort.
MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
THINKING_MARKERS = [("<think>", "</think>"), ("[Start thinking]", "[End thinking]")]

# The chat template defaults to reasoning_effort='xhigh' and injects "think carefully
# through the task, validate key assumptions, consider plausible alternatives...".
# On a substantive question that spends the whole 1024-token turn budget thinking:
# measured on this Space, "explain how SQL injection works, with a vulnerable code
# sample" gave off -> 4330 chars of answer, low -> 1204 reasoning + 3032 answer,
# xhigh -> 4556 chars of reasoning and *no answer at all*. Hence 'off' by default --
# it is also the mode the published refusal numbers were measured in.
REASONING_CHOICES = [("Off — answer immediately (default)", "off"),
                     ("Low — brief thinking", "low"),
                     ("Medium", "medium"),
                     ("High — the model's own default, often truncates", "xhigh")]
DEFAULT_REASONING = "off"

# The template turns reasoning_effort into a system instruction. Kept here so the effort
# setting still works if the loaded template ever stops accepting the kwarg -- see the
# startup probe below. 'medium' is the model's own behaviour, with nothing injected.
REASONING_INSTRUCTIONS = {
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, "
           "moving directly to the conclusion without unnecessary elaboration.",
    "medium": "",
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the "
             "task, validate key assumptions, consider plausible alternatives, and "
             "prioritize correctness, consistency, and clarity in the final answer.",
}

# ZeroGPU currently backs Spaces with an RTX PRO 6000 Blackwell (96 GB), so the 4-bit
# path this was written for is no longer forced: bf16 is ~54 GB and fits, and avoids
# bitsandbytes needing sm_120 kernels at load time. Set QUANTIZE=4bit to go back to NF4.
QUANTIZE = os.environ.get("QUANTIZE", "bf16").lower()

load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
if QUANTIZE == "4bit":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

# Log the resolved model id: MODEL_ID is overridable by a Space variable, and an
# override is invisible from the code alone -- this Space spent a while serving a
# different model than its title claimed because of exactly that.
print(f"[app] loading MODEL_ID={MODEL_ID} "
      f"({'env override' if os.environ.get('MODEL_ID') else 'app.py default'})", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
model.eval()

# TextIteratorStreamer only needs .decode; the processor proxies it to the tokenizer,
# but hand it the tokenizer directly so a processor without the proxy still streams.
tokenizer = getattr(processor, "tokenizer", processor)


def template_honours_reasoning_effort():
    """Does the loaded chat template actually act on a `reasoning_effort` kwarg?

    transformers logs "Keyword argument `reasoning_effort` is not a valid argument for
    this processor" for every call -- that warning is about the kwarg *also* being
    forwarded to processor.__call__, not about the template, which receives it either
    way. Rather than trust that reading, check the rendered prompt directly.
    """
    try:
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            add_generation_prompt=True, tokenize=False, reasoning_effort="low",
        )
        if isinstance(rendered, (list, tuple)):
            rendered = rendered[0]
        return "set to low" in rendered
    except Exception as exc:
        print(f"[app] reasoning_effort probe failed ({exc}); using system-prompt fallback")
        return False


HONOURS_EFFORT = template_honours_reasoning_effort()
print(f"[app] chat template honours reasoning_effort={HONOURS_EFFORT}; "
      f"effort is applied via {'template kwarg' if HONOURS_EFFORT else 'system message'}")


# transformers' image loader delegates to torchvision.decode_image, which accepts only
# jpeg/png/webp/gif and raises a bare RuntimeError on anything else -- the user saw a
# 30-line traceback for an ordinary phone photo. Normalising through PIL first means
# BMP/TIFF/ICO (and HEIC, when pillow-heif is installed) work instead of failing, and
# anything genuinely unreadable produces a message that says what to do about it.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF = True
except Exception:
    _HEIF = False

NATIVE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
SUPPORTED_HINT = ("JPEG, PNG, WebP, GIF, BMP, TIFF"
                  + (", HEIC/HEIF" if _HEIF else ""))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".gif", ".bmp", ".dib",
                  ".tif", ".tiff", ".ico", ".ppm", ".pgm", ".pbm", ".tga",
                  ".heic", ".heif", ".avif"}
# Gradio turns a long paste into an attached pasted_text.txt, and users drop source
# files and notes into the box as well. Those are text, not images -- feeding them to
# the image path is what produced "cannot identify image file 'pasted_text.txt'".
TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
                 ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini",
                 ".cfg", ".conf", ".env", ".xml", ".html", ".htm", ".css", ".js",
                 ".jsx", ".ts", ".tsx", ".py", ".pyi", ".c", ".h", ".cc", ".cpp",
                 ".hpp", ".java", ".kt", ".go", ".rs", ".rb", ".php", ".swift",
                 ".sh", ".bash", ".zsh", ".sql", ".r", ".jl", ".lua", ".pl",
                 ".diff", ".patch", ".srt", ".vtt", ".tex"}
MAX_FILE_CHARS = 30_000
_image_cache = {}


def prepare_image(path):
    """Return a path the processor can definitely decode, or raise a clear gr.Error."""
    cached = _image_cache.get(path)
    if cached and os.path.exists(cached):
        return cached

    name = os.path.basename(path or "file")
    try:
        with Image.open(path) as probe:
            probe.verify()                      # catches truncated/corrupt files
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            if fmt in NATIVE_FORMATS and img.mode in ("RGB", "L"):
                _image_cache[path] = path
                return path
            converted = img.convert("RGB")
        handle, out = tempfile.mkstemp(suffix=".png")
        os.close(handle)
        converted.save(out, format="PNG")
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(
            f"Couldn't read the image '{name}'. Supported formats: {SUPPORTED_HINT}. "
            f"If it's an AVIF or HEIC photo, re-save it as JPEG or PNG and try again."
        ) from exc

    _image_cache[path] = out
    return out


def looks_like_text(path):
    """Sniff an extension-less or unknown attachment: decodable UTF-8 with no NULs."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character straddling the 8 KB cut is not a binary signal.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def read_text_file(path):
    """Inline an attached text file, truncated so one paste can't eat the context."""
    name = os.path.basename(path or "file")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read(MAX_FILE_CHARS + 1)
    except OSError as exc:
        raise gr.Error(f"Couldn't read the attached file '{name}'.") from exc

    if len(body) > MAX_FILE_CHARS:
        body = body[:MAX_FILE_CHARS].rstrip() + "\n\n[... truncated]"
    # A long paste arrives as pasted_text.txt; it is the message itself, so it goes in
    # verbatim. A real attachment keeps its name so the model can refer to it.
    if name == "pasted_text.txt":
        return body
    return f"--- attached file: {name} ---\n{body}\n--- end of {name} ---"


def file_part(path):
    """Turn an attached file into an image or a text part, whichever it actually is."""
    if not path:
        return None
    suffix = os.path.splitext(path)[1].lower()
    if suffix in IMAGE_SUFFIXES:
        return {"type": "image", "image": prepare_image(path)}
    if suffix in TEXT_SUFFIXES or looks_like_text(path):
        return {"type": "text", "text": read_text_file(path)}
    # Unknown binary: try it as an image so odd-but-valid photos still work, and let
    # prepare_image produce the readable error if it isn't one.
    return {"type": "image", "image": prepare_image(path)}


def content_parts(content):
    """Normalise one history turn's content into chat-template parts.

    Gradio 6 stores a multimodal turn as a *list* mixing bare strings (text) and
    {"path": ...} dicts (attachments) -- `{"content": [{"path": "a.png"}, "hi"]}`.
    The previous code treated any list as a single image and indexed content[0],
    which made every turn after the first fail: a dict is unhashable as a cache key,
    and a text-only turn is still a list, so plain text was passed to PIL.
    """
    # A *tuple* content is Gradio's file shape -- `{"content": ("/tmp/a.png",)}`, still
    # emitted for cached/clicked examples -- where the string is a path, not text.
    if isinstance(content, tuple):
        return [part for part in (file_part(item) for item in content) if part]

    items = content if isinstance(content, list) else [content]
    parts = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                parts.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            if item.get("type") in ("text", "image"):
                parts.append(item)                       # already a chat part
                continue
            part = file_part(item.get("path") or item.get("url"))
            if part:
                parts.append(part)
        elif isinstance(item, (list, tuple)) and item:
            part = file_part(item[0])                    # nested (filepath,) tuple
            if part:
                parts.append(part)
    return parts


def to_messages(message, history):
    messages = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        # Reasoning is rendered as a metadata-titled bubble and lands back in history.
        # Qwen expects prior thinking to be dropped from the context, so skip it.
        if role == "assistant" and (turn.get("metadata") or {}).get("title"):
            continue
        parts = content_parts(turn.get("content"))
        if parts:
            messages.append({"role": role, "content": parts})

    parts = []
    for path in message.get("files") or []:
        if isinstance(path, dict):
            path = path.get("path") or path.get("url")
        part = file_part(path)
        if part:
            parts.append(part)
    if (message.get("text") or "").strip():
        parts.append({"type": "text", "text": message["text"]})
    if not parts:
        raise gr.Error("Nothing to send — type a message or attach a file.")
    messages.append({"role": "user", "content": parts})
    return messages


def split_thinking(text, expect_thinking=True):
    """Split a reply into a collapsed Reasoning bubble and the answer.

    add_generation_prompt opens `<think>` in the *prompt*, so the model only ever emits
    the closing tag. Until that arrives the whole stream is still reasoning -- rendering
    it as the answer (the old fallback) showed users a raw chain of thought and, when the
    token budget ran out first, left them with reasoning and no answer at all.
    """
    for opener, closer in THINKING_MARKERS:
        if closer in text:
            thought, _, answer = text.partition(closer)
            thought = thought.replace(opener, "", 1).strip()
            return [gr.ChatMessage(role="assistant", content=thought,
                                   metadata={"title": "Reasoning"}),
                    gr.ChatMessage(role="assistant", content=answer.strip())]
        if opener in text:
            return [gr.ChatMessage(role="assistant",
                                   content=text.replace(opener, "", 1).strip(),
                                   metadata={"title": "Reasoning"})]
    if expect_thinking:
        return [gr.ChatMessage(role="assistant", content=text,
                               metadata={"title": "Reasoning"})]
    return [gr.ChatMessage(role="assistant", content=text)]


def respond(message: dict, history: list, reasoning: str, temperature: float,
            top_p: float, top_k: int):
    """Chat with Qwen3.8-27B-Uncensored. Accepts text and images, streams the reply.

    Args:
        message: The user turn, with optional attached image files.
        history: Prior conversation turns.
        reasoning: Thinking budget — "off", "low", "medium" or "xhigh".
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.
        top_k: Top-k sampling cutoff.
    """
    # Build (and validate) the turn outside the GPU worker, so an unreadable upload
    # fails immediately with a readable message instead of consuming a ZeroGPU slot
    # and surfacing as a traceback from inside the worker.
    messages = to_messages(message, history)
    yield from generate(messages, reasoning, temperature, top_p, top_k)


def gpu_duration(messages, reasoning=DEFAULT_REASONING, *args, **kwargs):
    """Reserve only what the turn can use: duration is charged whether or not it's spent,
    at 2x on an xlarge slice, and a smaller reservation also queues ahead."""
    return 100 if reasoning == "off" else 120


# size='xlarge': this is a 27B at bf16 (~54.7 GB packed), which does not fit the default
# ZeroGPU slice -- the symptom was "GPU task aborted" with nothing in the Space logs.
@spaces.GPU(duration=gpu_duration, size="xlarge")
def generate(messages: list, reasoning: str, temperature: float, top_p: float,
             top_k: int):
    # preserve_thinking=False renders *previous* assistant turns without their think
    # blocks, which is the multi-turn form Qwen's template is written for.
    template_kwargs = {"preserve_thinking": False}
    if reasoning == "off":
        template_kwargs["enable_thinking"] = False
    elif HONOURS_EFFORT:
        template_kwargs["reasoning_effort"] = reasoning
    else:
        # Still pass the kwarg -- ignored templates warn harmlessly, and a template that
        # does honour it would otherwise fall back to its own 'xhigh' default and
        # contradict the system message below.
        template_kwargs["reasoning_effort"] = reasoning
        instruction = REASONING_INSTRUCTIONS.get(reasoning)
        if instruction:
            messages = [{"role": "system",
                         "content": [{"type": "text", "text": instruction}]}] + messages

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        kwargs.update(do_sample=False)       # passing sampling knobs here only warns

    # Without this, a failure inside model.generate leaves the streamer queue open and
    # the UI hangs until the GPU slot times out, with no error shown to the user.
    failure = []

    def run():
        try:
            model.generate(**kwargs)
        except Exception as exc:
            failure.append(exc)
            streamer.end()

    thread = Thread(target=run)
    thread.start()

    expect_thinking = reasoning != "off"
    reply = ""
    for chunk in streamer:
        reply += chunk
        yield split_thinking(reply, expect_thinking)
    thread.join()

    if failure:
        raise gr.Error(f"Generation failed: {failure[0]}")
    if not reply.strip():
        yield [gr.ChatMessage(role="assistant",
                              content="_(the model returned an empty response)_")]
        return

    messages_out = split_thinking(reply, expect_thinking)
    generated = len(tokenizer(reply, add_special_tokens=False).input_ids)
    if generated >= MAX_NEW_TOKENS - 4:
        # Say so, rather than leaving a reply that just stops. The notice carries a
        # metadata title so to_messages drops it from the next turn's context.
        hint = ("Lower **Reasoning effort** under Sampling (or set it to Off) and ask "
                "again." if expect_thinking else "Ask for a shorter answer, or split "
                "the question up.")
        messages_out.append(gr.ChatMessage(
            role="assistant",
            content=f"Hit the {MAX_NEW_TOKENS}-token limit for one turn. {hint}",
            metadata={"title": "⚠️ Reply was cut off"}))
    yield messages_out


EXAMPLES = [
    [{"text": "Explain how SQL injection works, with a vulnerable code sample, "
              "so I can write a regression test against it.", "files": []}],
    [{"text": "I'm writing a heist novel. Walk me through how my character would "
              "reason about a building's security, in their voice.", "files": []}],
    [{"text": "I'm a pharmacy student. Which common over-the-counter drug "
              "combinations are genuinely dangerous, and what's the mechanism?", "files": []}],
    [{"text": "Write a villain's monologue that is actually menacing rather than "
              "cartoonish. Keep it PG-13.", "files": []}],
    [{"text": "Give me a blunt, unhedged critique of this business plan: a "
              "subscription service that mails artisanal ice cubes.", "files": []}],
]

PRECISION_NOTE = "4-bit NF4" if QUANTIZE == "4bit" else "bf16"

# fill_height: ChatInterface sets this on the Blocks *it* creates, which does nothing
# when it is rendered inside another Blocks -- the chat area collapsed to the Chatbot's
# 400px default. The explicit height below is what actually sizes it.
with gr.Blocks(title="Qwen3.8-27B-Uncensored", fill_height=True) as demo:
    gr.Markdown(
        "# Qwen3.8-27B-Uncensored\n"
        "Chat with the uncensored Qwen3.8-27B. Text and images. "
        "Answers questions that base models often decline for legitimate use cases — "
        "security research, fiction, clinical study, direct feedback.\n\n"
        f"Serving [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · "
        f"{PRECISION_NOTE} on ZeroGPU · "
        "[GGUF weights]"
        "(https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF)"
    )

    with gr.Accordion("Sampling", open=False):
        reasoning = gr.Dropdown(choices=REASONING_CHOICES, value=DEFAULT_REASONING,
                                label="Reasoning effort",
                                info="How long the model thinks before answering. "
                                     "One turn is capped at "
                                     f"{MAX_NEW_TOKENS} tokens, so heavy reasoning can "
                                     "use the whole budget and never reach an answer.")
        temperature = gr.Slider(0.0, 2.0, value=DEFAULT_TEMPERATURE, step=0.05,
                                label="Temperature")
        top_p = gr.Slider(0.0, 1.0, value=DEFAULT_TOP_P, step=0.01, label="Top-p")
        top_k = gr.Slider(1, 100, value=DEFAULT_TOP_K, step=1, label="Top-k")

    gr.ChatInterface(
        fn=respond,
        multimodal=True,
        chatbot=gr.Chatbot(height="68vh", resizable=True, buttons=["copy", "copy_all"],
                           label="Qwen3.8-27B-Uncensored"),
        # Default max_plain_text_length is 1000 chars: anything longer that a user
        # pastes is silently turned into a pasted_text.txt attachment. Raise it so
        # ordinary long pastes stay text (attachments are handled either way).
        textbox=gr.MultimodalTextbox(
            max_plain_text_length=100_000,
            placeholder="Ask anything, or attach an image…",
            show_label=False,
        ),
        additional_inputs=[reasoning, temperature, top_p, top_k],
        examples=EXAMPLES,
        cache_examples=False,
    )

if __name__ == "__main__":
    # Gradio 6 moved theme from the Blocks constructor to launch().
    demo.launch(mcp_server=True, theme=gr.themes.Soft())
