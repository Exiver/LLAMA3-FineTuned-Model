"""
Streamlit UI for FathyElghoneimy/llama-3-8b-lora-code-gen
A LoRA-fine-tuned Llama-3-8B adapter for code-generation instructions.

Run this inside Google Colab (GPU runtime) — see README_STREAMLIT.md for the
exact Colab cells to launch it with a public URL.
"""

import time
import streamlit as st
import torch

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Llama-3-8B Code Gen (LoRA Fine-Tuned)",
    page_icon="🦙",
    layout="centered",
)

BASE_MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER_ID = "FathyElghoneimy/llama-3-8b-lora-code-gen"
MAX_SEQ_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_ID,          # loads base model + applies the LoRA adapter
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(instruction: str, extra_input: str) -> str:
    """Same Alpaca-style formatting used during training/evaluation."""
    if extra_input.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{extra_input}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, do_sample):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LENGTH).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt back off so we only show the generated response
    if "### Response:" in full_text:
        return full_text.split("### Response:", 1)[1].strip()
    return full_text[len(prompt):].strip()


# ----------------------------------------------------------------------------
# Sidebar — generation settings + project info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_new_tokens = st.slider("Max new tokens", 32, 512, 200, step=16)
    do_sample = st.toggle("Sampling (creative mode)", value=False)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1, disabled=not do_sample)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05, disabled=not do_sample)

    st.divider()
    st.header("📦 Model")
    st.caption(f"Base: `{BASE_MODEL_ID}`")
    st.caption(f"Adapter: `{ADAPTER_ID}`")
    st.caption("Loaded in 4-bit via Unsloth")

    st.divider()
    st.header("🖥️ Device")
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.caption(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
        if vram_gb < 8:
            st.warning(
                "You're on a GPU with < 8GB VRAM. If you hit an out-of-memory "
                "error, lower **Max new tokens** and close other GPU apps "
                "(browser tabs, other models, etc.)."
            )
    else:
        st.warning("No GPU detected — running on CPU will be very slow for an 8B model.")

    st.divider()
    with st.expander("📊 About this fine-tune"):
        st.markdown(
            """
            LoRA-fine-tuned on a curated 640-example code-generation dataset
            (derived from Alpaca). Only **0.52%** of parameters were trained.

            **Eval vs. base model (50 held-out prompts):**
            | Metric | Base | Fine-tuned |
            |---|---|---|
            | ROUGE-L | 0.219 | **0.390** |
            | BERTScore F1 | 0.844 | **0.906** |
            | Judge: Correctness | 3.57 | **4.29** |
            | Judge: Quality | 3.00 | **4.00** |
            """
        )

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.title("🦙 Llama-3-8B — Code Generation Assistant")
st.caption("LoRA fine-tuned on a curated code-generation dataset · Unsloth · Colab T4")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("gen_form"):
    instruction = st.text_area(
        "Instruction",
        placeholder="e.g. Write a Python function that checks if a string is a palindrome.",
        height=100,
    )
    extra_input = st.text_area(
        "Input (optional)",
        placeholder="Extra context or data for the instruction, if any.",
        height=70,
    )
    submitted = st.form_submit_button("🚀 Generate", use_container_width=True)

if submitted:
    if not instruction.strip():
        st.warning("Please enter an instruction first.")
    else:
        with st.spinner("Loading model (first run only)..."):
            model, tokenizer = load_model()

        prompt = build_prompt(instruction, extra_input)

        with st.spinner("Generating..."):
            start = time.time()
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens, temperature, top_p, do_sample,
            )
            elapsed = time.time() - start

        st.subheader("Response")
        st.code(response, language="python")
        st.caption(f"Generated in {elapsed:.1f}s")

        st.session_state.history.insert(0, {
            "instruction": instruction,
            "input": extra_input,
            "response": response,
        })

if st.session_state.history:
    st.divider()
    st.subheader("🕘 History")
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"{i+1}. {item['instruction'][:70]}"):
            if item["input"]:
                st.markdown(f"**Input:** {item['input']}")
            st.code(item["response"], language="python")

st.divider()
st.caption(
    "Model: [FathyElghoneimy/llama-3-8b-lora-code-gen](https://huggingface.co/FathyElghoneimy/llama-3-8b-lora-code-gen) · "
    "Dataset: [alpaca-code-generation-curated](https://huggingface.co/datasets/FathyElghoneimy/alpaca-code-generation-curated)"
)
