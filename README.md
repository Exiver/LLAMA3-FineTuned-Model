# Llama-3-8B LoRA Fine-Tuning for Code Generation

Fine-tuning and rigorous evaluation of **Llama-3-8B** on a curated code-generation dataset using **LoRA** (via Unsloth), benchmarked against the base model with ROUGE-L, BERTScore, and an LLM-as-judge rubric.

- 🤗 Adapter: [`FathyElghoneimy/llama-3-8b-lora-code-gen`](https://huggingface.co/FathyElghoneimy/llama-3-8b-lora-code-gen)
- 🤗 Dataset: [`FathyElghoneimy/alpaca-code-generation-curated`](https://huggingface.co/datasets/FathyElghoneimy/alpaca-code-generation-curated)
- Base model: `unsloth/llama-3-8b-bnb-4bit`
- Hardware: Google Colab, single Tesla T4 (16GB)

---

## 📁 Project Structure

| Notebook | Purpose |
|---|---|
| `DataSet_custumization.ipynb` | Builds a clean, code-generation-focused dataset from `tatsu-lab/alpaca` and pushes it to the Hub |
| `Fine_Tuned_Model.ipynb` | LoRA fine-tuning of Llama-3-8B with Unsloth + TRL's `SFTTrainer` |
| `base_VS_FineTunedModel.ipynb` | Head-to-head evaluation: base model vs. fine-tuned adapter (ROUGE-L, BERTScore, LLM-as-judge) |

---

## 1. Dataset Curation

Starting from `tatsu-lab/alpaca`, the dataset was filtered and cleaned to focus specifically on code-generation tasks:

- **Domain filtering** — kept only instructions containing code-related keywords (`python`, `javascript`, `html`, `css`, `c++`, `java`, `function`, `algorithm`, `code`, `script`, `debug`, `sql`)
- **De-duplication** on `(instruction, input, output)` triples
- **Text cleaning** — trimmed whitespace and collapsed excessive newlines
- **Quality filter** — discarded examples with an output shorter than 30 characters
- **Downsampled** to 800 examples (seed = 42) and split:

| Split | Examples |
|---|---|
| Train | 640 |
| Validation | 80 |
| Test | 80 |

Final dataset published as [`FathyElghoneimy/alpaca-code-generation-curated`](https://huggingface.co/datasets/FathyElghoneimy/alpaca-code-generation-curated).

---

## 2. Fine-Tuning

LoRA fine-tuning was performed with **Unsloth** for 2x faster training on a single T4 GPU.

**LoRA config**
| Param | Value |
|---|---|
| Rank (`r`) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | 41,943,040 / 8,072,204,288 (**0.52%**) |

**Training config**
| Param | Value |
|---|---|
| Epochs | 4 |
| Effective batch size | 8 (2 × 4 grad. accumulation) |
| Learning rate | 2e-4 (linear schedule) |
| Optimizer | `adamw_8bit` |
| Precision | fp16 |
| Total steps | 320 |

The fine-tuned adapter was pushed to the Hub as `FathyElghoneimy/llama-3-8b-lora-code-gen`.

---

## 3. Evaluation Methodology

50 held-out test examples (random seed = 42) were used to compare the **base model** vs. the **fine-tuned adapter** on identical prompts:

1. **ROUGE-L** — n-gram/sequence overlap with the reference response
2. **BERTScore (F1)** — semantic similarity using `roberta-large` embeddings
3. **LLM-as-judge** — `gemini-2.5-flash` scored each response 1–5 on **correctness**, **completeness**, and **quality**, given the instruction and reference. Candidate order (base vs. fine-tuned) was **randomized per example** to avoid position bias.

Both models were loaded, run, and then explicitly freed from GPU memory between passes (`del model; gc.collect(); torch.cuda.empty_cache()`) to fit evaluation within the T4's VRAM budget.

---

## 📊 Results

| Metric | Base Model | Fine-Tuned Model | Δ | % Change |
|---|---|---|---|---|
| ROUGE-L | 0.2191 | **0.3897** | +0.1706 | **+77.8%** |
| BERTScore-F1 | 0.8436 | **0.9064** | +0.0629 | **+7.5%** |
| Judge — Correctness (1–5) | 3.57 | **4.29** | +0.71 | **+20.0%** |
| Judge — Completeness (1–5) | 3.14 | **3.86** | +0.71 | **+22.7%** |
| Judge — Quality (1–5) | 3.00 | **4.00** | +1.00 | **+33.3%** |

> ⚠️ Note: the LLM-judge run hit Gemini free-tier rate limits (429 errors) partway through, so the judge scores above are averaged over the subset of the 50 examples that returned a valid score, not the full set.

---

## 💡 Insights

- **Fine-tuning delivered the largest lift on lexical overlap (ROUGE-L, +77.8%)**, indicating the LoRA adapter learned to reproduce the dataset's specific formatting and phrasing conventions much more closely than the base model.
- **BERTScore improved more modestly (+7.5%)**, which makes sense — the base Llama-3-8B already produces semantically reasonable answers; fine-tuning mainly sharpened surface-level alignment with the target style rather than teaching wholly new capabilities.
- **The LLM-judge agreed with the automatic metrics directionally**, rating the fine-tuned model higher across all three qualitative dimensions, with **quality (readability/idiomatic style, +33.3%)** showing the biggest jump. This suggests the adapter nudged the model toward cleaner, more consistent code style rather than just "correctness."
- **Correctness and completeness improved by comparable margins (~20–23%)**, implying the fine-tuned model isn't just more stylistically consistent — it's also more reliably solving the actual instruction and covering what's asked, not only producing prettier prose.
- **Small-scale LoRA (0.52% of parameters trained) was enough to produce a measurable, consistent improvement** across every metric — a strong signal that parameter-efficient fine-tuning is well-suited to this kind of narrow domain adaptation (general Alpaca-style instructions → code-generation-specific responses).
- **Evaluation robustness matters**: randomizing candidate order (A/B) before sending to the judge model was a deliberate step to prevent positional bias from inflating either model's score — a common pitfall in LLM-as-judge setups.
- **Practical constraint uncovered**: the free-tier Gemini API rate limit (5 requests/min) became a bottleneck for judge-based evaluation at scale, meaning the judge numbers here reflect a partial sample of the 50 test prompts rather than the complete set — worth budgeting for a paid tier or a self-hosted judge model in future iterations.

---

## 🛠️ Tech Stack

`Unsloth` · `TRL (SFTTrainer)` · `PEFT / LoRA` · `Transformers` · `bitsandbytes (4-bit)` · `Hugging Face Hub & Datasets` · `evaluate` (ROUGE) · `bert-score` · `Gemini 2.5 Flash` (LLM-as-judge) · `pandas` / `matplotlib`

---

## 🔮 Future Work

- Re-run the LLM-judge evaluation on the **full 50-example set** using a paid API tier (or a self-hosted judge such as a local Llama/Qwen model) to eliminate the rate-limit gap.
- Expand the training set beyond 640 examples and explore more epochs / higher LoRA rank to see if gains continue to scale.
- Add a production-facing demo (e.g., a small Gradio/Streamlit app or API endpoint) to showcase the fine-tuned adapter interactively.
- Benchmark against additional baselines (e.g., a full fine-tune or a different open model) for broader context.
