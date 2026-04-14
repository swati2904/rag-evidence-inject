# RAG evidence-present injection benchmark (TRIM)

Code for multi-document RAG under **indirect prompt injection** when **labeled supporting passages are still in the candidate pool** (“evidence-present” settings), plus **trust-role isolation (TRIM)** preprocessing and evaluation scripts.

## Definitions

| Term | Meaning here |
|------|----------------|
| **Gold document** | Passage marked as supporting the reference answer for that benchmark item. |
| **Poison document** | Synthetic adversarial passage inserted into the same retrieval pool for evaluation. |
| **Evidence-present** | The gold passage remains in the pool so the experiment targets generation under conflicting context, not retrieval miss alone. |

## Setup

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1

pip install -e .
```

`pip install -e .` installs `accelerate`, used with `device_map="auto"` in `generation/runner.py` for `--hf-fallback`. Large checkpoints need sufficient disk (on the order of tens of GB for 7B-class weights). Model and tokenizer names are in `configs/pilot.yaml`. Benchmark poison text is synthetic.

## Running experiments

**1. Retrieval sanity (BM25 top-*k*; no LLM)** — fraction of examples where gold appears in top-*k* on clean pools:

```bash
python scripts/run_clean_sanity.py --config configs/pilot.yaml --limit 40
```

Optional dense retrieval (downloads `BAAI/bge-large-en-v1.5`):

```bash
python scripts/run_clean_sanity.py --limit 20 --dense
```

**2. Pilot benchmark** — OpenAI-compatible HTTP API (e.g. vLLM) unless `--hf-fallback`:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
python scripts/run_pilot.py --config configs/pilot.yaml
```

Dry run without an LLM:

```bash
python scripts/run_pilot.py --no-llm --max-examples 10
```

**3. TRIM ablation**

```bash
python scripts/run_ablation.py --subset 16 --no-llm
```

**4. LaTeX tables from logged JSON**

```bash
python scripts/export_tables.py
cd paper && pdflatex main.tex
```

## Repository layout

| Path | Purpose |
|------|---------|
| `benchmark/datasets.py` | Load pilot slices (e.g. KILT NQ, HotpotQA-style rows). |
| `benchmark/corpus_builder.py` | Build per-question document pools (gold, distractors, poison) and ranks. |
| `benchmark/templates.py` | Poison text patterns / families. |
| `retrieval/` | BM25, dense FAISS index, hybrid blend, helpers to turn ranked IDs into context. |
| `generation/` | Prompt assembly and inference (HTTP API or HF weights). |
| `defense/trim.py` | Span masking for directive-like text in poison passages. |
| `evaluation/` | EM/F1, rule-based ASR, JSONL run logging. |
| `scripts/` | CLI entry points wired to `configs/*.yaml`. |
| `configs/` | `pilot.yaml`, `main_experiment.yaml`. |
| `paper/` | LaTeX article source; table snippets under `paper/tables/`. |
| `load_config.py` | YAML loader with `${VAR:-default}` substitution. |

## Natural Questions (KILT)

KILT NQ rows often provide **identifiers without full Wikipedia paragraph text**. `benchmark/datasets.py` can build **short surrogate passages** from metadata so retrieval and poisoning run end-to-end. For experiments that require **full gold passage text**, swap in a corpus that stores passages explicitly (e.g. a Wikipedia join).
