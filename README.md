# Improving Automated Verification of Image–Text Claims Through Structured Reasoning

This repo documents the code for our submission within the [FEVER-9 Shared Task](https://fever.ai/task.html).

## Files
- `run_pipeline.py` — CLI entry point that loads the dataset and runs retrieval + generation.
- `EvidenceRetriever.py` — text (BM25 + embedding refinement) and image (SigLIP) retrieval.
- `GenerationLLMHandler.py` — wrapper around a HF vision-language model (default: Qwen/Qwen3-VL-8B-Instruct).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you run in an offline environment, make sure you have already downloaded the required HF models and the NLTK tokenizer data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Run

```bash
python run_pipeline.py   --image-dir ./images   --text-store-dir ./text_related/text_related_store_text_train   --image-store-dir ./text_related/image_related_store_text_train   --output-path ./outputs/submission.json
```
You can find the information on how to download the corresponding image and knowledge store here: https://fever.ai/task.html 

### Notes
- Model downloads are handled by Hugging Face Transformers / SentenceTransformers. If you need private model access, set `HUGGINGFACE_HUB_TOKEN` in your environment.
