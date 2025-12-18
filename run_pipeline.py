import os
import json
import argparse
import re
from datasets import load_dataset
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from GenerationLLMHandler import GenerationLLMHandler
from EvidenceRetriever import EvidenceRetriever  
import json
from typing import List, Tuple, Dict


# -------------------------
# Question parsing
# -------------------------


def parse_questions(raw: str):
    """
    Parse JSON output into:
      - all_questions: list[str]
      - text_questions: list[str]
      - image_questions: list[str]
      - image_q_with_indices: list[{"text": str, "indices": List[int]}]

    Robust to:
    - ```json ... ``` fences
    - trailing explanation text
    - duplicated questions
    """

    if not raw or not raw.strip():
        return [], [], [], []

    s = raw.strip()

    # 1) Strip markdown fences like ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        # remove opening fence with optional language
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        # remove trailing ``` if present
        if s.endswith("```"):
            s = s[:-3].strip()

    # 2) Try to load as JSON directly
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON array between first '[' and last ']'
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                # give up, return everything as a single text question
                q = raw.strip()
                return ([q] if q else []), ([q] if q else []), [], []
        else:
            q = raw.strip()
            return ([q] if q else []), ([q] if q else []), [], []

    if not isinstance(data, list):
        # Not the expected structure, treat whole thing as one text question
        q = raw.strip()
        return ([q] if q else []), ([q] if q else []), [], []

    questions_meta: List[Dict] = []
    seen_questions = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        q_text = str(item.get("question", "")).strip()
        if not q_text:
            continue

        # Deduplicate by lowercased question text
        key = q_text.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)

        q_type = str(item.get("type", "text")).strip().lower()
        images = item.get("images", [])

        if not isinstance(images, list):
            images = []

        # normalize indices to int list
        indices = []
        for v in images:
            try:
                indices.append(int(v))
            except (TypeError, ValueError):
                continue

        # Normalize type
        if q_type == "image":
            q_type_norm = "Image-related"
        else:
            q_type_norm = "Text-related"

        questions_meta.append(
            {
                "text": q_text,
                "type": q_type_norm,
                "indices": indices,
            }
        )

    all_questions = [q["text"] for q in questions_meta]
    text_questions = [q["text"] for q in questions_meta if q["type"] == "Text-related"]
    image_questions = [q["text"] for q in questions_meta if q["type"] == "Image-related"]

    image_q_with_indices = [
        {"text": q["text"], "indices": q["indices"]}
        for q in questions_meta
        if q["type"] == "Image-related"
    ]

    return all_questions, text_questions, image_questions, image_q_with_indices



# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(
    image_dir: str,
    text_store_dir: str,
    image_store_dir: str,
    output_path: str,
    split: str = "validation",
    num_instances: int | None = None,
):
    # 1. Load dataset
    print(f"Loading dataset Rui4416/AVerImaTeC [{split}] ...")
    ds = load_dataset("Rui4416/AVerImaTeC")
    # currently you only take 1 example – keep as you had it
    data = ds[split]

    if num_instances is not None:
        data = data.select(range(min(num_instances, len(data))))

    print(f"Number of instances: {len(data)}")

    # 2. Load models
    print("Loading generation LLM handler...")
    gen = GenerationLLMHandler(image_root=image_dir, model_name='Qwen/Qwen3-VL-8B-Instruct')

    print("Loading embedding model...")
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading retriever...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = EvidenceRetriever(
        text_store_dir=text_store_dir,
        image_store_dir=image_store_dir,
        emb_model=emb,
        siglip_model_name="google/siglip-base-patch16-384",
        device=device,
    )

    outputs = []

    # 3. Iterate over claims
    for idx, example in enumerate(tqdm(data, desc="Batches")):
        claim = example["claim_text"]

       

        # Raw entries from the dataset; may already contain "./images/" or "images/"
        claim_imgs_raw = example.get("claim_images", []) or []
        

        # Strip any directory (./images/, images/, whatever) → keep only filename
        claim_imgs = [os.path.basename(str(p)) for p in claim_imgs_raw]
       

        # Build actual paths on disk for the LLM (Qwen2-VL)
        img_paths = [os.path.join(image_dir, img_name) for img_name in claim_imgs]
        

        
        

        

        # -------------------------------
        # Step 1: Question generation
        # -------------------------------
        raw_q = gen.generate_question(
            claim_text=claim,
            image_refs=claim_imgs,  # full valid paths, e.g. "./images/6787....jpg"
            max_tokens=512,
            temperature=0.2,
        )
        

        all_questions, text_questions, image_questions, image_q_with_indices = parse_questions(raw_q)
        

        # -------------------------------
        # Step 2: Retrieve evidence from
        #         BOTH text & image channels
        # -------------------------------
        all_evid_texts = []
        all_evid_urls = []
        all_evid_scores = []

        # ---- 2A. Text-based retrieval ----
        if text_questions:
            text_query = claim + " " + " ".join(text_questions)

            text_evids, text_urls, text_scores = retriever.retrieve_evidences(
                claim_text=claim,
                claim_image=None,
                claim_id=str(idx),
                question=text_query,
                strategy="Text-search",
            )

            all_evid_texts.extend(text_evids)
            all_evid_urls.extend(text_urls)
            all_evid_scores.extend(text_scores)

        # ---- 2B. Image-based retrieval ----
        if image_q_with_indices and img_paths:
            for iq in image_q_with_indices:
                q_text = iq["text"]
                indices = iq["indices"]

                img_path_for_q = None
                for k in indices:
                    pos = k - 1
                    if 0 <= pos < len(img_paths):
                        img_path_for_q = img_paths[pos]
                        break
                if img_path_for_q is None:
                    continue

                img_query = claim + " " + q_text

                image_hits = retriever.retrieve_evidences(
                    claim_text=claim,
                    claim_image=img_path_for_q,
                    claim_id=str(idx),
                    question=img_query,
                    strategy="Image-search",
                )

                # Merge results into the global evidence lists
                for hit in image_hits:
                    all_evid_texts.append(hit.get("text", ""))
                    all_evid_urls.append(hit.get("url", ""))
                    all_evid_scores.append(hit.get("score", 0.0))


        # Optional: deduplicate evidence texts (simple)
        seen = set()
        dedup_evid_texts = []
        dedup_evid_urls = []
        dedup_evid_scores = []
        for t, u, s in zip(all_evid_texts, all_evid_urls, all_evid_scores):
            key = (t, u)
            if key in seen:
                continue
            seen.add(key)
            dedup_evid_texts.append(t)
            dedup_evid_urls.append(u)
            dedup_evid_scores.append(s)

        all_evid_texts = dedup_evid_texts
        all_evid_urls = dedup_evid_urls
        all_evid_scores = dedup_evid_scores
        # Build evidence items for output
        evidence_items = []
        for t, u, s in zip(all_evid_texts, all_evid_urls, all_evid_scores):
            evidence_items.append({
                "text": t,
                "images": []   # always empty for now
            })

        if not all_evid_texts:
            joined_evidence = "No relevant evidence was retrieved."
        else:
            joined_evidence = "\n".join(all_evid_texts)

        # -------------------------------
        # Step 3: Predict verdict
        # -------------------------------
        verdict = gen.generate_prediction(
            claim=claim,
            retrieved_evidences=joined_evidence,
            image_refs=claim_imgs,
            max_tokens=64        )

        # -------------------------------
        # Step 4: Generate justification
        # -------------------------------
        justification = gen.generate_justification(
            claim=claim,
            retrieved_evidences=joined_evidence,
            verdict=verdict,
            image_refs=claim_imgs,
            max_tokens=256,
            
        )

        # -------------------------------
        # Step 5: Collect output
        # -------------------------------
        out_item = {
            "id": str(idx),  # normalized filenames
            "questions": all_questions,
            "evidence": evidence_items,
            "verdict": verdict,
            "justification": justification,
        }
        outputs.append(out_item)

    # 4. Save results as a single JSON file (list of dicts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(outputs)} items to {output_path}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./images",
        help="Directory containing the claim images referenced in the dataset.",
    )
    parser.add_argument(
        "--text-store-dir",
        type=str,
        default="./text_related/text_related_store_text_train",
        help="Directory of the text evidence store.",
    )
    parser.add_argument(
        "--image-store-dir",
        type=str,
        default="./text_related/image_related_store_text_train",
        help="Directory of the image evidence store metadata.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./outputs/submission.json",
        help="Where to write the JSON output.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (validation / test / train).",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="If given, only process the first N instances.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        image_dir=args.image_dir,
        text_store_dir=args.text_store_dir,
        image_store_dir=args.image_store_dir,
        output_path=args.output_path,
        split=args.split,
        num_instances=args.num_instances,
    )
