#!/usr/bin/env python
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Try using project logger, fallback to basic logging
try:
    from logger import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class GenerationLLMHandler:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        image_root: str = "./images",
        use_qwen_vl: bool = True,
        max_new_tokens: int = 512,
        **_,
    ):
        """
        Handler for a Qwen3-VL vision-language model in Transformers.
        """

        # Device: prefer CUDA if available, otherwise CPU.
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Loading {model_name} on device={self.device}")

        # Load processor & model (Qwen3-VL needs trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=None,  # load on CPU first, then move
            trust_remote_code=True,
        )
        self.model.to(self.device)

        self.image_root = image_root
        self.use_qwen_vl = use_qwen_vl
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------
    # Utility: resolve image paths
    # ---------------------------------------------------------
    def resolve_image_paths(self, refs: Optional[List[str]]) -> List[str]:
        if not refs:
            return []
        abs_paths = []
        for r in refs:
            if os.path.isabs(r):
                abs_paths.append(r)
            else:
                abs_paths.append(os.path.join(self.image_root, r))
        return abs_paths

    # ---------------------------------------------------------
    # Build messages for Qwen3-VL
    # ---------------------------------------------------------
    def _build_messages(self, prompt_text: str, image_paths: List[str]):
        """
        Build a chat-like message structure with optional images.
        Qwen3-VL expects each image as a URI/path in the "image" field.
        """
        content = []

        # Add images first
        for path in image_paths:
            uri = Path(path).absolute().as_uri()
            content.append({"type": "image", "image": uri})

        # Then text
        content.append({"type": "text", "text": prompt_text})

        return [{"role": "user", "content": content}]

    # ---------------------------------------------------------
    # Question generation
    # ---------------------------------------------------------
    def generate_question(
        self,
        claim_text: str,
        image_refs: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        imgs = self.resolve_image_paths(image_refs)
        number_of_images = len(imgs)

        if not self.use_qwen_vl:
            return "Text-only model not implemented."

        prompt = f"""
You are a professional fact-checker. Your task is to analyze a claim and the associated {number_of_images} images (if any),
and generate a list of verification questions in a STRICT JSON format.

Each question must:
- Focus on a single, clear factual statement (atomic or meta-level).
- Be legitimate, clearly phrased, and directly related to the claim.
- Be verifiable using retrieved evidence (text or images).
- Be labeled as either text-related or image-related.

For EACH question, you MUST decide:
- "type": "text"  - it is about ONLY with the textual claim.
- "type": "image" - it is about AT LEAST one of the provided images.

Images are numbered in the order given: 1, 2, 3, ...

OUTPUT FORMAT (VERY IMPORTANT):
You MUST output ONLY a valid JSON array. No extra text, no comments, no explanation.

Example format (structure only, not content):

[
  {{
    "id": 1,
    "question": "Question text here",
    "type": "text",
    "images": []
  }},
  {{
    "id": 2,
    "question": "Question text here",
    "type": "image",
    "images": [1, 2]
  }}
]

Rules:
- Generate between 3 and 5 questions.
- You must NOT generate repeated questions.
- If there is at least one image, at least one question MUST be of "type": "image".
- The questions can be about different aspects of the claim.
- "id" must be 1, 2, 3, ... in order.
- If "type" is "text", "images" MUST be [].
- If "type" is "image", "images" MUST be a non-empty list of 1-based image indices.

Claim:
{claim_text}

Now output ONLY the JSON array described above. Do NOT wrap it in backticks and do NOT add any text before or after it.
""".strip()

        # 1) Build messages with images + text (your helper already does this)
        messages = self._build_messages(prompt, imgs)

        # 2) Convert to a chat prompt string (this inserts proper image tokens)
        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 3) Convert (text + images) to tensors
        inputs = self.processor(
            text=chat_prompt,
            images=imgs if imgs else None,
            return_tensors="pt",
        ).to(self.device)

        # 4) Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
            )

        # 5) Decode only the newly generated tokens (strip the prompt)
        generated_ids = output_ids
        if "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]

        out_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        return out_text



    # ---------------------------------------------------------
    # Prediction (Supported / Refuted / Conflicting / Not Enough Evidence)
    # ---------------------------------------------------------
    def generate_prediction(
        self,
        claim: str,
        retrieved_evidences: str,
        image_refs: Optional[List[str]] = None,
        max_tokens: int = 8,
    ):
        imgs = self.resolve_image_paths(image_refs)

        prompt = f"""
You are a professional fact checker.
Given a claim, evidence and images, classify the claim's veracity into EXACTLY ONE of:

- Supported
- Refuted
- Conflicting
- Not Enough Evidence

Definitions:
- Supported: The evidence clearly supports the claim.
- Refuted: The evidence clearly contradicts the claim.
- Conflicting: The evidence both supports AND refutes parts of the claim.
- Not Enough Evidence: Evidence is insufficient, unrelated, or too weak to decide.

Answer with ONLY one of these four labels, nothing else.

Claim:
{claim}

Evidence:
{retrieved_evidences}
""".strip()

        messages = self._build_messages(prompt, imgs)

        # 1) Build chat prompt as STRING
        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 2) Convert (text + images) to model inputs (TENSORS)
        inputs = self.processor(
            text=chat_prompt,
            images=imgs if imgs else None,
            return_tensors="pt",
        ).to(self.device)

        # 3) Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        # 4) Decode only generated tokens (after the prompt)
        generated_ids = output_ids
        if "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]

        raw = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        norm = raw.lower()

        # Robust mapping to the four official labels
        if "conflict" in norm:
            return "Conflicting"
        if "support" in norm and "refut" not in norm:
            return "Supported"
        if "refut" in norm or "false" in norm:
            return "Refuted"
        if "not enough" in norm or "insufficient" in norm or "unknown" in norm:
            return "Not Enough Evidence"

        # Fallback: try exact label, then default
        if raw in ["Supported", "Refuted", "Conflicting", "Not Enough Evidence"]:
            return raw

        return "Not Enough Evidence"
    
    # ---------------------------------------------------------
    # Justification generation
    # ---------------------------------------------------------
    def generate_justification(
        self,
        claim: str,
        retrieved_evidences: str,
        verdict: str,
        image_refs: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a short natural-language justification for the given verdict,
        grounded in the retrieved evidence (and images if provided).
        """
        imgs = self.resolve_image_paths(image_refs)

        prompt = f"""
You are a professional fact checker.

Your task:
Given a claim, a FINAL veracity label, and the retrieved evidence (and optionally images),
write a brief justification that explains why this label is appropriate.

Requirements:
- The justification MUST be consistent with the given label: "{verdict}".
- Base the explanation ONLY on the provided evidence (and images), do not invent facts.
- Refer explicitly to the key pieces of evidence.
- Be concise: 2–4 sentences.
- Do NOT restate the label; focus on the reasoning.

Claim:
{claim}

Final label:
{verdict}

Evidence:
{retrieved_evidences}
""".strip()

        messages = self._build_messages(prompt, imgs)

        # 1) Build chat prompt as STRING
        chat_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 2) Convert (text + images) to model inputs (TENSORS)
        inputs = self.processor(
            text=chat_prompt,
            images=imgs if imgs else None,
            return_tensors="pt",
        ).to(self.device)

        # 3) Generate justification
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
        )

        # 4) Decode only generated tokens (after the prompt)
        generated_ids = output_ids
        if "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        return text



