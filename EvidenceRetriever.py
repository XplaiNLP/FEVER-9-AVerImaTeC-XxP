import os
import numpy as np
from logger import logger
from rank_bm25 import BM25Okapi
import json
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import SiglipModel, SiglipImageProcessor, AutoTokenizer

gte_base = SentenceTransformer("thenlper/gte-base")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with safe normalization (handles zero vectors)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


class MMRetriever:
    """
    Multimodal retriever using SigLIP.
    Expects knowledge-store JSON files named {claim_id}.json with one JSON object per line:
      {"url": "https://...", "url2text": ["text chunk 1", "text chunk 2", ...], ...}
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        knowledge_store_dir_path: str = None,
        device: str = None,
    ):
        if knowledge_store_dir_path is None:
            raise ValueError("knowledge_store_dir_path must be provided to MMRetriever")
        self.knowledge_store_dir_path = knowledge_store_dir_path
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer / model / image processor once
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Use the model's own max positional length as global text limit
        self.max_text_tokens = getattr(self.model.config, "max_position_embeddings", 64)
        self.tokenizer.model_max_length = int(1e6)

    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.model.config.projection_dim, dtype=np.float32)

        # Respect the model's max sequence length
        max_len = self.max_text_tokens

        inputs = self.tokenizer(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        return out[0].cpu().numpy()

    def embed_image(self, image_path: str) -> np.ndarray:
        if not image_path:
            return None
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        return out[0].cpu().numpy()

    def embed_pair(self, text: str, image_path: str = None) -> np.ndarray:
        """Return combined text+image embedding (average if image provided)."""
        t = self.embed_text(text)
        if image_path:
            i = self.embed_image(image_path)
            if i is None:
                return t
            # average; can be changed to weighted average if desired
            return (t + i) / 2.0
        return t

    def chunk_text_token_aware(self, text: str, max_tokens: int = None):
        """
        Split a text string into chunks of <= max_tokens token ids,
        returning decoded text chunks.
        """
        if not text:
            return []

        # default: leave a little room for special tokens if any
        if max_tokens is None:
            max_tokens = max(1, self.max_text_tokens - 4)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return []

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            sub_tokens = tokens[i : i + max_tokens]
            # decode back to text using the tokenizer instance
            chunk = self.tokenizer.decode(
                sub_tokens,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk.strip())
        return chunks

    def MM_retrieve(
        self,
        query_text: str,
        query_image: str,
        claim_id: str,
        top_k: int = 5,
        top_n_per_url: int = 3,
    ):
        """
        Read the knowledge file {claim_id}.json, build chunk list, compute similarity to query embedding,
        aggregate per URL and return top_k URLs with aggregated score and concatenated top chunks.
        """
        knowledge_store_file_path = os.path.join(
            self.knowledge_store_dir_path, f"{claim_id}.json"
        )
        if not os.path.exists(knowledge_store_file_path):
            raise FileNotFoundError(
                f"Knowledge file not found: {knowledge_store_file_path}"
            )

        chunks = []
        with open(knowledge_store_file_path, "r", encoding="utf-8") as json_file:
            for line in json_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping malformed JSON line in {knowledge_store_file_path}"
                    )
                    continue

                url = entry.get("url")
                url2text = entry.get("url2text", [])
                if isinstance(url2text, str):
                    url2text = [url2text]
                for text_piece in url2text:
                    for ch in self.chunk_text_token_aware(text_piece):
                        chunks.append({"text": ch, "image": None, "url": url})

        if len(chunks) == 0:
            return []

        # query embedding 
        q_emb = self.embed_pair(query_text or "", query_image)

        url_scores = {}
        url_to_texts = {}

        for c in chunks:
            c_emb = self.embed_pair(c["text"], c["image"])
            score = cosine_sim(q_emb, c_emb)
            url = c.get("url")
            url_scores.setdefault(url, []).append(score)
            url_to_texts.setdefault(url, []).append(c["text"])

        # Aggregate top-N scores per URL
        url_agg_scores = {}
        for url, s_list in url_scores.items():
            top_scores = sorted(s_list, reverse=True)[:top_n_per_url]
            url_agg_scores[url] = float(np.mean(top_scores)) if top_scores else 0.0

        # Sort and return top_k
        top_urls = sorted(
            url_agg_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for url, score in top_urls:
            top_texts = url_to_texts.get(url, [])[: max(3, top_n_per_url)]
            full_text = "\n".join(top_texts)
            results.append({"url": url, "score": score, "text": full_text})
        return results


class BM25Retriever:
    """
    BM25 + embedding refinement retriever.
    Expects same knowledge file format as MMRetriever.
    embedding_model should be a sentence-transformers-like object with .encode(list_or_str, normalize_embeddings=True/False).
    """

    def __init__(self, emb_llm=None, knowledge_store_dir_path: str = None):
        if knowledge_store_dir_path is None:
            raise ValueError("knowledge_store_dir_path must be provided to BM25Retriever")
        # default to a global gte_base if emb_llm not provided 
        if emb_llm is None:
            try:
                emb_llm = gte_base  # global defined earlier
            except NameError:
                raise ValueError("No embedding model provided and global gte_base not found.")
        self.embedding_model = emb_llm
        self.knowledge_store_dir_path = knowledge_store_dir_path
        self.bm25_top_k = 2000
        self.cos_sim_top_k = 10
        self.concatenate_sentences = True
        self.concat_sentence_amount = 4

    def retrieve_evidences(self, claim: str, claim_id: str):
        bm25_top_k_sentences, bm25_urls = self.get_bm25_results_for_claim(claim, claim_id)
        evidences, urls, similarities = self.get_top_k_cos_sim_results_for_claim(claim, bm25_top_k_sentences, bm25_urls)
        return evidences, urls, similarities

    def get_bm25_results_for_claim(self, claim: str, claim_id: str):
        knowledge_store_file_path = os.path.join(self.knowledge_store_dir_path, f"{claim_id}.json")
        sentences = []
        urls = []

        if not os.path.exists(knowledge_store_file_path):
            raise FileNotFoundError(f"Knowledge file not found: {knowledge_store_file_path}")

        with open(knowledge_store_file_path, "r", encoding="utf-8") as json_file:
            for line in json_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {knowledge_store_file_path}")
                    continue
                url2text = entry.get("url2text", [])
                url = entry.get("url")
                if isinstance(url2text, str):
                    url2text = [url2text]
                if self.concatenate_sentences:
                    # take groups of concat_sentence_amount items
                    for i in range(0, len(url2text), self.concat_sentence_amount):
                        chunk = url2text[i : i + self.concat_sentence_amount]
                        concatenated_sentence = " ".join(chunk)
                        sentences.append(concatenated_sentence)
                        urls.append(url)
                else:
                    for s in url2text:
                        sentences.append(s)
                        urls.append(url)

        if len(sentences) == 0:
            return [], []

        # BM25 expects tokenized docs
        tokenized_docs = [nltk.word_tokenize(doc) for doc in sentences]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(claim))
        top_k_idx = np.argsort(scores)[::-1][: min(self.bm25_top_k, len(scores))]
        return [sentences[i] for i in top_k_idx], [urls[i] for i in top_k_idx]

    def get_top_k_cos_sim_results_for_claim(self, claim: str, bm25_top_k_sentences: list, urls: list):
        cos_sim_top_k_sentences = self.retrieve_top_k_sentences_with_cos_sim(claim, bm25_top_k_sentences, urls)
        retrieved_sentences = [s[0] + "\n" for s in cos_sim_top_k_sentences]
        similarities = [s[1] for s in cos_sim_top_k_sentences]
        urls_out = [s[2] + "\n" for s in cos_sim_top_k_sentences]
        return retrieved_sentences, urls_out, similarities

    def retrieve_top_k_sentences_with_cos_sim(self, query: str, sentences: list, urls: list):
        if len(sentences) == 0:
            return []

        # For memory safety, encode in batches if list is large
        # sentence-transformers supports normalize_embeddings=True
        query_embedding = np.asarray(self.embedding_model.encode(query, normalize_embeddings=True))
        sentence_embeddings = np.asarray(self.embedding_model.encode(sentences, normalize_embeddings=True))
        similarities = [cosine_sim(query_embedding, sentence_embeddings[i]) for i in range(len(sentence_embeddings))]
        sentence_similarity_pairs = list(zip(sentences, similarities, urls))
        sorted_pairs = sorted(sentence_similarity_pairs, key=lambda x: x[1], reverse=True)
        return sorted_pairs[: self.cos_sim_top_k]

class EvidenceRetriever:
    """
    Wrapper that holds a BM25Retriever and MMRetriever.
    retrieve_evidences(..., strategy="Text-search"|"Image-search")
    """

    def __init__(self, text_store_dir: str, image_store_dir: str, emb_model=None, siglip_model_name: str = "google/siglip-base-patch16-384", device: str = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # instantiate retrievers once
        self.text_retriever = BM25Retriever(emb_llm=emb_model, knowledge_store_dir_path=text_store_dir)
        self.mm_retriever = MMRetriever(model_name=siglip_model_name, knowledge_store_dir_path=image_store_dir, device=self.device)

    def retrieve_evidences(self, claim_text: str, claim_image: str, claim_id: str, question: str = None, strategy: str = "Text-search"):
        """
        strategy: "Text-search" or "Image-search"
        Returns:
          - for Text-search: (evidences:list[str], urls:list[str], similarities:list[float])
          - for Image-search: results: list of dicts [{"url","score","text"}, ...]
        """
        query_text=" ".join([claim_text, question]).strip()
        if strategy == "Text-search":
            evidences, urls, similarities = self.text_retriever.retrieve_evidences(query_text, claim_id)
            # optionally print or log
            logger.info(f"BM25 retrieved {len(evidences)} evidences for claim_id={claim_id}")
            return evidences, urls, similarities

        elif strategy == "Image-search":
            results = self.mm_retriever.MM_retrieve(query_text, claim_image, claim_id, top_k=5, top_n_per_url=3)
            logger.info(f"MM retrieved {len(results)} results for claim_id={claim_id}")
            return results

        else:
            raise ValueError(f"Unknown strategy: {strategy}")