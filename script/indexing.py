import os
import json
import random
from typing import List, Dict, Any

import tiktoken
from tqdm.auto import tqdm
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

from model import get_embedding, embedding_client
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

_enc = tiktoken.get_encoding("cl100k_base")

class LCEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        chunks_per_doc = [chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP) for t in texts]
        flat_chunks: List[str] = []
        counts: List[int] = []
        for cs in chunks_per_doc:
            counts.append(len(cs))
            flat_chunks.extend(cs)
        flat_embs = embed_texts_batch(flat_chunks, batch_size=16)
        out: List[List[float]] = []
        i = 0
        for c in counts:
            embs = flat_embs[i:i + c]
            i += c
            out.append(aggregate_embeddings(embs))
        return out

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = _enc.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    out = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        seg = tokens[start:end]
        if not seg:
            break
        out.append(_enc.decode(seg))
        if end >= len(tokens):
            break
        start += step
    return out


def aggregate_embeddings(vs: List[List[float]]) -> List[float]:
    if not vs:
        return []
    dim = len(vs[0])
    acc = [0.0] * dim
    for v in vs:
        for i in range(dim):
            acc[i] += v[i]
    return [x / len(vs) for x in acc]


def embed_texts_batch(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if not batch:
            continue
        resp = embedding_client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def embed_document(text: str) -> List[float]:
    parts = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if len(parts) == 1:
        return get_embedding(parts[0])
    embs = [get_embedding(p) for p in parts]
    return aggregate_embeddings(embs)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except Exception:
            continue
    return lines


def group_by_id(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        rid = r.get("id")
        if rid is None:
            continue
        docs = r.get("documents")
        if isinstance(docs, list):
            out[str(rid)] = []
            for d in docs:
                text = d.get("text") or ""
                doc_id = d.get("doc_id") or d.get("id")
                if text and doc_id is not None:
                    out[str(rid)].append({"doc_id": str(doc_id), "text": text})
            continue
        doc_id = r.get("doc_id") or r.get("docId")
        text = r.get("text") or r.get("content")
        if doc_id is not None and text:
            out.setdefault(str(rid), []).append({"doc_id": str(doc_id), "text": text})
    return out


def build_faiss_for_id(rid: str, docs: List[Dict[str, Any]], index_root: str) -> str:
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    seen = set()
    for d in docs:
        doc_id = d.get("doc_id")
        text = d.get("text")
        if not doc_id or not text:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        texts.append(text)
        metas.append({"id": rid, "doc_id": doc_id})
    if not texts:
        return ""
    vs = FAISS.from_texts(texts, embedding=LCEmbeddings(), metadatas=metas)
    out_dir = os.path.join(index_root, f"id_{rid}")
    os.makedirs(out_dir, exist_ok=True)
    vs.save_local(out_dir)
    return out_dir


def build_indices(data_path: str, index_root: str, sample_size: int = 200, seed: int = 42) -> List[str]:
    records = load_dataset(data_path)
    groups = group_by_id(records)
    ids = list(groups.keys())
    if not ids:
        return []
    random.seed(seed)
    picked = ids if len(ids) <= sample_size else random.sample(ids, sample_size)
    os.makedirs(index_root, exist_ok=True)
    out_paths = []
    for rid in tqdm(picked, desc="Build FAISS", unit="id"):
        p = build_faiss_for_id(rid, groups.get(rid, []), index_root)
        if p:
            out_paths.append(p)
    return out_paths

if __name__ == "__main__":
    data_path = "../data/valid.json"
    index_root = "faiss_indexes"
    build_indices(data_path, index_root, sample_size=200, seed=42)