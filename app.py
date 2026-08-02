# -*- coding: utf-8 -*-
"""RAG API（輕量版）：讀取預先算好的向量 JSON -> numpy 算 cosine 相似度 -> 呼叫 OpenRouter LLM 生成回答。
不依賴 chromadb，避免免費方案記憶體超限。"""
import json
import os
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

VECTORS_FILE = Path(__file__).parent / "vectors.json"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

app = FastAPI(title="711 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("載入向量資料...")
_data = json.loads(VECTORS_FILE.read_text(encoding="utf-8"))
DOCS = _data["documents"]
METAS = _data["metadatas"]
EMB_MATRIX = np.array(_data["embeddings"], dtype=np.float32)
EMB_NORMS = EMB_MATRIX / np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True)

print("載入 embedding 模型 (ONNX)...")
model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"RAG API 準備就緒，{len(DOCS)} 個切塊")


class AskRequest(BaseModel):
    question: str


def search(query: str, k: int = 4):
    q_emb = next(model.embed([query]))
    q_norm = q_emb / np.linalg.norm(q_emb)
    scores = EMB_NORMS @ q_norm
    top_idx = np.argsort(-scores)[:k]
    return [(DOCS[i], METAS[i], float(scores[i])) for i in top_idx]


@app.get("/")
def health():
    return {"status": "ok", "chunks": len(DOCS)}


@app.post("/api/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "請輸入問題。", "sources": []}

    results = search(question)
    context = "\n\n".join(f"[來源：{m['source']}]\n{d}" for d, m, _ in results)
    sources = sorted({m["source"] for _, m, _ in results})

    if not OPENROUTER_API_KEY:
        return {
            "answer": "（後端尚未設定 OPENROUTER_API_KEY，先回傳檢索到的原始片段）\n\n" + context[:800],
            "sources": sources,
        }

    system_prompt = (
        "你是 7-ELEVEN 門市月營收 AI 智慧回推系統的客服助理。"
        "只根據下面提供的參考資料回答使用者問題，不要編造資料裡沒有的數字或結論。"
        "如果參考資料不足以回答，請誠實告知使用者。用繁體中文回答，簡潔清楚。"
    )
    user_prompt = f"參考資料：\n{context}\n\n使用者問題：{question}"

    async with httpx.AsyncClient(timeout=30) as http_client:
        resp = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

    return {"answer": answer, "sources": sources}
