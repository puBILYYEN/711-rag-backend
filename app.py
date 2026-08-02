# -*- coding: utf-8 -*-
"""RAG API：接收問題 -> Chroma 檢索相關切塊 -> 呼叫 OpenRouter LLM 生成回答。"""
import os
from pathlib import Path

import chromadb
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "711_project_docs"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

app = FastAPI(title="711 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("載入 embedding 模型...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(COLLECTION_NAME)
print("RAG API 準備就緒")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/api/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "請輸入問題。", "sources": []}

    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=4)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[來源：{m['source']}]\n{d}" for d, m in zip(docs, metas)
    )
    sources = sorted({m["source"] for m in metas})

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
