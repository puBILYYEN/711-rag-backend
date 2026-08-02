# -*- coding: utf-8 -*-
"""RAG API（輕量版）：讀取預先算好的向量 JSON -> numpy 算 cosine 相似度 -> 呼叫 OpenRouter LLM 生成回答。
不依賴 chromadb，避免免費方案記憶體超限。"""
import csv
import io
import json
import os
import statistics
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
model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
print(f"RAG API 準備就緒，{len(DOCS)} 個切塊")


class AskRequest(BaseModel):
    question: str


class ValidateRequest(BaseModel):
    filename: str
    content: str


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "（後端尚未設定 OPENROUTER_API_KEY，無法生成報告）"
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
        return data["choices"][0]["message"]["content"]


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
    answer = await call_llm(system_prompt, user_prompt)
    return {"answer": answer, "sources": sources}


def analyze_csv(content: str) -> dict:
    """純程式統計檢查，不靠 AI：缺值/型別/離群值。"""
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return {"error": "空檔案，沒有任何內容"}

    header = rows[0]
    data_rows = rows[1:]
    n = len(data_rows)
    col_stats = []

    for ci, col_name in enumerate(header):
        values = [r[ci] if ci < len(r) else "" for r in data_rows]
        blanks = sum(1 for v in values if v.strip() == "")
        numeric_vals = []
        non_numeric_non_blank = 0
        for v in values:
            v = v.strip()
            if v == "":
                continue
            try:
                numeric_vals.append(float(v))
            except ValueError:
                non_numeric_non_blank += 1

        stat = {
            "column": col_name,
            "missing_count": blanks,
            "missing_pct": round(blanks / n * 100, 1) if n else 0,
            "non_numeric_count": non_numeric_non_blank,
        }
        if numeric_vals and non_numeric_non_blank == 0:
            stat["numeric"] = True
            stat["min"] = min(numeric_vals)
            stat["max"] = max(numeric_vals)
            stat["mean"] = round(statistics.mean(numeric_vals), 2)
            if len(numeric_vals) > 1:
                sd = statistics.stdev(numeric_vals)
                mean = statistics.mean(numeric_vals)
                outliers = sum(1 for v in numeric_vals if sd > 0 and abs(v - mean) > 3 * sd)
                stat["outlier_count_3sigma"] = outliers
            if any(v < 0 for v in numeric_vals):
                stat["has_negative"] = True
            if any(v == 0 for v in numeric_vals):
                stat["zero_count"] = sum(1 for v in numeric_vals if v == 0)
        else:
            stat["numeric"] = False

        col_stats.append(stat)

    duplicate_rows = n - len(set(tuple(r) for r in data_rows))

    return {
        "row_count": n,
        "columns": header,
        "duplicate_rows": duplicate_rows,
        "column_stats": col_stats,
    }


@app.post("/api/validate_csv")
async def validate_csv(req: ValidateRequest):
    stats = analyze_csv(req.content)
    if "error" in stats:
        return {"status": "error", "report": stats["error"], "stats": stats}

    system_prompt = (
        "你是資料品質稽核助理，任務是評估使用者剛上傳的原始 CSV 資料。"
        "你的評估風格要參考本專案團隊的方法論原則：\n"
        "1. 缺值不能當作 0 處理或忽略不提——0 有其字面意義（例如「全台最便宜」），"
        "誤填會扭曲後續分析，缺值一律要明確點出筆數與比例。\n"
        "2. 誠實揭露優於隱藏：資料有問題就直接講、不要粉飾，並具體指出可能造成的後續影響。\n"
        "3. 異常值/離群值要點出來，並提醒使用者根查是資料輸入錯誤還是真實特例。\n"
        "4. 不要編造資料裡沒有的資訊，只根據提供的統計數據評估。\n"
        "輸出格式：先一句話總結資料品質等級（良好/需留意/建議先修正再使用），"
        "接著條列具體發現（每點附數字），最後給使用者具體建議。全程繁體中文，簡潔。"
    )
    user_prompt = f"檔名：{req.filename}\n\n統計結果（JSON）：\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
    report = await call_llm(system_prompt, user_prompt)

    has_issues = any(
        c.get("missing_pct", 0) > 0 or c.get("non_numeric_count", 0) > 0 or c.get("outlier_count_3sigma", 0) > 0
        for c in stats["column_stats"]
    ) or stats["duplicate_rows"] > 0

    return {
        "status": "needs_attention" if has_issues else "clean",
        "report": report,
        "stats": stats,
    }
