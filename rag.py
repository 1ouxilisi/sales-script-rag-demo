# -*- coding: utf-8 -*-
"""
销售接待与异议处理话术 RAG 演示原型
- 向量化：Ollama bge-m3
- 对话：Ollama qwen3:8b（请求时 think=false）
- 检索：numpy 余弦相似度（无 FAISS/Chroma 重依赖）
用法：
  py rag.py --run      # 跑 questions.txt 里的演示问题，结果写入 answers.md
  py rag.py            # 进入交互式问答
"""
import os
import re
import glob
import json
import sys
import math
import numpy as np
import ollama

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "docs")
CACHE_NPZ = os.path.join(BASE, "embeddings.npz")
CACHE_JSON = os.path.join(BASE, "chunks.json")
QUESTIONS_FILE = os.path.join(BASE, "questions.txt")
ANSWERS_FILE = os.path.join(BASE, "answers.md")

EMBED_MODEL = "bge-m3:latest"
CHAT_MODEL = "qwen3:8b"
CHUNK_SIZE = 320
CHUNK_OVERLAP = 60
TOP_K = 3

# 注意：OLLAMA_HOST 在 Ollama 里是“服务监听地址”，不要直接拿来当客户端连接地址，
# 否则监听被设成 0.0.0.0 时客户端会连不上。客户端地址用独立的 OLLAMA_API。
OLLAMA_API = os.environ.get("OLLAMA_API", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_API)


# ---------- 文档切分 ----------
def split_chunks(text: str, source: str):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= CHUNK_SIZE:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append({"source": source, "text": buf})
            buf = p
    if buf:
        chunks.append({"source": source, "text": buf})
    # 简单重叠：相邻块借尾部一段，提升召回
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0 and CHUNK_OVERLAP > 0:
            tail = chunks[i - 1]["text"][-CHUNK_OVERLAP:]
            c = {"source": c["source"], "text": tail + "\n" + c["text"]}
        overlapped.append(c)
    return overlapped


def load_chunks():
    chunks = []
    patterns = ["**/*.md", "**/*.txt"]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(DOCS_DIR, pat), recursive=True)
    for path in sorted(set(files)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks += split_chunks(text, os.path.basename(path))
    return chunks


def cache_signature(chunks):
    return json.dumps([[c["source"], len(c["text"])] for c in chunks], ensure_ascii=False)


def embed(text: str) -> np.ndarray:
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(resp["embedding"], dtype=np.float32)


def build_index(chunks):
    """带缓存：文档没变就复用向量，避免每次重复向量化。"""
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON):
        try:
            with open(CACHE_JSON, encoding="utf-8") as f:
                saved = json.load(f)
            if saved["signature"] == cache_signature(chunks):
                data = np.load(CACHE_NPZ)
                print("已复用本地向量缓存。")
                return chunks, data["embeddings"]
        except Exception as e:
            print(f"缓存读取失败，重建索引：{e}")

    print(f"正在向量化 {len(chunks)} 个文本块（模型 {EMBED_MODEL}）...")
    embs = np.vstack([embed(c["text"]) for c in chunks])
    np.savez(CACHE_NPZ, embeddings=embs)
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump({"signature": cache_signature(chunks), "chunks": chunks}, f, ensure_ascii=False)
    return chunks, embs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve(chunks, embs, question, top_k=TOP_K):
    q_emb = embed(question)
    scores = [cosine(q_emb, e) for e in embs]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = [{"source": chunks[i]["source"], "text": chunks[i]["text"], "score": round(scores[i], 3)} for i in order]
    return hits


SYSTEM_PROMPT = (
    "你是门店销售接待与异议处理的 AI 话术助手，服务销售顾问。"
    "只能依据给定的接待流程、话术口径与异议应对资料回答，逐条引用资料原句；"
    "资料里的数字、时限、金额、百分比必须原样照抄，禁止换算、四舍五入、补全位数或自行估算。"
    "第一步先判断资料是否已给出明确的口径、边界或禁忌：若资料已覆盖，必须直接按资料作答并说清该口径，"
    "不得仅因话题涉及竞品、智驾或优惠就转人工。"
    "例如被问能否贬低竞品，应答“对比竞品只陈述本车型客观参数，不贬低、不编造对手缺点”；"
    "被问能否承诺脱手自动驾驶，应答“只说明官方公布的适用范围，不承诺脱手自动驾驶或未开放功能”。"
    "不得口头让价或承诺折扣，不贬低竞品，不承诺贷款审批结果、利率、保值残值、具体提车日期或促销礼包。"
    "可以先说明资料中的原则口径；只有资料未给出结论、或客户要求具体数字与个案承诺时才转人工，"
    "例如要求给出具体残值数字、在两款车之间比较谁加速更快或谁更好的优劣结论（即使检索到“不贬低竞品”的原则句，"
    "也不能用该原则句代替对比结论，必须转人工）、审批一定通过的结论或具体利率数值、"
    "具体提车日期、特殊定制、合同条款、投诉争议，或代写营销文案等资料外创作，"
    "才回复“这个问题需要转人工顾问”，不要自行下结论或给数据。回答简洁，分点说明，不要重复罗列，不要复述问题。"
)


def answer(question: str, chunks, embs):
    hits = retrieve(chunks, embs, question)
    context = "\n\n".join(f"【{h['source']}】{h['text']}" for h in hits)
    user_prompt = f"资料：\n{context}\n\n问题：{question}\n请根据资料回答："
    kwargs = dict(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1, "num_predict": 512, "repeat_penalty": 1.1},
    )
    try:
        resp = client.chat(think=False, **kwargs)
    except TypeError:
        resp = client.chat(**kwargs)
    msg = resp["message"]
    text = (msg.get("content") or msg.get("thinking") or "").strip()
    return text, hits


def read_questions():
    if not os.path.exists(QUESTIONS_FILE):
        return []
    qs = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                qs.append(line)
    return qs


def run_batch():
    chunks, embs = build_index(load_chunks())
    lines = ["# 演示问答结果（本地 RAG 原型）", "", f"- 对话模型：{CHAT_MODEL}；向量化模型：{EMBED_MODEL}", ""]
    for q in read_questions():
        print(f"\n问：{q}")
        try:
            a, hits = answer(q, chunks, embs)
            print(f"答：{a}")
        except Exception as e:
            a, hits = f"（生成失败：{e}）", []
            print(f"答：{a}")
        lines += [f"## 问：{q}", "", f"**答：** {a}", "", "**检索来源：**"]
        for h in hits:
            lines.append(f"- {h['source']}（相似度 {h['score']}）")
        lines.append("")
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n结果已写入：{ANSWERS_FILE}")


def run_interactive():
    chunks, embs = build_index(load_chunks())
    print("进入交互式问答，输入 quit 退出。")
    while True:
        q = input("\n问：").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        a, hits = answer(q, chunks, embs)
        print(f"答：{a}")
        print("来源：" + "；".join(f"{h['source']}({h['score']})" for h in hits))


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_batch()
    else:
        run_interactive()
