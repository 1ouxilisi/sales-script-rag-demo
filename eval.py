# -*- coding: utf-8 -*-
"""
RAG 评测脚本：召回命中率 + 回答关键词准确率 + 拒答测试。
用法：python eval.py
输出：控制台汇总 + eval_report.md
"""
import json
import os
import re
import time

import rag

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(BASE, "eval_set.json")
REPORT_FILE = os.path.join(BASE, "eval_report.md")


def norm(s: str) -> str:
    """归一化：去空白、逗号顿号，小写，否定词统一为“不”，便于关键词比对。"""
    s = re.sub(r"[\s,，、]", "", s or "").lower()
    s = re.sub(r"不能|不应|不可|不该|不得", "不", s)
    return s


def hit_source(expect: str, hits: list) -> bool:
    if not expect:
        return True
    return any(expect in h["source"] for h in hits)


def answer_pass(item: dict, answer: str) -> tuple[bool, list[str]]:
    if item.get("reject"):
        ok = ("转人工" in answer) or ("人工顾问" in answer)
        return ok, ["拒答转人工"] if ok else []
    ans = norm(answer)
    missing = [k for k in item["keywords"] if norm(k) not in ans]
    return (len(missing) == 0), item["keywords"] if not missing else missing


def main():
    chunks, embs = rag.build_index(rag.load_chunks())
    with open(EVAL_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    recall_ok = answer_ok = 0

    for item in cases:
        t0 = time.time()
        answer, hits = rag.answer(item["question"], chunks, embs)
        secs = round(time.time() - t0, 1)

        if item.get("reject"):
            rec = None  # 拒答题不考核知识库召回
        else:
            rec = hit_source(item["expect_source"], hits)
        ok, missing = answer_pass(item, answer)
        if rec is not None:
            recall_ok += int(rec)
        answer_ok += int(ok)

        rows.append({
            "id": item["id"],
            "q": item["question"],
            "recall": rec,
            "answer_ok": ok,
            "missing": missing,
            "answer": answer,
            "sources": "；".join(f"{h['source']}({h['score']})" for h in hits),
            "secs": secs,
        })
        rec_mark = "—" if rec is None else ("✓" if rec else "✗")
        print(f"[{item['id']:>2}] 召回={rec_mark} 回答={'✓' if ok else '✗'} "
              f"{secs}s  {item['question']}")

    n = len(cases)
    n_recall = sum(1 for c in cases if not c.get("reject"))
    recall_rate = recall_ok / n_recall if n_recall else 0
    answer_rate = answer_ok / n

    # 写报告
    lines = [
        "# RAG 评测报告",
        "",
        f"- 评测题数：{n}",
        f"- 对话模型：{rag.CHAT_MODEL}；向量模型：{rag.EMBED_MODEL}",
        f"- 召回命中率：**{recall_ok}/{n_recall} = {recall_rate:.0%}**（拒答题不考核召回）",
        f"- 回答准确率（关键词/拒答）：**{answer_ok}/{n} = {answer_rate:.0%}**",
        "",
        "| # | 问题 | 召回 | 回答 | 缺失关键词 | 耗时 | 来源 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        rec_mark = "—" if r["recall"] is None else ("✓" if r["recall"] else "✗")
        lines.append(
            f"| {r['id']} | {r['q']} | {rec_mark} | "
            f"{'✓' if r['answer_ok'] else '✗'} | {', '.join(r['missing']) or '—'} | "
            f"{r['secs']}s | {r['sources']} |"
        )
    lines += ["", "## 逐题回答明细", ""]
    for r in rows:
        lines += [f"### {r['id']}. {r['q']}", "", f"**答：** {r['answer']}", ""]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n召回率 {recall_rate:.0%}  回答准确率 {answer_rate:.0%}")
    print(f"报告已写入：{REPORT_FILE}")


if __name__ == "__main__":
    main()
