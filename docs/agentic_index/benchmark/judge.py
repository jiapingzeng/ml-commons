# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Blind judging with the configured judge model (any of the three model types).

Before judging, each answer is normalized (normalize()): index names, tool names (in the real corpus only the agentic
index tool names, because ListIndexTool, IndexMappingTool and SearchIndexTool are documented features there) and
phrases such as "knowledge base" or "agentic index" are replaced with neutral words, and citations are renumbered [1],
[2], ... The judge sees only the question, the gold answer (with its grading notes), the gold source text with its
metadata, the normalized answer and the text of the passages the answer cites. Distinct (question, normalized answer,
cited passages) items get a random blind id and are judged in shuffled order, so the judge cannot tell which setup or
model wrote an answer. Loop-cap hits and errors are wrong without a judge call.

Scores are only comparable between runs judged by the same judge model: <work>/judge/judge_info.json records it, and
this script refuses to add verdicts from a different judge (use --reset to start over).
Outputs: <work>/judge/judge_blind_<corpus>.json (blind items and verdicts) and judge_map_<corpus>.json (execution ->
blind id). Optional manual overrides: <work>/judge/manual_overrides.json, {"<corpus>": {"<blind id>": {"verdict": ...,
"reason": ...}}}.
Usage: python3 judge.py <synth|real|route|all> [tag glob, default *] [--reset]"""

import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import sys
import uuid

import analyze as A
import common as C

DOMAIN = {"synth": "a support corpus for a fictional software product (Nimbus)",
          "real": "OpenSearch documentation, release notes and GitHub issues",
          "route": "a software company's data: Nimbus product documents, OpenSearch release notes and customer "
                   "tenants' support tickets"}
TOOL_NAMES = ["ListIndexTool", "IndexMappingTool", "SearchIndexTool", "agentic_index_retrieve", "agentic_index_ask"]
# In the real corpus the three built-in tool names are documented OpenSearch features that gold answers name, so only
# the agentic index tool names are neutralized there.
TOOL_NAMES_BY_CORPUS = {"synth": TOOL_NAMES, "real": ["agentic_index_retrieve", "agentic_index_ask"],
                        "route": TOOL_NAMES}
MAX_PASSAGE_CHARS = 1600
MAX_CITED = 8
JUDGE_META_FIELDS = ["product", "product_line", "product_version", "api_version", "plans", "plan", "tier", "applies_to",
                     "component", "effective_date", "release_date", "published", "published_date", "updated_at",
                     "last_updated"]

PROMPT = """You grade answers from a search assistant over {domain}.

Question: {q}

Gold answer: {expect}

Grading notes: {notes}

Gold source passages:
{gold}

Answer to grade:
<answer>
{answer}
</answer>

Passages the answer cites (by its [n] markers):
{cited}

Grade only the facts, not style or citations. Scale:
- correct: states every gold fact needed to answer all parts of the question, with no wrong claim that matters.
- partial: answers some parts or states some gold facts, but misses a part, is vague, or adds a wrong secondary claim.
- wrong: misses the main fact, states a wrong main fact, or says the information is unavailable when the gold answer \
gives it.
For a question whose gold answer is that the information is unavailable, an answer that says so is correct; an answer \
that presents an unofficial source's claim as the answer is wrong.

Return only JSON: {{"verdict": "correct" | "partial" | "wrong", "note": "<one short sentence; empty if correct>"}}"""


def index_names(K):
    return sorted(set(K.X.indices) | set(K.X.aliases), key=len, reverse=True)


def normalize(answer, K, names):
    """Neutral wording and renumbered citations; returns (text, cited ids in [n] order)."""
    cited = []

    def renumber(m):
        out = []
        for part in re.split(r"[,;]\s*|\s+and\s+", m.group(1)):
            p = re.sub(r"^(?:[\w.-]+/)", "", part.strip().strip("`*'\" "))
            if p in K.X.docs:
                if p not in cited:
                    cited.append(p)
                out.append(str(cited.index(p) + 1))
        return "[" + ", ".join(out) + "]" if out else m.group(0)

    text = re.sub(r"\[([^\[\]]+)\]", renumber, answer)
    for t in TOOL_NAMES_BY_CORPUS[K.name]:
        text = re.sub(rf"`?{re.escape(t)}`?", "search", text)
    for n in names:
        e = re.escape(n)
        text = re.sub(rf"\b(?:the\s+)?[`'\"]{e}[`'\"](?:\s+(?:index|indices|indexes|collection))?", "the documents",
                      text)
        text = re.sub(rf"\b(?:the\s+)?{e}\s+(?:index|indices|indexes|collection)\b", "the documents", text)
    text = re.sub(r"(?i)\bknowledge[\s-]base\b", "documents", text)
    text = re.sub(r"(?i)\bagentic[\s_-]index\b", "search", text)
    return text, cited


def passage_block(K, ids, numbered):
    lines = []
    for i, pid in enumerate(ids[:MAX_CITED], 1):
        c = K.chunks.get(pid)
        head = f"[{i}] " if numbered else "- "
        if c is None:  # a record (ticket, invoice, ...) is shown as its field list
            src = K.X.docs.get(pid)
            if src is None:
                continue
            lines.append(f"{head}record | {src.get('status', 'record')}\n" + A.L.doc_text(src)[:MAX_PASSAGE_CHARS])
            continue
        status = "unofficial" if pid in K.unofficial else "official"
        meta = f"{c.get('doc_title') or c.get('title')} | {status} | date {c.get('date')}"
        if c.get("version"):
            meta += f" | version {c['version']}"
        src = K.X.docs.get(pid) or {}
        extra = [f"{k} {src[k]}" for k in JUDGE_META_FIELDS if src.get(k) not in (None, "", [])]
        if extra:
            meta += " | " + " | ".join(extra)
        lines.append(f"{head}{meta}\n{c['text'][:MAX_PASSAGE_CHARS]}")
    return "\n\n".join(lines) or "(none)"


def judge(m, text):
    err = "no valid verdict"
    for _ in range(6):
        try:
            out = C.chat(m, text, max_tokens=int(m.get("judge_max_tokens", 1024)))
            j = json.loads(re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", out, re.S)[-1])
            if j["verdict"] in ("correct", "partial", "wrong"):
                return {"verdict": j["verdict"], "note": j.get("note", "")}
        except Exception as e:
            err = repr(e)
    return {"verdict": "unjudged", "note": err[:200]}


def judge_identity(cfg):
    m = cfg.models[cfg.judge]
    return {"key": cfg.judge, "type": m["type"],
            "model": m.get("model") or m.get("model_id") or m.get("endpoint"), "label": cfg.label(cfg.judge)}


def build(cfg, corpus, tag):
    K = A.Corpus(cfg, corpus)
    names = index_names(K)
    blind_path = cfg.wpath("judge", f"judge_blind_{corpus}.json")
    map_path = cfg.wpath("judge", f"judge_map_{corpus}.json")
    blind = C.load_json(blind_path, {})
    mapping = C.load_json(map_path, {})
    by_item = {(b["qi"], b["answer"], tuple(b["cited"])): bid for bid, b in blind.items()}
    for r in A.load(cfg, corpus, tag):
        key = A.exec_key(r)
        if r.get("error") or not r.get("answer"):
            mapping[key] = {"rule": "wrong", "note": "agent error: " + str(r.get("error"))[:150]}
            continue
        if A.CAP_TEXT in r["answer"]:
            mapping[key] = {"rule": "wrong", "note": "hit the loop cap (15 tool calls)"}
            continue
        text, cited = normalize(r["answer"].strip(), K, names)
        item = (r["qi"], text, tuple(cited))
        if item not in by_item:
            bid = uuid.uuid4().hex[:10]
            blind[bid] = {"qi": r["qi"], "answer": text, "cited": cited}
            by_item[item] = bid
        mapping[key] = {"blind": by_item[item]}
    return K, blind, mapping, blind_path, map_path


def run_corpus(cfg, corpus, tag):
    K, blind, mapping, blind_path, map_path = build(cfg, corpus, tag)
    todo = [bid for bid, b in blind.items() if b.get("verdict") in (None, "unjudged")]
    random.Random(11).shuffle(todo)
    print(f"{corpus}: {len(blind)} distinct answers, {len(todo)} to judge", flush=True)
    m = cfg.models[cfg.judge]
    with cf.ThreadPoolExecutor(cfg.judge_workers) as pool:
        futs = {}
        for bid in todo:
            b = blind[bid]
            q = K.questions[b["qi"] - 1]
            text = PROMPT.format(domain=DOMAIN[corpus], q=q["q"], expect=q["expect"], notes=q.get("grading") or "None.",
                                 gold=passage_block(K, q["gold"] + q["ok"], False), answer=b["answer"],
                                 cited=passage_block(K, b["cited"], True))
            futs[pool.submit(judge, m, text)] = bid
        for n, fut in enumerate(cf.as_completed(futs), 1):
            blind[futs[fut]].update(fut.result(), by="judge")
            if n % 200 == 0:
                print("judged", n, flush=True)
                C.dump_json(blind, blind_path)
    C.dump_json(blind, blind_path)
    C.dump_json(dict(sorted(mapping.items())), map_path)
    counts = {}
    for b in blind.values():
        counts[b["verdict"]] = counts.get(b["verdict"], 0) + 1
    print(corpus, counts)


def verdicts(cfg, corpus):
    """execution key -> {"verdict", "note", "by"}; manual overrides win."""
    blind = C.load_json(cfg.wpath("judge", f"judge_blind_{corpus}.json"), {})
    mapping = C.load_json(cfg.wpath("judge", f"judge_map_{corpus}.json"), {})
    overrides = C.load_json(cfg.wpath("judge", "manual_overrides.json"), {}).get(corpus, {})
    out = {}
    for key, m in mapping.items():
        if "rule" in m:
            out[key] = {"verdict": m["rule"], "note": m["note"], "by": "rule"}
            continue
        b = blind[m["blind"]]
        v = {"verdict": b.get("verdict", "unjudged"), "note": b.get("note", ""), "by": "judge", "blind": m["blind"]}
        if m["blind"] in overrides:
            o = overrides[m["blind"]]
            v.update(verdict=o["verdict"], note=o["reason"], by="manual", judge_verdict=b.get("verdict"))
        out[key] = v
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    cfg = C.Config()
    jdir = cfg.wpath("judge")
    if "--reset" in sys.argv and os.path.isdir(jdir):
        shutil.rmtree(jdir)
    os.makedirs(jdir, exist_ok=True)
    ident, info_path = judge_identity(cfg), os.path.join(jdir, "judge_info.json")
    prev = C.load_json(info_path)
    if prev and prev != ident:
        raise SystemExit(f"{jdir} holds verdicts from judge {prev}, not {ident}. Scores from different judges are "
                         "not comparable; use --reset to judge everything again with the configured judge.")
    C.dump_json(ident, info_path)
    corpora = C.CORPORA if args[0] == "all" else [args[0]]
    tag = args[1] if len(args) > 1 else "*"
    for corpus in corpora:
        run_corpus(cfg, corpus, tag)


if __name__ == "__main__":
    main()
