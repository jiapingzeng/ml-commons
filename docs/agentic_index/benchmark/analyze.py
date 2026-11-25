# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-execution features: tool calls, exploration calls, failed calls, zero-hit searches, indices touched, routing
hit, LLM calls and tokens (the agentic index's own LLM calls come from the proxy log), latency, loop-cap hits and
citations. Every document of the cluster is known by _id (layout.py), so citations of any index count."""

import fnmatch
import glob
import json
import os
import re

import common as C
import layout as L
import run as R

AGENTIC_TOOLS = {"agentic_index_retrieve": "_retrieve", "agentic_index_ask": "_ask"}
CAP_TEXT = "Agent reached maximum iterations"
LEAF_QUERIES = {"term", "terms", "match", "match_phrase", "match_phrase_prefix", "range", "prefix", "wildcard",
                "fuzzy", "regexp", "match_bool_prefix", "neural"}


class Cluster:
    """Index names, aliases, data streams, field sets and every document (shared by the corpora)."""
    _inst = None

    @classmethod
    def get(cls, cfg):
        if cls._inst is None:
            cls._inst = cls(cfg)
        return cls._inst

    def __init__(self, cfg):
        meta = C.load_json(cfg.wpath("indices.json"))
        self.indices = [r["index"] for r in meta["indices"]]
        self.aliases = {}
        for r in meta["indices"]:
            for a in r["aliases"]:
                self.aliases.setdefault(a, []).append(r["index"])
        self.backing = {r["index"]: r["backing"] for r in meta["indices"] if r["data_stream"]}
        self.doc_index, self.docs = {}, {}
        for s in L.layout(cfg.work):
            for _id, src in s.docs:
                self.doc_index[_id] = s.name
                self.docs[_id] = src
        self.fields = {i: set(f) for i, f in C.load_json(cfg.wpath("fields.json")).items()}
        for ds, back in self.backing.items():
            self.fields[ds] = set().union(*(self.fields.get(b, set()) for b in back))
        self.unofficial = {i for i, s in self.docs.items() if s.get("status") == "unofficial"}

    def resolve(self, target):
        """Concrete index names a search target (name, list, wildcard, alias, data stream) reaches."""
        out = set()
        for t in [x.strip() for x in str(target).split(",") if x.strip()]:
            if t.startswith("-"):
                continue
            if t in self.aliases:
                out |= set(self.aliases[t])
            elif "*" in t or "?" in t:
                out |= {i for i in self.indices if fnmatch.fnmatch(i, t)}
                for a, v in self.aliases.items():
                    if fnmatch.fnmatch(a, t):
                        out |= set(v)
            elif t in self.indices:
                out.add(t)
            elif t.startswith(".ds-"):
                out |= {ds for ds, b in self.backing.items() if t in b}
        return out

    def known(self, target):
        t = str(target).strip()
        return t in self.indices or t in self.aliases or "*" in t or t.startswith(".ds-")


class Corpus:
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.name = name
        self.questions = R.load_questions(name)
        self.X = Cluster.get(cfg)
        self.chunks = dict(L.chunks_synth())
        self.chunks.update(L.chunks_real(cfg.work))
        self.other_ids = set(self.X.docs) - set(self.chunks)
        self.unofficial = self.X.unofficial
        self.proxy = None

    def gold_indices(self, qi):
        q = self.questions[qi - 1]
        return {self.X.doc_index[g] for g in q["gold"] + q["ok"] if g in self.X.doc_index}

    def proxy_log(self):
        if self.proxy is None:
            path = self.cfg.wpath("logs", "proxy.jsonl")
            rows = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
            self.proxy = {"rows": rows, "used": set()}
        return self.proxy


def exec_key(r):
    return f"{r['tag']}|{r['setup']}|{r['model']}|{r['qi']}"


def load(cfg, corpus, tag="run[0-9]*"):
    """Executions of the given tags (a glob); a client-side error is superseded by a later execution of the same item."""
    rows = {}
    for path in sorted(glob.glob(cfg.wpath("results", corpus, tag, "*.jsonl"))):
        tagname = os.path.basename(os.path.dirname(path))
        for line in open(path):
            r = json.loads(line)
            r.setdefault("tag", tagname)
            k = (tagname, r["setup"], r["model"], r["qi"])
            if k not in rows or str(rows[k].get("error", "")).startswith("client:"):
                rows[k] = r
    return list(rows.values())


def _json(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _targets(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    v = _json(value) if isinstance(value, str) and value.strip().startswith("[") else None
    if isinstance(v, list):
        return [str(x) for x in v]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _fields(dsl, out):
    if isinstance(dsl, dict):
        for k, v in dsl.items():
            if k in LEAF_QUERIES and isinstance(v, dict):
                out.update(f for f in v if not f.startswith("_") and f != "boost")
            elif k == "fields" and isinstance(v, list):
                out.update(str(f) for f in v)
            elif k in ("field", "default_field") and isinstance(v, str):
                out.add(v)
            elif k == "exists" and isinstance(v, dict) and "field" in v:
                out.add(v["field"])
            elif k == "sort":
                for it in (v if isinstance(v, list) else [v]):
                    if isinstance(it, dict):
                        out.update(it)
                    elif isinstance(it, str):
                        out.add(it)
            _fields(v, out)
    elif isinstance(dsl, list):
        for x in dsl:
            _fields(x, out)


def _clean(field):
    return re.sub(r"\^[\d.]+$", "", field).replace("*", "")


def _hits(response):
    j = _json(response)
    try:
        return j["inference_results"][0]["output"][0]["dataAsMap"]["hits"]["total"]["value"]
    except Exception:
        return 0


def _hit_indices(response):
    j = _json(response)
    try:
        return {h["_index"] for h in j["inference_results"][0]["output"][0]["dataAsMap"]["hits"]["hits"]}
    except Exception:
        return set()


def _inner(C, row, mode, question):
    log = C.proxy_log()
    tag = f"{row['setup']}-{row['model']}"
    for i, e in enumerate(log["rows"]):
        if i in log["used"] or e.get("tag") != tag or not e["path"].endswith(mode):
            continue
        if e["request"].get("question") == question and row["t0"] - 1 <= e["t0"] <= row["t1"] + 1:
            log["used"].add(i)
            return e
    return None


def _ds_name(index, X):
    for ds, back in X.backing.items():
        if index in back:
            return ds
    return index


def features(row, C):
    X = C.X
    f = {"tool_calls": 0, "exploration": 0, "search_calls": 0, "agentic_calls": 0, "failed_tool_calls": 0,
         "failures": [], "indices": set(), "zero_hit_searches": 0, "inner_llm_calls": 0, "inner_input_tokens": 0,
         "inner_output_tokens": 0, "inner_rounds": 0, "inner_unmatched": 0, "routing": [], "route_llm_calls": 0,
         "hit_indices": set()}
    for s in row.get("steps", []):
        origin = s["origin"]
        if origin == "LLM":
            continue
        f["tool_calls"] += 1
        inp = _json(s.get("input")) or {}
        resp = s.get("response") or ""
        failed = resp.startswith("Failed to run the tool")
        if origin == "ListIndexTool":
            f["exploration"] += 1
        elif origin == "IndexMappingTool":
            f["exploration"] += 1
            if "There were no results" in resp or "index_not_found" in resp:
                failed = True
                f["failures"].append("mapping of missing index")
        elif origin == "SearchIndexTool":
            targets = _targets(inp.get("index")) if isinstance(inp, dict) else []
            reached = set()
            for t in targets:
                reached |= X.resolve(t)
            f["indices"] |= reached
            query = inp.get("query") if isinstance(inp, dict) else None
            query = _json(query) if isinstance(query, str) else query
            q = query.get("query") if isinstance(query, dict) else None
            is_sample = q is None or (isinstance(q, dict) and set(q) == {"match_all"})
            has_aggs = isinstance(query, dict) and ("aggs" in query or "aggregations" in query)
            if is_sample and not has_aggs:
                f["exploration"] += 1
            else:
                f["search_calls"] += 1
            used = set()
            _fields(query, used)
            known = set().union(*(X.fields.get(i, set()) for i in reached)) if reached else set()
            unknown = sorted({u for u in used if _clean(u) not in known and not _clean(u).startswith("_")
                              and not any(k.startswith(_clean(u)) for k in known)})
            if failed or '"error"' in resp[:300]:
                failed = True
                f["failures"].append(f"search error on {targets}: {resp[:160]}")
            elif unknown and known:
                failed = True
                f["failures"].append(f"guessed field(s) {unknown} on {targets}")
            if not failed and not is_sample and _hits(resp) == 0:
                f["zero_hit_searches"] += 1
            f["hit_indices"] |= {_ds_name(i, X) for i in _hit_indices(resp)}
            for t in targets:
                if not X.known(t) and not t.startswith(".") and not t.startswith("-"):
                    failed = True
                    f["failures"].append(f"search on missing index {t}")
        elif origin in AGENTIC_TOOLS:
            f["agentic_calls"] += 1
            j = _json(resp) or {}
            try:
                dm = j["inference_results"][0]["output"][0]["dataAsMap"]
                assert "passages" in dm
                f["hit_indices"] |= {_ds_name(p.get("index"), X) for p in dm["passages"] if p.get("index")}
            except Exception:
                failed = True
                f["failures"].append(f"{origin} error: {resp[:160]}")
            e = _inner(C, row, AGENTIC_TOOLS[origin], inp.get("question") if isinstance(inp, dict) else None)
            if e is None:
                f["inner_unmatched"] += 1
            else:
                tr = (e.get("response") or {}).get("trace") or {}
                f["inner_llm_calls"] += tr.get("llm_call_count", 0)
                f["inner_input_tokens"] += tr.get("prompt_tokens", 0)
                f["inner_output_tokens"] += tr.get("completion_tokens", 0)
                f["inner_rounds"] += len(tr.get("rounds", []))
                rt = tr.get("routing") or {}
                sel = [_ds_name(x["index"], X) for x in rt.get("selected", [])
                       if not str(x.get("reason") or "").startswith("added by the sufficiency check")]
                added = [_ds_name(x["index"], X) for x in rt.get("selected", [])
                         if str(x.get("reason") or "").startswith("added by the sufficiency check")]
                searched = set()
                for rd in tr.get("rounds", []):
                    for sq in rd.get("subqueries", []):
                        if sq.get("index"):
                            searched.add(_ds_name(sq["index"], X))
                f["indices"] |= searched
                f["routing"].append({"selected": sel, "added": added, "searched": sorted(searched),
                                     "routed": rt.get("routed"),
                                     "catalog_index_count": rt.get("catalog_index_count"),
                                     "profiled_index_count": rt.get("profiled_index_count"),
                                     "catalog_prefiltered": rt.get("catalog_prefiltered"),
                                     "catalog_shown": None})
                f["route_llm_calls"] += sum(1 for c in tr.get("llm_calls", []) if c.get("step") == "route")
        if failed:
            f["failed_tool_calls"] += 1
    f["indices"] = sorted(i for i in f["indices"] if i)
    f["hit_indices"] = sorted(i for i in f["hit_indices"] if i)
    gold_idx = C.gold_indices(row["qi"])
    f["gold_indices"] = sorted(gold_idx)
    f["retrieved_gold"] = bool(set(f["hit_indices"]) & gold_idx) if gold_idx else None
    if gold_idx:
        if row["setup"].startswith("ai_"):
            sel = set().union(*(set(r["selected"]) for r in f["routing"])) if f["routing"] else set()
            searched = set().union(*(set(r["searched"]) for r in f["routing"])) if f["routing"] else set()
            f["route_hit"] = bool(sel & gold_idx) if f["routing"] else None
            f["route_hit_first"] = bool(set(f["routing"][0]["selected"]) & gold_idx) if f["routing"] else None
            f["searched_hit"] = bool(searched & gold_idx) if f["routing"] else None
            added = set().union(*(set(r["added"]) for r in f["routing"])) if f["routing"] else set()
            f["followup_rescue"] = bool(added & gold_idx) and not f["route_hit"] if f["routing"] else None
        else:
            f["route_hit"] = bool(set(f["indices"]) & gold_idx)
            f["route_hit_first"] = None
            f["searched_hit"] = f["route_hit"]
    else:
        f["route_hit"] = f["route_hit_first"] = f["searched_hit"] = None
    usage = (row.get("token_usage") or {}).get("per_turn_usage", [])
    f["agent_llm_calls"] = len(usage) or sum(1 for s in row.get("steps", []) if s["origin"] == "LLM")
    f["agent_input_tokens"] = sum(u.get("input_tokens", 0) or 0 for u in usage)
    f["agent_output_tokens"] = sum(u.get("output_tokens", 0) or 0 for u in usage)
    f["first_turn_input_tokens"] = (usage[0].get("input_tokens") if usage else None)
    f["llm_calls"] = f["agent_llm_calls"] + f["inner_llm_calls"]
    f["input_tokens"] = f["agent_input_tokens"] + f["inner_input_tokens"]
    f["output_tokens"] = f["agent_output_tokens"] + f["inner_output_tokens"]
    f["tokens"] = f["input_tokens"] + f["output_tokens"]
    f["indices_touched"] = len(f["indices"])
    answer = row.get("answer") or ""
    f["error"] = row.get("error")
    f["cap_hit"] = CAP_TEXT in answer
    f["cited"] = citations(answer, C)
    q = C.questions[row["qi"] - 1]
    support = set(q["gold"]) | set(q["ok"])
    f["cited_unofficial"] = [c for c in f["cited"] if c in C.unofficial and c not in support]
    f["precision"] = (sum(1 for c in f["cited"] if c in support) / len(f["cited"])) if f["cited"] and support else None
    f["gold_recall"] = (sum(1 for g in q["gold"] if g in f["cited"]) / len(q["gold"])) if q["gold"] else None
    f["latency_ms"] = row.get("latency_ms")
    return f


def citations(answer, C):
    ids = []
    for group in re.findall(r"\[([^\[\]]+)\]", answer or ""):
        for part in re.split(r"[,;]\s*|\s+and\s+", group):
            part = part.strip().strip("`*'\" ")
            part = re.sub(r"^(?:[\w.-]+/)", "", part)
            if part in C.X.docs:
                ids.append(part)
    return list(dict.fromkeys(ids))
