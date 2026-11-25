# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Builds the benchmark cluster (layout.py) on the baseline and the agentic cluster: settings, the embedding model,
the hybrid normalization pipeline, every index (mapping, english analyzer on relevant text fields, one embedding field
per embedded index via an ingest pipeline, `_meta.description` on about a fifth of them), two data streams, the
aliases and the data. Query-insights exporters are turned off first so no top_queries-* index appears.
Writes <work>/cluster_<role>.json (url, embedding model id), <work>/indices.json (index list with purposes, doc
counts, aliases, embedded and described flags), <work>/fields.json and <work>/index_list.md.
Usage: python3 build_cluster.py [baseline|agentic ...]   (default: both; built once if both URLs are equal)"""

import json
import sys

import requests

import common as C
import layout as L


def quiet_insights(base):
    C.call(base, "PUT", "/_cluster/settings", {"persistent": {
        f"search.insights.top_queries.{m}.enabled": False for m in ("latency", "cpu", "memory")}})
    for r in C.call(base, "GET", "/_cat/indices?format=json&h=index&expand_wildcards=all"):
        if r["index"].startswith("top_queries"):
            C.call(base, "DELETE", "/" + r["index"], ok=(200, 404))


def mapping_of(s):
    m = json.loads(json.dumps(s.mapping)) if s.mapping else {}
    if s.embed_field:
        m["properties"][s.embed_field + "_embedding"] = C.KNN
        # vectors stay searchable but out of _source, so search hits and passages do not carry them
        m["_source"] = {"excludes": [s.embed_field + "_embedding"]}
    if L.has_description(s.name):
        m["_meta"] = {"description": s.purpose}
    return m


def index_settings(s):
    st = {"number_of_replicas": 0}
    if s.embed_field:
        st.update({"knn": True, "default_pipeline": f"emb-{s.embed_field}", "search.default_pipeline": "hybrid-norm"})
    return st


def bulk_create(base, index, docs):
    """Data streams accept only op_type create."""
    lines = []
    for _id, src in docs:
        lines.append(json.dumps({"create": {"_index": index, "_id": _id}}))
        lines.append(json.dumps(src))
    r = requests.post(base + "/_bulk", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"},
                      timeout=900)
    r.raise_for_status()
    if r.json()["errors"]:
        raise RuntimeError(json.dumps([i for i in r.json()["items"] if i["create"].get("error")][:3])[:2000])


def build(base, cfg):
    quiet_insights(base)
    C.settings(base, cfg)
    emb = C.embedding_model(base)
    C.hybrid_pipeline(base)
    specs = L.layout(cfg.work)
    for f in sorted({s.embed_field for s in specs if s.embed_field}):
        C.embedding_pipeline(base, f"emb-{f}", emb, {f: f + "_embedding"})
    docs = []
    for s in specs:
        if s.data_stream:
            C.call(base, "PUT", f"/_index_template/{s.name}", {
                "index_patterns": [s.name], "data_stream": {}, "priority": 200,
                "template": {"settings": index_settings(s), "mappings": mapping_of(s)}})
            C.call(base, "DELETE", f"/_data_stream/{s.name}", ok=(200, 404))
            bulk_create(base, s.name, s.docs)
            continue
        C.call(base, "DELETE", f"/{s.name}", ok=(200, 404))
        body = {"settings": {"index": index_settings(s)}}
        if s.mapping:
            body["mappings"] = mapping_of(s)
        C.call(base, "PUT", f"/{s.name}", body)
        docs += [(s.name, _id, src) for _id, src in s.docs]
    C.bulk(base, docs)
    C.call(base, "POST", "/_aliases", {"actions": [{"add": {"index": p, "alias": a}} for a, p in L.ALIASES.items()]})
    quiet_insights(base)
    return emb, specs


def _fields_of(props, prefix=""):
    for name, d in props.items():
        path = prefix + name
        if "properties" in d:
            yield from _fields_of(d["properties"], path + ".")
        else:
            yield path
            for sub in d.get("fields") or {}:
                yield f"{path}.{sub}"


def write_lists(base, specs, cfg):
    aliases = {}
    for r in C.call(base, "GET", "/_cat/aliases?format=json"):
        aliases.setdefault(r["index"], []).append(r["alias"])
    ds = {d["name"]: [i["index_name"] for i in d["indices"]] for d in C.call(base, "GET", "/_data_stream")["data_streams"]}
    rows = []
    for s in specs:
        rows.append({"index": s.name, "family": s.family, "purpose": s.purpose, "docs": len(s.docs),
                     "relevant": s.relevant, "embedded": bool(s.embed_field), "described": L.has_description(s.name),
                     "data_stream": s.data_stream, "backing": ds.get(s.name, []), "aliases": sorted(aliases.get(s.name, []))})
    C.dump_json({"indices": rows, "aliases": L.ALIASES, "scope": L.SCOPE}, cfg.wpath("indices.json"))
    m = C.call(base, "GET", "/_mapping")
    C.dump_json({i: sorted(set(_fields_of(v["mappings"].get("properties", {})))) for i, v in m.items()
                 if not i.startswith(".") or i.startswith(".ds-")}, cfg.wpath("fields.json"))
    n_desc = sum(r["described"] for r in rows)
    md = ["# Benchmark cluster: index list\n",
          f"{len(rows)} indices ({sum(r['data_stream'] for r in rows)} of them data streams), "
          f"{sum(r['docs'] for r in rows):,} documents. Relevant (hold corpus documents): "
          f"{sum(r['relevant'] for r in rows)}. Embedding field: {sum(r['embedded'] for r in rows)}. "
          f"`_meta.description`: {n_desc} ({n_desc / len(rows):.0%}).\n",
          "Aliases: " + ", ".join(f"`{a}` -> `{p}`" for a, p in L.ALIASES.items()) + ".\n",
          "Agentic Index (scoped) target: `" + ",".join(L.SCOPE) + "`.\n",
          "| Index | Docs | Relevant | Embedding | _meta.description | Aliases | Purpose |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['index']}{' (data stream)' if r['data_stream'] else ''} | {r['docs']} | "
                  f"{'yes' if r['relevant'] else ''} | {'yes' if r['embedded'] else ''} | "
                  f"{'yes' if r['described'] else ''} | {', '.join(r['aliases'])} | {r['purpose']} |")
    open(cfg.wpath("index_list.md"), "w").write("\n".join(md) + "\n")


def main():
    cfg = C.Config()
    roles = [a for a in sys.argv[1:] if a in ("baseline", "agentic")] or ["baseline", "agentic"]
    built = {}
    for role in roles:
        base = cfg.clusters[role]
        if base not in built:
            built[base] = build(base, cfg)
            print(role, base, "built", len(built[base][1]), "indices", flush=True)
        emb, specs = built[base]
        C.dump_json({"url": base, "embedding_model_id": emb}, cfg.wpath(f"cluster_{role}.json"))
        if role == "baseline":
            write_lists(base, specs, cfg)


if __name__ == "__main__":
    main()
