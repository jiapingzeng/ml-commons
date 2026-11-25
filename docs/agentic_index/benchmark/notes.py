# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Schema notes for the "Baseline + schema notes" setup, generated from the baseline cluster.

Per index, the text the agentic index profile renders (ml-commons IndexProfile.render(): at most 50 profiled fields,
the top 25 values of each keyword field cut to 100 characters, min/max of each date field, at most 8,000 characters),
plus the mapping `_meta.description` (first 150 characters, as in the agentic index catalog) when the index has one.
Data streams are rendered by name. A header lists the cluster's aliases. Nothing is hand-written.

The notes for every index are far larger than an agent prompt allows (<work>/notes_size.json), so they are capped per
question the way the agentic index catalog is (ml-commons IndexCatalog): when the entries exceed 12,000 characters,
indices are ranked by lexical overlap of the question with their name (3 points per term), field names (2) and
keyword values (1), and the best-ranked entries that fit are shown, followed by "(N more indices not shown)".
Usage: python3 notes.py   (writes <work>/notes.json and <work>/notes_size.json)"""

import json
import re

import requests

import common as C

MAX_TERMS, MAX_FIELDS, MAX_VALUE_CHARS, MAX_RENDERED_CHARS = 25, 50, 100, 8000
ORDER = ["TEXT", "KEYWORD", "DATE", "BOOLEAN", "NUMBER"]
NUMBER_TYPES = {"long", "integer", "short", "byte", "double", "float", "half_float", "scaled_float", "unsigned_long"}
MAX_CHARS = 12000
DESCRIPTION_CHARS = 150
STOPWORDS = set("the and an as at be by do in is it of on or to we were been for are was what which who when where how "
                "does did with from that this there have has our any all about into between most many much show list "
                "give tell".split())


def kind_of(t):
    if t in ("keyword", "constant_keyword"):
        return "KEYWORD"
    if t in ("text", "match_only_text"):
        return "TEXT"
    if t in ("date", "date_nanos"):
        return "DATE"
    if t == "boolean":
        return "BOOLEAN"
    if t == "knn_vector":
        return "VECTOR"
    return "NUMBER" if t in NUMBER_TYPES else None


def collect(props, prefix, fields, no_dv, subs):
    for name, d in props.items():
        path = f"{prefix}.{name}" if prefix else name
        t = d.get("type")
        if "properties" in d and t in (None, "object"):
            collect(d["properties"], path, fields, no_dv, subs)
            continue
        k = kind_of(t)
        if k is None:
            continue
        fields[path] = k
        if str(d.get("doc_values")) == "False" or d.get("doc_values") is False:
            no_dv.add(path)
        for sub, sd in (d.get("fields") or {}).items():
            if kind_of(sd.get("type")) == "KEYWORD":
                fields[f"{path}.{sub}"] = "KEYWORD"
                subs.append(f"{path}.{sub}")
                if sd.get("doc_values") is False:
                    no_dv.add(f"{path}.{sub}")


def truncate(v, n):
    return v if len(v) <= n else v[:n] + "..."


def render(base, index):
    m = requests.get(f"{base}/{index}/_mapping", timeout=60).json()
    props = next(iter(m.values()))["mappings"].get("properties", {})
    fields, no_dv, subs = {}, set(), []
    collect(props, "", fields, no_dv, subs)
    selected = []
    for k in ORDER:
        selected += [f for f, kk in fields.items() if kk == k and f not in subs and f not in selected]
    selected += [s for s in subs if s not in selected]
    profiled = selected[:MAX_FIELDS]
    of = lambda k: [f for f in profiled if fields[f] == k]  # noqa: E731
    aggs = {}
    for f in of("KEYWORD"):
        if f not in no_dv:
            aggs["terms__" + f] = {"terms": {"field": f, "size": MAX_TERMS}}
    for f in of("DATE"):
        if f not in no_dv:
            aggs["min__" + f] = {"min": {"field": f}}
            aggs["max__" + f] = {"max": {"field": f}}
    a = requests.post(f"{base}/{index}/_search", json={"size": 0, "track_total_hits": False, "aggs": aggs},
                      timeout=120).json().get("aggregations", {})
    out = [f"Index: {index}"]
    if of("TEXT"):
        out.append("Text fields (keyword/semantic match): " + ", ".join(of("TEXT")))
    out.append("Exact-match keyword fields:")
    for f in of("KEYWORD"):
        b = a.get("terms__" + f, {}).get("buckets", [])
        line = f"- {f}"
        if b:
            vals = [truncate(str(x.get("key_as_string", x["key"])), MAX_VALUE_CHARS) for x in b]
            line += " values: " + " | ".join(vals) + (" | ..." if len(vals) >= MAX_TERMS else "")
        out.append(line)
    out.append("Date fields:")
    for f in of("DATE"):
        lo, hi = a.get("min__" + f, {}), a.get("max__" + f, {})
        line = f"- {f}"
        if lo.get("value") is not None and hi.get("value") is not None:
            line += f" (from {lo.get('value_as_string')} to {hi.get('value_as_string')})"
        out.append(line)
    if of("NUMBER"):
        out.append("Numeric fields: " + ", ".join(of("NUMBER")))
    if of("BOOLEAN"):
        out.append("Boolean fields: " + ", ".join(of("BOOLEAN")))
    omitted = sum(1 for k in fields.values() if k != "VECTOR") - len(profiled)
    if omitted > 0:
        out.append(f"({omitted} more fields not profiled)")
    s = "\n".join(out) + "\n"
    return s if len(s) <= MAX_RENDERED_CHARS else s[:MAX_RENDERED_CHARS] + "\n... (profile truncated)\n"


def indices(base):
    rows = requests.get(f"{base}/_cat/indices?format=json&h=index", timeout=60).json()
    return sorted(r["index"] for r in rows if not r["index"].startswith(".") and not r["index"].startswith("top_queries"))


def terms(text):
    out = []
    for w in re.split(r"[^0-9A-Za-z]+", text.lower()):
        if len(w) < 2 or w in STOPWORDS:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        if w not in out:
            out.append(w)
    return out


def entry(base, index):
    text = render(base, index)
    m = requests.get(f"{base}/{index}/_mapping", timeout=60).json()
    meta = next(iter(m.values()))["mappings"].get("_meta") or {}
    desc = " ".join(str(meta.get("description", "")).split())
    if desc:
        desc = desc if len(desc) <= DESCRIPTION_CHARS else desc[:DESCRIPTION_CHARS] + "..."
        text = text.replace(f"Index: {index}\n", f"Index: {index}\nDescription: {desc}\n", 1)
    fields, values = [], []
    for line in text.splitlines():
        if line.startswith("Text fields"):
            fields += [f.strip() for f in line.split(":", 1)[1].split(",")]
        elif line.startswith("- "):
            name, _, vals = line[2:].partition(" values: ")
            fields.append(name.split(" (")[0])
            values += [v.strip() for v in vals.split(" | ") if v.strip() and v.strip() != "..."]
        elif line.startswith(("Numeric fields", "Boolean fields")):
            fields += [f.strip() for f in line.split(":", 1)[1].split(",")]
    return {"index": index, "text": text, "fields": fields, "values": values}


def alias_header(base):
    rows = requests.get(f"{base}/_cat/aliases?format=json", timeout=60).json()
    by = {}
    for r in rows:
        by.setdefault(r["alias"], []).append(r["index"])
    if not by:
        return ""
    return "Aliases: " + "; ".join(f"{a} -> {len(v)} indices ({', '.join(sorted(v)[:3])}{', ...' if len(v) > 3 else ''})"
                                   for a, v in sorted(by.items())) + "\n\n"


def score(e, qterms):
    name = set(terms(e["index"]))
    fields = {t for f in e["fields"] for t in terms(f)}
    values = {t for v in e["values"] for t in terms(v)}
    return sum((3 if t in name else 0) + (2 if t in fields else 0) + (1 if t in values else 0) for t in qterms)


def capped(notes, question, max_chars=MAX_CHARS):
    """(text, shown index names, prefiltered) for one question, as IndexCatalog.render does."""
    entries = notes["entries"]
    max_chars -= len(notes["aliases"])  # the alias header counts against the budget
    total = sum(len(e["text"]) + 1 for e in entries)
    prefiltered = total > max_chars
    order = entries
    if prefiltered:
        q = terms(question)
        order = sorted(entries, key=lambda e: -score(e, q))  # stable: ties keep name order
    out, shown = [], []
    size = 0
    for e in order:
        if size + len(e["text"]) + 1 > max_chars:
            continue
        out.append(e["text"])
        shown.append(e["index"])
        size += len(e["text"]) + 1
    text = "\n".join(out)
    if len(shown) < len(entries):
        text += f"\n({len(entries) - len(shown)} more indices not shown)\n"
    return notes["aliases"] + text, shown, prefiltered


def main():
    cfg = C.Config()
    base = cfg.clusters["baseline"]
    ds = [d["name"] for d in requests.get(f"{base}/_data_stream", timeout=60).json()["data_streams"]]
    entries = [entry(base, i) for i in sorted(set(indices(base)) | set(ds))]
    notes = {"aliases": alias_header(base), "entries": entries}
    C.dump_json(notes, cfg.wpath("notes.json"))
    full = notes["aliases"] + "\n".join(e["text"] for e in entries)
    size = {"indices": len(entries), "chars": len(full), "approx_tokens_chars_div_4": len(full) // 4,
            "largest": sorted(((len(e["text"]), e["index"]) for e in entries), reverse=True)[:5]}
    C.dump_json(size, cfg.wpath("notes_size.json"))
    print(json.dumps(size))


if __name__ == "__main__":
    main()
