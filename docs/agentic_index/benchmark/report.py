# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tables, paired differences, routing analysis and review listings.

Throughput runs (tag run<N>, or --tag=<glob>): score, routing accuracy, indices touched, tool calls, failed calls,
tokens. Serial subset (tag serial): latency, one request at a time.
Paired differences: per question, the mean over its runs per setup; then the per-question difference; mean difference
with a 95% percentile bootstrap CI over questions (10,000 resamples). * = CI excludes 0.
Outputs in <work>/report/: tables.md, summary.json, routing.md, review_<corpus>.md.
Usage: python3 report.py [--tag=run*] [--serial-tag=serial]"""

import collections
import json
import os
import random
import re
import statistics
import sys

import analyze as A
import common as C
import judge as J
import notes as N

SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
METRICS = [("score", "Score", 2), ("retrieved_gold", "Gold index retrieved", 2), ("tokens", "Tokens", 0),
           ("tool_calls", "Tool calls", 2), ("failed_tool_calls", "Failed calls", 2)]
B = 10000
TITLES = {"synth": "Synthetic corpus (Nimbus)", "real": "Real corpus (OpenSearch docs, release notes, GitHub issues)",
          "route": "Route questions (the right index among similar ones)"}


class Report:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = cfg.agents + cfg.variant_keys()
        self.setups = [s for s in C.SETUPS if s in cfg.setups]
        comps = []
        for m in cfg.agents:
            comps += [("ai_all", m, "baseline", m), ("ai_all", m, "baseline_notes", m), ("ai_scoped", m, "ai_all", m),
                      ("ai_all10", m, "ai_all", m)]
        for v in cfg.inner_variants:
            k, a = f"{v['agent']}__{v['inner']}", v["agent"]
            comps += [("ai_all", k, "baseline", a), ("ai_all", k, "baseline_notes", a), ("ai_all", k, "ai_all", a)]
        self.comparisons = comps

    def comp_label(self, a, ma, b, mb):
        right = C.SETUP_NAMES[b] + ("" if mb == ma else f" ({self.cfg.label(mb)})")
        return f"{C.SETUP_NAMES[a]} - {right}"


def boot(diffs, seed=1):
    rnd = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(diffs) / n, means[int(0.025 * B)], means[int(0.975 * B) - 1]


def fmt(v, d):
    if v is None:
        return "-"
    return f"{v:+,.{d}f}" if d else f"{v:+,.0f}"


def rows_for(cfg, corpus, tag):
    K = A.Corpus(cfg, corpus)
    V = J.verdicts(cfg, corpus)
    out = []
    for r in A.load(cfg, corpus, tag):
        key = A.exec_key(r)
        f = A.features(r, K)
        v = V.get(key, {"verdict": "unjudged"})
        f.update(score=SCORE.get(v["verdict"]), verdict=v["verdict"], note=v.get("note"), by=v.get("by"),
                 blind=v.get("blind"), key=key, qi=r["qi"], run=r["run"], tag=r["tag"], setup=r["setup"],
                 model=r["model"], kind=K.questions[r["qi"] - 1]["kind"], answer=r.get("answer"),
                 notes_shown=r.get("notes_shown"), corpus=corpus, question=r["question"])
        out.append(f)
    return K, out


def agg(rows, setup, model):
    return [r for r in rows if r["setup"] == setup and r["model"] == model]


def per_q(rows, metric):
    d = {}
    for r in rows:
        v = r.get(metric)
        if v is not None:
            d.setdefault((r["corpus"], r["qi"]), []).append(float(v))
    return {q: sum(v) / len(v) for q, v in d.items()}


def pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def f2(v):
    return "-" if v is None else f"{v:.2f}"


def main_table(R, md, summary, corpus, K, rows):
    nq = len({r["qi"] for r in rows})
    md.append(f"## {TITLES[corpus]}, {nq} questions\n")
    md.append(f"| Model | Setup | Score / {nq} (runs) | Routing hit | Gold index retrieved | Indices touched | Tool calls | "
              "Failed calls / exec | Zero-hit searches / exec | LLM calls (agent + index) | Tokens in+out | Cap hits | "
              "Errors | Unjudged | Cite precision | Gold recall | Unofficial cited / exec |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in R.models:
        for s in R.setups:
            rs = agg(rows, s, m)
            if not rs:
                continue
            tags = sorted({r["tag"] for r in rs})
            per_run = [sum(r["score"] or 0 for r in rs if r["tag"] == k) for k in tags]
            n = len(rs)
            mean = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
            row = {"score": statistics.mean(per_run), "per_run": per_run,
                   "route_hit": mean_or_none(r["route_hit"] for r in rs),
                   "retrieved": mean_or_none(r["retrieved_gold"] for r in rs),
                   "indices": mean("indices_touched"), "tool_calls": mean("tool_calls"), "failed": mean("failed_tool_calls"),
                   "zero_hit": mean("zero_hit_searches"), "agent_llm": mean("agent_llm_calls"),
                   "inner_llm": mean("inner_llm_calls"), "tokens": mean("tokens"),
                   "cap_hits": sum(1 for r in rs if r["cap_hit"]), "errors": sum(1 for r in rs if r["error"]),
                   "unjudged": sum(1 for r in rs if r["score"] is None),
                   "precision": mean_or_none(r["precision"] for r in rs), "recall": mean_or_none(r["gold_recall"] for r in rs),
                   "unofficial": sum(len(r["cited_unofficial"]) for r in rs) / n,
                   "inner_unmatched": sum(r["inner_unmatched"] for r in rs), "n": n}
            summary[corpus]["main"][f"{m}|{s}"] = row
            md.append(f"| {R.cfg.label(m)} | {C.SETUP_NAMES[s]} | {row['score']:.1f} "
                      f"({', '.join(f'{x:g}' for x in per_run)}) | {f2(row['route_hit'])} | {f2(row['retrieved'])} | "
                      f"{row['indices']:.1f} | {row['tool_calls']:.2f} | {row['failed']:.2f} | {row['zero_hit']:.2f} | "
                      f"{row['agent_llm'] + row['inner_llm']:.2f} ({row['agent_llm']:.2f} + {row['inner_llm']:.2f}) | "
                      f"{row['tokens']:,.0f} | {row['cap_hits']} | {row['errors']} | {row['unjudged']} | "
                      f"{f2(row['precision'])} | {f2(row['recall'])} | {row['unofficial']:.2f} |")
    md.append("")


def paired_table(R, md, rows, title):
    md.append(title + "\n")
    md.append("| Model | Comparison | Questions | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |")
    md.append("|---|---|---|---|---|---|---|---|")
    out = {}
    for a, ma, b, mb in R.comparisons:
        ra, rb = agg(rows, a, ma), agg(rows, b, mb)
        if not ra or not rb:
            continue
        cells, res, nq = [], {}, 0
        for metric, _, d in METRICS:
            pa, pb = per_q(ra, metric), per_q(rb, metric)
            qs = sorted(set(pa) & set(pb))
            if not qs:
                cells.append("-")
                continue
            nq = max(nq, len(qs))
            mu, lo, hi = boot([pa[q] - pb[q] for q in qs])
            sig = "*" if lo > 0 or hi < 0 else ""
            res[metric] = {"mean": mu, "lo": lo, "hi": hi, "sig": bool(sig), "n": len(qs)}
            cells.append(f"{fmt(mu, d)} [{fmt(lo, d)}, {fmt(hi, d)}]{sig}")
        out[f"{a}:{ma}-{b}:{mb}"] = res
        md.append(f"| {R.cfg.label(ma)} | {R.comp_label(a, ma, b, mb)} | {nq} | " + " | ".join(cells) + " |")
    md.append("")
    return out


def latency_section(R, md, summary_part, serial, title):
    md.append(title + "\n")
    md.append("| Model | Setup | p50 s | p90 s | mean s | Score on subset |")
    md.append("|---|---|---|---|---|---|")
    stats = {}
    for m in R.models:
        for s in R.setups:
            rs = agg(serial, s, m)
            if not rs:
                continue
            lat = [r["latency_ms"] / 1000 for r in rs]
            sc = sum(r["score"] or 0 for r in rs)
            stats[f"{m}|{s}"] = {"p50": pct(lat, .5), "p90": pct(lat, .9), "mean": statistics.mean(lat), "n": len(rs),
                                 "score": sc}
            md.append(f"| {R.cfg.label(m)} | {C.SETUP_NAMES[s]} | {pct(lat, .5):.1f} | {pct(lat, .9):.1f} | "
                      f"{statistics.mean(lat):.1f} | {sc:g} / {len(rs)} |")
    md.append("")
    md.append("Paired latency differences, seconds per question.\n")
    md.append("| Model | Comparison | Latency s |")
    md.append("|---|---|---|")
    paired = {}
    for a, ma, b, mb in R.comparisons:
        ra, rb = agg(serial, a, ma), agg(serial, b, mb)
        if not ra or not rb:
            continue
        pa, pb = per_q(ra, "latency_ms"), per_q(rb, "latency_ms")
        qs = sorted(set(pa) & set(pb))
        if not qs:
            continue
        mu, lo, hi = boot([(pa[q] - pb[q]) / 1000 for q in qs])
        sig = "*" if lo > 0 or hi < 0 else ""
        paired[f"{a}:{ma}-{b}:{mb}"] = {"mean": mu, "lo": lo, "hi": hi, "sig": bool(sig)}
        md.append(f"| {R.cfg.label(ma)} | {R.comp_label(a, ma, b, mb)} | {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig} |")
    md.append("")
    summary_part["latency"], summary_part["latency_paired"] = stats, paired


def kind_table(R, md, summary, corpus, K, rows):
    kinds = sorted({r["kind"] for r in rows})
    md.append(f"Score by question kind ({corpus}), mean per execution:\n")
    md.append("| Model | Setup | " + " | ".join(f"{k} ({sum(1 for q in K.questions if q['kind'] == k)})" for k in kinds)
              + " |")
    md.append("|---|---|" + "---|" * len(kinds))
    for m in R.models:
        for s in R.setups:
            rs = agg(rows, s, m)
            if not rs:
                continue
            vals = []
            for k in kinds:
                ks = [r["score"] or 0 for r in rs if r["kind"] == k]
                vals.append(f"{sum(ks) / len(ks):.2f}" if ks else "-")
                summary[corpus]["kinds"][f"{m}|{s}|{k}"] = sum(ks) / len(ks) if ks else None
            md.append(f"| {R.cfg.label(m)} | {C.SETUP_NAMES[s]} | " + " | ".join(vals) + " |")
    md.append("")


# ---------------- routing analysis ----------------

def catalog_estimate(cfg, question, X, max_chars=12000, entry_chars=400):
    """Estimated routing catalog membership for one question with mapping-only profiles (no cached keyword values):
    IndexCatalog ranking by name (3) and field (2) terms, entries cut to 400 characters, filled to 12,000."""
    meta = {r["index"]: r for r in C.load_json(cfg.wpath("indices.json"))["indices"]}
    q = N.terms(question)
    entries = []
    for name in X.indices:
        back = X.backing.get(name)
        idx = back[0] if back else name
        fields = sorted(f for f in X.fields.get(idx, set()) if not f.endswith("_embedding") and not f.endswith(".keyword"))
        if not fields:
            continue
        desc = meta[name]["purpose"][:150] if meta[name]["described"] else ""
        text = f"- {idx}: " + (f"description {desc}; " if desc else "") + "fields " + ", ".join(fields)
        nt, ft = set(N.terms(idx)), {t for f in fields for t in N.terms(f)}
        sc = sum((3 if t in nt else 0) + (2 if t in ft else 0) for t in q)
        entries.append((sc, name, min(len(text), entry_chars)))
    shown, size = set(), 0
    for sc, name, ln in sorted(entries, key=lambda e: -e[0]):
        if size + ln + 1 > max_chars:
            continue
        shown.add(name)
        size += ln + 1
    return shown


def family(index):
    return re.sub(r"[-.](\d{4}\.(\d{2}|q\d)|\d+\.\d+|platform|sync|gateway|analytics|acme|globex|initech|umbrella|"
                  r"hooli|stark|wayne|wonka)(?=$|-)", "-*", index)


def routing_analysis(R, all_rows):
    cfg = R.cfg
    md = ["# Routing analysis\n",
          "Routing hit: for Agentic Index, whether the routing step of any agentic call of an execution selected an index "
          "that holds a gold or ok document (indices a follow-up round added are counted separately); for the Baselines, "
          "whether the agent searched such an index (a pattern or alias counts for every index it reaches). Gold index "
          "retrieved: a returned hit or passage came from such an index. Questions without gold documents are left "
          "out.\n"]
    X = A.Cluster.get(cfg)
    scope = set().union(*(X.resolve(p) for p in A.L.SCOPE))
    for corpus in C.CORPORA:
        rows = [r for r in all_rows if r["corpus"] == corpus and r["route_hit"] is not None]
        if not rows:
            continue
        md.append(f"## {TITLES[corpus]}\n")
        md.append("| Model | Setup | Routing hit (any call) | First call hit | Follow-up added the gold index | "
                  "Gold index searched | Gold index retrieved | Executions with gold index outside scope | "
                  "Notes showed a gold index |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for m in R.models:
            for s in R.setups:
                rs = agg(rows, s, m)
                if not rs:
                    continue
                outside = sum(1 for r in rs if not set(r["gold_indices"]) & scope) if s == "ai_scoped" else None
                notes = (mean_or_none(float(bool(set(r["notes_shown"] or []) & set(r["gold_indices"]))) for r in rs)
                         if s == "baseline_notes" else None)
                md.append(f"| {cfg.label(m)} | {C.SETUP_NAMES[s]} | {f2(mean_or_none(r['route_hit'] for r in rs))} | "
                          f"{f2(mean_or_none(r['route_hit_first'] for r in rs))} | "
                          f"{f2(mean_or_none(r.get('followup_rescue') for r in rs))} | "
                          f"{f2(mean_or_none(r['searched_hit'] for r in rs))} | "
                          f"{f2(mean_or_none(r['retrieved_gold'] for r in rs))} | "
                          f"{'-' if outside is None else f'{outside} of {len(rs)}'} | {f2(notes)} |")
        md.append("")
    md.append("## Why first-call routing missed (Agentic Index)\n")
    md.append("Each miss is the first agentic call of an execution whose routing selected no index holding a gold or ok "
              "document. Categories:\n"
              "- near miss: selected an index of the same family as a gold index (another month, release, tenant or "
              "product), but not the right one;\n"
              "- not in catalog (estimated): no gold index was in the estimated routing catalog (lexical prefilter, "
              "mapping-only entries; cached keyword values are ignored, so this is approximate);\n"
              "- in catalog, not chosen: a gold index was in the estimated catalog and the LLM chose other indices.\n")
    md.append("| Corpus | Setup | Model | First calls with gold | Misses | Near miss | Not in catalog (est.) | "
              "In catalog, not chosen |")
    md.append("|---|---|---|---|---|---|---|---|")
    cat_cache, examples = {}, []
    for corpus in C.CORPORA:
        for s in ("ai_all", "ai_all10", "ai_scoped"):
            for m in R.models:
                rs = [r for r in all_rows if r["corpus"] == corpus and r["setup"] == s and r["model"] == m
                      and r["route_hit_first"] is not None]
                if not rs:
                    continue
                c = collections.Counter()
                for r in rs:
                    if r["route_hit_first"]:
                        continue
                    c["miss"] += 1
                    gold = set(r["gold_indices"])
                    sel = set(r["routing"][0]["selected"])
                    if {family(g) for g in gold} & {family(x) for x in sel}:
                        c["near"] += 1
                        kind = "near miss"
                    else:
                        key = (corpus, r["qi"])
                        if key not in cat_cache:
                            cat_cache[key] = catalog_estimate(cfg, r["question"], X)
                        cat = cat_cache[key] & scope if s == "ai_scoped" else cat_cache[key]
                        if gold & cat:
                            c["chosen_other"] += 1
                            kind = "in catalog, not chosen"
                        else:
                            c["not_in_catalog"] += 1
                            kind = "not in catalog"
                    examples.append((corpus, s, m, r["qi"], kind, sorted(gold), sorted(sel)))
                md.append(f"| {corpus} | {C.SETUP_NAMES[s]} | {cfg.label(m)} | {len(rs)} | {c['miss']} | {c['near']} | "
                          f"{c['not_in_catalog']} | {c['chosen_other']} |")
    md.append("")
    md.append("## Questions with the most first-call misses (Agentic Index (all), every model)\n")
    groups = {}
    for e in examples:
        if e[1] == "ai_all":
            g = groups.setdefault((e[0], e[3]), {"n": 0, "kinds": collections.Counter(), "gold": e[5],
                                                  "selected": collections.Counter()})
            g["n"] += 1
            g["kinds"][e[4]] += 1
            g["selected"].update(e[6])
    md.append("| Corpus | Question | Misses | Categories | Gold indices | Most often selected instead |")
    md.append("|---|---|---|---|---|---|")
    for (corpus, qi), g in sorted(groups.items(), key=lambda x: -x[1]["n"])[:15]:
        cats = ", ".join(f"{k} {v}" for k, v in g["kinds"].most_common())
        sel = ", ".join(f"{k} ({v})" for k, v in g["selected"].most_common(4))
        md.append(f"| {corpus} | q{qi} | {g['n']} | {cats} | {', '.join(g['gold'])} | {sel} |")
    md.append("")
    open(cfg.wpath("report", "routing.md"), "w").write("\n".join(md) + "\n")


def write_review(cfg, corpus, K, rows):
    """Every run-to-run disagreement and a random 10% sample, for a manual spot check of the judge."""
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["setup"], r["qi"]), []).append(r)
    dis = {k: v for k, v in groups.items() if len({x["verdict"] for x in v}) > 1}
    rnd = random.Random(f"review-{corpus}")
    sample = rnd.sample(rows, max(1, len(rows) // 10)) if rows else []
    out = [f"# Review, {corpus}: {len(dis)} disagreeing groups, {len(sample)} sampled executions\n"]

    def item(r):
        a = (r["answer"] or "").replace("\n", " ")
        return (f"- {r['key']} verdict={r['verdict']} ({r['by']}) blind={r['blind']} note={r['note']!r}\n"
                f"  answer: {a[:700]}")

    out.append("## Disagreements\n")
    for (m, s, qi), v in sorted(dis.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        q = K.questions[qi - 1]
        out.append(f"### q{qi} {m} {s}: {q['q']}\nGold: {q['expect']}\n")
        out += [item(r) for r in sorted(v, key=lambda x: x["tag"])]
        out.append("")
    out.append("## Random 10% sample\n")
    for r in sorted(sample, key=lambda x: (x["qi"], x["model"], x["setup"])):
        q = K.questions[r["qi"] - 1]
        out.append(f"### q{r['qi']} {q['q']}\nGold: {q['expect']}\n" + item(r) + "\n")
    open(cfg.wpath("report", f"review_{corpus}.md"), "w").write("\n".join(out) + "\n")
    return len(dis), len(sample)


def main():
    args = dict((a.lstrip("-").split("=", 1) + [""])[:2] for a in sys.argv[1:])
    cfg = C.Config()
    R = Report(cfg)
    tag, serial_tag = args.get("tag", "run*"), args.get("serial-tag", "serial")
    os.makedirs(cfg.wpath("report"), exist_ok=True)
    md, summary, all_rows, all_serial = [f"# Benchmark tables (judge: {cfg.label(cfg.judge)})\n"], {}, [], []
    for corpus in C.CORPORA:
        K, rows = rows_for(cfg, corpus, tag)
        _, serial = rows_for(cfg, corpus, serial_tag)
        summary[corpus] = {"main": {}, "paired": {}, "kinds": {}}
        if rows:
            all_rows += rows
            main_table(R, md, summary, corpus, K, rows)
            summary[corpus]["paired"] = paired_table(
                R, md, rows, f"Paired differences per question ({corpus}), mean over the runs per question, 95% "
                             "bootstrap CI over questions; * = CI excludes 0.")
            kind_table(R, md, summary, corpus, K, rows)
            summary[corpus]["review"] = write_review(cfg, corpus, K, rows + serial)
        if serial:
            all_serial += serial
            latency_section(R, md, summary[corpus], serial,
                            f"Latency, serial subset ({corpus}: {len({r['qi'] for r in serial})} questions, one "
                            "request at a time)")
    if all_rows:
        nq = len({(r["corpus"], r["qi"]) for r in all_rows})
        summary["pooled"] = {"paired": paired_table(R, md, all_rows, f"Paired differences pooled over all {nq} "
                                                    "questions, 95% bootstrap CI over questions.")}
        routing_analysis(R, all_rows)
    if all_serial:
        summary.setdefault("pooled", {})
        latency_section(R, md, summary["pooled"], all_serial, "Latency pooled over the serial subset "
                        f"({len({(r['corpus'], r['qi']) for r in all_serial})} questions)")
    open(cfg.wpath("report", "tables.md"), "w").write("\n".join(md) + "\n")
    C.dump_json(summary, cfg.wpath("report", "summary.json"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
