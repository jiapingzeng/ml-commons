# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Runs questions through the setup agents.

Throughput mode: per (run, agent model), `workers` threads split the questions of every corpus; for one question a
thread runs every setup of that model back to back in a rotating order, so paired executions see the same endpoint
load. Each run starts its question list at a different offset, so the same question rarely runs in two runs at once.
Executions already saved under the same tag are skipped, so an interrupted run resumes where it stopped.
Raw results: <work>/results/<corpus>/<tag>/<setup>-<model>.jsonl (one execution per line, appended); tag is run<N>.
Serial mode (--serial): the latency subset (data/latency_subset.json: 20 synthetic, 20 real and all 15 route
questions), the configured serial_models, every setup, one request at a time. Tag: serial.
Usage: python3 run.py [--corpora=synth,real,route] [--runs=1,2,3] [--models=a,b] [--setups=...]
       [--questions=synth:1,real:2] [--workers=4] [--serial] [--tag=smoke]"""

import json
import os
import sys
import threading
import time

import requests

import common as C
import notes as N
import prompts as P

LOCK = threading.Lock()


def load_questions(corpus):
    return json.load(open(os.path.join(C.DATA, f"questions_{corpus}.json")))["questions"]


class Harness:
    def __init__(self, cfg):
        self.cfg = cfg
        self.questions = {c: load_questions(c) for c in C.CORPORA}
        self.sb = C.load_json(cfg.wpath("state_baseline.json"))
        self.sa = C.load_json(cfg.wpath("state_agentic.json"))
        self.prompts = C.load_json(cfg.wpath("prompts.json"))
        self.notes = C.load_json(cfg.wpath("notes.json"))
        self.total_indices = len(self.notes["entries"])

    def agent(self, setup, model, question):
        """(base url, agent id, system prompt, extra row fields) or None."""
        if setup in ("baseline", "baseline_notes"):
            ids = self.sb["agents"].get(setup, {})
            if model not in ids:
                return None
            if setup == "baseline":
                return self.sb["url"], ids[model], self.prompts["baseline"], {}
            notes, shown, _ = N.capped(self.notes, question)
            sysp = P.baseline_notes(self.prompts["cluster_line"], self.sb["embedding_model_id"], notes, len(shown),
                                    self.total_indices)
            return self.sb["url"], ids[model], sysp, {"notes_shown": shown, "notes_chars": len(notes)}
        ids = self.sa["agents"].get(setup, {})
        if model not in ids:
            return None
        return self.sa["url"], ids[model], self.prompts["ai"], {}


def traces(base, message_id):
    # the traces API returns at most 100 entries per call; 31 iterations stay well below that
    for _ in range(5):
        r = requests.get(f"{base}/_plugins/_ml/memory/message/{message_id}/traces?max_results=100", timeout=60)
        if r.status_code == 200:
            return r.json().get("traces", [])
        time.sleep(1)
    return []


def run_one(base, agent_id, system_prompt, question):
    t0 = time.time()
    try:
        r = requests.post(f"{base}/_plugins/_ml/agents/{agent_id}/_execute",
                          json={"parameters": {"question": question, "system_prompt": system_prompt, "verbose": True,
                                               "include_token_usage": True}}, timeout=900)
    except Exception as e:
        return {"error": f"client: {e!r}"[:2000], "t0": t0, "t1": time.time(),
                "latency_ms": int((time.time() - t0) * 1000)}
    t1 = time.time()
    out = {"t0": t0, "t1": t1, "latency_ms": int((t1 - t0) * 1000)}
    if r.status_code != 200:
        out["error"] = r.text[:3000]
        return out
    responses = []
    for o in r.json()["inference_results"][0]["output"]:
        if o.get("name") == "parent_interaction_id":
            out["message_id"] = o["result"]
        elif o.get("name") == "memory_id":
            out["memory_id"] = o["result"]
        elif o.get("name") == "token_usage":
            out["token_usage"] = o["dataAsMap"]
        elif o.get("name") == "response":
            responses.append(o.get("result"))
    out["answer"] = responses[-1] if responses else None
    if out.get("message_id"):
        tr = traces(base, out["message_id"])
        out["steps"] = [{"origin": x.get("origin"), "input": x.get("input"), "response": x.get("response"),
                         "trace_number": x.get("trace_number")} for x in sorted(tr, key=lambda x: x["trace_number"])]
    return out


def variants_for(H, model, setups):
    v = [(s, model) for s in setups if H.agent(s, model, "") is not None]
    if "ai_all" in setups:
        v += [("ai_all", k) for k in H.cfg.variant_keys() if k.split("__")[0] == model]
    return v


def main():
    args = dict((a.lstrip("-").split("=", 1) + [""])[:2] for a in sys.argv[1:])
    cfg = C.Config()
    H = Harness(cfg)
    corpora = [c for c in args.get("corpora", ",".join(C.CORPORA)).split(",") if c in H.questions]
    runs = [int(x) for x in args.get("runs", ",".join(map(str, cfg.runs))).split(",")]
    models = args.get("models", ",".join(cfg.agents)).split(",")
    setups = args.get("setups", ",".join(cfg.setups)).split(",")
    workers = int(args.get("workers", cfg.workers))
    serial = "serial" in args

    def save(corpus, tag, row):
        out_dir = cfg.wpath("results", corpus, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{row['setup']}-{row['model']}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    def execute(corpus, tag, run, qi, s, m):
        q = H.questions[corpus][qi - 1]["q"]
        base, agent_id, sysp, extra = H.agent(s, m, q)
        res = run_one(base, agent_id, sysp, q)
        res.update(extra, question=q, qi=qi, setup=s, model=m, run=run, tag=tag, corpus=corpus)
        with LOCK:
            save(corpus, tag, res)
            print(f"[{corpus} {tag} {s}-{m} q{qi}] {res['latency_ms']}ms steps={len(res.get('steps', []))} "
                  f"{(res.get('answer') or res.get('error') or '')[:70]!r}", flush=True)

    if serial:
        subset = json.load(open(os.path.join(C.DATA, "latency_subset.json")))
        tag = args.get("tag", "serial")
        serial_models = [m for m in models if m in cfg.serial_models] if "models" in args else cfg.serial_models
        for corpus in corpora:
            for qi in subset[corpus]:
                for model in serial_models:
                    v = variants_for(H, model, setups)
                    rot = qi % len(v)
                    for s, m in v[rot:] + v[:rot]:
                        execute(corpus, tag, 0, qi, s, m)
        return

    done = set()
    for corpus in corpora:
        for run in runs:
            tag = args.get("tag", f"run{run}")
            d = cfg.wpath("results", corpus, tag)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                for line in open(os.path.join(d, fn)):
                    row = json.loads(line)
                    if not str(row.get("error", "")).startswith("client:"):
                        done.add((corpus, tag, row["model"], row["setup"], row["qi"]))
    print("resuming with", len(done), "saved executions", flush=True)

    if args.get("questions"):
        allq = [(c, int(i)) for c, i in (x.split(":") for x in args["questions"].split(","))]
    else:
        allq = [(c, i) for c in corpora for i in range(1, len(H.questions[c]) + 1)]

    def worker(run, model, my_q):
        v = variants_for(H, model, setups)
        tag = args.get("tag", f"run{run}")
        for corpus, qi in my_q:
            rot = (qi + run) % len(v)
            for s, m in v[rot:] + v[:rot]:
                if (corpus, tag, m, s, qi) in done:
                    continue
                execute(corpus, tag, run, qi, s, m)

    threads = []
    for run in runs:
        off = ((run - 1) * len(allq) // 3) % max(1, len(allq))
        order = allq[off:] + allq[:off]
        for model in models:
            for w in range(workers):
                t = threading.Thread(target=worker, args=(run, model, order[w::workers]))
                threads.append(t)
                t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
