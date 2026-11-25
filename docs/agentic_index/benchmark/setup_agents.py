# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Registers the chat models and agents.

Baseline cluster: per agent model, the Baseline and Baseline + schema notes agents (ListIndexTool, IndexMappingTool,
SearchIndexTool with the raw response).
Agentic cluster: the agentic index's own chat models, and per setup and agent model an _ask and a _retrieve
ConnectorTool through the logging proxy (aiproxy.py):
  ai_all     indices ["*"] (every non-system index and data stream), max_indices_selected 5
  ai_all10   indices ["*"], max_indices_selected 10
  ai_scoped  indices layout.SCOPE, max_indices_selected 5
  inner variants: ai_all with one model as the agent and another as the agentic index's own model
System prompts are passed at execute time; Baseline + schema notes gets per-question notes (notes.capped).
Writes <work>/state_baseline.json, <work>/state_agentic.json and <work>/prompts.json.
Usage: python3 setup_agents.py"""

import common as C
import layout as L
import prompts as P

BASELINE_TOOLS = [{"type": "ListIndexTool", "name": "ListIndexTool"},
                  {"type": "IndexMappingTool", "name": "IndexMappingTool"},
                  # the raw response shows hits.total, so a successful zero-hit search is explicit
                  {"type": "SearchIndexTool", "name": "SearchIndexTool", "parameters": {"return_raw_response": "true"}}]
COVERS = {"ai_all": "every index of the configured cluster",
          "ai_all10": "every index of the configured cluster",
          "ai_scoped": "the knowledge indices of the configured cluster: release notes, documentation, pricing and "
                       "support content (index patterns " + ", ".join(L.SCOPE) + ")"}
TARGET = {"ai_all": (["*"], 5), "ai_all10": (["*"], 10), "ai_scoped": (L.SCOPE, 5)}


def variants(cfg):
    """(setup, model key, agent model, inner model) for every agentic setup."""
    out = [(s, a, a, a) for s in ("ai_all", "ai_scoped", "ai_all10") if s in cfg.setups for a in cfg.agents]
    if "ai_all" in cfg.setups:
        out += [("ai_all", f"{v['agent']}__{v['inner']}", v["agent"], v["inner"]) for v in cfg.inner_variants]
    return out


def main():
    cfg = C.Config()
    n = len(C.load_json(cfg.wpath("indices.json"))["indices"])
    line = L.CLUSTER_LINE.format(n=n)
    prompts = {"cluster_line": line}

    st_b = C.load_json(cfg.wpath("cluster_baseline.json"))
    base = st_b["url"]
    models = {k: C.register_agent_model(base, k, cfg.models[k]) for k in cfg.agents}
    st_b.update(models=models, agents={})
    for setup in ("baseline", "baseline_notes"):
        if setup in cfg.setups:
            st_b["agents"][setup] = {k: C.register_agent(base, f"{setup}-{k}", m, BASELINE_TOOLS)
                                     for k, m in models.items()}
    prompts["baseline"] = P.baseline(line, st_b["embedding_model_id"])
    C.dump_json(st_b, cfg.wpath("state_baseline.json"))

    st_a = C.load_json(cfg.wpath("cluster_agentic.json"))
    base = st_a["url"]
    vs = variants(cfg)
    llms = {k: C.register_index_llm(base, k, cfg.models[k]) for k in sorted({v[3] for v in vs})}
    models = {k: C.register_agent_model(base, k, cfg.models[k]) for k in sorted({v[2] for v in vs})}
    st_a.update(index_llm=llms, models=models, connectors={}, agents={}, inner={})
    prompts["ai"] = P.agentic(line)
    prompts["tools"] = {}
    for setup, key, agent, inner in vs:
        indices, max_sel = TARGET[setup]
        tag = f"{setup}-{key}"
        conns = {mode: C.call(base, "POST", "/_plugins/_ml/connectors/_create", C.agentic_index_connector(
            cfg, indices, mode, llms[inner]["model_id"], st_a["embedding_model_id"], tag, max_sel))["connector_id"]
                 for mode in ("_retrieve", "_ask")}
        retrieve, ask = P.ai_descriptions(COVERS[setup])
        prompts["tools"][setup] = {"retrieve": retrieve, "ask": ask}
        st_a["connectors"][tag] = conns
        st_a["inner"][tag] = inner
        st_a["agents"].setdefault(setup, {})[key] = C.register_agent(
            base, tag, models[agent], C.agentic_tools(conns, retrieve, ask))
    C.dump_json(st_a, cfg.wpath("state_agentic.json"))
    C.dump_json(prompts, cfg.wpath("prompts.json"))
    print("agents registered:", {s: sorted(v) for s, v in {**st_b["agents"], **st_a["agents"]}.items()})


if __name__ == "__main__":
    main()
