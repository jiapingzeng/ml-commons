# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: configuration, OpenSearch REST calls, chat model connectors for the three model types
(openai, sagemaker, bedrock), direct chat calls for the judge, and the agentic index ConnectorTools.

Configuration comes from one JSON file (BENCH_CONFIG, default ./config.json). Any string value may reference
environment variables as ${VAR} or ${VAR:-default}. Secrets are never written to disk: API keys are read from the
environment variable named by api_key_env, and AWS credentials come from the default AWS credential chain (or the
optional aws_profile) at run time and are sent only to the local clusters."""

import json
import os
import re
import subprocess
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EMBED_MODEL = ("huggingface/sentence-transformers/all-MiniLM-L6-v2", "1.0.2")
# ml-commons counts LLM turns and tool results as separate iterations: 31 = 15 tool calls plus a final answer turn.
MAX_ITERATION = 31
SETUPS = ["baseline", "baseline_notes", "ai_all", "ai_scoped", "ai_all10"]
SETUP_NAMES = {"baseline": "Baseline", "baseline_notes": "Baseline + schema notes", "ai_all": "Agentic Index (all)",
               "ai_scoped": "Agentic Index (scoped)", "ai_all10": "Agentic Index (all, max 10)"}
CORPORA = ["synth", "real", "route"]
MODEL_TYPES = ("openai", "sagemaker", "bedrock")
KEY_RE = re.compile(r"^[A-Za-z0-9_.]+$")


# ---------------- configuration ----------------

def _expand(v):
    if isinstance(v, str):
        def sub(m):
            name, sep, default = m.group(1).partition(":-")
            return os.environ.get(name) or (default if sep else "")
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?)\}", sub, v)
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


class Config:
    def __init__(self, path=None):
        path = path or os.environ.get("BENCH_CONFIG", os.path.join(HERE, "config.json"))
        if not os.path.exists(path):
            raise SystemExit(f"Config file {path} not found. Copy config.example.json to config.json and edit it, or "
                             "set BENCH_CONFIG.")
        self.path = path
        c = _expand(json.load(open(path)))
        self.raw = c
        self.clusters = c["clusters"]
        self.models = c["models"]
        for k, m in self.models.items():
            if not KEY_RE.match(k):
                raise SystemExit(f"Model key {k!r}: use letters, digits, '.' and '_' only (no '-').")
            if m.get("type") not in MODEL_TYPES:
                raise SystemExit(f"Model {k}: type must be one of {', '.join(MODEL_TYPES)}.")
        self.agents = c["agents"]
        self.inner_variants = c.get("inner_variants", [])
        for k in self.agents + [v[x] for v in self.inner_variants for x in ("agent", "inner")] + [c["judge"]]:
            if k not in self.models:
                raise SystemExit(f"{k!r} is used in agents, inner_variants or judge but is not defined under models.")
        self.setups = c.get("setups", ["baseline", "baseline_notes", "ai_all", "ai_scoped"])
        bad = [s for s in self.setups if s not in SETUPS]
        if bad:
            raise SystemExit(f"Unknown setups {bad}; choose from {SETUPS}.")
        self.judge = c["judge"]
        self.runs = c.get("runs", [1, 2, 3])
        self.workers = int(c.get("workers", 4))
        self.judge_workers = int(c.get("judge_workers", 8))
        self.serial_models = c.get("serial_models", self.agents)
        self.proxy = c.get("proxy", {})
        self.work = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), c.get("work_dir", "work")))
        os.makedirs(self.work, exist_ok=True)

    def wpath(self, *parts):
        return os.path.join(self.work, *parts)

    def label(self, key):
        """Display name of an agent model key or an inner variant key (agent__inner)."""
        if "__" in key:
            a, i = key.split("__", 1)
            return f"{self.label(a)} agent, {self.label(i)} inside"
        return self.models.get(key, {}).get("label", key)

    def variant_keys(self):
        return [f"{v['agent']}__{v['inner']}" for v in self.inner_variants]

    def proxy_host(self):
        h = self.proxy.get("host", "auto")
        if h != "auto":
            return h
        out = subprocess.run(["docker", "network", "inspect", "bridge", "--format",
                              "{{(index .IPAM.Config 0).Gateway}}"], capture_output=True, text=True, check=True)
        return out.stdout.strip()

    def proxy_port(self):
        return int(self.proxy.get("port", 9422))


# ---------------- OpenSearch REST ----------------

def call(base, method, path, body=None, ok=(200, 201), timeout=300):
    r = requests.request(method, base + path, json=body, timeout=timeout)
    if r.status_code not in ok:
        raise RuntimeError(f"{method} {base}{path} -> {r.status_code}: {r.text[:1500]}")
    return r.json() if r.text else {}


def wait_task(base, task_id, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        t = call(base, "GET", f"/_plugins/_ml/tasks/{task_id}")
        if t["state"] == "COMPLETED":
            return t
        if t["state"] in ("FAILED", "COMPLETED_WITH_ERROR"):
            raise RuntimeError(f"task {task_id} failed: {t}")
        time.sleep(3)
    raise TimeoutError(task_id)


def bulk(base, docs, refresh=True):
    for i in range(0, len(docs), 200):
        lines = []
        for index, _id, src in docs[i:i + 200]:
            lines.append(json.dumps({"index": {"_index": index, "_id": _id}}))
            lines.append(json.dumps(src))
        r = requests.post(base + "/_bulk", data="\n".join(lines) + "\n",
                          headers={"Content-Type": "application/x-ndjson"}, timeout=900)
        r.raise_for_status()
        if r.json()["errors"]:
            bad = [it for it in r.json()["items"] if it["index"].get("error")][:3]
            raise RuntimeError(json.dumps(bad)[:2000])
    if refresh:
        call(base, "POST", "/_refresh")


def endpoint_regexes(cfg):
    """trusted_connector_endpoints_regex entries for every configured model and the logging proxy."""
    out = [f"^http://{re.escape(cfg.proxy_host())}:{cfg.proxy_port()}/.*$"]
    for m in cfg.models.values():
        if m["type"] == "openai":
            u = re.match(r"^(https?://[^/]+)", m["url"]).group(1)
            out.append("^" + re.escape(u) + "/.*$")
        elif m["type"] == "sagemaker":
            out.append("^https://runtime\\.sagemaker\\..*\\.amazonaws\\.com/.*$")
        else:
            out.append("^https://bedrock-runtime\\..*\\.amazonaws\\.com/.*$")
    return sorted(set(out))


def settings(base, cfg):
    call(base, "PUT", "/_cluster/settings", {"persistent": {
        "plugins.ml_commons.agentic_index_enabled": True,
        "plugins.ml_commons.only_run_on_ml_node": False,
        "plugins.ml_commons.native_memory_threshold": 100,
        # the logging proxy and a local LLM are on private addresses
        "plugins.ml_commons.connector.private_ip_enabled": True,
        "plugins.ml_commons.trusted_connector_endpoints_regex": endpoint_regexes(cfg),
    }})


def embedding_model(base):
    emb = call(base, "POST", "/_plugins/_ml/models/_register?deploy=true",
               {"name": EMBED_MODEL[0], "version": EMBED_MODEL[1], "model_format": "TORCH_SCRIPT"})
    model_id = wait_task(base, emb["task_id"])["model_id"]
    for _ in range(200):
        if call(base, "GET", f"/_plugins/_ml/models/{model_id}")["model_state"] == "DEPLOYED":
            return model_id
        time.sleep(3)
    raise TimeoutError("embedding model deploy")


def embedding_pipeline(base, name, model_id, field_map):
    call(base, "PUT", f"/_ingest/pipeline/{name}", {"processors": [{"text_embedding": {
        "model_id": model_id, "field_map": field_map}}]})


def hybrid_pipeline(base, name="hybrid-norm"):
    call(base, "PUT", f"/_search/pipeline/{name}", {"phase_results_processors": [{"normalization-processor": {
        "normalization": {"technique": "min_max"},
        "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.5, 0.5]}}}}]})
    return name


KNN = {"type": "knn_vector", "dimension": 384, "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}


# ---------------- chat model connectors ----------------

def aws_credentials(m):
    import boto3
    c = boto3.Session(profile_name=m.get("aws_profile") or None, region_name=m["region"]).get_credentials()
    if c is None:
        raise SystemExit("No AWS credentials found for a sagemaker or bedrock model; configure the AWS credential chain "
                         "or set aws_profile.")
    c = c.get_frozen_credentials()
    out = {"access_key": c.access_key, "secret_key": c.secret_key}
    if c.token:
        out["session_token"] = c.token
    return out


def api_key(m):
    return os.environ.get(m.get("api_key_env") or "", "") or "EMPTY"


def openai_extras(m, max_tokens=True):
    """Request fields after messages (and tools): max_tokens, temperature, thinking toggle, extra_body."""
    e = {}
    if max_tokens:
        e["max_tokens"] = int(m.get("max_tokens", 2048))
    if m.get("temperature", 0) is not None:
        e["temperature"] = m.get("temperature", 0)
    if "thinking" in m:
        e["chat_template_kwargs"] = {"enable_thinking": bool(m["thinking"])}
    e.update(m.get("extra_body") or {})
    return e


def _splice(prefix, extras, suffix="}"):
    return prefix + ("" if not extras else ", " + json.dumps(extras)[1:-1]) + suffix


def _model_field(m):
    return f'"model": {json.dumps(m["model"])}, ' if m.get("model") else ""


def agent_body(m):
    """Request body for the conversational agent (native function calling)."""
    if m["type"] in ("openai", "sagemaker"):
        prefix = ('{' + _model_field(m) + '"messages": [{"role": "system", "content": "${parameters.system_prompt}"}, '
                  '${parameters._chat_history:-}{"role": "user", "content": "${parameters.prompt}"}'
                  '${parameters._interactions:-}]${parameters.tool_configs:-}')
        return _splice(prefix, openai_extras(m))
    inf = {"maxTokens": int(m.get("max_tokens", 2048))}
    if m.get("temperature") is not None:
        inf["temperature"] = m["temperature"]
    tail = {"inferenceConfig": inf}
    if not m.get("thinking", False):
        tail["additionalModelRequestFields"] = {"thinking": {"type": "disabled"}}
    return _splice('{"system": [{"text": "${parameters.system_prompt}"}], "messages": [${parameters._chat_history:-}'
                   '{"role": "user", "content": [{"text": "${parameters.prompt}"}]}${parameters._interactions:-}]'
                   '${parameters.tool_configs:-}', tail)


INTERFACE = {"openai": "openai/v1/chat/completions", "sagemaker": "openai/v1/chat/completions",
             "bedrock": "bedrock/converse/claude"}

# The agentic index sends a JSON messages array and max_tokens. Bedrock Converse takes the system prompt separately and
# content blocks per message, so its connector converts messages in a pre-process script and maps the Converse
# response back to choices[0].message.content and usage in a post-process script. The output keys differ from the
# input keys because ml-commons re-applies input parameters after pre-processing. Painless strings only support \\ and
# quote escapes, so newline, CR and tab are real characters inside the quotes.
PAINLESS_ESC = ("String esc(def x) { String s = x == null ? '' : x.toString(); "
                "return s.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\n', '\\\\n')"
                ".replace('\r', '\\\\r').replace('\t', '\\\\t'); }\n")
CONVERSE_PRE = PAINLESS_ESC + (
    "String sys = ''; StringBuilder b = new StringBuilder('['); int n = 0;\n"
    "for (def m : params.messages) {\n"
    "  if (m.role == 'system') { sys = sys + m.content; }\n"
    "  else { if (n > 0) { b.append(','); } n++;\n"
    "    b.append('{\"role\":\"' + m.role + '\",\"content\":[{\"text\":\"' + esc(m.content) + '\"}]}'); }\n"
    "}\n"
    "b.append(']');\n"
    "String mt = params.max_tokens.toString(); if (mt.endsWith('.0')) { mt = mt.substring(0, mt.length() - 2); }\n"
    "return '{\"parameters\":{\"system_text\":\"' + esc(sys) + '\",\"bedrock_messages\":' + b.toString()"
    " + ',\"max_tok\":\"' + mt + '\"}}';")
CONVERSE_POST = PAINLESS_ESC + (
    "StringBuilder b = new StringBuilder();\n"
    "for (def c : params.output.message.content) { if (c.text != null) { b.append(c.text); } }\n"
    "return '{\"name\":\"response\",\"dataAsMap\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"'"
    " + esc(b.toString()) + '\"}}],\"usage\":{\"prompt_tokens\":' + params.usage.inputTokens"
    " + ',\"completion_tokens\":' + params.usage.outputTokens + '}}}';")


def index_llm_body(m):
    """Request body for the agentic index's own LLM calls."""
    if m["type"] in ("openai", "sagemaker"):
        return _splice('{' + _model_field(m) + '"messages": ${parameters.messages}, '
                       '"max_tokens": ${parameters.max_tokens}', openai_extras(m, max_tokens=False))
    inf = '"inferenceConfig": {"maxTokens": ${parameters.max_tok}' + (
        f', "temperature": {json.dumps(m["temperature"])}' if m.get("temperature") is not None else "") + '}'
    tail = {} if m.get("thinking", False) else {"additionalModelRequestFields": {"thinking": {"type": "disabled"}}}
    return _splice('{"system": [{"text": "${parameters.system_text}"}], "messages": ${parameters.bedrock_messages}, '
                   + inf, tail)


def connector(name, m, body, convert_messages=False):
    action = {"action_type": "predict", "method": "POST", "headers": {"content-type": "application/json"},
              "request_body": body}
    out = {"name": name, "description": f"benchmark chat model ({m['type']})", "version": 1,
           "client_config": {"read_timeout": int(m.get("read_timeout_ms", 120000)), "connection_timeout": 10000,
                             "max_retry_times": int(m.get("max_retry_times", 3))},
           "actions": [action]}
    if m["type"] == "openai":
        action["url"] = m["url"]
        action["headers"]["Authorization"] = "Bearer ${credential.openAI_key}"
        out.update(protocol="http", parameters={}, credential={"openAI_key": api_key(m)})
    elif m["type"] == "sagemaker":
        action["url"] = f"https://runtime.sagemaker.{m['region']}.amazonaws.com/endpoints/{m['endpoint']}/invocations"
        out.update(protocol="aws_sigv4", credential=aws_credentials(m),
                   parameters={"region": m["region"], "service_name": "sagemaker"})
    else:
        action["url"] = f"https://bedrock-runtime.{m['region']}.amazonaws.com/model/{m['model_id']}/converse"
        params = {"region": m["region"], "service_name": "bedrock"}
        if convert_messages:
            action["pre_process_function"] = CONVERSE_PRE
            action["post_process_function"] = CONVERSE_POST
            params["pre_process_function.process_remote_inference_input"] = "true"
            params["pre_process_function.convert_remote_inference_param_to_object"] = "true"
        out.update(protocol="aws_sigv4", credential=aws_credentials(m), parameters=params)
    return out


def connector_credential(m):
    if m["type"] == "openai":
        return {"openAI_key": api_key(m)}
    return aws_credentials(m)


def register_model(base, name, conn):
    c = call(base, "POST", "/_plugins/_ml/connectors/_create", conn)
    model = call(base, "POST", "/_plugins/_ml/models/_register?deploy=true",
                 {"name": name, "function_name": "remote", "connector_id": c["connector_id"]})
    wait_task(base, model["task_id"])
    return {"connector_id": c["connector_id"], "model_id": model["model_id"]}


def register_agent_model(base, key, m):
    return dict(register_model(base, f"agent-{key}", connector(f"agent-{key}", m, agent_body(m))), type=m["type"])


def register_index_llm(base, key, m):
    return dict(register_model(base, f"index-{key}", connector(f"index-{key}", m, index_llm_body(m), True)),
                type=m["type"])


def register_agent(base, name, model, tools):
    """The system prompt is passed as an execute parameter (it overrides llm.parameters), which avoids the 32 KB
    keyword limit of the agent's flat_object llm field for long schema notes."""
    return call(base, "POST", "/_plugins/_ml/agents/_register", {
        "name": name, "type": "conversational", "description": "agentic index benchmark",
        "llm": {"model_id": model["model_id"], "parameters": {
            "max_iteration": MAX_ITERATION, "system_prompt": "(set at execute time)",
            "prompt": "${parameters.question}"}},
        "memory": {"type": "conversation_index"},
        "parameters": {"_llm_interface": INTERFACE[model["type"]]},
        "tools": tools})["agent_id"]


# ---------------- direct chat calls (judge) ----------------

def chat(m, prompt, max_tokens=1024, timeout=300):
    """One user message to a configured model from the harness; returns the reply text."""
    if m["type"] == "bedrock":
        client = _bedrock_client(m)
        kwargs = {"inferenceConfig": {"maxTokens": max_tokens}}
        if m.get("temperature") is not None:
            kwargs["inferenceConfig"]["temperature"] = m["temperature"]
        r = client.converse(modelId=m["model_id"], messages=[{"role": "user", "content": [{"text": prompt}]}],
                            **kwargs)
        return "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    body = {"messages": [{"role": "user", "content": prompt}], **openai_extras(m, max_tokens=False),
            "max_tokens": max_tokens}
    if m.get("model"):
        body["model"] = m["model"]
    if m["type"] == "openai":
        r = requests.post(m.get("client_url") or m["url"], json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key(m)}"})
        r.raise_for_status()
        j = r.json()
    else:
        r = _sagemaker_client(m).invoke_endpoint(EndpointName=m["endpoint"], ContentType="application/json",
                                                 Body=json.dumps(body))
        j = json.loads(r["Body"].read())
    text = j["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S)


_CLIENTS = {}


def _boto_client(service, m):
    key = (service, m.get("aws_profile"), m["region"])
    if key not in _CLIENTS:
        import boto3
        from botocore.config import Config as BotoConfig
        _CLIENTS[key] = boto3.Session(profile_name=m.get("aws_profile") or None, region_name=m["region"]).client(
            service, config=BotoConfig(read_timeout=300, retries={"max_attempts": 8, "mode": "adaptive"}))
    return _CLIENTS[key]


def _bedrock_client(m):
    return _boto_client("bedrock-runtime", m)


def _sagemaker_client(m):
    return _boto_client("sagemaker-runtime", m)


# ---------------- agentic index ConnectorTools ----------------

# Trims the _ask/_retrieve response the agent sees to answer + passages (id, index, title, text, metadata) + timed_out,
# and rewrites the answer's [P#] passage markers to document _ids (the passages no longer carry passage_id). Each
# passage keeps its metadata (everything PassageStore keeps, at most 20 fields and 1,000 characters) as one
# "field: value; ..." string, so the agent can check dates and plans itself.
# Painless string literals have no \n escape, so control characters are built with (char) casts.
TRIM_POST_PROCESS = r"""
String esc(def v) {
  if (v == null) { return ''; }
  String s = v.toString();
  s = s.replace('\\', '\\\\');
  s = s.replace('"', '\\"');
  s = s.replace(String.valueOf((char)10), '\\n');
  s = s.replace(String.valueOf((char)13), '\\r');
  s = s.replace(String.valueOf((char)9), '\\t');
  s = s.replace(String.valueOf((char)8), '');
  s = s.replace(String.valueOf((char)12), '');
  return s;
}
boolean digit(char c) { return c >= (char)48 && c <= (char)57; }
boolean word(char c) { return digit(c) || (c >= (char)65 && c <= (char)90) || (c >= (char)97 && c <= (char)122) || c == (char)95 || c == (char)45; }
Map ids = new HashMap();
List ps = params.passages == null ? new ArrayList() : params.passages;
for (def p : ps) { ids.put(p.passage_id, p.doc_id); }
StringBuilder sb = new StringBuilder();
sb.append('{"name":"response","dataAsMap":{');
if (params.containsKey('answer')) {
  def a = params.answer;
  if (a == null) {
    sb.append('"answer":null,');
  } else {
    String t = a.toString();
    StringBuilder o = new StringBuilder();
    int i = 0;
    int n = t.length();
    while (i < n) {
      char c = t.charAt(i);
      if (c == (char)80 && i + 1 < n && digit(t.charAt(i + 1)) && (i == 0 || !word(t.charAt(i - 1)))) {
        int j = i + 1;
        while (j < n && digit(t.charAt(j))) { j++; }
        String label = t.substring(i, j);
        if (ids.containsKey(label) && (j >= n || !word(t.charAt(j)))) {
          o.append(ids.get(label).toString());
          i = j;
          continue;
        }
      }
      o.append(t.substring(i, i + 1));
      i++;
    }
    sb.append('"answer":"').append(esc(o.toString())).append('",');
  }
}
sb.append('"passages":[');
boolean first = true;
for (def p : ps) {
  if (!first) { sb.append(','); }
  first = false;
  StringBuilder m = new StringBuilder();
  def md = p.metadata;
  if (md != null) {
    for (def e : md.entrySet()) {
      if (m.length() > 0) { m.append('; '); }
      m.append(e.getKey()).append(': ').append(e.getValue());
    }
  }
  sb.append('{"id":"').append(esc(p.doc_id)).append('","index":"').append(esc(p.index));
  sb.append('","title":"').append(esc(p.title)).append('","metadata":"').append(esc(m.toString()));
  sb.append('","text":"').append(esc(p.text)).append('"}');
}
sb.append('],"timed_out":').append(params.timed_out == true ? 'true' : 'false').append('}}');
return sb.toString();
"""


def agentic_index_connector(cfg, indices, mode, llm_model_id, embedding_model_id, tag, max_indices_selected=5, k=5):
    """ConnectorTool connector for _ask/_retrieve over a list of index patterns (the request body's indices field),
    through the logging proxy; tag (the first path segment) tells the proxy log which setup and model made the call."""
    body = json.dumps({"question": "QUESTION", "indices": indices, "llm_model_id": llm_model_id,
                       "embedding_model_id": embedding_model_id, "k": k,
                       "max_indices_selected": max_indices_selected})
    body = body.replace('"QUESTION"', '"${parameters.question}"')
    action = {"action_type": "execute", "method": "POST",
              "url": f"http://{cfg.proxy_host()}:{cfg.proxy_port()}/{tag}/_plugins/_ml/agentic_index/{mode}",
              "headers": {"content-type": "application/json"}, "request_body": body,
              "post_process_function": TRIM_POST_PROCESS}
    return {"name": f"agentic-index{mode}-{tag}", "description": f"Agentic index {mode} ({tag})", "version": 1,
            "protocol": "http", "parameters": {}, "credential": {"unused": "none"},
            "client_config": {"read_timeout": 300000, "connection_timeout": 10000, "max_retry_times": 0},
            "actions": [action]}


QUESTION_SCHEMA = json.dumps({"type": "object", "properties": {"question": {
    "type": "string", "description": "The question in natural language, with every constraint (product, plan, "
                                     "dates, error codes) it needs."}}, "required": ["question"]})


def agentic_tools(conn_ids, retrieve_desc, ask_desc):
    return [
        {"type": "ConnectorTool", "name": "agentic_index_retrieve", "description": retrieve_desc,
         "parameters": {"connector_id": conn_ids["_retrieve"]},
         "attributes": {"input_schema": QUESTION_SCHEMA, "strict": False}},
        {"type": "ConnectorTool", "name": "agentic_index_ask", "description": ask_desc,
         "parameters": {"connector_id": conn_ids["_ask"]},
         "attributes": {"input_schema": QUESTION_SCHEMA, "strict": False}},
    ]


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def dump_json(obj, path, indent=1):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
