# Agentic Index benchmark

This directory holds a benchmark harness for the experimental agentic index `_ask` and `_retrieve` APIs on this
branch. It compares an agent that uses the agentic index tools with agents that explore the cluster with the existing
search tools, on one realistic cluster of 162 indices. It is not part of the plugin: nothing here is built by Gradle,
checked by spotless or run in CI.

The guide assumes you run it on one machine that serves a chat model locally through an OpenAI-compatible
chat-completions endpoint with native tool calling (for example vLLM at `http://<host>:8000/v1/chat/completions`). The
same harness also runs models on a SageMaker endpoint that serves vLLM, and models on Bedrock through the Converse API.

- [What the benchmark measures](#what-the-benchmark-measures)
- [Prerequisites](#prerequisites)
- [Step by step](#step-by-step)
- [Expected runtime and request volume](#expected-runtime-and-request-volume)
- [Reading the results](#reading-the-results)
- [Known pitfalls](#known-pitfalls)
- [Files](#files)

## What the benchmark measures

### Setups

Every setup runs as an ml-commons `conversational` agent with native function calling, the same model, the same
system prompt apart from its search section, and at most 15 tool calls. An execution that hits the cap is scored
wrong.

| Setup | Key | In plain language |
|---|---|---|
| Baseline | `baseline` | The agent lists the indices, reads mappings and writes its own Query DSL (ListIndexTool, IndexMappingTool, SearchIndexTool). It sees the whole cluster and has to find the right indices itself. |
| Baseline + schema notes | `baseline_notes` | Baseline, plus notes about the indices in the system prompt: fields, top keyword values and date ranges, generated from the cluster. Notes for all 162 indices would be about 47k tokens, so per question the notes are capped at 12,000 characters with the indices that best match the question's words. The prompt says how many other indices exist. |
| Agentic Index (all) | `ai_all` | The agent has only two tools, `agentic_index_retrieve` and `agentic_index_ask`, over `indices: ["*"]`. The agentic index picks the relevant indices from a catalog, plans keyword and semantic subqueries, checks whether the evidence is enough and returns cited passages (or a cited answer). `max_indices_selected` is 5. |
| Agentic Index (all, max 10) | `ai_all10` | The same with `max_indices_selected` 10. Off by default. |
| Agentic Index (scoped) | `ai_scoped` | The same tools on a fixed knowledge scope, `release-notes*,docs-*,pricing*,support-*` (55 indices). Questions whose answer is outside the scope (tenant tickets, incident reports) cannot be answered. |
| Inner variants | `ai_all` with model key `<agent>__<inner>` | Agentic Index (all) with one model as the agent and another as the agentic index's own LLM, for example a small agent with a larger model inside. Optional. |

Every setup gets the same one-line description of the cluster and the same rule about official and unofficial
sources. The agentic index tools use `k` 5, hybrid search where an index has a vector field, and the default 2-minute
deadline.

### Cluster and questions

`build_cluster.py` builds the same cluster twice, on two single-node clusters, one for the Baselines and one for
Agentic Index, so the agentic index statistics cache and load on one side do not affect the other. `layout.py`
defines it:

- 162 indices (2 of them data streams), 9,372 documents. 69 indices hold the corpus documents, spread the way they
  accumulate in practice: per product, per month or release, per team and per repository, with aliases over the
  families. 93 indices are unrelated: orders, invoices, logs, metrics, per-tenant copies, tickets, audit events, the
  OpenSearch Dashboards sample data and more, with overlapping vocabulary (error codes, plans, SSO, prices).
- Field names differ by source. About a fifth of the indices have a `_meta.description`. Every relevant index and 13
  unrelated ones have one embedding field (`<text field>_embedding`, all-MiniLM-L6-v2) and a hybrid search pipeline.
- Every index with prose text has `status` official or unofficial.

Question sets (`data/questions_*.json`), each question with a gold answer, grading notes and gold passage ids:

| Corpus | Questions | Content |
|---|---|---|
| `synth` | 40 | Nimbus, a fictional software company: manuals, pricing sheets, release notes, help center, tickets, forum, wiki and more (`data/corpus_synth.json`, all values synthetic). |
| `real` | 60 | Public OpenSearch documentation, release notes 2.10 to 3.10 and GitHub issues, rebuilt from pinned commits by `fetch_real_corpus.py`. |
| `route` | 15 | The answer sits in one index among similar ones: one month's release notes, one dated pricing sheet, one tenant's tickets. |

### Metrics

Per (model, setup) and corpus, `report.py` prints:

- **Score**: correct 1, partial 0.5, wrong 0, summed over the questions and averaged over the runs.
- **Routing hit**: for Agentic Index, the routing step of any agentic call selected an index that holds a gold
  document; for the Baselines, the agent searched such an index (a pattern or alias counts for every index it
  reaches). **Gold index retrieved**: a returned hit or passage came from such an index; this one is comparable
  across setups.
- **Indices touched**: concrete indices searched. A Baseline `*` search counts every index.
- **Tool calls**, **failed calls** (Query DSL errors, guessed fields, missing indices) and **zero-hit searches** per
  execution.
- **LLM calls (agent + index)** and **tokens in+out** per execution: the agent's own calls plus the agentic index's
  internal calls, which come from the logging proxy.
- **Cap hits** (15 tool calls), **errors**, **unjudged** executions, **cite precision** (cited documents that are gold
  or acceptable), **gold recall** (gold documents cited) and unofficial documents cited.
- **Latency** (serial subset only): p50, p90 and mean seconds per question.

Paired differences compare two setups on the same questions; see [Reading the results](#reading-the-results).

## Prerequisites

- Docker, with about 14 GB of free memory for two clusters at the default 6 GB heap (`BENCH_HEAP` changes it), and
  about 3 GB of disk for the image (2.5 GB).
- JDK 21 to build the plugin from this branch.
- Python 3.10 or later, git, and network access to github.com (corpus sources) and artifacts.opensearch.org (the
  clusters download the embedding model).
- `pip install -r requirements.txt` (`requests`; `boto3` only for SageMaker or Bedrock models).
- A chat model with native tool calling behind an OpenAI-compatible endpoint. For vLLM, start it with tool calling
  enabled and reachable from the containers, for example:

  ```
  vllm serve <model> --served-model-name <served-model-name> --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser <parser for your model, for example hermes> --api-key "$LOCAL_LLM_API_KEY"
  ```

  If you turn thinking on, also start vLLM with the reasoning parser of your model (for example
  `--reasoning-parser qwen3`), so reasoning text does not end up in the message content. Bind vLLM to an address the
  containers reach (`0.0.0.0` or the docker bridge address) and keep the host firewall closed to other machines, or
  set an API key.
- A judge model: any configured model of the three types. Use a strong model that is not under test. Scores are only
  comparable between runs judged by the same judge.

## Step by step

Run every command from this directory (`docs/agentic_index/benchmark`) unless it says otherwise. Generated files go to
the work directory (`work/` by default; set `work_dir` in the config).

### 1. Build the plugin zip and the image

From the repository root:

```
./gradlew :opensearch-ml-plugin:bundlePlugin -x test
cp plugin/build/distributions/opensearch-ml-3.10.0.0-SNAPSHOT.zip docs/agentic_index/benchmark/docker/
```

Then, from this directory:

```
docker build -t opensearch-ml:agentic-index-bench docker
```

The image starts from `opensearchstaging/opensearch:3.10.0` and replaces the bundled ml-commons with the zip. The
skills plugin depends on ml-commons and is removed first. To use another 3.10.0 base image, pass
`--build-arg OPENSEARCH_IMAGE=<image>`; the plugin version must match the OpenSearch version.

### 2. Configure

```
cp config.example.json config.json
```

Edit `config.json`, or leave the `${VAR:-default}` placeholders and set the variables. Any string value may use
`${VAR}` or `${VAR:-default}`. Set `BENCH_CONFIG` to use a config file elsewhere.

| Field | Meaning |
|---|---|
| `clusters.baseline`, `clusters.agentic` | Cluster URLs as seen from this machine. They may be the same URL to run one cluster; two are recommended. |
| `proxy.host`, `proxy.port` | Where the logging proxy listens. `auto` uses the docker bridge gateway (`docker network inspect bridge`), which the containers reach and other machines do not. |
| `work_dir` | Output directory, relative to the config file. |
| `models.<key>` | One entry per model. Keys use letters, digits, `.` and `_`. Fields by type below. |
| `agents` | Model keys that run as agents in every setup. |
| `inner_variants` | Optional `[{"agent": "<key>", "inner": "<key>"}]`: Agentic Index (all) with another model inside. |
| `setups` | Any of `baseline`, `baseline_notes`, `ai_all`, `ai_scoped`, `ai_all10`. |
| `judge` | Model key of the judge. |
| `runs`, `workers` | Throughput runs (default 3) and parallel threads per agent model (default 4). |
| `judge_workers` | Parallel judge calls (default 8). |
| `serial_models` | Agent models for the serial latency subset. |

Model fields:

| Type | Fields |
|---|---|
| `openai` | `url` (as the cluster reaches it), `client_url` (as this machine reaches it, for the judge; defaults to `url`), `model` (served model name), `api_key_env` (name of the environment variable holding the key; the key is sent as `Authorization: Bearer`), `thinking` (`false` sends `chat_template_kwargs.enable_thinking: false`, `true` sends `true`, omit to send nothing), `temperature` (default 0, `null` to omit), `max_tokens` (default 2048), `extra_body` (more request fields). |
| `sagemaker` | `endpoint`, `region`, optional `aws_profile`; `model`, `thinking`, `temperature`, `max_tokens` and `extra_body` as for `openai`. The endpoint must serve the OpenAI chat-completions API (vLLM). |
| `bedrock` | `model_id` (model or inference profile id), `region`, optional `aws_profile`, `thinking` (`false` disables extended thinking), `temperature` (`null` to omit; some models reject it), `max_tokens`. |
| any | `label` (display name), `read_timeout_ms` (default 120000), `max_retry_times` (default 3), `judge_max_tokens` (default 1024). |

For a local model on the docker host, the cluster reaches it as `host.docker.internal` (`start_clusters.sh` maps it to
the host), and the harness reaches it as `127.0.0.1`:

```
export LOCAL_LLM_URL=http://host.docker.internal:8000/v1/chat/completions
export LOCAL_LLM_CLIENT_URL=http://127.0.0.1:8000/v1/chat/completions
export LOCAL_LLM_MODEL=<served-model-name>
export LOCAL_LLM_API_KEY=<key, or any value if vLLM runs without --api-key>
```

AWS credentials for SageMaker and Bedrock models come from the AWS credential chain (or `aws_profile`) when
`setup_agents.py` creates the connectors and when the judge runs. Use temporary credentials with only
`sagemaker:InvokeEndpoint` on the endpoint or `bedrock:InvokeModel` on the model. They are sent only to your local
clusters and never written to the work directory.

### 3. Start the clusters

```
bash start_clusters.sh
```

It starts `aibench-baseline` and `aibench-agentic` with the security plugin off (this is a local benchmark), HTTP
bound to `127.0.0.1` only, on ports `BENCH_BASELINE_PORT` (9321) and `BENCH_AGENTIC_PORT` (9322). If you change the
ports, change `clusters` in the config to match.

### 4. Fetch the real corpus

```
python3 fetch_real_corpus.py
```

It clones `documentation-website` and `opensearch-build` at the commits pinned in `data/real_corpus_manifest.json`
(blobless, sparse), downloads the OpenSearch Dashboards sample data at a pinned commit, fetches the 248 GitHub issues of
the snapshot through the search API, and writes `work/corpus_real.json` (3,918 passages). It ends by checking every
gold passage of the real questions against the snapshot's text hashes and stops if one differs. Documentation and
release-note passages rebuild identically to the snapshot; GitHub issue fields such as state, labels and comment
count can drift; no question has its answer in an issue. Unauthenticated GitHub search allows 10 requests per minute,
so this takes a few minutes; set `GITHUB_TOKEN` to speed it up.

### 5. Build the cluster and register the embedding model

```
python3 build_cluster.py
python3 notes.py
```

`build_cluster.py` applies these cluster settings on both clusters, registers and deploys the embedding model, creates
the pipelines, indices, data streams and aliases, and loads the data (about 9,400 documents; embedding takes a few
minutes on CPU):

```
PUT /_cluster/settings
{"persistent": {
  "plugins.ml_commons.agentic_index_enabled": true,
  "plugins.ml_commons.only_run_on_ml_node": false,
  "plugins.ml_commons.native_memory_threshold": 100,
  "plugins.ml_commons.connector.private_ip_enabled": true,
  "plugins.ml_commons.trusted_connector_endpoints_regex": [
    "^http://<proxy host>:9422/.*$",
    "^http://host\\.docker\\.internal:8000/.*$",
    "^https://runtime\\.sagemaker\\..*\\.amazonaws\\.com/.*$",
    "^https://bedrock-runtime\\..*\\.amazonaws\\.com/.*$"]}}

POST /_plugins/_ml/models/_register?deploy=true
{"name": "huggingface/sentence-transformers/all-MiniLM-L6-v2", "version": "1.0.2", "model_format": "TORCH_SCRIPT"}
```

The endpoint list is built from the configured models: one entry per `openai` URL (scheme, host and port), plus the
SageMaker and Bedrock patterns when such models are configured, plus the proxy. `connector.private_ip_enabled` is
needed because the proxy and a local LLM have private addresses; ml-commons rejects connector URLs that resolve to
private addresses otherwise. `notes.py` writes the schema notes for Baseline + schema notes.

### 6. Start the logging proxy and register the agents

In a second terminal (it runs in the foreground):

```
python3 aiproxy.py
```

The agentic index tools call the cluster's own `_ask` and `_retrieve` through this proxy. The proxy logs every full
response (trace, routing, token counts) to `work/logs/proxy.jsonl`, because the agent only sees a trimmed response.

Then:

```
python3 setup_agents.py
```

It creates, per model, the connectors below, the agents of every setup and the agentic index ConnectorTools, and
writes `work/state_*.json` and `work/prompts.json`. System prompts are passed at execute time.

Connector created for the agent, (a) a generic OpenAI-compatible endpoint (the local model case):

```
POST /_plugins/_ml/connectors/_create
{
  "name": "agent-qwen", "version": 1, "protocol": "http",
  "parameters": {},
  "credential": {"openAI_key": "<value of LOCAL_LLM_API_KEY>"},
  "client_config": {"read_timeout": 120000, "connection_timeout": 10000, "max_retry_times": 3},
  "actions": [{
    "action_type": "predict", "method": "POST",
    "url": "http://host.docker.internal:8000/v1/chat/completions",
    "headers": {"content-type": "application/json", "Authorization": "Bearer ${credential.openAI_key}"},
    "request_body": "{\"model\": \"<served-model-name>\", \"messages\": [{\"role\": \"system\", \"content\": \"${parameters.system_prompt}\"}, ${parameters._chat_history:-}{\"role\": \"user\", \"content\": \"${parameters.prompt}\"}${parameters._interactions:-}]${parameters.tool_configs:-}, \"max_tokens\": 2048, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}"
  }]
}
```

The agent registered on it uses `"_llm_interface": "openai/v1/chat/completions"` and `max_iteration` 31. The
agentic index's own model uses the same connector with this request body, because the agentic index sends a JSON
`messages` array and `max_tokens`:

```
"request_body": "{\"model\": \"<served-model-name>\", \"messages\": ${parameters.messages}, \"max_tokens\": ${parameters.max_tokens}, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}"
```

(b) A SageMaker endpoint that serves vLLM: the same request bodies, with `"protocol": "aws_sigv4"`, AWS credentials,
`"parameters": {"region": "<region>", "service_name": "sagemaker"}` and the URL
`https://runtime.sagemaker.<region>.amazonaws.com/endpoints/<endpoint>/invocations`. The
[SageMaker vLLM chat connector blueprint](../../remote_inference_blueprints/sagemaker_vllm_chat_connector_blueprint.md)
describes this connector for the agentic index step by step. Config entry:

```
"sm_model": {"type": "sagemaker", "endpoint": "<sagemaker-endpoint-name>", "region": "<aws-region>",
             "thinking": false, "temperature": 0}
```

(c) Bedrock Converse: `"protocol": "aws_sigv4"`, `"service_name": "bedrock"`, the URL
`https://bedrock-runtime.<region>.amazonaws.com/model/<model id>/converse`, and the agent interface
`bedrock/converse/claude`. For the agentic index's own model the connector converts the `messages` array to Converse
`system` and `messages` in a pre-process script and maps the Converse response back to `choices[0].message.content`
and `usage` in a post-process script (`common.py`, `CONVERSE_PRE` and `CONVERSE_POST`). Config entry:

```
"judge": {"type": "bedrock", "model_id": "<bedrock-model-or-inference-profile-id>", "region": "<aws-region>",
          "thinking": false, "temperature": null}
```

### 7. Run a smoke subset first

Three questions, every setup, one run, under a separate tag so they do not count in the full results:

```
python3 run.py --questions=synth:5,real:9,route:1 --runs=1 --workers=1 --tag=smoke
python3 judge.py all smoke
python3 report.py --tag=smoke
```

Check that every execution has an answer, that the agentic setups made `agentic_index_*` calls, that
`work/logs/proxy.jsonl` has entries, and that `work/report/tables.md` shows no errors or unjudged executions. Each line
`run.py` prints shows the latency, the number of trace steps and the start of the answer. With one question per
corpus the confidence intervals collapse to a single value, so ignore the `*` marks in the smoke report.

### 8. Throughput runs

```
python3 run.py
```

It runs every question of the three corpora through every setup for each configured run (`--runs=1,2,3` by default
from the config), `workers` threads per agent model. For one question, a thread runs every setup back to back in a
rotating order, so paired executions see the same endpoint load. Results go to `work/results/<corpus>/run<N>/`. If it
stops, run it again: executions already saved are skipped. `--models`, `--setups`, `--corpora` and `--questions`
narrow a run.

### 9. Serial latency run

```
python3 run.py --serial
```

The latency subset (`data/latency_subset.json`: 20 synthetic, 20 real and the 15 route questions), the
`serial_models`, every setup, one request at a time, so latency is not distorted by parallel load. Results go to
`work/results/<corpus>/serial/`. Do not run anything else against the model endpoint at the same time.

### 10. Judge and report

```
python3 judge.py all
python3 report.py
```

`judge.py` normalizes each answer (index and tool names neutralized, citations renumbered), gives every distinct
answer a random id and judges them in shuffled order, so the judge does not know the setup or model. Loop-cap hits and
errors are scored wrong without a judge call. `work/judge/judge_info.json` records the judge; `judge.py` refuses to mix
verdicts from different judges (`--reset` starts over). Optional manual corrections go in
`work/judge/manual_overrides.json`.

`report.py` writes to `work/report/`:
- `tables.md`: per corpus the main table, paired differences and score by question kind; latency tables for the
  serial subset; pooled paired differences.
- `routing.md`: routing hit rates and why first-call routing missed.
- `review_<corpus>.md`: every run-to-run verdict disagreement and a random 10% sample, for a manual spot check of the
  judge.
- `summary.json`: every number in the tables.

### 11. Credentials and cleanup

Temporary AWS credentials expire. Before they do, stop `run.py` (it resumes), then:

```
python3 refresh_credentials.py
```

ml-commons does not apply a connector update while a model is deployed, so this undeploys each model, updates the
connector and deploys it again. Pass `openai` to push a new API key.

To clean up, stop the proxy with Ctrl-C (or let `cleanup.sh` stop it), then:

```
bash cleanup.sh            # removes the containers; --image also removes the image
```

Set `BENCH_WORK` if `work_dir` is not `./work` (cleanup reads the proxy's pid file there) and `BENCH_PREFIX` if you
changed it for `start_clusters.sh`. Results stay in the work directory. Remove `work/src_real/` to delete the source
clones.

## Expected runtime and request volume

Request volume per question, from the earlier run of this benchmark with a 27B model served by vLLM (thinking off),
summed over the four default setups: about 30 LLM calls and 255k tokens (input and output). By setup: Baseline 7.3
calls and 126k tokens, Baseline + schema notes 5.2 and 71k, Agentic Index (all) 9.0 and 29k, Agentic Index (scoped)
9.1 and 29k. Most tokens are prompt tokens that repeat between turns, so prefix caching helps.

For one agent model and the default config:

| Step | Executions | LLM calls | Tokens |
|---|---|---|---|
| Smoke subset | 12 | about 90 | about 0.8M |
| One throughput run (115 questions) | 460 | about 3,500 | about 29M |
| Three throughput runs | 1,380 | about 10,500 | about 88M |
| Serial subset (55 questions) | 220 | about 1,700 | about 14M |
| Judge | up to one call per distinct answer, about 1,600 | about 1,600 | 3-6k each |

Each inner variant adds about a quarter of these numbers.

Runtime depends on the model server. In the earlier run, a vLLM deployment on 8 GPUs and Bedrock together completed
about 49 executions per minute for three agent models with 4 workers each (about 16 per model), so three throughput
runs for one model take about 85 minutes at that rate. Serial executions averaged 5-26 seconds depending on the
setup. Measure your own rate with the smoke subset and the first minutes of the throughput run: the executions in the
table divided by your executions per minute gives the time. Raise `workers` while the server keeps latency stable.

## Reading the results

The paired tables answer "is setup A better than setup B on the same questions". For each question, the harness
averages each setup over its runs, takes the difference A - B, and reports the mean difference over questions with a
95% bootstrap confidence interval (10,000 resamples of the questions). For example, `+0.11 [+0.05, +0.16]*` in the
Score column means A scored 0.11 more per question on average (on a 0-1 scale), and the interval excludes 0, which the
`*` marks as significant. `-0.02 [-0.06, +0.02]` means no measurable difference: the interval includes 0. Token and
tool-call columns read the same way in their own units.

- The interval is over questions, so it reflects how much the result depends on which questions were asked. With few
  questions (route has 15, the smoke subset 3) the intervals are wide.
- Paired differences are more sensitive than comparing the two scores in the main table, because each question is its
  own control.
- Compare runs only when they use the same judge, the same questions and the same cluster build.

## Known pitfalls

These are limits of ml-commons and of the APIs that shaped the harness. Most are already handled; they matter if you
change it.

- **Only the first tool call of a turn runs.** The ml-commons agent runs the first tool call when a model returns
  several in one turn, so every system prompt says "Call one tool at a time".
- **`max_iteration` counts LLM turns and tool results separately.** 31 allows 15 tool calls plus a final answer; the
  harness sets it and scores a cap hit as wrong.
- **The traces API returns at most 100 entries** (`max_results=100`). With 31 iterations an execution stays below it.
- **Exclude embedding fields from `_source`.** Otherwise every search hit and passage carries a 384-dimension vector
  into the prompt. The mappings put `<field>_embedding` in `_source.excludes`.
- **`${parameters` in corpus text breaks requests.** ml-commons rejects any LLM payload that contains `${parameters`
  (`Connector.validatePayload`), and connector examples in the docs contain it. `fetch_real_corpus.py` writes it as
  `$ {parameters`.
- **System prompts over about 32 KB must be passed at execute time.** The agent's `llm` field is a flat object with a
  32 KB keyword limit, and Baseline + schema notes prompts exceed it. Agents are registered with a placeholder prompt
  and `run.py` passes `system_prompt` with each execution.
- **SearchIndexTool needs `return_raw_response`** so that the agent sees `hits.total.value` and a successful zero-hit
  search is explicit. It also does not accept a comma-separated index list (the string is taken as one index name), so
  the Baseline prompt only offers one index, a wildcard pattern or an alias.
- **A connector must be undeployed to update its credentials.** `refresh_credentials.py` undeploys, updates and
  redeploys.
- **Agentic index payload errors.** A few executions of the earlier run (6 of 4,830) failed with "Some parameter
  placeholder not filled in payload" or an invalid payload after the tool returned long documentation passages. They
  are scored wrong and appear in the Errors column.
- **PassageStore keeps at most 20 metadata fields per passage.** The Nimbus documents drop four noise fields so that
  plan, version and status stay within the cap.
- **Thinking.** With thinking on and no reasoning parser, reasoning text lands in the message content and can use up
  `max_tokens` before the answer or the tool call. Use the server's reasoning parser, or keep thinking off.

## Files

| File | Purpose |
|---|---|
| `config.example.json` | Config template with placeholders. |
| `common.py` | Config loading, REST helpers, connectors for the three model types, the judge's chat call, the agentic index ConnectorTools. |
| `fetch_real_corpus.py` | Fetches the pinned public sources and builds `work/corpus_real.json`. |
| `layout.py`, `data_synth.py` | The 162-index cluster layout and the synthetic corpus fields. |
| `build_cluster.py`, `notes.py` | Builds the cluster; writes the schema notes. |
| `prompts.py`, `setup_agents.py` | System prompts and tool descriptions; registers models, connectors and agents. |
| `aiproxy.py` | Logging proxy for the agentic index calls. |
| `run.py` | Throughput and serial runs. |
| `judge.py`, `analyze.py`, `report.py` | Blind judging, per-execution features, tables and statistics. |
| `refresh_credentials.py` | Pushes fresh credentials into the connectors. |
| `start_clusters.sh`, `cleanup.sh`, `docker/Dockerfile` | Clusters and image. |
| `data/corpus_synth.json` | Synthetic corpus (399 passages, all values synthetic). |
| `data/questions_synth.json`, `questions_real.json`, `questions_route.json` | Question sets with gold answers, grading notes and gold passage ids. |
| `data/latency_subset.json` | Question numbers of the serial subset. |
| `data/real_corpus_manifest.json` | Pinned commits, the issue list and gold passage hashes of the real corpus. |
