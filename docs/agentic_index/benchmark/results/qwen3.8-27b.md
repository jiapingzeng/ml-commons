# Agentic Index benchmark: Qwen3.8 27B

Results of the full benchmark in [the benchmark guide](../README.md) with Qwen3.8 27B as both the agent model and the
agentic index's own LLM, run on 2026-11-25. Three throughput runs plus the serial latency subset: 1,600 executions,
no errors, every answer judged.

- [Summary](#summary)
- [Setup](#setup)
- [Results](#results)
- [Observations](#observations)
- [Caveats](#caveats)
- [Reproducing](#reproducing)
- [Appendix: generated tables](#appendix-generated-tables)

## Summary

| Setup | Score (115 questions, 0-1) | Tokens / execution | LLM calls / execution | Tool calls / execution | Cap hits |
|---|---|---|---|---|---|
| Baseline | 0.67 | 163k | 8.9 | 7.6 | 103 of 345 (30%) |
| Baseline + schema notes | 0.74 | 114k | 7.1 | 5.8 | 88 of 345 (26%) |
| Agentic Index (all) | **0.86** | 87k | 20.9 | 3.6 | 39 of 345 (11%) |
| Agentic Index (scoped) | **0.85** | 80k | 18.6 | 3.2 | 37 of 345 (11%) |

Scores are means over the three runs. LLM calls count the agent's calls plus the agentic index's internal calls.

- Agentic Index (all) scores +0.18 per question over Baseline and +0.12 over Baseline + schema notes. Both differences
  are significant, and Agentic Index uses fewer tokens and tool calls than either.
- Agentic Index (scoped) ties Agentic Index (all) overall: it is slightly better on the synthetic corpus and worse on
  the route questions whose answer is outside its scope, as designed.
- Typical latency is the same for every setup (p50 4-5 s), but Agentic Index has a long tail (p90 86-112 s against
  17-24 s), so the mean latency of Agentic Index (all) is 11-14 s higher.
- Qwen3.8 rarely gives up: 19% of all executions hit the 15-tool-call cap, which is scored wrong. The Baselines hit it
  two to three times as often as Agentic Index, which explains a large part of the score difference.

## Setup

| Item | Value |
|---|---|
| Code | Branch `agentic-index-poc` at `2e22aa21a`; plugin zip built from it on the `opensearchstaging/opensearch:3.10.0` image. |
| Agent and agentic index model | Qwen3.8 27B (`qwen3.8-27b`) on one local GPU behind an OpenAI-compatible chat-completions endpoint with native tool calling; at most 2 concurrent requests. Thinking off, temperature 0, `max_tokens` 2048. |
| Judge | Claude Opus 5.5 on Amazon Bedrock (`us.anthropic.claude-opus-5-5`). |
| Setups | `baseline`, `baseline_notes`, `ai_all`, `ai_scoped` (the defaults). |
| Cluster | One single-node cluster with a 4 GB heap for both the Baselines and Agentic Index; see [Caveats](#caveats). |
| Throughput runs | 3 runs of 115 questions x 4 setups (1,380 executions), 2 concurrent executions throughout: run 1 with 2 workers, runs 2 and 3 together with 1 worker each. |
| Serial subset | 55 questions x 4 setups (220 executions), one request at a time. |
| Judging | 1,104 distinct answers (all runs, the serial subset and the smoke subset) judged blind; cap hits scored wrong by rule. No unjudged or errored executions. |
| Wall time | Run 1: 1 h 36 min. Runs 2 and 3: 3 h 01 min. Serial subset: 1 h 03 min (runs 2 and 3 were judged at the same time). |

## Results

### Score by corpus

Score per corpus, mean of the three runs (individual runs in parentheses).

| Setup | Synthetic (40) | Real (60) | Route (15) |
|---|---|---|---|
| Baseline | 29.7 (27.5, 30.5, 31) | 36.5 (35, 37.5, 37) | 11.3 (11.5, 10.5, 12) |
| Baseline + schema notes | 30.5 (29, 31.5, 31) | 40.5 (41.5, 40.5, 39.5) | 13.7 (14, 14, 13) |
| Agentic Index (all) | 34.7 (35, 36, 33) | 49.5 (47.5, 53, 48) | 14.2 (14.5, 13.5, 14.5) |
| Agentic Index (scoped) | 37.3 (36.5, 37.5, 38) | 49.8 (49.5, 50.5, 49.5) | 11.0 (11, 11, 11) |

### Paired differences

Mean difference per question over all 115 questions with a 95% bootstrap confidence interval over questions; `*`
marks an interval that excludes 0. See [Reading the results](../README.md#reading-the-results).

| Comparison | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |
|---|---|---|---|---|---|
| Agentic Index (all) - Baseline | +0.18 [+0.11, +0.25]* | +0.05 [+0.01, +0.10]* | -76,548 [-110,291, -46,172]* | -4.01 [-4.76, -3.28]* | -0.42 [-0.53, -0.32]* |
| Agentic Index (all) - Baseline + schema notes | +0.12 [+0.05, +0.19]* | +0.04 [-0.00, +0.09] | -27,759 [-54,611, -629]* | -2.27 [-3.22, -1.29]* | -0.68 [-0.97, -0.43]* |
| Agentic Index (scoped) - Agentic Index (all) | -0.00 [-0.05, +0.05] | -0.03 [-0.07, +0.01] | -7,068 [-25,663, +12,560] | -0.36 [-0.95, +0.24] | +0.00 [+0.00, +0.00] |

Score differences per corpus:

| Comparison | Synthetic | Real | Route |
|---|---|---|---|
| Agentic Index (all) - Baseline | +0.12 [+0.02, +0.23]* | +0.22 [+0.12, +0.31]* | +0.19 [+0.02, +0.39]* |
| Agentic Index (all) - Baseline + schema notes | +0.10 [-0.01, +0.22] | +0.15 [+0.04, +0.26]* | +0.03 [-0.08, +0.16] |
| Agentic Index (scoped) - Agentic Index (all) | +0.07 [-0.01, +0.15] | +0.01 [-0.04, +0.05] | -0.21 [-0.43, -0.01]* |

### Latency

Serial subset (55 questions, one request at a time), seconds per execution.

| Setup | p50 | p90 | Mean | Score on subset |
|---|---|---|---|---|
| Baseline | 4.6 | 24.2 | 9.8 | 37 / 55 |
| Baseline + schema notes | 4.2 | 17.1 | 7.6 | 39.5 / 55 |
| Agentic Index (all) | 4.7 | 86.2 | 21.1 | 46.5 / 55 |
| Agentic Index (scoped) | 5.1 | 111.8 | 29.9 | 43.5 / 55 |

Paired latency differences: Agentic Index (all) - Baseline +11.4 s [+4.1, +19.6]*, Agentic Index (all) - Baseline +
schema notes +13.6 s [+5.6, +22.6]*, Agentic Index (scoped) - Agentic Index (all) +8.7 s [-1.0, +19.7].

The agentic index API itself was fast and reliable over all runs (logging proxy, 2,759 calls):

| Call | Calls | p50 | p90 | Max | Timed out | Non-200 |
|---|---|---|---|---|---|---|
| `_retrieve` | 1,838 | 7.8 s | 12.6 s | 50 s | 0 | 0 |
| `_ask` | 921 | 6.5 s | 13.3 s | 39 s | 0 | 0 |

### Cap hits

Executions that reached 15 tool calls without a final answer, over the three throughput runs (345 per setup), by
what the trace shows:

| Setup | Cap hits | Question has no answer in the data | Gold document returned before the cap | Gold document never returned |
|---|---|---|---|---|
| Baseline | 103 | 30 | 42 | 31 |
| Baseline + schema notes | 88 | 30 | 35 | 23 |
| Agentic Index (all) | 39 | 17 | 12 | 10 |
| Agentic Index (scoped) | 37 | 13 | 11 | 13 |

"Gold document returned" means a gold or acceptable document id appeared in a tool response of the execution.

### Routing

| Setup | Synthetic | Real | Route |
|---|---|---|---|
| Baseline (gold index searched) | 1.00 | 0.82 | 1.00 |
| Baseline + schema notes (gold index searched) | 0.99 | 0.85 | 1.00 |
| Agentic Index (all), any call / first call | 0.98 / 0.72 | 0.82 / 0.68 | 0.98 / 0.98 |
| Agentic Index (scoped), any call / first call | 0.91 / 0.87 | 0.87 / 0.82 | 0.71 / 0.71 |

For Agentic Index (scoped) on the route questions, 12 of 42 executions had their gold index outside the scope.

## Observations

- **Accuracy.** Agentic Index leads on every corpus. The gap is largest on the real OpenSearch corpus (+0.22 over
  Baseline, +0.15 over schema notes), where Baseline touches 19 indices per execution on average (a wildcard search
  counts every index) and often reaches the cap. On the route questions schema notes are as good as Agentic Index (all): the notes show a gold index in 71%
  of route executions, and the index names there identify the month, release or tenant.
- **Questions with no answer.** Both Baselines score 0.00 on the `no_answer` questions in every corpus: they keep
  searching until the cap instead of saying the information is unavailable. Agentic Index answers some of them
  correctly (0.33-1.00 depending on corpus and setup), and its cap hits on such questions are about half as frequent.
- **Persistence.** About 40% of the Baseline cap hits and 30% of the Agentic Index cap hits had already returned a
  gold document. Qwen3.8 keeps
  refining its query rather than answering from the evidence it has, in every setup.
- **Cost.** Agentic Index sends 47-51% fewer tokens than Baseline and 24-30% fewer than schema notes, but makes more
  than twice as many LLM calls, because each agentic index call runs about 4.5 LLM calls of its own (routing,
  planning, sufficiency, answer). On a local GPU, call count matters for latency as much as token count.
- **Latency.** When the agent stops after one or two agentic calls, latency matches the Baselines. When it loops, each
  call costs about 8 s against about 1 s for a Baseline search, which produces the long p90.
- **Scoped.** Restricting the scope reduces tokens and tool calls on the synthetic corpus and does not change the real
  corpus. On the route questions it misses the tenant-ticket answers outside its scope (0.00 on `route_tenant`) and
  then searches longer, which increases its tokens there.

## Caveats

- **One cluster.** The guide recommends separate clusters for the Baselines and Agentic Index. This run used one
  single-node cluster with a 4 GB heap for both because the benchmark host did not have memory for two. The setups
  therefore shared one node's caches and CPU; with 2 concurrent executions the load was light.
- **The tool-call cap.** Cap hits are scored wrong, and they make up 19% of all executions. Scores here mix answer
  quality with how often the model stops in time, and that depends on the model: compare setups within this run, not
  absolute numbers with runs of other models.
- **Request volume.** The guide's request-volume figures come from an earlier run with another model (Agentic Index
  (all): 9.0 LLM calls and 29k tokens per question). Qwen3.8 made about 2.3 times as many LLM calls with Agentic
  Index (all), so plan for more traffic than the guide estimates.
- **Thinking off.** Qwen3.8 ran without thinking. A run with thinking on, or with a system prompt that tells the agent
  to answer once searches stop adding evidence, may change the cap-hit rates and the gaps between setups.
- **Judge.** Scores are comparable only with runs judged by the same judge (Claude Opus 5.5).

## Reproducing

Follow the [step-by-step guide](../README.md#step-by-step) with one agent model. The model entries of `config.json`
for this run, with placeholders for the endpoint and region:

```json
"models": {
 "qwen38": {
  "type": "openai",
  "label": "Qwen3.8 27B",
  "url": "http://host.docker.internal:<port>/v1/chat/completions",
  "client_url": "http://127.0.0.1:<port>/v1/chat/completions",
  "model": "qwen3.8-27b",
  "thinking": false,
  "temperature": 0,
  "max_tokens": 2048
 },
 "judge": {
  "type": "bedrock",
  "label": "Claude Opus 5.5 (judge)",
  "model_id": "us.anthropic.claude-opus-5-5",
  "region": "<aws-region>",
  "temperature": null
 }
},
"agents": ["qwen38"],
"judge": "judge",
"workers": 2,
"serial_models": ["qwen38"]
```

For one cluster, set `clusters.baseline` and `clusters.agentic` to the same URL; `build_cluster.py` then builds it
once. The runs:

```
python3 run.py --questions=synth:5,real:9,route:1 --runs=1 --workers=1 --tag=smoke
python3 run.py --runs=1
python3 run.py --runs=2,3 --workers=1
python3 run.py --serial
python3 judge.py all
python3 report.py
```

Set `workers` to the number of concurrent requests the model server handles; more workers only queue requests.

## Appendix: generated tables

The tables below are `work/report/tables.md` from `report.py` for this run, unchanged apart from heading levels.

### Synthetic corpus (Nimbus), 40 questions

| Model | Setup | Score / 40 (runs) | Routing hit | Gold index retrieved | Indices touched | Tool calls | Failed calls / exec | Zero-hit searches / exec | LLM calls (agent + index) | Tokens in+out | Cap hits | Errors | Unjudged | Cite precision | Gold recall | Unofficial cited / exec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 29.7 (27.5, 30.5, 31) | 1.00 | 1.00 | 2.6 | 7.35 | 0.49 | 0.63 | 8.57 (8.57 + 0.00) | 121,523 | 26 | 0 | 0 | 0.69 | 0.81 | 0.00 |
| Qwen3.8 27B | Baseline + schema notes | 30.5 (29, 31.5, 31) | 0.99 | 0.98 | 3.2 | 6.53 | 1.24 | 1.51 | 7.76 (7.76 + 0.00) | 107,449 | 27 | 0 | 0 | 0.73 | 0.78 | 0.05 |
| Qwen3.8 27B | Agentic Index (all) | 34.7 (35, 36, 33) | 0.98 | 1.00 | 3.7 | 3.53 | 0.00 | 0.00 | 20.64 (4.65 + 15.99) | 78,543 | 14 | 0 | 0 | 0.81 | 0.83 | 0.01 |
| Qwen3.8 27B | Agentic Index (scoped) | 37.3 (36.5, 37.5, 38) | 0.91 | 0.99 | 2.8 | 2.34 | 0.00 | 0.00 | 13.77 (3.39 + 10.38) | 48,706 | 6 | 0 | 0 | 0.80 | 0.88 | 0.00 |

Paired differences per question (synth), mean over the runs per question, 95% bootstrap CI over questions; * = CI excludes 0.

| Model | Comparison | Questions | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | 40 | +0.12 [+0.02, +0.23]* | +0.00 [+0.00, +0.00] | -42,980 [-71,543, -18,510]* | -3.82 [-4.87, -2.78]* | -0.49 [-0.67, -0.33]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | 40 | +0.10 [-0.01, +0.22] | +0.02 [+0.00, +0.05] | -28,906 [-61,212, +4,391] | -3.00 [-4.33, -1.67]* | -1.24 [-1.68, -0.84]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | 40 | +0.07 [-0.01, +0.15] | -0.01 [-0.03, +0.00] | -29,837 [-52,692, -10,516]* | -1.19 [-2.12, -0.40]* | +0.00 [+0.00, +0.00] |

Score by question kind (synth), mean per execution:

| Model | Setup | cross_source (8) | distractor (6) | filter (7) | lookup (8) | no_answer (4) | no_answer->lookup (1) | numeric_date (6) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 0.79 | 1.00 | 0.67 | 1.00 | 0.00 | 0.00 | 0.78 |
| Qwen3.8 27B | Baseline + schema notes | 0.77 | 0.92 | 0.76 | 0.96 | 0.00 | 1.00 | 0.81 |
| Qwen3.8 27B | Agentic Index (all) | 0.98 | 0.97 | 0.81 | 1.00 | 0.33 | 0.33 | 0.94 |
| Qwen3.8 27B | Agentic Index (scoped) | 1.00 | 1.00 | 0.93 | 0.96 | 0.58 | 1.00 | 0.97 |

Latency, serial subset (synth: 20 questions, one request at a time)

| Model | Setup | p50 s | p90 s | mean s | Score on subset |
|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 6.5 | 24.4 | 9.6 | 15 / 20 |
| Qwen3.8 27B | Baseline + schema notes | 6.2 | 13.5 | 7.3 | 15 / 20 |
| Qwen3.8 27B | Agentic Index (all) | 7.1 | 93.3 | 23.4 | 17 / 20 |
| Qwen3.8 27B | Agentic Index (scoped) | 6.7 | 103.2 | 23.2 | 16.5 / 20 |

Paired latency differences, seconds per question.

| Model | Comparison | Latency s |
|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | +13.83 [+2.21, +28.18]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | +16.15 [+3.40, +32.14]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | -0.25 [-10.33, +12.40] |

### Real corpus (OpenSearch docs, release notes, GitHub issues), 60 questions

| Model | Setup | Score / 60 (runs) | Routing hit | Gold index retrieved | Indices touched | Tool calls | Failed calls / exec | Zero-hit searches / exec | LLM calls (agent + index) | Tokens in+out | Cap hits | Errors | Unjudged | Cite precision | Gold recall | Unofficial cited / exec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 36.5 (35, 37.5, 37) | 0.82 | 0.81 | 19.4 | 8.18 | 0.42 | 1.12 | 9.54 (9.54 + 0.00) | 209,114 | 67 | 0 | 0 | 0.74 | 0.60 | 0.00 |
| Qwen3.8 27B | Baseline + schema notes | 40.5 (41.5, 40.5, 39.5) | 0.85 | 0.84 | 3.8 | 5.99 | 0.44 | 0.83 | 7.31 (7.31 + 0.00) | 136,709 | 57 | 0 | 0 | 0.76 | 0.66 | 0.00 |
| Qwen3.8 27B | Agentic Index (all) | 49.5 (47.5, 53, 48) | 0.82 | 0.90 | 2.6 | 3.99 | 0.00 | 0.00 | 23.75 (5.12 + 18.63) | 105,187 | 24 | 0 | 0 | 0.77 | 0.78 | 0.01 |
| Qwen3.8 27B | Agentic Index (scoped) | 49.8 (49.5, 50.5, 49.5) | 0.87 | 0.93 | 2.4 | 3.51 | 0.00 | 0.00 | 20.19 (4.63 + 15.56) | 90,488 | 23 | 0 | 0 | 0.76 | 0.78 | 0.00 |

Paired differences per question (real), mean over the runs per question, 95% bootstrap CI over questions; * = CI excludes 0.

| Model | Comparison | Questions | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | 60 | +0.22 [+0.12, +0.31]* | +0.10 [+0.02, +0.18]* | -103,927 [-161,257, -51,260]* | -4.19 [-5.31, -3.10]* | -0.42 [-0.58, -0.27]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | 60 | +0.15 [+0.04, +0.26]* | +0.07 [-0.02, +0.16] | -31,523 [-78,790, +14,093] | -2.01 [-3.57, -0.46]* | -0.44 [-0.89, -0.09]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | 60 | +0.01 [-0.04, +0.05] | +0.03 [+0.00, +0.06] | -14,699 [-36,795, +7,382] | -0.48 [-1.12, +0.17] | +0.00 [+0.00, +0.00] |

Score by question kind (real), mean per execution:

| Model | Setup | cross_source (12) | distractor (4) | filter (12) | lookup (19) | no_answer (5) | numeric (8) |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 0.44 | 0.46 | 0.94 | 0.64 | 0.00 | 0.73 |
| Qwen3.8 27B | Baseline + schema notes | 0.64 | 0.54 | 0.81 | 0.84 | 0.00 | 0.62 |
| Qwen3.8 27B | Agentic Index (all) | 0.85 | 0.46 | 0.92 | 0.90 | 0.40 | 0.92 |
| Qwen3.8 27B | Agentic Index (scoped) | 0.75 | 0.62 | 0.93 | 0.91 | 0.47 | 0.94 |

Latency, serial subset (real: 20 questions, one request at a time)

| Model | Setup | p50 s | p90 s | mean s | Score on subset |
|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 7.7 | 27.3 | 13.1 | 10.5 / 20 |
| Qwen3.8 27B | Baseline + schema notes | 3.7 | 25.0 | 10.4 | 11.5 / 20 |
| Qwen3.8 27B | Agentic Index (all) | 6.9 | 101.1 | 28.1 | 16 / 20 |
| Qwen3.8 27B | Agentic Index (scoped) | 6.6 | 113.2 | 30.9 | 16 / 20 |

Paired latency differences, seconds per question.

| Model | Comparison | Latency s |
|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | +14.96 [+0.96, +31.64]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | +17.70 [+2.35, +36.43]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | +2.81 [-7.44, +14.85] |

### Route questions (the right index among similar ones), 15 questions

| Model | Setup | Score / 15 (runs) | Routing hit | Gold index retrieved | Indices touched | Tool calls | Failed calls / exec | Zero-hit searches / exec | LLM calls (agent + index) | Tokens in+out | Cap hits | Errors | Unjudged | Cite precision | Gold recall | Unofficial cited / exec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 11.3 (11.5, 10.5, 12) | 1.00 | 1.00 | 1.8 | 5.76 | 0.27 | 0.40 | 6.98 (6.98 + 0.00) | 90,646 | 10 | 0 | 0 | 0.97 | 0.83 | 0.00 |
| Qwen3.8 27B | Baseline + schema notes | 13.7 (14, 14, 13) | 1.00 | 1.00 | 1.3 | 3.27 | 0.16 | 0.49 | 4.36 (4.36 + 0.00) | 43,745 | 4 | 0 | 0 | 0.98 | 0.98 | 0.00 |
| Qwen3.8 27B | Agentic Index (all) | 14.2 (14.5, 13.5, 14.5) | 0.98 | 1.00 | 1.8 | 1.91 | 0.00 | 0.00 | 10.29 (2.93 + 7.36) | 34,101 | 1 | 0 | 0 | 0.92 | 1.00 | 0.00 |
| Qwen3.8 27B | Agentic Index (scoped) | 11.0 (11, 11, 11) | 0.71 | 0.71 | 3.3 | 4.27 | 0.00 | 0.00 | 24.84 (5.44 + 19.40) | 118,272 | 8 | 0 | 0 | 0.87 | 0.71 | 0.24 |

Paired differences per question (route), mean over the runs per question, 95% bootstrap CI over questions; * = CI excludes 0.

| Model | Comparison | Questions | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | 15 | +0.19 [+0.02, +0.39]* | +0.00 [+0.00, +0.00] | -56,545 [-126,904, +852] | -3.84 [-6.09, -2.02]* | -0.27 [-0.56, -0.04]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | 15 | +0.03 [-0.08, +0.16] | +0.00 [+0.00, +0.00] | -9,645 [-29,839, +9,073] | -1.36 [-2.89, -0.11]* | -0.16 [-0.36, +0.00] |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | 15 | -0.21 [-0.43, -0.01]* | -0.29 [-0.50, -0.07]* | +84,171 [+531, +181,749]* | +2.36 [-0.07, +5.13] | +0.00 [+0.00, +0.00] |

Score by question kind (route), mean per execution:

| Model | Setup | no_answer (1) | route_month (3) | route_pricing (2) | route_product (2) | route_tenant (3) | route_version (4) |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 0.00 | 1.00 | 1.00 | 0.83 | 0.22 | 1.00 |
| Qwen3.8 27B | Baseline + schema notes | 0.00 | 1.00 | 1.00 | 1.00 | 0.89 | 1.00 |
| Qwen3.8 27B | Agentic Index (all) | 0.67 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| Qwen3.8 27B | Agentic Index (scoped) | 1.00 | 1.00 | 1.00 | 0.50 | 0.00 | 1.00 |

Latency, serial subset (route: 15 questions, one request at a time)

| Model | Setup | p50 s | p90 s | mean s | Score on subset |
|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 3.1 | 13.8 | 5.6 | 11.5 / 15 |
| Qwen3.8 27B | Baseline + schema notes | 2.1 | 9.5 | 4.2 | 13 / 15 |
| Qwen3.8 27B | Agentic Index (all) | 3.7 | 6.8 | 8.8 | 13.5 / 15 |
| Qwen3.8 27B | Agentic Index (scoped) | 3.7 | 136.7 | 37.4 | 11 / 15 |

Paired latency differences, seconds per question.

| Model | Comparison | Latency s |
|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | +3.25 [-2.99, +12.97] |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | +4.65 [-1.32, +14.13] |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | +28.55 [+2.08, +58.69]* |

Paired differences pooled over all 115 questions, 95% bootstrap CI over questions.

| Model | Comparison | Questions | Score | Gold index retrieved | Tokens | Tool calls | Failed calls |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | 115 | +0.18 [+0.11, +0.25]* | +0.05 [+0.01, +0.10]* | -76,548 [-110,291, -46,172]* | -4.01 [-4.76, -3.28]* | -0.42 [-0.53, -0.32]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | 115 | +0.12 [+0.05, +0.19]* | +0.04 [-0.00, +0.09] | -27,759 [-54,611, -629]* | -2.27 [-3.22, -1.29]* | -0.68 [-0.97, -0.43]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | 115 | -0.00 [-0.05, +0.05] | -0.03 [-0.07, +0.01] | -7,068 [-25,663, +12,560] | -0.36 [-0.95, +0.24] | +0.00 [+0.00, +0.00] |

Latency pooled over the serial subset (55 questions)

| Model | Setup | p50 s | p90 s | mean s | Score on subset |
|---|---|---|---|---|---|
| Qwen3.8 27B | Baseline | 4.6 | 24.2 | 9.8 | 37 / 55 |
| Qwen3.8 27B | Baseline + schema notes | 4.2 | 17.1 | 7.6 | 39.5 / 55 |
| Qwen3.8 27B | Agentic Index (all) | 4.7 | 86.2 | 21.1 | 46.5 / 55 |
| Qwen3.8 27B | Agentic Index (scoped) | 5.1 | 111.8 | 29.9 | 43.5 / 55 |

Paired latency differences, seconds per question.

| Model | Comparison | Latency s |
|---|---|---|
| Qwen3.8 27B | Agentic Index (all) - Baseline | +11.36 [+4.08, +19.56]* |
| Qwen3.8 27B | Agentic Index (all) - Baseline + schema notes | +13.58 [+5.64, +22.58]* |
| Qwen3.8 27B | Agentic Index (scoped) - Agentic Index (all) | +8.72 [-0.95, +19.68] |

