# SageMaker vLLM chat model connector blueprint for agentic index

This blueprint connects ml-commons to a SageMaker endpoint that serves a chat model with vLLM and the OpenAI
chat-completions API (a vLLM-served chat model). The experimental agentic index `_ask` and `_retrieve` APIs use it
as their LLM. The connector returns the raw chat completion, so the agentic index can read
`choices[0].message.content` and the token `usage` of every call.

## Request format

The agentic index sends two parameters:

- `messages`: a JSON array of chat messages (`[{"role":"system","content":"..."},{"role":"user","content":"..."}]`).
  Because the value is valid JSON, ml-commons inserts it into the request body unescaped.
- `max_tokens`: the completion budget for the step.

Qwen3-family templates think by default. If the endpoint runs without a vLLM reasoning parser, the reasoning text
comes back inside `message.content` and can use up `max_tokens` before the answer. Send
`"chat_template_kwargs": {"enable_thinking": false}` as below. The agentic index also strips any `<think>...</think>`
block from the content.

## 1. Create the connector

Replace `my-vllm-endpoint` with your endpoint name. Use temporary credentials with permission for `sagemaker:InvokeEndpoint` on the endpoint only, and rotate them with
`PUT /_plugins/_ml/connectors/<connector_id>` before they expire. Do not store long-lived keys in connectors.

```json
POST /_plugins/_ml/connectors/_create
{
  "name": "vllm-chat",
  "description": "vLLM-served chat model on SageMaker (chat completions)",
  "version": 1,
  "protocol": "aws_sigv4",
  "credential": {
    "access_key": "<AWS ACCESS KEY>",
    "secret_key": "<AWS SECRET KEY>",
    "session_token": "<AWS SESSION TOKEN>"
  },
  "parameters": {
    "region": "us-west-2",
    "service_name": "sagemaker"
  },
  "client_config": {
    "read_timeout": 120000,
    "connection_timeout": 10000,
    "max_retry_times": 2
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-vllm-endpoint/invocations",
      "headers": {
        "content-type": "application/json"
      },
      "request_body": "{\"messages\": ${parameters.messages}, \"max_tokens\": ${parameters.max_tokens}, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}"
    }
  ]
}
```

## 2. Register and deploy the model

```json
POST /_plugins/_ml/models/_register?deploy=true
{
  "name": "vllm-chat",
  "function_name": "remote",
  "connector_id": "<connector_id>"
}
```

## 3. Test the model

```json
POST /_plugins/_ml/models/<model_id>/_predict
{
  "parameters": {
    "messages": "[{\"role\":\"user\",\"content\":\"Reply with JSON {\\\"ok\\\":true} only.\"}]",
    "max_tokens": "50"
  }
}
```

Sample response (truncated):

```json
{
  "inference_results": [
    {
      "output": [
        {
          "name": "response",
          "dataAsMap": {
            "object": "chat.completion",
            "choices": [
              { "index": 0, "message": { "role": "assistant", "content": "{\"ok\":true}" }, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 22, "completion_tokens": 6, "total_tokens": 28 }
          }
        }
      ],
      "status_code": 200
    }
  ]
}
```

## 4. Ask a question over an index

Enable the experimental API, then call `_ask` (cited answer) or `_retrieve` (cited passages only).

```json
PUT /_cluster/settings
{ "persistent": { "plugins.ml_commons.agentic_index_enabled": true } }

POST /_plugins/_ml/agentic_index/support-docs/_ask
{
  "question": "Which plans include SSO, and what changed in the last pricing update?",
  "llm_model_id": "<model_id>",
  "embedding_model_id": "<text embedding model_id>",
  "embedding_field": "embedding"
}
```

Optional request fields: `text_fields`, `max_rounds`, `max_subqueries`, `k`, `use_index_insight`, and `timeout`
(for example `"30s"`). The request stops at its deadline and returns the passages found so far with
`"timed_out": true` (ask also returns `"answer": null`). The deadline defaults to, and is capped by, the cluster setting
`plugins.ml_commons.agentic_index_timeout` (default `2m`). The target must resolve to one index; wildcards and aliases
that point to several indices are rejected.

## Permissions

The APIs run every step as the calling user. With the security plugin enabled, the caller needs:

- Cluster permissions `cluster:admin/opensearch/ml/agentic_index/query` and `cluster:admin/opensearch/ml/predict`.
- Index permissions on the target index: `indices:admin/mappings/get` and `indices:data/read/search`.
- Access to the model group of the LLM and of the embedding model, when model access control is enabled.
- With `use_index_insight`, `cluster:admin/opensearch/ml/index_insight/get`. If index insight is unavailable, the
  request continues without it and reports a warning in the trace.

## Using the same endpoint with QueryPlanningTool

`QueryPlanningTool` sends `system_prompt` and `user_prompt` instead of `messages`. Create a second connector on the
same endpoint with this request body, and set `"response_filter": "$.choices[0].message.content"` in the tool
parameters:

```
"request_body": "{\"messages\": [{\"role\": \"system\", \"content\": \"${parameters.system_prompt}\"}, {\"role\": \"user\", \"content\": \"${parameters.user_prompt}\"}], \"max_tokens\": 2048, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}"
```
