# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pushes fresh credentials into every chat model connector (agent models on both clusters and the agentic index's
own models): AWS credentials from the credential chain for sagemaker and bedrock models, the API key for openai models.
ml-commons does not apply a connector update while a model uses it, so each model is undeployed, updated and
redeployed: run it only while no execution is in flight.
Usage: python3 refresh_credentials.py [openai|sagemaker|bedrock ...]   (default: sagemaker and bedrock)"""

import sys

import common as C


def main():
    cfg = C.Config()
    types = set(sys.argv[1:]) or {"sagemaker", "bedrock"}
    n = 0
    for role in ("baseline", "agentic"):
        st = C.load_json(cfg.wpath(f"state_{role}.json"), {})
        base = st.get("url")
        for group in ("models", "index_llm"):
            for key, m in (st.get(group) or {}).items():
                if m["type"] not in types:
                    continue
                C.call(base, "POST", f"/_plugins/_ml/models/{m['model_id']}/_undeploy")
                C.call(base, "PUT", f"/_plugins/_ml/connectors/{m['connector_id']}",
                       {"credential": C.connector_credential(cfg.models[key])})
                C.wait_task(base, C.call(base, "POST", f"/_plugins/_ml/models/{m['model_id']}/_deploy")["task_id"])
                n += 1
    print("refreshed", n, "connectors")


if __name__ == "__main__":
    main()
