# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass-through logging proxy between the agentic index ConnectorTools and the agentic cluster's own agentic index API.

Connector URLs are http://<proxy host>:<port>/<tag>/_plugins/...; the proxy strips the tag (setup-model, for example
ai_all-qwen), forwards the request unchanged to the agentic cluster and logs the full response (trace, routing, token
counts) with the tag to <work>/logs/proxy.jsonl. The agent only sees the response after the connector's
post_process_function trims it, so this log is the only record of the agentic index's own LLM calls.
It binds to the proxy host only (by default the docker bridge gateway, which the containers reach), never to 0.0.0.0.
Usage: python3 aiproxy.py   (runs in the foreground; stop it with Ctrl-C or cleanup.sh)"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import common as C

LOCK = threading.Lock()


def handler(cluster, log):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            parts = self.path.split("/", 2)
            if len(parts) < 3 or not parts[2].startswith("_plugins/_ml/agentic_index/"):
                self.send_error(404)
                return
            tag, rest = parts[1], parts[2]
            t0 = time.time()
            try:
                r = requests.post(f"{cluster}/{rest}", data=body, headers={"Content-Type": "application/json"},
                                  timeout=600)
                status, content = r.status_code, r.content
            except Exception as e:  # report upstream failures to the tool instead of hanging
                status, content = 502, json.dumps({"error": repr(e)[:500]}).encode()
            t1 = time.time()
            try:
                resp = json.loads(content)
            except ValueError:
                resp = {"_raw": content[:5000].decode(errors="replace")}
            with LOCK, open(log, "a") as f:
                f.write(json.dumps({"t0": t0, "t1": t1, "tag": tag, "path": "/" + rest, "status": status,
                                    "request": json.loads(body or b"{}"), "response": resp}) + "\n")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *a):
            pass

    return H


def main():
    cfg = C.Config()
    os.makedirs(cfg.wpath("logs"), exist_ok=True)
    host, port, cluster = cfg.proxy_host(), cfg.proxy_port(), cfg.clusters["agentic"]
    s = ThreadingHTTPServer((host, port), handler(cluster, cfg.wpath("logs", "proxy.jsonl")))
    open(cfg.wpath("proxy.pid"), "w").write(str(os.getpid()))
    print(f"proxy {host}:{port} -> {cluster}", flush=True)
    s.serve_forever()


if __name__ == "__main__":
    main()
