#!/usr/bin/env bash
#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Starts the baseline and agentic clusters from the benchmark image: single node, security plugin off, HTTP bound to
# 127.0.0.1 only. host.docker.internal resolves to the docker host inside the containers, for a local LLM server.
# Environment: BENCH_IMAGE (default opensearch-ml:agentic-index-bench), BENCH_BASELINE_PORT (9321),
# BENCH_AGENTIC_PORT (9322), BENCH_HEAP (6g), BENCH_PREFIX (aibench).
set -euo pipefail
image="${BENCH_IMAGE:-opensearch-ml:agentic-index-bench}"
prefix="${BENCH_PREFIX:-aibench}"
heap="${BENCH_HEAP:-6g}"
for spec in "baseline:${BENCH_BASELINE_PORT:-9321}" "agentic:${BENCH_AGENTIC_PORT:-9322}"; do
  name="$prefix-${spec%%:*}"
  port="${spec##*:}"
  docker rm -f "$name" > /dev/null 2>&1 || true
  docker run -d --name "$name" -p "127.0.0.1:$port:9200" --add-host host.docker.internal:host-gateway \
    -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
    -e "OPENSEARCH_JAVA_OPTS=-Xms$heap -Xmx$heap" "$image" > /dev/null
  echo "started $name on 127.0.0.1:$port"
done
for port in "${BENCH_BASELINE_PORT:-9321}" "${BENCH_AGENTIC_PORT:-9322}"; do
  for _ in $(seq 1 100); do
    curl -sf "http://127.0.0.1:$port/_cluster/health" > /dev/null && break
    sleep 3
  done
  curl -sf "http://127.0.0.1:$port/_cluster/health" > /dev/null || { echo "cluster on port $port did not start"; exit 1; }
done
echo ready
