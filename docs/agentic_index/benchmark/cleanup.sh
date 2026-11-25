#!/usr/bin/env bash
#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stops the logging proxy and removes the benchmark containers. With --image it also removes the benchmark image.
# Results under the work directory are kept. Environment: BENCH_PREFIX (aibench), BENCH_IMAGE, BENCH_WORK (./work).
set -uo pipefail
prefix="${BENCH_PREFIX:-aibench}"
work="${BENCH_WORK:-$(dirname "$0")/work}"
if [ -f "$work/proxy.pid" ]; then
  kill "$(cat "$work/proxy.pid")" 2> /dev/null && echo "stopped proxy"
  rm -f "$work/proxy.pid"
fi
for role in baseline agentic; do
  docker rm -f "$prefix-$role" > /dev/null 2>&1 && echo "removed $prefix-$role"
done
if [ "${1:-}" = "--image" ]; then
  docker rmi "${BENCH_IMAGE:-opensearch-ml:agentic-index-bench}" > /dev/null && echo "removed image"
fi
exit 0
