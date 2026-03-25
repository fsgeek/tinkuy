#!/usr/bin/env bash
# Run the PRD planning benchmark through the Tinkuy projective gateway.
#
# Prerequisites:
#   - Gateway running on port 8340 (or set TINKUY_PORT)
#   - ANTHROPIC_API_KEY set in environment
#
# Usage:
#   ./scripts/run-prd-benchmark.sh

set -euo pipefail

TINKUY_PORT="${TINKUY_PORT:-8340}"
BENCHMARK_DIR="/home/tony/projects/bladnman/planning_benchmark"

# Verify gateway is alive
if ! curl -sf "http://127.0.0.1:${TINKUY_PORT}/v1/tinkuy/health" > /dev/null 2>&1; then
    echo "ERROR: Gateway not responding on port ${TINKUY_PORT}" >&2
    echo "Start it with: uv run python -m tinkuy serve --port ${TINKUY_PORT} --data-dir .tinkuy-data/experiments/prd-benchmark-01" >&2
    exit 1
fi

echo "=== PRD Planning Benchmark ==="
echo "  Gateway: http://127.0.0.1:${TINKUY_PORT}"
echo "  Benchmark: ${BENCHMARK_DIR}"
echo "  Output: ${BENCHMARK_DIR}/results/PLAN.md"
echo ""

# Route through the Tinkuy gateway
export ANTHROPIC_BASE_URL="http://127.0.0.1:${TINKUY_PORT}"

exec claude \
    --print \
    --output-format text \
    --model claude-sonnet-4-20250514 \
    --max-turns 50 \
    -p "Read 1-START_HERE.md and follow its instructions exactly. Read all PRD documents, then write the implementation plan to results/PLAN.md." \
    "${BENCHMARK_DIR}"
