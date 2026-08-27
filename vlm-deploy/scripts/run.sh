#!/usr/bin/env bash
# =============================================================================
# run.sh — Start the vLLM server using docker compose or docker run
# =============================================================================
# This script loads .env and starts the container.
#
# Usage:
#   ./scripts/run.sh          # start with docker compose
#   ./scripts/run.sh --stop   # stop the container
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Load .env if it exists
# ---------------------------------------------------------------------------
if [ -f "${PROJECT_DIR}/.env" ]; then
    echo "Loading environment from ${PROJECT_DIR}/.env"
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Handle --stop
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--stop" ]; then
    echo "Stopping vLLM server..."
    cd "$PROJECT_DIR" && docker compose down
    exit 0
fi

# ---------------------------------------------------------------------------
# Start with docker compose
# ---------------------------------------------------------------------------
echo "Starting vLLM server with docker compose..."
cd "$PROJECT_DIR" && docker compose up -d

echo ""
echo "Container started. Check logs with:"
echo "  docker compose logs -f"
echo ""
echo "API endpoint: http://localhost:${PORT:-8000}"
