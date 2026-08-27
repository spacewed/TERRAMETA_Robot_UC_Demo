#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${TM_NET_IFACE:-enP2p1s0f1np1}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  IFACE="$1"
  shift
fi

exec python3 "$SCRIPT_DIR/tsn_probe_profile.py" status --iface "$IFACE" "$@"
