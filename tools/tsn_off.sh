#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFACE="${TM_NET_IFACE:-enP2p1s0f1np1}"

if [[ $# -gt 0 && "$1" != --* ]]; then
  IFACE="$1"
  shift
fi

NEEDS_ROOT=1
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    NEEDS_ROOT=0
  fi
done

if [[ "$NEEDS_ROOT" -eq 1 && "$(id -u)" -ne 0 ]]; then
  exec sudo -E "$0" "$IFACE" "$@"
fi

exec python3 "$SCRIPT_DIR/tsn_probe_profile.py" restore --iface "$IFACE" "$@"
