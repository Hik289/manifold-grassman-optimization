#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nohup bash agg_daemon.sh >/dev/null 2>&1 &
echo "Aggregator PID: $!"
