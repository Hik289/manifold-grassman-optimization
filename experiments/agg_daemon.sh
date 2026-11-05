#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
while true; do
  if python3 aggregate.py > runs/.results_tmp.json 2>runs/_agg_err.txt; then
    mv runs/.results_tmp.json runs/results_running.json
  fi
  sleep 180
done
