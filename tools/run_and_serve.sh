#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
make
./cuda_micro_bench --json
python3 -m http.server 8000
