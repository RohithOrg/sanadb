#!/usr/bin/env bash
# Soak the store: writer + checkpoints + in-process and child-process
# readers for N seconds (default 60), then exact agreement with the model.
# Usage: scripts/soak.sh [seconds]
set -euo pipefail
cd "$(dirname "$0")/.."
SANADB_SOAK_SECS="${1:-60}" cargo test --release --test db -- --ignored soak_readers_writer_and_checkpoints --nocapture
