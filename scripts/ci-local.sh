#!/usr/bin/env bash
# Run the Linux CI steps in Docker before pushing: clippy and the test suite
# on x86-64 (emulated on Apple Silicon, so slow, but it compiles the AVX2
# paths and runs x86-only lints) and on arm64. Usage: scripts/ci-local.sh [amd64|arm64|all]
set -euo pipefail
cd "$(dirname "$0")/.."
which="${1:-all}"
run() {
  local arch="$1"
  echo "== linux/$arch"
  docker run --rm --platform "linux/$arch" \
    -v "$PWD":/w -w /w \
    -v sanadb-cargo-registry:/usr/local/cargo/registry \
    -v "sanadb-target-$arch":/w/target-docker \
    -e CARGO_TARGET_DIR=/w/target-docker \
    rust:1-slim bash -c '
      set -e
      rustup component add clippy >/dev/null 2>&1 || true
      cargo clippy --release --all-targets -- -D warnings
      cargo test --release
      cargo build --release --examples
    '
}
case "$which" in
  amd64|arm64) run "$which" ;;
  all) run arm64; run amd64 ;;
  *) echo "usage: $0 [amd64|arm64|all]"; exit 2 ;;
esac
