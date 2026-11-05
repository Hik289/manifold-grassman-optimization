#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/revision/third_party/SubTrack"
COMMIT="f85bfb369a4a73b26f2e46459abc1e20863185f0"

if [ ! -d "$TARGET/.git" ]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone https://github.com/criticalml-uw/SubTrack.git "$TARGET"
fi
git -C "$TARGET" fetch origin "$COMMIT"
git -C "$TARGET" checkout --detach "$COMMIT"
printf 'Pinned external SubTrack++ source: %s\n' "$(git -C "$TARGET" rev-parse HEAD)"
