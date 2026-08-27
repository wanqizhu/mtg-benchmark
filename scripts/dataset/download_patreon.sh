#!/bin/bash
# Raw dump of every Possibility Storm Patreon post (images + post text).
# Auth: reads your existing Patreon session from a local browser cookie store.
# Log into patreon.com in that browser first. No password is passed to this script.
#
# Usage:
#   scripts/dataset/download_patreon.sh [chrome|firefox|safari]
#   PATREON_OUT=... scripts/dataset/download_patreon.sh chrome

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BROWSER="${1:-chrome}"
DIR="${PATREON_OUT:-$ROOT/datasets/patreon-raw}"
CONF="$ROOT/scripts/dataset/gallery-dl.conf"

mkdir -p "$DIR"

gallery-dl \
  --config "$CONF" \
  --cookies-from-browser "$BROWSER" \
  --user-agent browser \
  -d "$DIR" \
  --download-archive "$DIR/archive.txt" \
  --sleep-request 1-2 \
  "https://www.patreon.com/c/mtgpuzzles/posts" \
  2>&1 | tee -a "$DIR/download.log"
