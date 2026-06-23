#!/usr/bin/env bash
# Download full StatsBomb Open Data into data/statsbomb/
# Usage: bash scripts/download_statsbomb.sh [git|zip]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/statsbomb"
METHOD="${1:-git}"

echo "SportSwarm-CPP | StatsBomb Open Data downloader"
echo "Target: $DEST"
echo "Method: $METHOD"
echo ""

# Remove broken partial downloads
if [[ -d "$DEST" ]]; then
  echo "Removing incomplete folder: $DEST"
  rm -rf "$DEST"
fi
mkdir -p "$ROOT/data"

download_git() {
  echo "=== Git shallow clone (full repo, ~2–3 GB) ==="
  echo "Tip: if this fails, run: bash scripts/download_statsbomb.sh zip"
  echo ""

  export GIT_HTTP_VERSION=HTTP/1.1
  local attempt
  for attempt in 1 2 3; do
    echo "--- Attempt $attempt/3 ---"
    if git -c http.version=HTTP/1.1 \
           -c http.postBuffer=524288000 \
           clone --depth 1 --progress \
           https://github.com/statsbomb/open-data.git "$DEST"; then
      return 0
    fi
    echo "Attempt $attempt failed. Cleaning up..."
    rm -rf "$DEST"
    sleep 5
  done
  return 1
}

download_zip() {
  echo "=== ZIP download (browser-equivalent, supports resume) ==="
  local zip="$ROOT/data/open-data-master.zip"
  local url="https://github.com/statsbomb/open-data/archive/refs/heads/master.zip"

  echo "Downloading to $zip"
  echo "(If interrupted, re-run this script — curl will resume)"
  curl -L --retry 5 --retry-delay 5 -C - -o "$zip" "$url"

  echo "Extracting..."
  rm -rf "$DEST"
  unzip -q "$zip" -d "$ROOT/data"
  mv "$ROOT/data/open-data-master" "$DEST"
  rm -f "$zip"
}

case "$METHOD" in
  git)
    download_git || {
      echo ""
      echo "Git clone failed after 3 attempts."
      echo "Trying ZIP method automatically..."
      download_zip
    }
    ;;
  zip)
    download_zip
    ;;
  *)
    echo "Unknown method: $METHOD"
    echo "Usage: bash scripts/download_statsbomb.sh [git|zip]"
    exit 1
    ;;
esac

echo ""
echo "=== Verify ==="
for sub in competitions events lineups matches; do
  if [[ -d "$DEST/data/$sub" ]]; then
    count=$(find "$DEST/data/$sub" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  data/$sub : $count files"
  else
    echo "  data/$sub : MISSING"
  fi
done

echo ""
echo "Done. BLF path: data/statsbomb/data/events"
echo "Next: python3 scripts/blf_from_statsbomb.py --data-dir data/statsbomb/data/events"
echo "      python3 main.py configs/football_full.yaml"
