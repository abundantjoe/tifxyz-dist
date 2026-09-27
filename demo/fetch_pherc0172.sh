#!/bin/bash
# Fetch the public PHerc0172 (Scroll 5) segment meshes (x/y/z.tif + meta.json only) traced on
# volume 20241024131838 from the Vesuvius Challenge open-data bucket (CC BY-NC 4.0).
# Usage: demo/fetch_pherc0172.sh OUT_DIR    -> OUT_DIR/<segment>/{x,y,z}.tif, OUT_DIR/manifest.tsv
set -euo pipefail
OUT=${1:?usage: fetch_pherc0172.sh OUT_DIR}
B=https://vesuvius-challenge-open-data.s3.amazonaws.com
VOL=20241024131838
mkdir -p "$OUT"
curl -sS --retry 4 "$B/?list-type=2&delimiter=/&prefix=PHerc0172/segments/&max-keys=1000" \
  | grep -o '<Prefix>PHerc0172/segments/[^<]*/' | sed 's#<Prefix>##' > "$OUT/.segs"
for p in $(cat "$OUT/.segs"); do
  key=$(curl -sS --retry 4 "$B/?list-type=2&prefix=${p}mesh/&max-keys=100" \
        | grep -o "<Key>[^<]*-on-${VOL}-[^<]*\.tifxyz/x\.tif" | head -1 | sed 's#<Key>##') || true
  [ -z "$key" ] && continue
  d=${key%/x.tif}; id=$(basename "$p")
  mkdir -p "$OUT/$id"
  for f in x.tif y.tif z.tif meta.json; do
    [ -s "$OUT/$id/$f" ] || curl -sS --fail --retry 4 -o "$OUT/$id/$f" "$B/$d/$f"
  done
  printf '%s\ts3://vesuvius-challenge-open-data/%s\t%s\n' "$id" "$d" \
    "$(cat "$OUT/$id"/{x,y,z}.tif | sha256sum | cut -c1-16)" >> "$OUT/manifest.tsv"
done
sort -u -o "$OUT/manifest.tsv" "$OUT/manifest.tsv"
wc -l "$OUT/manifest.tsv"
