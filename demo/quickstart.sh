#!/bin/bash
# Quick start on public data: certified clearance between two public PHerc0172 segments
# (w062 and w078, volume 20241024131838). Downloads ~6 MB of mesh coordinates.
set -euo pipefail
OUT=${1:-quickstart_data}
B=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0172/segments
for s in 20250917143559-w062_20250917143559205_flatboi/mesh/20250917143559-on-20241024131838-7.91um.tifxyz \
         20250926112011-w078_20250926112011918_flatboi/mesh/20250926112011-on-20241024131838-7.91um.tifxyz; do
  d="$OUT/$(basename "$s")"; mkdir -p "$d"
  for f in x.tif y.tif z.tif meta.json; do [ -s "$d/$f" ] || curl -sS --fail --retry 4 -o "$d/$f" "$B/$s/$f"; done
done
tifxyz-dist clearance "$OUT/20250917143559-on-20241024131838-7.91um.tifxyz" \
                      "$OUT/20250926112011-on-20241024131838-7.91um.tifxyz"
