#!/bin/bash
# apt-get install, but download the .debs first with parallel curl: the proxy throttles each connection,
# so up to 24 files at once, and files over 16 MB in 8 ranged pieces. apt still checks every hash.
set -euo pipefail
cd /var/cache/apt/archives
apt-get install -y --no-install-recommends --print-uris "$@" \
  | sed -n "s/^'\([^']*\)' \([^ ]*\) \([0-9]*\) .*/\1 \2 \3/p" > /tmp/apt-uris
echo "prefetching $(wc -l < /tmp/apt-uris) packages, $(awk '{s+=$3} END {printf "%.0f", s/1e6}' /tmp/apt-uris) MB"

fetch() {
  local url=$1 file=$2 size=$3 parts=8 i
  if [ "$size" -lt 16000000 ]; then
    curl -sSfL --retry 5 -o "partial/$file" "$url"
  else
    local chunk=$(( (size + parts - 1) / parts ))
    for ((i = 0; i < parts; i++)); do
      curl -sSfL --retry 5 -r "$((i * chunk))-$(( (i + 1) * chunk - 1 ))" -o "partial/$file.$i" "$url" &
    done
    wait
    for ((i = 0; i < parts; i++)); do cat "partial/$file.$i"; done > "partial/$file"
    rm -f "partial/$file".[0-9]
    [ "$(stat -c %s "partial/$file")" = "$size" ] || { rm -f "partial/$file"; return 1; }
  fi
  mv "partial/$file" "$file"
}
export -f fetch
xargs -r -P 24 -n 3 bash -c 'fetch "$@" || echo "prefetch failed: $1"' _ < /tmp/apt-uris
apt-get install -y --no-install-recommends "$@"
rm -f /tmp/apt-uris
