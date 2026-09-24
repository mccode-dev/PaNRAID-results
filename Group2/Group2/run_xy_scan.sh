#!/bin/bash
# Guide-entrance (x,y) offset scan for eqsans_1_fixed.instr
# Grid: -20..+20 mm, 1 mm step, in both x and y  -> 41 x 41 = 1681 points
# gx,gy are passed in metres; one output folder per (x,y) combination.

set -u
PROJ="$(cd "$(dirname "$0")" && pwd)"
OUT="$PROJ/scan_guide_xy"
NCOUNT=${NCOUNT:-1e6}
JOBS=${JOBS:-10}

mkdir -p "$OUT"

run_one() {
  name=$1; gx=$2; gy=$3
  d="$OUT/$name"
  [ -f "$d/mccode.sim" ] && return 0   # already done, skip
  rm -rf "$d"
  "$PROJ/eqsans_1_fixed.out" --ncount="$NCOUNT" --dir="$d" gx="$gx" gy="$gy" \
     > "$OUT/.log_$name" 2>&1
}
export -f run_one
export PROJ OUT NCOUNT

xargs -P "$JOBS" -n 3 bash -c 'run_one "$0" "$1" "$2"' < /tmp/eqsans_joblist.txt

echo "scan complete: $(ls -d "$OUT"/x*_y* 2>/dev/null | wc -l) folders"
