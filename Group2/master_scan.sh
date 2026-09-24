#!/bin/bash
# =============================================================================
# EQ-SANS guide-entrance (x,y) alignment scan -- end-to-end master script.
#
#   ./master_scan.sh                 # full run: build -> joblist -> scan -> analyse
#   XMIN=-5 XMAX=5 STEP=1 ./master_scan.sh    # smaller grid
#   NCOUNT=1e5 JOBS=4 ./master_scan.sh        # faster/lighter
#   ./master_scan.sh analyse         # re-make plots only, no re-running
#
# Self-contained: generates its own job list inside the project directory,
# and is resumable -- points that already have mccode.sim are skipped, so
# re-running fills in gaps rather than redoing finished work.
# =============================================================================
set -u

PROJ="$(cd "$(dirname "$0")" && pwd)"
INSTR="$PROJ/eqsans_1_fixed.instr"
BIN="$PROJ/eqsans_1_fixed.out"
OUT="$PROJ/scan_guide_xy"
JOBLIST="$PROJ/scan_joblist.txt"

# Grid in mm (gx,gy are converted to metres for McStas below)
XMIN=${XMIN:--20}; XMAX=${XMAX:-20}
YMIN=${YMIN:--20}; YMAX=${YMAX:-20}
STEP=${STEP:-1}
NCOUNT=${NCOUNT:-1e6}
JOBS=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}

mkdir -p "$OUT"

# --- 1. build the instrument if the binary is missing or out of date --------
if [ ! -x "$BIN" ] || [ "$INSTR" -nt "$BIN" ]; then
  echo "[build] compiling $(basename "$INSTR")"
  ( cd "$PROJ" && mcrun "$INSTR" gx=0 gy=0 -n 1e3 --dir="$(mktemp -d)" >/dev/null 2>&1 )
  [ -x "$BIN" ] || { echo "[build] FAILED - no $BIN"; exit 1; }
fi

# --- 2. generate the job list (dirname, gx[m], gy[m]) ----------------------
python3 - "$XMIN" "$XMAX" "$YMIN" "$YMAX" "$STEP" "$JOBLIST" <<'PY'
import sys
xmin,xmax,ymin,ymax,step = (int(v) for v in sys.argv[1:6])
out = sys.argv[6]
rows = []
for x in range(xmin, xmax+1, step):
    for y in range(ymin, ymax+1, step):
        rows.append("x%+03d_y%+03d %.4f %.4f" % (x, y, x/1000.0, y/1000.0))
open(out, "w").write("\n".join(rows) + "\n")
print("[joblist] %d points -> %s" % (len(rows), out))
PY

# --- 3. run the scan in parallel (skip completed points) -------------------
run_one() {
  name=$1; gx=$2; gy=$3
  d="$OUT/$name"
  [ -f "$d/mccode.sim" ] && return 0
  rm -rf "$d"
  "$BIN" --ncount="$NCOUNT" --dir="$d" gx="$gx" gy="$gy" > "$OUT/.log_$name" 2>&1
}
export -f run_one
export OUT BIN NCOUNT

if [ "${1:-scan}" != "analyse" ]; then
  total=$(wc -l < "$JOBLIST")
  echo "[scan] $total points, ncount=$NCOUNT, $JOBS parallel jobs"
  time xargs -P "$JOBS" -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$JOBLIST"
  echo "[scan] complete: $(ls -d "$OUT"/x*_y* 2>/dev/null | wc -l) folders"
fi

# --- 4. build the map and plots -------------------------------------------
python3 "$PROJ/analyse_xy_scan.py"
