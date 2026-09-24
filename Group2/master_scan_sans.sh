#!/bin/bash
# =============================================================================
# EQ-SANS guide-entrance (x,y) scan, run TWICE with identical geometry:
#   scan_sans_nosample/   sample_on=0   empty beam through the sample aperture
#   scan_sans_sample/     sample_on=1   monodisperse hard spheres (R=100 AA)
#
# Both cases use the same sample aperture, so the only difference between the
# two datasets is the sample itself -- that is the point of the comparison.
#
#   ./master_scan_sans.sh                  # both scans
#   ./master_scan_sans.sh nosample         # only the empty-beam scan
#   ./master_scan_sans.sh sample           # only the sample scan
#   NCOUNT=5e6 ./master_scan_sans.sh       # lighter/faster
#
# Resumable: points that already have mccode.sim are skipped.
# =============================================================================
set -u

PROJ="$(cd "$(dirname "$0")" && pwd)"
INSTR="$PROJ/eqsans_sans.instr"
BIN="$PROJ/eqsans_sans.out"
JOBLIST="$PROJ/scan_joblist_sans.txt"

XMIN=${XMIN:--20}; XMAX=${XMAX:-20}
YMIN=${YMIN:--20}; YMAX=${YMAX:-20}
STEP=${STEP:-1}
NCOUNT=${NCOUNT:-1e7}
JOBS=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
SAMP_AP=${SAMP_AP:-0.01}      # sample aperture radius (m)
PHI=${PHI:-0.01}              # particle volume fraction
RSPH=${RSPH:-100}             # sphere radius (AA)
BS=${BS:-0}                   # beamstop radius (m); 0 = off (no direct beam exists)

# --- build if needed -------------------------------------------------------
if [ ! -x "$BIN" ] || [ "$INSTR" -nt "$BIN" ]; then
  echo "[build] compiling $(basename "$INSTR")"
  ( cd "$PROJ" && mcrun "$INSTR" -n 1e3 --dir="$(mktemp -d)" >/dev/null 2>&1 )
  [ -x "$BIN" ] || { echo "[build] FAILED"; exit 1; }
fi

# --- job list --------------------------------------------------------------
python3 - "$XMIN" "$XMAX" "$YMIN" "$YMAX" "$STEP" "$JOBLIST" <<'PY'
import sys
xmin,xmax,ymin,ymax,step = (int(v) for v in sys.argv[1:6])
rows=[]
for x in range(xmin,xmax+1,step):
    for y in range(ymin,ymax+1,step):
        rows.append("x%+03d_y%+03d %.4f %.4f" % (x,y,x/1000.0,y/1000.0))
open(sys.argv[6],"w").write("\n".join(rows)+"\n")
print("[joblist] %d points" % len(rows))
PY

run_one() {
  name=$1; gx=$2; gy=$3
  d="$OUT/$name"
  [ -f "$d/mccode.sim" ] && return 0
  rm -rf "$d"
  "$BIN" --ncount="$NCOUNT" --dir="$d" \
       gx="$gx" gy="$gy" sample_on="$SAMPLE_ON" \
       R_sphere="$RSPH" Phi="$PHI" bs_radius="$BS" samp_ap="$SAMP_AP" \
       > "$OUT/.log_$name" 2>&1
}
export -f run_one
export BIN NCOUNT RSPH PHI BS SAMP_AP

do_scan() {   # $1 = folder, $2 = sample_on
  export OUT="$PROJ/$1"
  export SAMPLE_ON="$2"
  mkdir -p "$OUT"
  echo "[scan] $1 (sample_on=$2), ncount=$NCOUNT, $JOBS jobs"
  time xargs -P "$JOBS" -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$JOBLIST"
  echo "[scan] $1 done: $(ls -d "$OUT"/x*_y* 2>/dev/null | wc -l) folders"
}

case "${1:-both}" in
  nosample) do_scan scan_sans_nosample 0 ;;
  sample)   do_scan scan_sans_sample   1 ;;
  *)        do_scan scan_sans_nosample 0
            do_scan scan_sans_sample   1 ;;
esac
