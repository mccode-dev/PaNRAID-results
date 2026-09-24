#!/bin/bash
# Verify a scan folder point-by-point and (optionally) clear incomplete points
# so that re-running the master script refills them.
#
#   ./verify_scan.sh scan_sans_nosample            # report only
#   ./verify_scan.sh scan_sans_sample --repair     # delete incomplete points
#
# A point is COMPLETE when its mccode.sim contains the detector "values:"
# lines. McStas writes mccode.sim at run START, so "file exists" is NOT a
# valid completeness test -- an interrupted run leaves a header-only file.
set -u

DIR=${1:?usage: verify_scan.sh <scan_folder> [--repair]}
REPAIR=${2:-}
EXPECT=${EXPECT:-auto}          # auto-infer detector entries per finished run

count_values() {   # grep -c prints 0 and exits 1 on no match; keep just the number
  local n
  n=$(grep -cE "^  values:" "$1" 2>/dev/null)
  echo "${n:-0}"
}

# Number of monitors differs per instrument (2 for the empty-beam instrument,
# 4 with the Q monitor), so infer it from the most complete point present.
if [ -z "${EXPECT:-}" ] || [ "${EXPECT:-auto}" = "auto" ]; then
  EXPECT=0
  for d in $(ls -d "$DIR"/x*_y* 2>/dev/null | head -50); do
    n=$(count_values "$d/mccode.sim")
    [ "$n" -gt "$EXPECT" ] && EXPECT=$n
  done
  [ "$EXPECT" -eq 0 ] && EXPECT=1
  echo "(expecting $EXPECT detector entries per point)"
fi

total=0; ok=0; bad=0
badlist=()
for d in "$DIR"/x*_y*; do
  [ -d "$d" ] || continue
  total=$((total+1))
  n=$(count_values "$d/mccode.sim")
  if [ "$n" -ge "$EXPECT" ]; then
    ok=$((ok+1))
  else
    bad=$((bad+1)); badlist+=("$d")
  fi
done

echo "$DIR: $total present, $ok complete, $bad incomplete"

if [ "$bad" -gt 0 ]; then
  printf '  incomplete: %s\n' "${badlist[@]:0:10}"
  [ "$bad" -gt 10 ] && echo "  ... and $((bad-10)) more"
  if [ "$REPAIR" = "--repair" ]; then
    # only safe once the scan is no longer running, or these are in-flight points
    if pgrep -f master_scan_sans.sh >/dev/null; then
      echo "  REFUSING to repair: a scan is still running (points may be in flight)."
      exit 1
    fi
    rm -rf "${badlist[@]}"
    echo "  removed $bad incomplete point(s) -- re-run the master script to refill"
  else
    echo "  (pass --repair to remove them, then re-run the master script)"
  fi
fi
