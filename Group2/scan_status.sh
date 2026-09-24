#!/bin/bash
# =============================================================================
# scan_status.sh -- one-shot status of the EQ-SANS (x,y) scans.
#
#   ./scan_status.sh                 # status of every scan folder found
#   ./scan_status.sh scan_sans_sample    # just one folder
#   ./scan_status.sh -w              # watch: refresh every 30 s until done
#
# Reports, per scan folder: points present / complete / incomplete, disk use,
# throughput and ETA. Completeness is judged by the detector "values:" entries
# in mccode.sim -- McStas creates that file at run START, so mere existence
# does NOT mean a point finished.
# =============================================================================
set -u

PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"
TOTAL=${TOTAL:-1681}
WATCH=0
FOLDERS=()

for a in "$@"; do
  case "$a" in
    -w|--watch) WATCH=1 ;;
    *) FOLDERS+=("$a") ;;
  esac
done
if [ ${#FOLDERS[@]} -eq 0 ]; then
  FOLDERS=(scan_guide_xy scan_sans_nosample scan_sans_sample)
fi

hms() {  # seconds -> "1h 23m"
  local s=${1%.*}
  printf "%dh %02dm" $((s/3600)) $(((s%3600)/60))
}

report_folder() {
  local d=$1
  [ -d "$d" ] || { printf "  %-22s %4d/%d (  0%%)  pending\n" "$d" 0 "$TOTAL"; return; }

  # One grep over all points: prints "path:count" per file.
  local stats
  stats=$(grep -cE "^  values:" "$d"/x*_y*/mccode.sim 2>/dev/null \
          | awk -F: '{n++; c[$2]++; if($2+0>mx) mx=$2+0}
                     END{print n+0, mx+0, c[mx]+0}')
  local present expect complete
  read -r present expect complete <<<"$stats"
  present=${present:-0}; expect=${expect:-0}; complete=${complete:-0}
  local incomplete=$((present-complete))

  # throughput from mccode.sim mtimes
  local rate="" eta=""
  if [ "$complete" -gt 5 ] && [ "$complete" -lt "$TOTAL" ]; then
    local span
    span=$(stat -f "%m" "$d"/x*_y*/mccode.sim 2>/dev/null \
           | awk 'NR==1{mn=mx=$1} {if($1<mn)mn=$1; if($1>mx)mx=$1} END{print mx-mn}')
    if [ "${span:-0}" -gt 0 ]; then
      rate=$(awk -v n="$complete" -v s="$span" 'BEGIN{printf "%.1f", n/(s/60)}')
      local left=$((TOTAL-complete))
      [ "$left" -lt 0 ] && left=0
      eta=$(awk -v l="$left" -v r="$rate" 'BEGIN{if(r>0) printf "%.0f", l*60/r; else print 0}')
    fi
  fi

  local pct=$(( present*100/TOTAL ))
  local size; size=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "  %-22s %4d/%d (%3d%%)  complete %4d  incomplete %2d  %6s" \
         "$d" "$present" "$TOTAL" "$pct" "$complete" "$incomplete" "${size:-–}"
  [ "$complete" -ge "$TOTAL" ] && printf "  DONE"
  [ -n "$rate" ] && printf "  %s pts/min" "$rate"
  if [ -n "$eta" ] && [ "$complete" -lt "$TOTAL" ]; then
    printf "  ETA %s" "$(hms "$eta")"
  fi
  printf "\n"
}

show() {
  echo "=== scan status  $(date '+%Y-%m-%d %H:%M:%S') ==="

  local running=0 phase="-"
  if ps -eo args | grep -q '[m]aster_scan_sans.sh'; then
    running=1; phase="sans (nosample -> sample)"
  elif ps -eo args | grep -q '[m]aster_scan.sh'; then
    running=1; phase="guide_xy"
  fi
  local workers
  workers=$(ps -eo args | grep -c '[e]qsans.*\.out')

  if [ "$running" -eq 1 ]; then
    echo "  job: RUNNING   phase: $phase   workers: $workers"
  else
    echo "  job: not running   (stray workers: $workers)"
  fi

  for d in "${FOLDERS[@]}"; do report_folder "$d"; done

  # nothing left to do?
  local alldone=1
  for d in "${FOLDERS[@]}"; do
    local c
    c=$(grep -cE "^  values:" "$d"/x*_y*/mccode.sim 2>/dev/null | awk -F: '{if($2+0>0)n++} END{print n+0}')
    [ "${c:-0}" -lt "$TOTAL" ] && alldone=0
  done
  [ "$alldone" -eq 1 ] && [ "$running" -eq 0 ] && echo "  -> all scans complete"
  return $alldone
}

if [ "$WATCH" -eq 1 ]; then
  while true; do
    show
    ps -eo args | grep -q '[m]aster_scan' || { echo "  (job finished)"; break; }
    echo
    sleep 30
  done
else
  show
fi
