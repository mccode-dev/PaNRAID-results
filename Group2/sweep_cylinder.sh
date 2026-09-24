#!/bin/bash
# =============================================================================
# Training-set sweep over SasView cylinder parameters on the EQ-SANS model.
#
#   ./sweep_cylinder.sh                  # 500 Latin-hypercube samples
#   N=2000 ./sweep_cylinder.sh           # bigger set
#   SAMPLING=random ./sweep_cylinder.sh  # plain uniform random instead of LHS
#   NCOUNT=3e6 ./sweep_cylinder.sh       # better statistics per sample
#   ./sweep_cylinder.sh manifest         # write the manifest only, run nothing
#
# One folder per sample under dataset_cylinder/, plus manifest.csv mapping
# folder -> every parameter (swept AND held fixed), which is what a training
# pipeline actually needs.
#
# Resumable: a point counts as done only when its mccode.sim carries the
# detector "values:" entries. McStas creates mccode.sim at run START, so mere
# file existence is NOT proof a run finished.
# =============================================================================
set -u

PROJ="$(cd "$(dirname "$0")" && pwd)"
INSTR="$PROJ/eqsans_cylinder.instr"
BIN="$PROJ/eqsans_cylinder.out"
OUT="$PROJ/dataset_cylinder"
MANIFEST="$OUT/manifest.csv"

N=${N:-2000}
SAMPLING=${SAMPLING:-lhs}         # lhs | random
SEED=${SEED:-1234}
NCOUNT=${NCOUNT:-5e6}          # noise ~2.9%/q-bin; pd_length is weak below this
JOBS=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
EXPECT=4                          # detector entries in a finished run

mkdir -p "$OUT"

# --- build -----------------------------------------------------------------
if [ ! -x "$BIN" ] || [ "$INSTR" -nt "$BIN" ]; then
  echo "[build] compiling $(basename "$INSTR")"
  ( cd "$PROJ" && mcrun "$INSTR" -n 1e3 --dir="$(mktemp -d)" >/dev/null 2>&1 )
  [ -x "$BIN" ] || { echo "[build] FAILED"; exit 1; }
fi

# --- manifest --------------------------------------------------------------
python3 - "$N" "$SAMPLING" "$SEED" "$MANIFEST" <<'PY'
import sys, csv, math, random
N, sampling, seed, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
rng = random.Random(seed)

# Swept: the parameters that actually shape I(q) and are identifiable in the
# q range this instrument covers (~0.002 - 0.15 1/AA).
#   name        lo     hi     scale
SWEPT = [
    ("radius",     20.0, 200.0,  "log"),
    ("length",    200.0, 1000.0, "log"),
    ("pd_radius",   0.0,   0.30, "lin"),
    ("pd_length",   0.0,   0.30, "lin"),
    ("sld",         0.5,   8.0,  "lin"),
]
# Held fixed, with the reason -- see the notes printed by the script.
FIXED = {"sld_solvent": 1.0, "model_scale": 1.0, "model_abs": 0.0}

def unit_samples(n, d):
    if sampling == "lhs":
        cols = []
        for _ in range(d):
            cut = [(i + rng.random()) / n for i in range(n)]
            rng.shuffle(cut)
            cols.append(cut)
        return [[cols[j][i] for j in range(d)] for i in range(n)]
    return [[rng.random() for _ in range(d)] for _ in range(n)]

U = unit_samples(N, len(SWEPT))
rows = []
for i, u in enumerate(U):
    rec = {"id": "cyl_%06d" % i}
    for uk, (name, lo, hi, scale) in zip(u, SWEPT):
        if scale == "log":
            v = math.exp(math.log(lo) + uk * (math.log(hi) - math.log(lo)))
        else:
            v = lo + uk * (hi - lo)
        rec[name] = round(v, 6)
    rec.update(FIXED)
    rows.append(rec)

cols = ["id"] + [s[0] for s in SWEPT] + list(FIXED)
with open(out, "w", newline="") as f:
    # default lineterminator is CRLF; the trailing \r would corrupt the last field
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print("[manifest] %d samples (%s, seed=%d) -> %s" % (N, sampling, seed, out))
PY

[ "${1:-run}" = "manifest" ] && exit 0

# --- run -------------------------------------------------------------------
run_one() {
  local line=$1
  local id radius length pdr pdl sld slds mscale mabs
  IFS=',' read -r id radius length pdr pdl sld slds mscale mabs <<<"$line"
  local d="$OUT/$id"
  # completeness check, not mere existence
  if [ -f "$d/mccode.sim" ]; then
    local n; n=$(grep -cE "^  values:" "$d/mccode.sim" 2>/dev/null); n=${n:-0}
    [ "$n" -ge "$EXPECT" ] && return 0
  fi
  rm -rf "$d"
  "$BIN" --ncount="$NCOUNT" --dir="$d" \
        radius="$radius" length="$length" \
        pd_radius="$pdr" pd_length="$pdl" \
        sld="$sld" sld_solvent="$slds" \
        model_scale="$mscale" model_abs="$mabs" \
        > "$OUT/.log_$id" 2>&1
}
export -f run_one
export OUT BIN NCOUNT EXPECT

echo "[sweep] $N samples, ncount=$NCOUNT, $JOBS parallel jobs"
tail -n +2 "$MANIFEST" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

done_n=$(grep -lE "^  values:" "$OUT"/cyl_*/mccode.sim 2>/dev/null | wc -l | tr -d ' ')
echo "[sweep] complete: $done_n / $N samples"
