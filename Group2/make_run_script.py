#!/usr/bin/env python3
"""Generate a flat, self-contained SLURM script for the EQ-SANS cylinder
campaign -- matching this project's own templates exactly:
  - Software/mcstas-GPU-test.sh                         (env setup / SBATCH header)
  - 22_September.../example_scripts/run_generated_sphere.sh   (flat structure)

No runtime path resolution, no manifest/CSV parsing inside the job: every
run is a literal, pre-written command line in the generated .sh file. This
is what those templates already do -- one #SBATCH header, one compile line,
then N explicit run lines -- and it sidesteps the whole class of "where does
this script think it's running from" problem entirely.

Usage:
  python3 make_run_script.py --n 500  --out submit_smoke.sh --data-dir dataset_smoke
  python3 make_run_script.py --n 5000 --out submit_full.sh  --data-dir dataset_campaign
  sbatch submit_smoke.sh          # on Juliet, from inside Group2/
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import qmc

MATERIALS = {  # sld [1e-6/Angs^2], approximate literature values
    "polystyrene": 1.41, "silica": 3.47, "gold": 4.50,
    "deuterated_polystyrene": 6.20, "nickel": 9.40,
}
SLD_SOLVENT_RANGE = (-0.56, 6.34)      # pure H2O -> pure D2O
MISALIGN_RANGE_MM = (-20.0, 20.0)      # characterized range, scan_sans_sample
ALIGNED_FRACTION = 0.15
RADIUS_RANGE = (20.0, 200.0)           # Angs, log-uniform
LENGTH_RANGE = (200.0, 1000.0)         # Angs, log-uniform
PD_RANGE = (0.0, 0.30)

COMMON_SETUP = """export MAMBA_EXE='/projects/m26216/INSTALL/bin/micromamba'
export MAMBA_ROOT_PREFIX='/projects/m26216/INSTALL/micromamba/'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate panraid

# Absolute path captured when THIS FILE WAS GENERATED (by make_run_script.py,
# run interactively, not inside a batch job -- so os.getcwd() there is
# unambiguous). Do not rely on SLURM's default working directory: it varies
# by cluster config, and guessing it wrong is exactly what broke earlier
# versions of this script.
cd {project_dir}
mkdir -p {data_dir}

# Skip a run if it already completed (mccode.sim carries "values:" only once
# a run finishes -- McStas writes the file at START, so mere existence is not
# proof of completion). Makes the whole campaign safely re-submittable.
run_if_needed() {{
  local dir=$1; shift
  if [ -f "$dir/mccode.sim" ] && grep -q "values:" "$dir/mccode.sim" 2>/dev/null; then
    return 0
  fi
  "$@"
}}
"""

COMPILE_GUARDED = """# Compile once. Guarded with a plain mkdir-lock (atomic on every POSIX
# filesystem, no dependency on flock being installed -- tested locally with
# 20 truly concurrent processes racing for it: exactly one compiled, the
# other 19 waited and then correctly saw the finished binary) so that
# multiple array tasks starting at once can't corrupt eqsans_cylinder.out by
# compiling it simultaneously.
if [ ! -x eqsans_cylinder.out ]; then
  if mkdir eqsans_compile.lockdir 2>/dev/null; then
    [ -x eqsans_cylinder.out ] || mcrun -c -n0 --format=McStas eqsans_cylinder.instr
    rmdir eqsans_compile.lockdir
  else
    while [ -d eqsans_compile.lockdir ]; do sleep 2; done
  fi
fi
"""

SINGLE_HEADER = """#!/bin/bash
#SBATCH --job-name=eqsans_ml_%j
#SBATCH --error=eqsans_ml_%j.err
#SBATCH --output=eqsans_ml_%j.out
#SBATCH --nodes 1
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={jobs}
#SBATCH --time {time}
#SBATCH --account m26216
set -euo pipefail

""" + COMMON_SETUP + COMPILE_GUARDED

ARRAY_HEADER = """#!/bin/bash
#SBATCH --job-name=eqsans_ml_%A_%a
#SBATCH --error=eqsans_ml_%A_%a.err
#SBATCH --output=eqsans_ml_%A_%a.out
#SBATCH --nodes 1
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={jobs}
#SBATCH --time {time}
#SBATCH --account m26216
#SBATCH --array=0-{last_task}
set -euo pipefail

""" + COMMON_SETUP + COMPILE_GUARDED + """
bash {chunk_prefix}_${{SLURM_ARRAY_TASK_ID}}.sh
"""

CHUNK_HEADER = """#!/bin/bash
# Chunk file: one array task's share of the campaign. Runnable standalone
# too (bash {chunk_name} from inside Group2/) -- it does not depend on being
# invoked through the array wrapper; the compile guard below is a cheap
# no-op if the wrapper already built the binary.
set -euo pipefail
""" + COMMON_SETUP + COMPILE_GUARDED


def lhs(n, bounds, seed):
    u = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n=n)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return qmc.scale(u, lo, hi)


def build_design(n, seed):
    rng = np.random.default_rng(seed)
    n_aligned = int(round(n * ALIGNED_FRACTION))
    n_mis = n - n_aligned
    gxgy = np.vstack([np.zeros((n_aligned, 2)), lhs(n_mis, [MISALIGN_RANGE_MM] * 2, seed)])
    aligned = np.array([True] * n_aligned + [False] * n_mis)
    order = rng.permutation(n)
    gxgy, aligned = gxgy[order], aligned[order]

    u = lhs(n, [(0, 1)] * 5, seed + 1)
    radius = np.exp(np.log(RADIUS_RANGE[0]) + u[:, 0] * math.log(RADIUS_RANGE[1] / RADIUS_RANGE[0]))
    length = np.exp(np.log(LENGTH_RANGE[0]) + u[:, 1] * math.log(LENGTH_RANGE[1] / LENGTH_RANGE[0]))
    pd_radius = PD_RANGE[0] + u[:, 2] * (PD_RANGE[1] - PD_RANGE[0])
    pd_length = PD_RANGE[0] + u[:, 3] * (PD_RANGE[1] - PD_RANGE[0])
    sld_solvent = SLD_SOLVENT_RANGE[0] + u[:, 4] * (SLD_SOLVENT_RANGE[1] - SLD_SOLVENT_RANGE[0])

    names = list(MATERIALS)
    material = [names[i] for i in rng.integers(0, len(names), size=n)]
    sld_material = np.array([MATERIALS[m] for m in material])

    rows = []
    for i in range(n):
        rows.append(dict(
            instance_id=f"eqs_{i:06d}", gx_mm=round(float(gxgy[i, 0]), 4),
            gy_mm=round(float(gxgy[i, 1]), 4), aligned=bool(aligned[i]),
            material=material[i], sld_material=round(float(sld_material[i]), 4),
            sld_solvent=round(float(sld_solvent[i]), 4),
            contrast=round(float(sld_material[i] - sld_solvent[i]), 4),
            radius=round(float(radius[i]), 4), length=round(float(length[i]), 4),
            pd_radius=round(float(pd_radius[i]), 5), pd_length=round(float(pd_length[i]), 5),
        ))
    return rows


def manifest_rows(design):
    """Expand each instance into its two runs (sample_on 0/1) for manifest.json,
    used later by train_regressor.py -- not needed by the .sh script itself."""
    rows = []
    for r in design:
        for sample_on in (0, 1):
            row = dict(r, run_id=f"{r['instance_id']}_s{sample_on}", sample_on=sample_on)
            row["targets_valid"] = {
                "gx_mm": True, "gy_mm": True,
                "radius": bool(sample_on), "length": bool(sample_on),
                "pd_radius": bool(sample_on), "pd_length": bool(sample_on),
                "contrast": bool(sample_on),
            }
            rows.append(row)
    return rows


SEC_PER_RUN = 4.5  # measured on this instrument at ncount=2e6, see chat history
CLUSTER_TOTAL_CORES = 336   # juliet2,3,4: 112 cores each (sinfo -o "%c" -p mesonet)
CLUSTER_TOTAL_GPUS = 23     # juliet3: a100:7, juliet[2,4]: a100:8 each (sinfo -o "%G")


def run_line(row, ncount):
    return (
        f"./eqsans_cylinder.out --ncount={ncount} "
        f"--dir=$DATA_DIR/{row['run_id']} --format=McStas --bufsiz=10000000 "
        f"gx={row['gx_mm']/1000.0} gy={row['gy_mm']/1000.0} sample_on={row['sample_on']} "
        f"radius={row['radius']} length={row['length']} "
        f"pd_radius={row['pd_radius']} pd_length={row['pd_length']} "
        f"sld={row['sld_material']} sld_solvent={row['sld_solvent']} "
        f"model_scale=1 model_abs=0"
    )


def write_batches(f, rows, ncount, jobs):
    for batch_start in range(0, len(rows), jobs):
        batch = rows[batch_start:batch_start + jobs]
        for row in batch:
            f.write(f"run_if_needed $DATA_DIR/{row['run_id']} {run_line(row, ncount)} &\n")
        f.write("wait\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="instrument/sample instances (x2 runs for sample_on)")
    ap.add_argument("--ncount", default="2e6")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default="submit_smoke.sh")
    ap.add_argument("--data-dir", default="dataset_smoke")
    ap.add_argument("--time", default="03:00:00", help="SBATCH --time budget")
    ap.add_argument("--jobs", type=int, default=16,
                    help="parallel runs at once per SLURM task/job; also requested as "
                         "--cpus-per-task.")
    ap.add_argument("--array-tasks", type=int, default=0,
                    help="if >0, split into this many SLURM array tasks (spread across "
                         "mesonet's 3 nodes) instead of one single-node job. 0 = single job.")
    args = ap.parse_args()

    project_dir = os.getcwd()
    if not os.path.isfile(os.path.join(project_dir, "eqsans_cylinder.instr")):
        sys.exit(
            f"eqsans_cylinder.instr not found in {project_dir}\n"
            f"Run this script from inside Group2/ (cd there first) -- the "
            f"generated .sh file hardcodes this directory as its 'cd' target, "
            f"so getting it right here is what makes the job find the "
            f"instrument later."
        )

    design = build_design(args.n, args.seed)
    rows = manifest_rows(design)
    n_aligned = sum(r["aligned"] for r in design)

    manifest_path = args.out.rsplit(".", 1)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(rows, f, indent=1)

    common_kwargs = dict(time=args.time, data_dir=args.data_dir, project_dir=project_dir, jobs=args.jobs)

    if args.array_tasks <= 0:
        # --- single job, internally batched (original design) -------------
        with open(args.out, "w") as f:
            f.write(SINGLE_HEADER.format(**common_kwargs))
            f.write(f"DATA_DIR={args.data_dir}\n\n")
            write_batches(f, rows, args.ncount, args.jobs)

        eta_min = len(rows) * SEC_PER_RUN / args.jobs / 60
        print(f"[make_run_script] {args.n} instances -> {len(rows)} runs, {args.jobs}-way parallel, 1 job")
        print(f"[make_run_script] aligned: {n_aligned} ({100*n_aligned/args.n:.0f}%)")
        print(f"[make_run_script] estimated wall-clock: ~{eta_min:.0f} min "
              f"(vs ~{len(rows)*SEC_PER_RUN/60:.0f} min sequential)")
        print(f"[make_run_script] wrote {args.out} ({len(rows)} run lines, "
              f"{math.ceil(len(rows)/args.jobs)} batches of <= {args.jobs})")
        print(f"[make_run_script] wrote {manifest_path} (for train_regressor.py)")
        print(f"\nOn Juliet:  sbatch {args.out}")

    else:
        # --- SLURM job array: spread across mesonet's multiple nodes -------
        T = args.array_tasks
        total_workers = T * args.jobs
        if total_workers > CLUSTER_TOTAL_CORES:
            print(f"[make_run_script] WARNING: {T} tasks x {args.jobs} jobs = {total_workers} "
                  f"concurrent workers > {CLUSTER_TOTAL_CORES} total cores on mesonet. "
                  f"Some tasks will queue behind others even with nothing else running.")
        if T > CLUSTER_TOTAL_GPUS:
            print(f"[make_run_script] WARNING: {T} array tasks each request --gres=gpu:1, "
                  f"but mesonet only has {CLUSTER_TOTAL_GPUS} GPUs total. "
                  f"At most {CLUSTER_TOTAL_GPUS} tasks can run at once regardless of --n.")

        chunk_prefix = args.out.rsplit(".", 1)[0] + "_chunk"
        chunk_size = math.ceil(len(rows) / T)
        for t in range(T):
            chunk_rows = rows[t * chunk_size:(t + 1) * chunk_size]
            chunk_name = f"{chunk_prefix}_{t}.sh"
            with open(chunk_name, "w") as f:
                f.write(CHUNK_HEADER.format(chunk_name=chunk_name, **common_kwargs))
                f.write(f"DATA_DIR={args.data_dir}\n\n")
                write_batches(f, chunk_rows, args.ncount, args.jobs)

        with open(args.out, "w") as f:
            f.write(ARRAY_HEADER.format(last_task=T - 1, chunk_prefix=chunk_prefix, **common_kwargs))

        per_task_runs = chunk_size
        eta_min = per_task_runs * SEC_PER_RUN / args.jobs / 60
        print(f"[make_run_script] {args.n} instances -> {len(rows)} runs, "
              f"split into {T} array tasks of <= {chunk_size} runs each")
        print(f"[make_run_script] aligned: {n_aligned} ({100*n_aligned/args.n:.0f}%)")
        print(f"[make_run_script] {args.jobs}-way parallel per task -> {total_workers} workers "
              f"if all {T} tasks run concurrently (cluster has {CLUSTER_TOTAL_CORES} cores, "
              f"{CLUSTER_TOTAL_GPUS} GPUs)")
        print(f"[make_run_script] estimated wall-clock per task: ~{eta_min:.0f} min "
              f"(campaign total if tasks run concurrently: about the same, not {T}x)")
        print(f"[make_run_script] wrote {args.out} (array wrapper, --array=0-{T-1})")
        print(f"[make_run_script] wrote {T} chunk files: {chunk_prefix}_0.sh .. {chunk_prefix}_{T-1}.sh")
        print(f"[make_run_script] wrote {manifest_path} (for train_regressor.py)")
        print(f"\nOn Juliet:  sbatch {args.out}")
