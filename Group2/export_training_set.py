#!/usr/bin/env python3
"""Turn the cylinder sweep into a regression training set, and test it.

  python3 export_training_set.py            # export + baseline regression
  python3 export_training_set.py export     # export only

Writes cylinder_trainingset.npz holding
    q      (n_q,)            common q grid [1/AA]
    X      (n_samples, n_q)  log10 I(q), interpolated onto that grid
    X_err  (n_samples, n_q)  relative error per bin
    y      (n_samples, n_p)  target parameters
    y_names                  parameter names
    ids                      sample ids

The baseline fit is deliberately simple (random forest on log I(q)). It is not
meant to be the final model -- it is there to show which targets carry enough
signal to be learned at all.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset_cylinder")
MANIFEST = os.path.join(DATA, "manifest.csv")
OUTNPZ = os.path.join(HERE, "cylinder_trainingset.npz")

TARGETS = ["radius", "length", "pd_radius", "pd_length", "sld"]
QMIN, QMAX, NQ = 0.003, 0.12, 60
QGRID = np.logspace(np.log10(QMIN), np.log10(QMAX), NQ)


def load_iq(folder):
    f = os.path.join(folder, "qdet.dat")
    if not os.path.exists(f):
        return None
    d = np.loadtxt(f, comments="#")
    q, I, E = d[:, 0], d[:, 1], d[:, 2]
    g = (q > 0) & (I > 0)
    if g.sum() < 10:
        return None
    Ii = np.interp(QGRID, q[g], I[g])
    Ei = np.interp(QGRID, q[g], E[g] / I[g])
    return Ii, Ei


def export():
    rows = list(csv.DictReader(open(MANIFEST)))
    X, Xe, y, ids = [], [], [], []
    skipped = 0
    for r in rows:
        got = load_iq(os.path.join(DATA, r["id"]))
        if got is None:
            skipped += 1
            continue
        Ii, Ei = got
        X.append(np.log10(Ii))
        Xe.append(Ei)
        y.append([float(r[t]) for t in TARGETS])
        ids.append(r["id"])
    X = np.array(X); Xe = np.array(Xe); y = np.array(y)
    np.savez_compressed(OUTNPZ, q=QGRID, X=X, X_err=Xe, y=y,
                        y_names=np.array(TARGETS), ids=np.array(ids))
    print("exported %d samples (%d skipped/unfinished), %d q-bins -> %s"
          % (len(X), skipped, NQ, os.path.basename(OUTNPZ)))
    print("median per-bin relative error: %.1f%%" % (100 * np.median(Xe)))
    return QGRID, X, y, TARGETS


def baseline(X, y, names):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    print("\nbaseline random forest, train %d / test %d" % (len(Xtr), len(Xte)))
    print("%-12s %8s %10s   %s" % ("target", "R2", "rel.err", "verdict"))

    # Absolute intensity is retained (log I), so amplitude-only parameters
    # such as sld remain learnable.
    for j, nm in enumerate(names):
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        m.fit(Xtr, ytr[:, j])
        p = m.predict(Xte)
        r2 = r2_score(yte[:, j], p)
        rel = np.median(np.abs(p - yte[:, j]) / np.maximum(np.abs(yte[:, j]), 1e-9))
        verdict = ("good" if r2 > 0.9 else
                   "usable" if r2 > 0.7 else
                   "weak" if r2 > 0.3 else "NOT learnable")
        print("%-12s %8.3f %9.1f%%   %s" % (nm, r2, 100 * rel, verdict))


if __name__ == "__main__":
    q, X, y, names = export()
    if len(sys.argv) < 2 or sys.argv[1] != "export":
        baseline(X, y, names)
