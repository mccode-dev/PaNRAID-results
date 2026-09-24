#!/usr/bin/env python3
"""Draw a to-scale (along z) schematic of eqsans_cylinder.instr as instrument_schematic.svg.

Distances and detector size are read from the instrument's DECLARE block; guide/
collimator lengths and apertures are the values in its active TRACE section.
Transverse sizes are drawn 4x larger than the z scale so millimetre-scale guides
are visible. Pure standard library; does not run McStas.

Usage:  python3 make_schematic.py [eqsans_cylinder.instr] [instrument_schematic.svg]
"""
import re
import sys

instr = sys.argv[1] if len(sys.argv) > 1 else "eqsans_cylinder.instr"
out = sys.argv[2] if len(sys.argv) > 2 else "instrument_schematic.svg"
text = open(instr).read()


def declared(name):
    m = re.search(rf"double\s+{name}\s*=\s*([0-9.]+)\s*;", text)
    if not m:
        raise SystemExit(f"could not find `double {name} = ...;` in {instr}")
    return float(m.group(1))


L1, L2 = declared("L1"), declared("L2")
GUIDE_START = declared("guide_start")
CH = [declared(f"chopper{i}_distance") for i in (1, 2, 3)]
DET_W, DET_H = declared("detector_width"), declared("detector_height")
Z_DET = L1 + L2

# guide segments (start, length) in metres from the moderator -- TRACE section
OPEN_L, COLL_L, G0_L, G1_L, G2_L, G3_L = 3.3, 0.27, 1.0, 2.0, 1.6, 0.468
opening = (GUIDE_START, OPEN_L)
collim = (GUIDE_START + OPEN_L, COLL_L)
guide0 = (collim[0] + COLL_L, G0_L)
guides = [guide0, (CH[0], G1_L), (CH[1], G2_L), (CH[2], G3_L)]
SLITS = [10.08, 10.18, 10.28]

W, H = 960, 850
X0, SZ = 48.0, 36.0            # px at z=0, px per metre along the beam
TR = 4 * SZ                    # px per metre transverse (4x exaggerated)
X = lambda z: X0 + SZ * z

INK, MUTED, LINE = "#1b1b1f", "#5b5b63", "#d9d5cb"
ACC, GUIDE, DET, PANEL = "#2f6f4f", "#cdbf8f", "#b3564a", "#ffffff"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
el = []


def add(s):
    el.append(s)


def text_(x, y, s, size=11, fill=INK, weight=400, anchor="start"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{s}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1, extra=""):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def line(x1, y1, x2, y2, stroke=INK, sw=1.2, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" {d}/>')


def marker(x, y, n):
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.5" fill="{ACC}"/>')
    text_(x, y + 4, str(n), 11, "#fff", 700, "middle")


def view(yc, half_det, mod_h, title, marks):
    text_(X0, yc - 122 if marks else yc - 108, title, 12, MUTED, 700)
    # scattered-neutron fan, sample -> detector
    add(f'<polygon points="{X(L1):.1f},{yc} {X(Z_DET):.1f},{yc - half_det:.1f} {X(Z_DET):.1f},{yc + half_det:.1f}" '
        f'fill="{ACC}" opacity="0.10"/>')
    line(X0, yc, X(Z_DET), yc, MUTED, 1.2, "4 4")
    rect(X0 - 4, yc - mod_h / 2, 8, mod_h, INK)
    gh = 0.04 * TR / 2
    for z0, l in [opening] + guides:
        rect(X(z0), yc - gh, SZ * l, 2 * gh, GUIDE, "#8d8360", 0.8)
    rect(X(collim[0]), yc - gh, SZ * COLL_L, 2 * gh, "#b9b9b9", "#6f6f6f", 0.8)
    for z in CH:
        rect(X(z) - 2.5, yc - 0.3 * TR, 5, 0.6 * TR, ACC)
    for z in SLITS:
        for s in (-1, 1):
            line(X(z), yc + s * 0.02 * TR, X(z), yc + s * (0.02 * TR + 9), INK, 1.6)
    add(f'<circle cx="{X(L1):.1f}" cy="{yc}" r="4.2" fill="{ACC}" stroke="{INK}" stroke-width="1"/>')
    rect(X(Z_DET) - 4, yc - half_det, 8, 2 * half_det, "#f4e3df", DET, 2)
    for k in range(1, 10):
        yy = yc - half_det + k * (2 * half_det) / 10
        line(X(Z_DET) - 4, yy, X(Z_DET) + 4, yy, DET, 0.6)
    for x, yrow, n, ytarget in marks:
        line(x, yrow + 8.5, x, ytarget, MUTED, 0.8, "2 2")
        marker(x, yrow, n)


# ---------------------------------------------------------------- header ----
text_(X0, 26, "EQ-SANS cylinder instrument (eqsans_cylinder.instr)", 15, INK, 700)
text_(X0, 44, "z along the beam is to scale; transverse sizes drawn 4&#215; larger. Neutron band 2&#8211;8 &#197; "
      "(SNS_source, 1.28&#8211;20.45 meV).", 11, MUTED)

YC1, YC2 = 190, 500
rowA, rowB = 112, 92
marks = [
    (X0, rowA, 1, YC1 - 9),
    (X(opening[0] + OPEN_L / 2), rowA, 2, YC1 - 4),
    (X(collim[0] + COLL_L / 2), rowB, 3, YC1 - 4),
    (X(guide0[0] + G0_L / 2), rowA, 4, YC1 - 4),
    (X(CH[0]), rowB, 5, YC1 - 0.3 * TR),
    (X(CH[0] + G1_L / 2), rowA, 6, YC1 - 4),
    (X(CH[1]), rowB, 7, YC1 - 0.3 * TR),
    (X(CH[1] + G2_L / 2), rowA, 8, YC1 - 4),
    (X(CH[2]), rowB, 9, YC1 - 0.3 * TR),
    (X(SLITS[1]), rowA, 10, YC1 - 12),
    (X(L1), rowA, 11, YC1 - 6),
]
view(YC1, DET_H / 2 * TR, 0.12 * TR, "SIDE VIEW (y vs z): detector height 1.4 m", marks)
marker(X(Z_DET) + 30, YC1, 12)
line(X(Z_DET) + 21.5, YC1, X(Z_DET) + 5, YC1, MUTED, 0.8, "2 2")
view(YC2, DET_W / 2 * TR, 0.10 * TR, "TOP VIEW (x vs z): detector width 1.0 m", [])

# ------------------------------------ inset: misalignment of the guide -------
rect(430, 306, 350, 110, "#faf9f6", LINE, 1, 'rx="8"')
cx, cy, s = 490.0, 374.0, 60.0    # 40 mm guide -> 60 px  (1.5 px/mm)
text_(444, 325, "Misalignment: guide entrance (2), looking downstream", 11, INK, 700)
rect(cx - s / 2, cy - s / 2, s, s, "none", MUTED, 1.2, 'stroke-dasharray="4 3"')
gx, gy = 15.0, -10.5              # +10 mm, +7 mm (y up)
rect(cx - s / 2 + gx, cy - s / 2 + gy, s, s, ACC, ACC, 2, 'fill-opacity="0.18"')
line(cx, cy, cx + gx, cy, DET, 1.8)
line(cx + gx, cy, cx + gx, cy + gy, DET, 1.8)
text_(cx + gx / 2, cy + 14, "gx", 11, DET, 700, "middle")
text_(cx + gx + 5, cy + gy / 2 + 4, "gy", 11, DET, 700)
text_(575, 352, "dashed: nominal (0, 0)", 10.5, MUTED)
text_(575, 369, "solid: shifted by (gx, gy)", 10.5, MUTED)
text_(575, 386, "gx, gy in &#177;20 mm;", 10.5, MUTED)
text_(575, 401, "15% of instances exactly 0", 10.5, MUTED)

# ---------------------------------------------------------------- ruler ----
YR = YC2 + DET_W / 2 * TR + 48
line(X0, YR, X(Z_DET), YR, INK, 1)
for z in (0, 5, 10, 15, 20, Z_DET):
    line(X(z), YR - 4, X(z), YR + 4, INK, 1)
    text_(X(z), YR + 18, f"{z:g} m" if z != Z_DET else f"{Z_DET:.2f} m", 10.5, MUTED, 400, "middle")
YD = YR + 44
for z0, z1, label in [(0, L1, f"L1 = {L1:.2f} m  (moderator &#8594; sample)"),
                      (L1, Z_DET, f"L2 = {L2:.2f} m  (sample &#8594; detector)")]:
    line(X(z0) + 2, YD, X(z1) - 2, YD, ACC, 1.4)
    for xx, d in ((X(z0) + 2, 1), (X(z1) - 2, -1)):
        add(f'<polyline points="{xx + 6 * d:.1f},{YD - 4} {xx:.1f},{YD} {xx + 6 * d:.1f},{YD + 4}" '
            f'fill="none" stroke="{ACC}" stroke-width="1.4"/>')
    text_((X(z0) + X(z1)) / 2, YD - 7, label, 11, ACC, 700, "middle")

# --------------------------------------------------------------- legend ----
legend = [
    (1, "SNS moderator", f"z = 0; 100 &#215; 120 mm source"),
    (2, "Guide &#8220;opening&#8221;", f"z = {opening[0]:.2f}&#8211;{opening[0] + OPEN_L:.2f} m; 40 &#215; 40 mm; carries (gx, gy)"),
    (3, "Collimator", f"z = {collim[0]:.2f}&#8211;{collim[0] + COLL_L:.2f} m; &#177;20 mm window"),
    (4, "Guide 0", f"z = {guide0[0]:.2f}&#8211;{guide0[0] + G0_L:.2f} m"),
    (5, "Chopper 1", f"z = {CH[0]:.2f} m; R = 0.30 m, 60 Hz, window 131.8&#176;"),
    (6, "Guide 1", f"z = {CH[0]:.2f}&#8211;{CH[0] + G1_L:.2f} m"),
    (7, "Chopper 2", f"z = {CH[1]:.2f} m; 60 Hz, window 160&#176;"),
    (8, "Guide 2", f"z = {CH[1]:.2f}&#8211;{CH[1] + G2_L:.2f} m"),
    (9, "Chopper 3 + Guide 3", f"z = {CH[2]:.2f} m; window 204&#176;; guide to {CH[2] + G3_L:.2f} m"),
    (10, "Slits (3)", "z = 10.08, 10.18, 10.28 m; &#177;20 mm"),
    (11, "Sample + aperture", f"z = {L1:.2f} m; SasView cylinder cell"),
    (12, "Detector", f"z = {Z_DET:.2f} m; {DET_W:g} &#215; {DET_H:g} m, 256 &#215; 256 px; I(q) monitor"),
]
YL = YD + 34
for i, (n, name, sub) in enumerate(legend):
    col, row = divmod(i, 6)
    x, y = X0 + col * 455, YL + row * 22
    marker(x + 9, y - 4, n)
    text_(x + 24, y, f"<tspan font-weight='700'>{name}</tspan> &#8212; {sub}", 11, INK)
text_(X0, YL + 6 * 22 + 8, "All guides: supermirror m = 3 (R0 0.99, Qc 0.0219 &#197;&#8315;&#185;, &#945; 5.707), gravity on. "
      "Disc-chopper windows are the theta_0 values in the instrument file.", 10.5, MUTED)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">'
       f'<rect width="{W}" height="{H}" fill="{PANEL}"/>' + "".join(el) + "</svg>")
open(out, "w").write(svg)
print(f"wrote {out}  (L1={L1}, L2={L2}, choppers={CH}, detector={DET_W}x{DET_H})")
