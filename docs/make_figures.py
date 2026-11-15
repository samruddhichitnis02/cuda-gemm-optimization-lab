"""Generate the animated SVG figures used in README.md.

Every number here is copied from results/ or Profiling_Results/ (see README).
Run:  python docs/make_figures.py      -> writes docs/images/*.svg
No dependencies beyond the Python standard library.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
FONT = 'system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif'

# Light / dark tokens. Kernel colors k1..k4 are one blue ramp (validated ordinal ramp,
# light end clears the surface in both modes); "ref" is the gray used for reference points.
STYLE = """
<style>
svg{--surface:#fcfcfb;--wash:#f0efec;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.12);
--k1:#86b6ef;--k2:#5598e7;--k3:#2a78d6;--k4:#1c5cab;--ref:#a8a7a0;
--s1:#2a78d6;--s2:#eb6834;--hl:rgba(42,120,214,.22)}
@media (prefers-color-scheme:dark){svg{--surface:#1a1a19;--wash:#262624;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#4a4a46;--border:rgba(255,255,255,.12);
--k1:#184f95;--k2:#256abf;--k3:#3987e5;--k4:#86b6ef;--ref:#6b6a65;
--s1:#3987e5;--s2:#d95926;--hl:rgba(57,135,229,.30)}}
text{font-family:FONT;fill:var(--ink)}
.title{font-size:22px;font-weight:700}
.sub{font-size:14px;fill:var(--ink2)}
.h{font-size:15px;font-weight:700}
.lbl{font-size:13px;fill:var(--ink2)}
.val{font-size:13px;font-weight:600;fill:var(--ink)}
.sm{font-size:12px;fill:var(--muted)}
.tick{font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}
.big{font-size:34px;font-weight:800;fill:var(--ink)}
.bg{fill:var(--surface);stroke:var(--border)}
.wash{fill:var(--wash)}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.cell{fill:var(--surface);stroke:var(--axis);stroke-width:1}
.box{fill:var(--wash);stroke:var(--axis);stroke-width:1.5}
.track{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:4 5;fill:none}
.k1{fill:var(--k1)}.k2{fill:var(--k2)}.k3{fill:var(--k3)}.k4{fill:var(--k4)}
.ref{fill:var(--ref)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}
.hl{fill:var(--hl);stroke:var(--k3);stroke-width:2}
.growx{transform-box:fill-box;transform-origin:left center;animation:growx 1.1s cubic-bezier(.2,.7,.2,1) backwards}
.growy{transform-box:fill-box;transform-origin:center bottom;animation:growy 1.1s cubic-bezier(.2,.7,.2,1) backwards}
.fade{animation:fade .6s ease-out backwards}
.rise{animation:rise .7s cubic-bezier(.2,.7,.2,1) backwards}
@keyframes growx{from{transform:scaleX(0)}}
@keyframes growy{from{transform:scaleY(0)}}
@keyframes fade{from{opacity:0}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}
EXTRA
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(name, w, h, body, label, extra_css=""):
    css = STYLE.replace("FONT", FONT).replace("EXTRA", extra_css)
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'
        f"<title>{esc(label)}</title>{css}"
        f'<rect class="bg" x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14"/>'
        f"{body}</svg>"
    )
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", name)


def text(x, y, s, cls="lbl", anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" {extra}>{esc(s)}</text>')


def hbar(x, y, w, h, cls, delay=0.0, r=4):
    w = max(w, 2)
    r = min(r, w / 2, h / 2)
    d = (f"M{x:.1f},{y:.1f} H{x+w-r:.1f} Q{x+w:.1f},{y:.1f} {x+w:.1f},{y+r:.1f} "
         f"V{y+h-r:.1f} Q{x+w:.1f},{y+h:.1f} {x+w-r:.1f},{y+h:.1f} H{x:.1f} Z")
    return f'<path d="{d}" class="{cls} growx" style="animation-delay:{delay:.2f}s"/>'


def vbar(x, base, w, h, cls, delay=0.0, r=4):
    h = max(h, 2)
    r = min(r, w / 2, h / 2)
    top = base - h
    d = (f"M{x:.1f},{base:.1f} V{top+r:.1f} Q{x:.1f},{top:.1f} {x+r:.1f},{top:.1f} "
         f"H{x+w-r:.1f} Q{x+w:.1f},{top:.1f} {x+w:.1f},{top+r:.1f} V{base:.1f} Z")
    return f'<path d="{d}" class="{cls} growy" style="animation-delay:{delay:.2f}s"/>'


def header(title, sub):
    return text(32, 40, title, "title") + text(32, 68, sub, "sub")


# ---------------------------------------------------------------- hero
def hero():
    W, H = 960, 250
    b = text(32, 46, "CUDA GEMM Optimization Lab", "title", extra='style="font-size:28px"')
    b += text(32, 80, "Making matrix multiplication fast on a laptop GPU (NVIDIA RTX 3060), one step at a time", "sub")
    tiles = [
        ("5.2×", "faster than my first", "GPU kernel (N = 4096)"),
        ("66%", "of NVIDIA's own cuBLAS", "library speed (N = 4096)"),
        ("~480×", "faster than one", "CPU core"),
        ("1.56×", "faster than PyTorch with", "a fused Triton kernel"),
    ]
    tw, gap = 212, 16
    for i, (big, l1, l2) in enumerate(tiles):
        x = 32 + i * (tw + gap)
        d = 0.25 + i * 0.25
        b += f'<g class="rise" style="animation-delay:{d:.2f}s">'
        b += f'<rect x="{x}" y="112" width="{tw}" height="110" rx="10" class="wash"/>'
        b += f'<rect x="{x}" y="112" width="5" height="110" rx="2" class="k{i+1 if i < 3 else 4}"/>'
        b += text(x + 22, 148, big, "big")
        b += text(x + 22, 182, l1, "lbl") + text(x + 22, 201, l2, "lbl")
        b += "</g>"
    svg("hero.svg", W, H, b, "CUDA GEMM Optimization Lab: 5.2x faster than naive, 66% of cuBLAS, about 480x faster than one CPU core, fused Triton kernel 1.56x faster than PyTorch")


# ---------------------------------------------------------------- matmul animation
def matmul():
    W, H = 960, 360
    cs, n = 38, 4
    ax, bx, cx, y0 = 140, 400, 660, 118
    step, hold = 0.6, 1.6
    T = n * n * step + hold
    css = ""
    b = header("What is matrix multiplication?",
               "Each answer box in C = one row of A times one column of B (multiply the pairs, then add them up)")
    for gx, name in ((ax, "A"), (bx, "B"), (cx, "C")):
        b += text(gx + cs * n / 2, y0 - 18, name, "h", "middle")
        for i in range(n):
            for j in range(n):
                b += f'<rect class="cell" x="{gx+j*cs}" y="{y0+i*cs}" width="{cs}" height="{cs}"/>'
    b += text((ax + cs * n + bx) / 2, y0 + cs * 2, "×", "title", "middle")
    b += text((bx + cs * n + cx) / 2, y0 + cs * 2, "=", "title", "middle")
    demo = 6  # the step shown when motion is reduced
    for s in range(n * n):
        i, j = divmod(s, n)
        a0 = s * step / T * 100
        a1 = (s + 1) * step / T * 100
        c0 = (s + 0.5) * step / T * 100
        css += (f"@keyframes h{s}{{0%,{a0:.2f}%{{opacity:0}}{a0+.01:.2f}%,{a1:.2f}%{{opacity:1}}"
                f"{a1+.01:.2f}%,100%{{opacity:0}}}}")
        css += (f"@keyframes c{s}{{0%,{c0:.2f}%{{opacity:0}}{c0+.01:.2f}%,96%{{opacity:1}}100%{{opacity:0}}}}")
        vis = 1 if s == demo else 0
        anim = f'style="animation:h{s} {T:.1f}s linear infinite" opacity="{vis}"'
        b += f'<rect class="hl" x="{ax}" y="{y0+i*cs}" width="{cs*n}" height="{cs}" rx="3" {anim}/>'
        b += f'<rect class="hl" x="{bx+j*cs}" y="{y0}" width="{cs}" height="{cs*n}" rx="3" {anim}/>'
        cvis = 1 if s <= demo else 0
        b += (f'<rect class="k3" x="{cx+j*cs+3}" y="{y0+i*cs+3}" width="{cs-6}" height="{cs-6}" rx="4" '
              f'style="animation:c{s} {T:.1f}s linear infinite" opacity="{cvis}"/>')
    b += text(W / 2, 300, "On a GPU, thousands of tiny workers (threads) fill in different answer boxes at the same time.", "lbl", "middle")
    b += text(W / 2, 324, "At N = 4096 that is 16.8 million answer boxes, each needing 4,096 multiply-adds.", "lbl", "middle")
    svg("matmul.svg", W, H, b, "Animation: each cell of C is computed from one row of A and one column of B", css)


# ---------------------------------------------------------------- strategy animation
def strategies():
    W, H = 960, 664
    css = ""
    b = header("Same math, smarter trips for data",
               "The GPU can do math much faster than it can fetch numbers. Each version cuts down the waiting.")
    gx0, gw = 40, 130          # global memory box
    sx0, sw = 410, 120         # shared memory box
    tx = 830                   # thread (worker) circle centre
    b += text(gx0 + gw / 2, 100, "Global memory", "h", "middle")
    b += text(gx0 + gw / 2, 119, "big but far & slow", "sm", "middle")
    b += text(sx0 + sw / 2, 100, "Shared memory", "h", "middle")
    b += text(sx0 + sw / 2, 119, "small, fast, shared by a team", "sm", "middle")
    b += text(tx, 100, "Thread + registers", "h", "middle")
    b += text(tx, 119, "registers = in its hands (fastest)", "sm", "middle")

    rows = [
        ("1 · Naive", "Each thread computes 1 answer and fetches every number from global memory itself.", 645, "k1", "naive"),
        ("2 · Shared-memory tiles", "A team copies a 32×32 tile into shared memory once, then everyone reads it from there.", 970, "k2", "tiled"),
        ("3 · Register tiling", "Each thread computes 16 answers (4×4) kept in registers, so every number it reads is reused 4 times.", 3173, "k3", "reg"),
        ("4 · + float4 loads", "Same plan, but numbers are grabbed 4 at a time: 5 → 2 load instructions per step.", 3375, "k4", "vec"),
    ]
    kid = 0

    def mover(dist, dur, delay, inner, cls_anim):
        nonlocal css, kid
        kid += 1
        name = f"m{kid}"
        css += (f"@keyframes {name}{{0%{{transform:translateX(0);opacity:0}}8%{{opacity:1}}"
                f"88%{{opacity:1}}100%{{transform:translateX({dist}px);opacity:0}}}}")
        return (f'<g style="animation:{name} {dur}s linear infinite;animation-delay:{delay:.2f}s">{inner}</g>')

    for r, (title, desc, gf, cls, kind) in enumerate(rows):
        y = 140 + r * 122
        ly = y + 82
        b += f'<line class="grid" x1="32" x2="{W-32}" y1="{y-4}" y2="{y-4}"/>'
        b += text(32, y + 18, title, "h")
        b += text(W - 32, y + 18, f"{gf:,} GFLOPS", "val", "end")
        b += text(32, y + 40, desc, "lbl")
        # boxes
        b += f'<rect class="box" x="{gx0}" y="{ly-22}" width="{gw}" height="44" rx="6"/>'
        b += text(gx0 + gw / 2, ly, "global", "sm", "middle")
        if kind == "naive":
            b += f'<line class="track" x1="{gx0+gw}" x2="{tx-24}" y1="{ly}" y2="{ly}"/>'
            b += text(sx0 + sw / 2, ly + 24, "(shared memory not used)", "sm", "middle")
        else:
            b += f'<line class="track" x1="{gx0+gw}" x2="{sx0}" y1="{ly}" y2="{ly}"/>'
            b += f'<line class="track" x1="{sx0+sw}" x2="{tx-24}" y1="{ly}" y2="{ly}"/>'
            b += f'<rect class="box" x="{sx0}" y="{ly-22}" width="{sw}" height="44" rx="6"/>'
            b += text(sx0 + sw / 2, ly, "shared", "sm", "middle")
        b += f'<circle cx="{tx}" cy="{ly}" r="20" class="{cls}"/>'
        # answers held by the thread
        if kind in ("reg", "vec"):
            for i in range(4):
                for j in range(4):
                    b += f'<rect class="{cls}" x="{tx+32+j*8}" y="{ly-16+i*8}" width="6.5" height="6.5" rx="1.5"/>'
            b += text(tx + 47, ly + 30, "16 answers", "sm", "middle")
        else:
            b += f'<rect class="{cls}" x="{tx+32}" y="{ly-4}" width="8" height="8" rx="1.5"/>'
            b += text(tx + 47, ly + 30, "1 answer", "sm", "middle")
        # data packets
        if kind == "naive":
            x0, dist, dur, cnt = gx0 + gw + 6, tx - 24 - (gx0 + gw) - 12, 2.8, 12
            for k in range(cnt):
                b += mover(dist, dur, -k * dur / cnt, f'<circle cx="{x0}" cy="{ly}" r="5" class="{cls}"/>', "")
        else:
            # tiles: global -> shared (big squares, slow)
            x0, dist, dur = gx0 + gw + 4, sx0 - (gx0 + gw) - 26, 3.2
            for k in range(2):
                b += mover(dist, dur, -k * dur / 2,
                           f'<rect x="{x0}" y="{ly-9}" width="18" height="18" rx="3" class="{cls}"/>', "")
            # shared -> thread
            x0, dist = sx0 + sw + 6, tx - 24 - (sx0 + sw) - 12
            if kind == "tiled":
                dur, cnt = 1.6, 10
                for k in range(cnt):
                    b += mover(dist, dur, -k * dur / cnt, f'<circle cx="{x0}" cy="{ly}" r="5" class="{cls}"/>', "")
            elif kind == "reg":
                dur, cnt = 1.6, 3
                for k in range(cnt):
                    b += mover(dist, dur, -k * dur / cnt, f'<circle cx="{x0}" cy="{ly}" r="5" class="{cls}"/>', "")
            else:
                dur, cnt = 1.6, 2
                bundle = "".join(f'<circle cx="{x0+dx}" cy="{ly+dy}" r="3.6" class="{cls}"/>'
                                 for dx, dy in ((0, -4.5), (9, -4.5), (0, 4.5), (9, 4.5)))
                for k in range(cnt):
                    b += mover(dist - 9, dur, -k * dur / cnt, bundle, "")
    b += text(W / 2, H - 22, "Dots = numbers moving. Fewer, shorter trips → threads spend more time doing math. Speeds measured at N = 4096.", "sm", "middle")
    svg("strategies.svg", W, H, b, "Animation: how naive, tiled, register-tiled and float4 kernels move data", css)


# ---------------------------------------------------------------- bar chart N=4096
def gflops_4096():
    W, H = 960, 430
    b = header("Speed of each version at N = 4096",
               "GFLOPS = billions of math operations per second · higher is better · blue = my kernels, gray = reference points")
    rows = [
        ("1 CPU core", 7, "ref", "7  (too small to see!)"),
        ("Naive", 645, "k1", "645  ·  1×"),
        ("Shared-memory tiles", 970, "k2", "970  ·  1.5×"),
        ("Register tiling", 3173, "k3", "3,173  ·  4.9×"),
        ("+ float4 loads (best)", 3375, "k4", "3,375  ·  5.2×"),
        ("NVIDIA cuBLAS", 5119, "ref", "5,119  ·  expert library"),
    ]
    x0, x1, vmax = 210, 800, 5500
    y0, rh, bh = 108, 42, 26
    sc = lambda v: (x1 - x0) * v / vmax
    yb = y0 + rh * len(rows)
    for t in range(0, 6000, 1000):
        x = x0 + sc(t)
        b += f'<line class="{"axis" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{y0-6}" y2="{yb}"/>'
        b += text(x, yb + 16, f"{t:,}", "tick", "middle")
    for i, (name, v, cls, lab) in enumerate(rows):
        y = y0 + i * rh + (rh - bh) / 2
        d = 0.2 + i * 0.15
        b += text(x0 - 12, y + bh / 2, name, "lbl", "end")
        b += hbar(x0, y, sc(v), bh, cls, d)
        b += f'<g class="fade" style="animation-delay:{d+0.8:.2f}s">' + text(x0 + max(sc(v), 2) + 10, y + bh / 2, lab, "val") + "</g>"
    b += text(x0 + sc(2750), yb + 38, "GFLOPS", "sm", "middle")
    svg("gflops_4096.svg", W, H, b, "Bar chart of GFLOPS at N=4096: CPU 7, naive 645, tiled 970, register tiling 3173, float4 3375, cuBLAS 5119")


# ---------------------------------------------------------------- grouped bars by N
def gflops_by_size():
    W, H = 960, 440
    b = header("Speed at three matrix sizes", "GFLOPS, median of 3 runs · higher is better")
    series = [("Naive", "k1"), ("Shared-memory tiles", "k2"), ("Register tiling", "k3"),
              ("+ float4 loads", "k4"), ("cuBLAS (reference)", "ref")]
    data = {1024: [670, 764, 3799, 4923, 5761], 2048: [668, 966, 3694, 3732, 7307],
            4096: [645, 970, 3173, 3375, 5119]}
    lx = 32
    for name, cls in series:
        b += f'<rect x="{lx}" y="94" width="12" height="12" rx="3" class="{cls}"/>'
        b += text(lx + 18, 100, name, "lbl")
        lx += 18 + len(name) * 7.2 + 26
    px0, px1, top, base, vmax = 80, 930, 138, 380, 8000
    sc = lambda v: (base - top) * v / vmax
    for t in range(0, 8001, 2000):
        y = base - sc(t)
        b += f'<line class="{"axis" if t == 0 else "grid"}" x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}"/>'
        b += text(px0 - 8, y, f"{t:,}", "tick", "end")
    gw = (px1 - px0) / 3
    bw, gap = 42, 2
    for g, (n, vals) in enumerate(data.items()):
        gx = px0 + g * gw + (gw - (5 * bw + 4 * gap)) / 2
        for k, v in enumerate(vals):
            x = gx + k * (bw + gap)
            d = 0.2 + g * 0.3 + k * 0.08
            b += vbar(x, base, bw, sc(v), series[k][1], d)
            b += f'<g class="fade" style="animation-delay:{d+0.8:.2f}s">' + text(x + bw / 2, base - sc(v) - 10, f"{v:,}", "tick", "middle", 'style="fill:var(--ink2)"') + "</g>"
        b += text(px0 + g * gw + gw / 2, base + 22, f"N = {n}", "h", "middle")
        b += text(px0 + g * gw + gw / 2, base + 42, f"({n} × {n} matrices)", "sm", "middle")
    svg("gflops_by_size.svg", W, H, b, "Grouped bar chart of GFLOPS for each kernel at N=1024, 2048 and 4096")


# ---------------------------------------------------------------- % of cuBLAS
def pct_cublas():
    W, H = 960, 240
    b = header("How close my best kernel gets to NVIDIA's cuBLAS",
               "My best hand-written kernel (+ float4 loads) as a share of cuBLAS speed, same session")
    rows = [(1024, 85), (2048, 51), (4096, 66)]
    x0, x1 = 150, 820
    for i, (n, p) in enumerate(rows):
        y = 104 + i * 40
        b += text(x0 - 14, y + 12, f"N = {n}", "lbl", "end")
        b += f'<rect x="{x0}" y="{y}" width="{x1-x0}" height="24" rx="5" class="wash"/>'
        b += hbar(x0, y, (x1 - x0) * p / 100, 24, "k4", 0.2 + i * 0.2, 5)
        b += text(x1 + 14, y + 12, f"{p}%", "val")
    b += f'<line x1="{x1}" x2="{x1}" y1="98" y2="{104+3*40-12}" class="axis" stroke-dasharray="3 3"/>'
    b += text(x1, 216, "100% = cuBLAS", "sm", "middle")
    svg("pct_cublas.svg", W, H, b, "Best kernel as percent of cuBLAS: 85% at N=1024, 51% at N=2048, 66% at N=4096")


# ---------------------------------------------------------------- profiler panels
def profile():
    W, H = 960, 414
    b = header("What the profiler showed (Nsight Compute, N = 2048)",
               "Why it got faster: less work spent fetching, more of the GPU's math power used")
    names = ["Naive", "Tiled", "Reg. tiling", "+ float4"]
    cls = ["k1", "k2", "k3", "k4"]
    panels = [
        (32, 460, "Instructions the GPU had to run", "millions · lower is better",
         [1467, 861, 465, 412], 1600, lambda v: f"{v:,} M"),
        (500, 928, "Share of the GPU's math power used", "% of FP32 peak · higher is better",
         [7, 7, 31, 43], 50, lambda v: ("~" if v == 7 else "") + f"{v}%"),
    ]
    top, base = 170, 340
    for p, (x0, x1, t, s, vals, vmax, fmt) in enumerate(panels):
        b += text(x0, 108, t, "h") + text(x0, 128, s, "sm")
        b += f'<line class="axis" x1="{x0}" x2="{x1}" y1="{base}" y2="{base}"/>'
        slot = (x1 - x0) / 4
        for k, v in enumerate(vals):
            bw = 56
            x = x0 + k * slot + (slot - bw) / 2
            h = (base - top) * v / vmax
            d = 0.2 + p * 0.5 + k * 0.12
            b += vbar(x, base, bw, h, cls[k], d)
            b += f'<g class="fade" style="animation-delay:{d+0.8:.2f}s">' + text(x + bw / 2, base - h - 12, fmt(v), "val", "middle") + "</g>"
            b += text(x + bw / 2, base + 18, names[k], "lbl", "middle")
    b += text(W / 2, 394, "The profiler locks the GPU clock at about 0.9 GHz, so its timings differ from the benchmark.", "sm", "middle")
    svg("profile.svg", W, H, b, "Profiler: instructions executed 1467, 861, 465, 412 million; share of FP32 peak about 7, 7, 31, 43 percent")


# ---------------------------------------------------------------- precision / tensor cores
def precision():
    W, H = 960, 300
    b = header("Smaller number formats unlock the Tensor Cores",
               "N = 4096 · TFLOPS = trillions of operations per second · blue = my kernel, gray = cuBLAS")
    rows = [("My best kernel (FP32)", 3.4, "k4", "3.4 TFLOPS"),
            ("cuBLAS FP32", 5.1, "ref", "5.1 TFLOPS  ·  1×"),
            ("cuBLAS TF32 (Tensor Cores)", 11.7, "ref", "11.7 TFLOPS  ·  2.3×"),
            ("cuBLAS FP16 (Tensor Cores)", 23.2, "ref", "23.2 TFLOPS  ·  4.5×")]
    x0, x1, vmax = 230, 760, 25
    y0, rh, bh = 102, 42, 26
    yb = y0 + rh * len(rows)
    for t in range(0, 26, 5):
        x = x0 + (x1 - x0) * t / vmax
        b += f'<line class="{"axis" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{y0-6}" y2="{yb}"/>'
        b += text(x, yb + 16, str(t), "tick", "middle")
    for i, (name, v, cls, lab) in enumerate(rows):
        y = y0 + i * rh + (rh - bh) / 2
        d = 0.2 + i * 0.2
        w = (x1 - x0) * v / vmax
        b += text(x0 - 12, y + bh / 2, name, "lbl", "end")
        b += hbar(x0, y, w, bh, cls, d)
        b += f'<g class="fade" style="animation-delay:{d+0.8:.2f}s">' + text(x0 + w + 10, y + bh / 2, lab, "val") + "</g>"
    svg("precision.svg", W, H, b, "cuBLAS at N=4096: FP32 5.1, TF32 11.7, FP16 23.2 TFLOPS; my best FP32 kernel 3.4")


# ---------------------------------------------------------------- fusion animation
def fusion():
    W, H = 960, 430
    T = 7.0
    b = header("Kernel fusion: fewer round trips to memory", "Computing  GELU(A × B + bias)  in two ways")
    css = ""
    # eager lane
    b += text(32, 106, "PyTorch eager: 3 separate kernels", "h")
    b += text(W - 32, 106, "memory trips: 3 writes + 2 reads", "val", "end")
    ops = [("matmul", 110), ("+ bias", 400), ("GELU", 690)]
    bw, by, my = 160, 126, 196
    for name, x in ops:
        b += f'<rect class="box" x="{x}" y="{by}" width="{bw}" height="40" rx="8"/>' + text(x + bw / 2, by + 20, name, "h", "middle")
    b += f'<rect x="90" y="{my}" width="760" height="30" rx="6" class="wash"/>' + text(862, my + 15, "GPU memory", "sm")
    pts = [(0, 0), (0, 45), (290, 45), (290, 0), (290, 45), (580, 45), (580, 0), (580, 45)]
    cx0, cy0 = 110 + bw / 2, by + 40
    n = len(pts)
    frames = []
    for i, (dx, dy) in enumerate(pts):
        pct = 5 + i * (80 / (n - 1))
        frames.append(f"{pct:.1f}%{{transform:translate({dx}px,{dy}px)}}")
    css += "@keyframes eag{0%{transform:translate(0,0);opacity:1}" + "".join(frames) + "92%{opacity:1}100%{transform:translate(580px,45px);opacity:0}}"
    b += f'<circle cx="{cx0}" cy="{cy0}" r="8" class="s2" style="animation:eag {T}s ease-in-out infinite"/>'
    # fused lane
    fy, fmy = 296, 366
    b += f'<line class="grid" x1="32" x2="{W-32}" y1="250" y2="250"/>'
    b += text(32, 276, "Triton fused: 1 kernel", "h")
    b += text(W - 32, 276, "memory trips: 1 write", "val", "end")
    b += f'<rect class="box" x="110" y="{fy}" width="740" height="40" rx="8"/>'
    b += text(480, fy + 20, "matmul  →  + bias  →  GELU     (all done in fast on-chip memory)", "h", "middle")
    b += f'<rect x="90" y="{fmy}" width="760" height="30" rx="6" class="wash"/>' + text(862, fmy + 15, "GPU memory", "sm")
    css += ("@keyframes fus{0%{transform:translate(0,0);opacity:1}5%{transform:translate(0,0)}"
            "45%{transform:translate(580px,0)}57%{transform:translate(580px,45px)}92%{opacity:1;transform:translate(580px,45px)}"
            "100%{opacity:0;transform:translate(580px,45px)}}")
    b += f'<circle cx="{cx0}" cy="{fy+40}" r="8" class="s1" style="animation:fus {T}s ease-in-out infinite"/>'
    b += text(W / 2, 414, "Each memory trip moves the whole N × N result. Fusing skips the in-between trips.", "sm", "middle")
    svg("fusion.svg", W, H, b, "Animation: eager PyTorch makes several memory round trips; the fused Triton kernel writes once", css)


# ---------------------------------------------------------------- fusion results
def fusion_results():
    W, H = 960, 430
    b = header("Speed compared with plain PyTorch (eager)",
               "FP16, median of 4 runs · above the dashed line = faster than PyTorch eager")
    series = [("torch.compile", "s2"), ("Triton fused kernel (mine)", "s1")]
    # speed vs eager = eager_ms / impl_ms (medians from results/fused_run1..4)
    data = {1024: (0.318 / 0.352, 0.318 / 0.204), 2048: (0.991 / 1.062, 0.991 / 0.932),
            4096: (6.509 / 6.460, 6.509 / 6.886)}
    lx = 32
    for name, cls in series:
        b += f'<rect x="{lx}" y="94" width="12" height="12" rx="3" class="{cls}"/>'
        b += text(lx + 18, 100, name, "lbl")
        lx += 18 + len(name) * 7.2 + 26
    b += f'<line x1="{lx}" x2="{lx+22}" y1="100" y2="100" stroke-width="1.5" stroke-dasharray="6 5" style="stroke:var(--ink2)"/>'
    b += text(lx + 30, 100, "PyTorch eager = 1.0× (baseline)", "lbl")
    px0, px1, top, base, vmax = 80, 930, 140, 370, 1.8
    sc = lambda v: (base - top) * v / vmax
    for t in (0, 0.5, 1.0, 1.5):
        y = base - sc(t)
        b += f'<line class="{"axis" if t == 0 else "grid"}" x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}"/>'
        b += text(px0 - 8, y, f"{t:.1f}×", "tick", "end")
    gw = (px1 - px0) / 3
    bw = 70
    for g, (n, vals) in enumerate(data.items()):
        gx = px0 + g * gw + (gw - (2 * bw + 2)) / 2
        for k, v in enumerate(vals):
            x = gx + k * (bw + 2)
            d = 0.2 + g * 0.3 + k * 0.12
            b += vbar(x, base, bw, sc(v), series[k][1], d)
            b += f'<g class="fade" style="animation-delay:{d+0.8:.2f}s">' + text(x + bw / 2, base - sc(v) + 16, f"{v:.2f}×", "val", "middle", 'style="fill:#fff"') + "</g>"
        b += text(px0 + g * gw + gw / 2, base + 22, f"N = {n}", "h", "middle")
    y1 = base - sc(1.0)
    b += f'<line x1="{px0}" x2="{px1}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="var(--ink2)" stroke-width="1.5" stroke-dasharray="6 5" style="stroke:var(--ink2)"/>'
    b += text(W / 2, 412, "Big win for small matrices, about even at 2048, slightly slower at 4096 where the matmul itself dominates.", "sm", "middle")
    svg("fusion_results.svg", W, H, b, "Speed vs PyTorch eager: Triton 1.56x, 1.06x, 0.95x; torch.compile 0.90x, 0.93x, 1.01x at N=1024, 2048, 4096")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (hero, matmul, strategies, gflops_4096, gflops_by_size, pct_cublas,
              profile, precision, fusion, fusion_results):
        f()
