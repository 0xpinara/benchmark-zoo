"""Figures for the paper, written straight into ``paper/figures/``.

Design constraints, in order of priority:

1. **Legible in greyscale at half column width.**  The paper is read in print
   and on screen, and a reviewer should not have to zoom.  Identity is therefore
   carried by line style and marker shape first, and by tone second; nothing
   depends on colour alone.  That also makes every figure safe under any form of
   colour vision deficiency by construction, rather than by validation.
2. **One vertical scale per panel.**  Two measures on different scales get two
   panels, never two axes, because the alignment of two scales is arbitrary and
   invents a relationship that is not in the data.
3. **Recessive axes and grid, thin marks, direct labels where there is room.**
   A legend is present whenever a panel has more than one series; a panel with
   one series is named by its title instead.
4. **No number printed on every point.**  Selective labels only, on the points a
   reader needs to read off.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..paths import FIGURES, ensure_dirs

# Greyscale tones, light to dark.  Used in fixed order, never cycled.
INK = "#111111"
TONES = ("#111111", "#555555", "#888888", "#bbbbbb")
STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")

# NeurIPS single column is about 5.5 inches; half width is 2.7.
FULL_WIDTH = 5.5
HALF_WIDTH = 2.7


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#666666",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.4,
            "grid.color": "#dddddd",
            "lines.linewidth": 1.2,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str) -> None:
    ensure_dirs()
    for ext in ("pdf", "png"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"wrote {FIGURES / (name + '.pdf')}")


def null_density(
    series: Dict[str, np.ndarray],
    name: str = "null_density",
    xlim: "tuple[float, float]" = (-5.0, 5.0),
    log_tail: bool = True,
) -> None:
    """The signature figure: measured null against the theoretical null.

    Left panel is the density on a linear scale, which shows the scale
    difference.  Right panel is the upper tail on a log scale, which is where a
    multiplicity correction actually operates and where a linear density shows
    nothing.  The theoretical standard normal is the reference in both, drawn as
    a light band so it recedes behind the measured curves.
    """
    setup_style()
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 1.9), gridspec_kw={"wspace": 0.28}
    )

    grid = np.linspace(xlim[0], xlim[1], 400)
    ax0.fill_between(
        grid, stats.norm.pdf(grid), color="#e8e8e8", zorder=0, linewidth=0
    )
    ax0.plot(grid, stats.norm.pdf(grid), color="#aaaaaa", lw=0.8, zorder=1)
    ax0.annotate(
        "theoretical\n$N(0,1)$",
        xy=(-2.3, stats.norm.pdf(2.3)),
        xytext=(-4.8, 0.155),
        color="#777777",
        fontsize=6.5,
        ha="left",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5,
                        shrinkA=1, shrinkB=1),
    )

    for i, (label, vals) in enumerate(series.items()):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        kde = stats.gaussian_kde(v)
        ax0.plot(
            grid,
            kde(grid),
            color=TONES[i % len(TONES)],
            linestyle=STYLES[i % len(STYLES)],
            label=f"{label} (sd {v.std(ddof=1):.2f})",
            zorder=3 + i,
        )
    ax0.set_xlim(*xlim)
    ax0.set_xlabel("$t$-statistic")
    ax0.set_ylabel("density")
    ax0.set_ylim(bottom=0)
    ax0.grid(axis="y")
    ax0.set_axisbelow(True)
    ax0.legend(frameon=False, loc="upper left", handlelength=1.8)

    cut = np.linspace(1.5, 5.0, 60)
    ax1.plot(
        cut,
        2.0 * stats.norm.sf(cut),
        color="#aaaaaa",
        lw=0.8,
        label="theoretical $N(0,1)$",
        zorder=1,
    )
    for i, (label, vals) in enumerate(series.items()):
        v = np.asarray(vals, dtype=float)
        v = np.abs(v[np.isfinite(v)])
        frac = np.array([(v > c).mean() for c in cut])
        keep = frac > 0
        ax1.plot(
            cut[keep],
            frac[keep],
            color=TONES[i % len(TONES)],
            linestyle=STYLES[i % len(STYLES)],
            label=label,
            zorder=3 + i,
        )
    if log_tail:
        ax1.set_yscale("log")
    ax1.set_xlabel("cutoff $c$")
    ax1.set_ylabel(r"$\Pr(|t| > c)$")
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower left", handlelength=1.8)
    save(fig, name)


def threshold_comparison(rows: Sequence[dict], name: str = "thresholds") -> None:
    """Dot plot of the thresholds each rule produces, one row per statistic.

    A dot plot rather than grouped bars: the quantity is a position on a common
    scale, not a magnitude from zero, and bars would imply the latter.
    """
    setup_style()
    labels = [f"{r['statistic']}, {r['weighting']}" for r in rows]
    keys = ("nominal", "calibrated", "max_t_joint", "bonferroni")
    pretty = {
        "nominal": "nominal 1.96",
        "calibrated": "calibrated, one test",
        "max_t_joint": "calibrated max-$t$ (correct)",
        "bonferroni": "Bonferroni, nominal null",
    }
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.30 * len(rows) + 0.95))
    y = np.arange(len(rows))
    # Every marker keeps a dark outline so that none of them disappears in
    # print; the fill distinguishes them further.
    fills = ("#111111", "white", "#777777", "white")
    for i, k in enumerate(keys):
        ax.scatter(
            [r[k] for r in rows],
            y,
            s=30,
            marker=MARKERS[i % len(MARKERS)],
            facecolors=fills[i % len(fills)],
            edgecolors=INK,
            linewidths=0.9,
            label=pretty[k],
            zorder=3,
        )
    for j, r in enumerate(rows):
        lo = min(r[k] for k in keys)
        hi = max(r[k] for k in keys)
        ax.plot([lo, hi], [j, j], color="#dddddd", lw=3.0, zorder=1, solid_capstyle="round")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("$|t|$ threshold at the 5 percent level")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    # Legend above the plot: the space there is empty, and putting it below
    # leaves a band of white between the axis label and the caption.
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4,
        handletextpad=0.4,
    )
    save(fig, name)


def survival_vs_n(
    curves: Dict[str, Sequence[dict]],
    name: str = "survival_vs_n",
    x_key: str = "n_total_tests",
    y_key: str = "survive_bonferroni",
    y_label: str = "predictors surviving",
    lower_bound: Optional[float] = None,
    lower_bound_label: str = "observed lower bound on $N$",
) -> None:
    """Survival as a function of the trial count, with any observed bound marked."""
    setup_style()
    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.35, 1.9))
    for i, (label, rows) in enumerate(curves.items()):
        xs = [r[x_key] for r in rows]
        ys = [r[y_key] for r in rows]
        ax.plot(
            xs,
            ys,
            color=TONES[i % len(TONES)],
            linestyle=STYLES[i % len(STYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markersize=3.5,
            label=label,
        )
    if lower_bound is not None:
        ax.axvline(lower_bound, color="#999999", lw=0.7, linestyle=(0, (1, 2)))
        # Anchored at the bottom of the axis and set outside the marker column,
        # so the label cannot sit on top of the leftmost data points.
        ax.annotate(
            lower_bound_label,
            xy=(lower_bound, ax.get_ylim()[0]),
            xytext=(4, 3),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=6,
            color="#777777",
        )
    ax.set_xscale("log")
    ax.set_xlabel("assumed number of trials $N$")
    ax.set_ylabel(y_label)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    if len(curves) > 1:
        ax.legend(frameon=False, handlelength=2.0)
    save(fig, name)


def saturation(rows: Sequence[dict], name: str = "saturation") -> None:
    """Null spread against remaining headroom, one point per benchmark.

    Two panels rather than one.  Left: both quantities on the accuracy scale,
    where the diagonal has a meaning -- a point on it is a benchmark where one
    draw from the null moves the metric by the whole remaining headroom.  With
    only a few benchmarks, all of them well below that line, the left panel alone
    would compress every point into a corner, so the right panel plots the ratio
    directly against the tuned baseline accuracy, which is the comparison the
    saturation question actually asks.

    Labels are placed on whichever side of the point has room, so that a name
    cannot run off the axis.
    """
    setup_style()
    head = np.array([r["headroom"] for r in rows], dtype=float)
    sig = np.array([r["sigma_delta"] for r in rows], dtype=float)
    base = np.array([r["baseline_accuracy"] for r in rows], dtype=float)
    ratio = sig / head
    names = [str(r["dataset"]) for r in rows]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, 2.1), gridspec_kw={"wspace": 0.32}
    )

    # --- left: sigma against headroom, with the equality line as reference
    ax0.scatter(head, sig, s=34, facecolors="white", edgecolors=INK,
                linewidths=1.1, zorder=3)
    _label_points(ax0, head, sig, names, log=True)
    lo = min(head.min(), sig.min()) / 2.0
    hi = max(head.max(), sig.max()) * 2.0
    ax0.plot([lo, hi], [lo, hi], color="#cccccc", lw=0.7, zorder=1)
    ax0.annotate(
        r"$\sigma_\Delta$ = headroom",
        xy=(hi, hi), xytext=(-4, -10), textcoords="offset points",
        ha="right", va="top", fontsize=6, color="#999999", rotation=45,
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("headroom above the tuned baseline")
    ax0.set_ylabel(r"null spread $\sigma_\Delta$")
    ax0.grid(True)
    ax0.set_axisbelow(True)

    # --- right: the ratio, which is what the saturation question asks about
    ax1.scatter(base, 100 * ratio, s=34, facecolors=INK, edgecolors=INK,
                linewidths=1.1, zorder=3)
    _label_points(ax1, base, 100 * ratio, names, log=False)
    pad_x = 0.08 * (base.max() - base.min() + 1e-9)
    ax1.set_xlim(base.min() - pad_x - 0.01, base.max() + pad_x + 0.01)
    lo_y, hi_y = (100 * ratio).min(), (100 * ratio).max()
    pad_y = 0.35 * (hi_y - lo_y + 1e-9)
    ax1.set_ylim(max(0.0, lo_y - pad_y), hi_y + pad_y)
    ax1.set_xlabel("tuned baseline accuracy")
    ax1.set_ylabel("null spread as a percentage\nof remaining headroom")
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)
    save(fig, name)


def _label_points(ax, x, y, labels, log: bool) -> None:
    """Put each label on whichever side of its point has room.

    With a handful of points a fixed offset eventually pushes a name off the
    axis, which is what happened the first time this figure was drawn.  Here the
    side is chosen from the point's position within the data range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mid = np.median(x)
    for xi, yi, lab in zip(x, y, labels):
        right = xi < mid
        ax.annotate(
            lab,
            xy=(xi, yi),
            xytext=(5 if right else -5, 4),
            textcoords="offset points",
            ha="left" if right else "right",
            va="bottom",
            fontsize=6.5,
            color=INK,
        )


def leaderboard_advances(
    advances: Sequence[dict],
    thresholds: Dict[str, float],
    name: str = "leaderboard_advances",
) -> None:
    """Each claimed advance on the leaderboard, against the deflated thresholds.

    One point per date on which the leaderboard best improved, with the size of
    the improvement on the vertical axis.  Horizontal reference lines are the
    deflated thresholds at the trial counts in the sensitivity grid, so a reader
    can see at a glance how many advances clear each of them.
    """
    setup_style()
    import datetime as dt

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.1))
    dates = [dt.date.fromisoformat(str(a["date"])[:10]) for a in advances]
    deltas = [100.0 * a["delta"] for a in advances]
    ax.scatter(
        dates, deltas, s=22, facecolors="white", edgecolors=INK, linewidths=1.0, zorder=4
    )
    for i, (label, thr) in enumerate(sorted(thresholds.items(), key=lambda kv: kv[1])):
        ax.axhline(
            100.0 * thr,
            color=TONES[min(i, len(TONES) - 1)],
            linestyle=STYLES[i % len(STYLES)],
            lw=0.9,
            label=label,
            zorder=2,
        )
    ax.set_ylabel("improvement over previous best\n(accuracy points)")
    ax.set_xlabel("submission date")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, handlelength=2.2, loc="upper right")
    fig.autofmt_xdate(rotation=0, ha="center")
    save(fig, name)
