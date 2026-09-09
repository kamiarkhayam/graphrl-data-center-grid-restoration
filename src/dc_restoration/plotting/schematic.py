from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle

from dc_restoration.paths import repository_root

ROOT = repository_root()

OUTPUTS = (
    ROOT
    / "data_processed"
    / "final_case_v0_1"
    / "restoration_env"
    / "rl_experiments_v0"
    / "outputs"
)

PLOT_READY = OUTPUTS / "v0_3_paper_figures_plot_ready_data_v1"

OUT = OUTPUTS / "main_text_figures_revision_v6_schematic_figure01_centered_legend"

FIG1_IN = PLOT_READY / "main_text_figures" / "fig01_system_map"

REGIONS = {
    "A": {
        "region_id": "BENCH_HOUSTON_CORE",
        "label": "Module A",
    },
    "B": {
        "region_id": "BENCH_SHIP_CHANNEL",
        "label": "Module B",
    },
    "C": {
        "region_id": "BENCH_COASTAL_SOUTHWEST",
        "label": "Module C",
    },
}


def ensure_dirs() -> None:
    for rel in [
        "figures/png",
        "figures/pdf",
        "figures/svg",
        "figures/preview",
        "plot_ready",
        "captions",
        "reports",
        "scripts",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)


def setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.3,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "figure.titlesize": 9.8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def add_panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.02) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0, fontweight="bold"
    )


def export_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {
        "png": OUT / "figures" / "png" / f"{stem}_600dpi.png",
        "pdf": OUT / "figures" / "pdf" / f"{stem}.pdf",
        "svg": OUT / "figures" / "svg" / f"{stem}.svg",
        "preview": OUT / "figures" / "preview" / f"{stem}_preview.png",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["preview"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def norm_transform(
    lon: pd.Series | list[float],
    lat: pd.Series | list[float],
    bbox: tuple[float, float, float, float],
    data_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    x0, y0, w, h = bbox
    xmin, xmax, ymin, ymax = data_bounds
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    scale = min(w / dx, h / dy)
    cx_data = (xmin + xmax) / 2.0
    cy_data = (ymin + ymax) / 2.0
    cx_box = x0 + w / 2.0
    cy_box = y0 + h / 2.0
    return cx_box + (lon_arr - cx_data) * scale, cy_box + (lat_arr - cy_data) * scale


def data_bounds_for_region(
    edges: pd.DataFrame,
    special: pd.DataFrame,
    region_id: str,
    include_special: bool = False,
) -> tuple[float, float, float, float]:
    e = edges[edges["region_id"].eq(region_id)].copy()
    xs = pd.concat([e["from_longitude"], e["to_longitude"]], ignore_index=True)
    ys = pd.concat([e["from_latitude"], e["to_latitude"]], ignore_index=True)
    if include_special:
        sp = special[special["region_id"].eq(region_id)]
        xs = pd.concat([xs, sp["longitude"]], ignore_index=True)
        ys = pd.concat([ys, sp["latitude"]], ignore_index=True)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    padx = max((xmax - xmin) * 0.035, 1e-5)
    pady = max((ymax - ymin) * 0.035, 1e-5)
    return xmin - padx, xmax + padx, ymin - pady, ymax + pady


def draw_edges(
    ax: plt.Axes,
    edges: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
    zorder: int = 2,
) -> None:
    for _, r in edges.iterrows():
        xs, ys = norm_transform(
            [r["from_longitude"], r["to_longitude"]],
            [r["from_latitude"], r["to_latitude"]],
            bbox,
            bounds,
        )
        ax.plot(xs, ys, color=color, lw=linewidth, alpha=alpha, linestyle=linestyle, zorder=zorder)


def anchor_zone_ellipse(
    ax: plt.Axes,
    zone: pd.Series,
    loads: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    scale_up: float = 1.12,
    label: bool = False,
) -> tuple[float, float]:
    try:
        load_ids = json.loads(zone.get("anchor_load_ids", "[]"))
    except Exception:
        load_ids = []
    pts = loads[loads["load_id"].isin(load_ids)].dropna(subset=["longitude", "latitude"])
    if pts.empty:
        xs, ys = norm_transform(
            [zone["centroid_longitude"]], [zone["centroid_latitude"]], bbox, bounds
        )
        cx, cy = float(xs[0]), float(ys[0])
        width, height = 0.09, 0.07
    else:
        xs, ys = norm_transform(pts["longitude"], pts["latitude"], bbox, bounds)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        width = max(float(np.ptp(xs)) * scale_up, 0.045)
        height = max(float(np.ptp(ys)) * scale_up, 0.045)
    ax.add_patch(
        Ellipse(
            (cx, cy),
            width=width,
            height=height,
            facecolor="#6DBB7D",
            edgecolor="#2C7D48",
            lw=0.85,
            alpha=0.18,
            zorder=5,
        )
    )
    if label:
        ax.annotate(
            "Prescribed anchor\nsupport zone",
            xy=(cx - width * 0.12, cy - height * 0.35),
            xytext=(max(cx - width * 0.75, 0.35), max(cy - height * 0.78, 0.19)),
            fontsize=7.1,
            ha="right",
            va="top",
            arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#2C7D48"},
            color="#2C6542",
            zorder=30,
        )
    return cx, cy


def draw_assets(
    ax: plt.Axes,
    special: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
) -> None:
    sp = special.dropna(subset=["longitude", "latitude"]).copy()
    spec = [
        ("critical_load", "o", "#B75A66", 13, 0.58, 7),
        ("substation", "s", "#222222", 58, 0.96, 9),
        ("dc_anchor", "*", "#0A6B4F", 170, 0.98, 10),
    ]
    for asset_type, marker, color, size, alpha, zorder in spec:
        part = sp[sp["asset_type"].eq(asset_type)]
        if part.empty:
            continue
        xs, ys = norm_transform(part["longitude"], part["latitude"], bbox, bounds)
        ax.scatter(
            xs,
            ys,
            s=size,
            marker=marker,
            c=color,
            edgecolors="white" if asset_type != "critical_load" else "none",
            linewidths=0.45 if asset_type != "critical_load" else 0,
            alpha=alpha,
            zorder=zorder,
        )


def box_to_global(
    box: tuple[float, float, float, float], pts: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    x0, y0, w, h = box
    return [(x0 + px * w, y0 + py * h) for px, py in pts]


def draw_polyline(ax: plt.Axes, pts: list[tuple[float, float]], **kwargs: Any) -> None:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, **kwargs)


def draw_module_schematic(
    ax: plt.Axes,
    box: tuple[float, float, float, float],
    label: str,
    pattern: str,
) -> tuple[float, float]:
    x0, y0, w, h = box
    ax.add_patch(
        Rectangle((x0, y0), w, h, facecolor="#FBFBFB", edgecolor="#AFAFAF", lw=0.85, zorder=1)
    )
    label_x = x0 + 0.018
    ax.text(label_x, y0 + h - 0.023, label, ha="left", va="top", fontsize=7.0, color="#222222")

    feeder_color = "#607D94"
    feeder_lw = 0.90
    branch_lw = 0.58
    node_color = "#536F85"

    if pattern == "A":
        source_local = (0.82, 0.19)
        anchor_local = (0.29, 0.72)
        trunks = [
            [(0.82, 0.19), (0.65, 0.34), (0.49, 0.52), (0.31, 0.71), (0.17, 0.79)],
            [(0.65, 0.34), (0.47, 0.30), (0.28, 0.26), (0.14, 0.18)],
            [(0.49, 0.52), (0.42, 0.67), (0.34, 0.82)],
            [(0.47, 0.30), (0.39, 0.15), (0.32, 0.09)],
        ]
        stubs = [
            [(0.31, 0.71), (0.25, 0.61), (0.18, 0.60)],
            [(0.49, 0.52), (0.56, 0.64), (0.62, 0.70)],
            [(0.28, 0.26), (0.22, 0.36), (0.18, 0.42)],
        ]
    elif pattern == "B":
        source_local = (0.18, 0.19)
        anchor_local = (0.75, 0.70)
        trunks = [
            [(0.18, 0.19), (0.34, 0.31), (0.52, 0.48), (0.70, 0.67), (0.85, 0.77)],
            [(0.34, 0.31), (0.48, 0.22), (0.66, 0.18), (0.82, 0.12)],
            [(0.52, 0.48), (0.59, 0.63), (0.62, 0.82)],
            [(0.48, 0.22), (0.58, 0.34), (0.71, 0.38)],
        ]
        stubs = [
            [(0.70, 0.67), (0.77, 0.57), (0.87, 0.54)],
            [(0.52, 0.48), (0.43, 0.58), (0.36, 0.70)],
            [(0.66, 0.18), (0.73, 0.27), (0.80, 0.30)],
        ]
    else:
        source_local = (0.50, 0.82)
        anchor_local = (0.73, 0.25)
        trunks = [
            [(0.50, 0.82), (0.50, 0.66), (0.51, 0.50), (0.52, 0.34), (0.52, 0.15)],
            [(0.50, 0.66), (0.35, 0.54), (0.22, 0.42), (0.12, 0.28)],
            [(0.51, 0.50), (0.66, 0.44), (0.81, 0.35), (0.88, 0.21)],
            [(0.52, 0.34), (0.40, 0.25), (0.29, 0.13)],
        ]
        stubs = [
            [(0.35, 0.54), (0.31, 0.68), (0.26, 0.75)],
            [(0.66, 0.44), (0.73, 0.58), (0.83, 0.65)],
            [(0.52, 0.34), (0.60, 0.23), (0.67, 0.12)],
        ]

    all_nodes: list[tuple[float, float]] = []
    for trunk in trunks:
        pts = box_to_global(box, trunk)
        all_nodes.extend(pts)
        draw_polyline(
            ax, pts, color=feeder_color, lw=feeder_lw, alpha=0.88, zorder=3, solid_capstyle="round"
        )
    for stub in stubs:
        pts = box_to_global(box, stub)
        all_nodes.extend(pts)
        draw_polyline(
            ax, pts, color=feeder_color, lw=branch_lw, alpha=0.78, zorder=3, solid_capstyle="round"
        )

    terminal_nodes = box_to_global(
        box, [trunk[-1] for trunk in trunks] + [stub[-1] for stub in stubs]
    )
    branch_nodes = box_to_global(box, [p for trunk in trunks for p in trunk[1:-1]])
    ax.scatter(
        [p[0] for p in branch_nodes],
        [p[1] for p in branch_nodes],
        s=5,
        c=node_color,
        alpha=0.38,
        linewidths=0,
        zorder=4,
    )
    ax.scatter(
        [p[0] for p in terminal_nodes],
        [p[1] for p in terminal_nodes],
        s=8,
        c=node_color,
        alpha=0.55,
        linewidths=0,
        zorder=4,
    )

    sx, sy = box_to_global(box, [source_local])[0]
    ax.scatter(
        [sx], [sy], s=55, marker="s", c="#222222", edgecolors="white", linewidths=0.38, zorder=8
    )
    ax.scatter(
        [sx],
        [sy],
        s=80,
        marker="s",
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.45,
        alpha=0.18,
        zorder=7,
    )

    ax_x, ax_y = box_to_global(box, [anchor_local])[0]
    ax.scatter(
        [ax_x],
        [ax_y],
        s=150,
        marker="*",
        c="#0A6B4F",
        edgecolors="white",
        linewidths=0.45,
        zorder=9,
    )
    return sx, sy


def make_figure1() -> dict[str, str]:
    nodes = read_csv(FIG1_IN / "nodes_for_map.csv")
    edges = read_csv(FIG1_IN / "edges_for_map.csv")
    special = read_csv(FIG1_IN / "special_assets_for_map.csv")
    zones = read_csv(FIG1_IN / "anchor_zones_for_map.csv")
    loads = read_csv(PLOT_READY / "source_copies" / "geometry" / "loads.csv")

    nodes_used = nodes[bool_series(nodes["coordinate_complete"])].copy()
    edges_used = edges[bool_series(edges["endpoint_mapping_complete"])].copy()
    special_used = special[bool_series(special["coordinate_complete"])].copy()
    zones_used = zones[bool_series(zones["coordinate_complete"])].copy()
    loads = loads.merge(nodes_used[["node_id", "longitude", "latitude"]], on="node_id", how="left")

    nodes_used.to_csv(OUT / "plot_ready" / "Figure01_nodes_used.csv", index=False)
    edges_used.to_csv(OUT / "plot_ready" / "Figure01_edges_used.csv", index=False)
    special_used.to_csv(OUT / "plot_ready" / "Figure01_special_assets_used.csv", index=False)
    zones_used.to_csv(OUT / "plot_ready" / "Figure01_anchor_zones_used.csv", index=False)

    fig = plt.figure(figsize=(7.1, 4.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.105)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    add_panel_label(ax_a, "(a)", x=0.005, y=0.965)
    ax_a.text(
        0.055, 0.955, "Composite-system construction schematic", ha="left", va="top", fontsize=8.8
    )

    module_boxes = {
        "A": (0.045, 0.58, 0.405, 0.30),
        "B": (0.55, 0.58, 0.405, 0.30),
        "C": (0.297, 0.18, 0.405, 0.30),
    }
    source_points: dict[str, tuple[float, float]] = {}
    for key, box in module_boxes.items():
        source_points[key] = draw_module_schematic(ax_a, box, REGIONS[key]["label"], key)

    hub = (0.500, 0.515)
    ax_a.scatter(
        [hub[0]],
        [hub[1]],
        s=45,
        marker="s",
        c="#3A3A3A",
        edgecolors="white",
        linewidths=0.35,
        zorder=10,
    )
    for sx, sy in source_points.values():
        ax_a.plot(
            [hub[0], sx], [hub[1], sy], color="#4F4F4F", lw=1.05, linestyle=(0, (4, 2)), zorder=2
        )
    ax_a.text(
        0.515,
        0.510,
        "Abstract backbone/\nsource connector",
        fontsize=6.9,
        color="#3F3F3F",
        ha="left",
        va="center",
    )

    ax_b.set_axis_off()
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    add_panel_label(ax_b, "(b)", x=0.005, y=0.965)
    ax_b.text(0.055, 0.955, "Representative module detail", ha="left", va="top", fontsize=8.8)
    region_id = REGIONS["A"]["region_id"]
    detail_box = (0.035, 0.110, 0.805, 0.755)
    bounds = data_bounds_for_region(edges_used, special_used, region_id, include_special=True)
    e_bench = edges_used[
        edges_used["region_id"].eq(region_id)
        & (~edges_used["edge_type"].astype(str).str.contains("synthetic", case=False, na=False))
    ]
    e_syn = edges_used[
        edges_used["region_id"].eq(region_id)
        & (edges_used["edge_type"].astype(str).str.contains("synthetic", case=False, na=False))
    ]
    draw_edges(
        ax_b, e_bench, detail_box, bounds, color="#536F85", linewidth=0.36, alpha=0.82, zorder=2
    )
    draw_edges(
        ax_b,
        e_syn,
        detail_box,
        bounds,
        color="#555555",
        linewidth=0.60,
        alpha=0.62,
        linestyle=(0, (3, 2)),
        zorder=3,
    )
    zone_a = zones_used[zones_used["region_id"].eq(region_id)]
    if not zone_a.empty:
        anchor_zone_ellipse(
            ax_b, zone_a.iloc[0], loads, detail_box, bounds, scale_up=1.18, label=True
        )
    draw_assets(ax_b, special_used[special_used["region_id"].eq(region_id)], detail_box, bounds)
    ax_b.add_patch(
        Rectangle(
            (0.025, 0.095), 0.835, 0.795, facecolor="none", edgecolor="#D0D0D0", lw=0.65, zorder=20
        )
    )

    stats = [
        ("Nodes", "7,114"),
        ("Edges", "7,215"),
        ("Loads", "4,179"),
        ("Critical loads", "204"),
        ("Source interfaces", "3"),
        ("Feeders", "27"),
        ("DC anchors", "3"),
    ]
    x_txt, y_txt = 0.875, 0.805
    ax_b.text(
        x_txt, y_txt + 0.075, "Testbed scale", fontsize=7.7, fontweight="bold", ha="left", va="top"
    )
    for i, (k, v) in enumerate(stats):
        ax_b.text(
            x_txt,
            y_txt - i * 0.052,
            f"{k}: {v}",
            fontsize=6.75,
            ha="left",
            va="top",
            color="#333333",
        )

    handles = [
        Line2D([0], [0], color="#607D94", lw=1.2, label="Benchmark-derived feeder topology"),
        Line2D(
            [0],
            [0],
            color="#4F4F4F",
            lw=1.2,
            linestyle=(0, (4, 2)),
            label="Abstract source/backbone connector",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markersize=6,
            label="Source/substation interface",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#0A6B4F",
            markeredgecolor="white",
            markersize=10,
            label="Prescribed DC support anchor",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B75A66",
            markersize=5,
            label="Synthetic critical load",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.500, 0.105),
        ncol=3,
        frameon=False,
        fontsize=6.6,
        handlelength=1.8,
        columnspacing=1.05,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.145, wspace=0.105)
    return export_figure(fig, "Figure01_testbed_prescribed_dc_support")
