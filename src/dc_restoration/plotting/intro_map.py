"""Regional introduction map with the manuscript's navy/gold figure styling."""

import json

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import box

from dc_restoration.paths import output_directory
from dc_restoration.spatial.prepare import geopandas, verify_prepared

STEM = "FigureIntro_Gulf_Atlantic_data_center_growth_and_hurricane_exposure"
COASTAL = ["TX", "LA", "MS", "AL", "FL", "GA", "SC", "NC", "VA"]
BLUE = "#0C457D"
GOLD = "#B89B4D"
TEXT = "#222222"
MUTED = "#666666"
NO_DATA = "#ECECEC"
# A sequential ramp anchored to the paper's gold, not its categorical palette.
HAZARD_COLORS = ["#FAF8F0", "#F1EBD7", "#E4D8AF", "#CEBC7B", GOLD, "#937A35", "#6B5726"]
SIZE_FACTOR = 8.5
EXTENT = (-1450000, 2650000, 150000, 2380000)
LABEL_POSITIONS = {
    "TX": (-1090000, 790000),
    "LA": (-80000, 610000),
    "MS": (295000, 1510000),
    "AL": (830000, 1630000),
    "FL": (1980000, 515000),
    "GA": (1710000, 1030000),
    "SC": (2100000, 1250000),
    "NC": (2100000, 1530000),
    "VA": (2100000, 1910000),
}


def setup_style():
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "axes.edgecolor": TEXT,
            "text.color": TEXT,
            "savefig.facecolor": "white",
        }
    )


def symbol_areas(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Symbol growth values must be finite and nonnegative")
    return values * SIZE_FACTOR


def symbols(ax, x, y, values):
    return ax.scatter(
        x,
        y,
        s=symbol_areas(values),
        facecolors=mcolors.to_rgba(BLUE, 0.09),
        edgecolors=BLUE,
        linewidths=0.9,
        zorder=5,
    )


def generate_map(config):
    gpd = geopandas()
    source_records = verify_prepared(config)
    data = config["prepared_dir"]
    counties = gpd.read_file(data / "county_hurricane_frequency.geojson").to_crs(5070)
    states = gpd.read_file(data / "state_data_center_growth.geojson").to_crs(5070)
    output = output_directory(config["output_dir"], require_empty=True)
    fig, selected, qa = draw_map(counties, states, source_count=len(source_records))
    try:
        fig.savefig(output / f"{STEM}_600dpi.PNG", dpi=600)
        fig.savefig(
            output / f"{STEM}.pdf",
            metadata={
                "Title": "Data-center demand growth and hurricane exposure along the Gulf and Atlantic coasts",
                "Author": "Kamiar Khayambashi; original cartography from EPRI, FEMA and U.S. Census Bureau data",
            },
        )
        fig.savefig(output / f"{STEM}.svg")
        fig.savefig(output / f"{STEM}_preview.png", dpi=220)
        selected[
            ["STUSPS", "NAME", "demand_2024_TWh", "demand_2030_TWh", "growth_TWh"]
        ].sort_values("STUSPS").to_csv(output / "regional_displayed_values.csv", index=False)
        (output / "figure_validation.json").write_text(
            json.dumps(qa, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        plt.close(fig)
    return qa


def draw_map(counties, states, *, source_count=0):
    setup_style()
    assert counties.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    selected = states[states.STUSPS.isin(COASTAL)].copy()
    assert len(selected) == 9 and selected.growth_TWh.notna().all()
    assert np.allclose(selected.demand_2030_TWh - selected.demand_2024_TWh, selected.growth_TWh)
    assert set(LABEL_POSITIONS) == set(COASTAL)
    viewport = box(EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3])
    assert selected.geometry.covered_by(viewport).all(), "Regional frame crops a selected state"

    fig = plt.figure(figsize=(7.2, 5.0), facecolor="white")
    ax = fig.add_axes([0.015, 0.175, 0.97, 0.80])
    ax.set_aspect("equal")
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT)
        spine.set_linewidth(0.65)
    bounds = np.array([0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5])
    rgb = np.array([mcolors.to_rgb(color) for color in HAZARD_COLORS])
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear_rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) < 0), "Hazard colors must darken with frequency"
    cmap = mcolors.ListedColormap(HAZARD_COLORS)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    visible = counties[counties.geometry.intersects(viewport)]
    valid = visible[visible.HRCN_AFREQ.notna()]
    states.plot(ax=ax, color=NO_DATA, edgecolor="none", zorder=1)
    valid.plot(
        ax=ax,
        column="HRCN_AFREQ",
        cmap=cmap,
        norm=norm,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        zorder=2,
    )
    states.boundary.plot(ax=ax, color="#9A9A9A", linewidth=0.35, zorder=3)
    selected.boundary.plot(ax=ax, color="#666666", linewidth=0.5, zorder=3)

    mapped = selected.sort_values("growth_TWh", ascending=False)
    symbols(ax, mapped.symbol_x, mapped.symbol_y, mapped.growth_TWh)
    labels = []
    for _, row in mapped.iterrows():
        tx, ty = LABEL_POSITIONS[row.STUSPS]
        label = ax.annotate(
            f"{row.NAME}\n+{row.growth_TWh:.1f}",
            xy=(row.symbol_x, row.symbol_y),
            xytext=(tx, ty),
            ha="left",
            va="center",
            fontsize=8.0,
            color=BLUE,
            linespacing=1.22,
            arrowprops={
                "arrowstyle": "-",
                "color": BLUE,
                "lw": 0.6,
                "shrinkA": 5,
                "shrinkB": np.sqrt(row.growth_TWh * SIZE_FACTOR) / 2 + 1,
                "connectionstyle": "arc3,rad=0",
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.7},
            zorder=8,
        )
        labels.append(label)

    for code in ["OK", "AR", "TN", "KY", "MO", "KS", "OH", "IN", "PA", "WV"]:
        row = states[states.STUSPS.eq(code)].iloc[0]
        px, py = row.symbol_x, row.symbol_y
        if viewport.contains(row.geometry.representative_point()):
            # Context-state names stay subordinate to the nine coastal-state labels.
            ax.text(
                px,
                py,
                code,
                fontsize=5.8,
                color="#888888",
                ha="center",
                va="center",
                zorder=4,
                path_effects=[pe.withStroke(linewidth=1.4, foreground="white", alpha=0.7)],
            )

    # The locator occupies land outside the coastal-state selection, above Texas.
    locator_bounds = (-1390000, 1770000, 1100000, 570000)
    locator_footprint = box(
        locator_bounds[0],
        locator_bounds[1],
        locator_bounds[0] + locator_bounds[2],
        locator_bounds[1] + locator_bounds[3],
    )
    assert selected.geometry.intersection(locator_footprint).area.sum() < 1, (
        "Locator hides selected states"
    )
    locator = ax.inset_axes(locator_bounds, transform=ax.transData, zorder=10)
    locator.set_facecolor("white")
    states.plot(ax=locator, color=NO_DATA, edgecolor="white", linewidth=0.22)
    selected.plot(ax=locator, color="#C2D2E0", edgecolor=BLUE, linewidth=0.34)
    locator.set_xlim(-2530000, 2520000)
    locator.set_ylim(180000, 3550000)
    locator.set_xticks([])
    locator.set_yticks([])
    for spine in locator.spines.values():
        spine.set_visible(True)
        spine.set_color("#BDBDBD")
        spine.set_linewidth(0.5)
    locator.text(
        0.025,
        0.975,
        "U.S. context",
        transform=locator.transAxes,
        fontsize=6.5,
        color=MUTED,
        va="top",
    )

    ax.text(
        2250000,
        925000,
        "Atlantic Ocean",
        fontsize=6.5,
        color="#888888",
        fontstyle="italic",
        rotation=45,
        ha="center",
    )
    ax.text(
        510000, 420000, "Gulf coast", fontsize=6.7, color="#888888", fontstyle="italic", ha="center"
    )
    x0, y0, width = 225000, 265000, 500000
    ax.plot([x0, x0 + width], [y0, y0], color=MUTED, lw=0.65)
    for x in (x0, x0 + width / 2, x0 + width):
        ax.plot([x, x], [y0 - 16000, y0 + 16000], color=MUTED, lw=0.65)
    ax.text(x0, y0 - 43000, "0", fontsize=6.4, ha="center", va="top", color=MUTED)
    ax.text(x0 + width, y0 - 43000, "500 km", fontsize=6.4, ha="center", va="top", color=MUTED)

    fig.add_artist(
        Line2D([0.045, 0.955], [0.163, 0.163], transform=fig.transFigure, color="#DCDCDC", lw=0.5)
    )
    fig.text(0.045, 0.133, "Coastal-state demand growth", fontsize=8.0, color=TEXT)
    fig.text(0.045, 0.105, "2024-2030, medium scenario (TWh/year)", fontsize=6.5, color=MUTED)
    lax = fig.add_axes([0.045, 0.017, 0.40, 0.078])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    symbols(lax, [0.055, 0.28, 0.60], [0.49, 0.49, 0.49], [5, 25, 100]).set_clip_on(False)
    for x, value in zip([0.055, 0.28, 0.60], [5, 25, 100]):
        lax.text(x + 0.08, 0.49, str(value), va="center", fontsize=7.0, color=TEXT)
    fig.text(0.50, 0.133, "Hurricane frequency", fontsize=8.0)
    fig.text(0.50, 0.105, "County average (events/year)", fontsize=6.5, color=MUTED)
    cax = fig.add_axes([0.50, 0.062, 0.455, 0.023])
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=bounds,
        spacing="uniform",
    )
    cb.set_ticklabels(["0", ".01", ".025", ".05", ".1", ".2", ".3", ".5"])
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor("#777777")
    cb.ax.tick_params(length=0, pad=2.5, labelsize=6.4, colors=MUTED)
    fig.add_artist(
        Rectangle(
            (0.50, 0.016),
            0.014,
            0.014,
            transform=fig.transFigure,
            facecolor=NO_DATA,
            edgecolor="#888888",
            lw=0.3,
        )
    )
    fig.text(0.52, 0.015, "Not applicable in FEMA", fontsize=6.4, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for i, label in enumerate(labels):
        bbox = label.get_bbox_patch().get_window_extent(renderer)
        for other in labels[i + 1 :]:
            if bbox.overlaps(other.get_bbox_patch().get_window_extent(renderer)):
                problems.append([label.get_text(), other.get_text()])
        if bbox.overlaps(locator.get_window_extent(renderer)):
            problems.append([label.get_text(), "locator overlap"])
        if not fig.bbox.contains(bbox.x0, bbox.y0) or not fig.bbox.contains(bbox.x1, bbox.y1):
            problems.append([label.get_text(), "outside figure"])
    assert not problems, problems

    qa = {
        "all_checks_passed": True,
        "source_hashes_checked": source_count,
        "canvas_inches": [7.2, 5.0],
        "raster_dpi": 600,
        "font": "DejaVu Serif",
        "coastal_states": COASTAL,
        "selection": "All nine Gulf/Atlantic coastal states from Texas through Virginia",
        "positive_growth_symbols": len(mapped),
        "label_overlap_check_passed": True,
        "locator_covers_no_selected_states": True,
        "selected_states_completely_inside_viewport": True,
        "symbol_size_rule": "scatter area = 8.5 * growth_TWh, no minimum-size inflation",
        "hazard_class_boundaries_events_per_year": bounds.tolist(),
        "map_extent_EPSG5070": EXTENT,
        "counties_intersecting_viewport": len(visible),
        "FEMA_null_handling": "gray, not applicable, never replaced with zero",
        "paper_style_reference": "dc_restoration/plotting/paper.py: setup_style and RESOURCE_COLOR",
        "navy": BLUE,
        "gold_anchor": GOLD,
        "hazard_colors": HAZARD_COLORS,
        "hazard_luminance_strictly_decreasing": bool(np.all(np.diff(luminance) < 0)),
        "boxed_map_frame_width_pt": 0.65,
    }
    return fig, selected, qa
