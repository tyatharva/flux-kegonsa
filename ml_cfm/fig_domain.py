"""The domain figure: the 3.66 km LES domain on Esri World Imagery, the solar-array rectangle
as parameterised, 5 m contours of the USGS 3DEP terrain, and an inset (placed over the lake) at
full tile resolution over the array with the parameterised rectangle and a translucent star at
the parameterised tower position, so both can be compared with what the imagery shows.

    python -m ml_cfm.fig_domain [--out results/ml_cfm/final_recipe/domain.png]
    python -m ml_cfm.fig_domain --overlay-case case_2025031921 --split test --allow-test \
        --out results/ml_cfm/final_recipe/domain_generative_test.png

With --overlay-case the 80% source-area outlines of fig_generative's first panel (every CFM
sample, the CFM mean, the LES target) are drawn on the imagery, transformed from the LES frame
to Web Mercator, so one figure serves as both the domain map and the generative figure.

Tiles come from https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
(the item linked in the request) and are cached under data/raw/esri_tiles/ (gitignored). All
drawing is in Web Mercator (EPSG:3857); the domain and the array are transformed from
EPSG:3071 with pyproj, so the model's map-aligned rectangle appears with its true rotation.
"""
import argparse
import io
import math
import os
import sys
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
CACHE = os.path.join(REPO, "data", "raw", "esri_tiles")
GRID = os.path.join(REPO, "data", "grid30_raised")
R = 6378137.0


def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def tile_of(x, y, z):
    n = 2 ** z
    return int((x + math.pi * R) / (2 * math.pi * R) * n), int((math.pi * R - y) / (2 * math.pi * R) * n)


def tile_bounds(tx, ty, z):
    n = 2 ** z
    w = 2 * math.pi * R / n
    x0 = -math.pi * R + tx * w
    y1 = math.pi * R - ty * w
    return x0, x0 + w, y1 - w, y1


def fetch(z, tx, ty):
    from PIL import Image
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{z}_{tx}_{ty}.jpg")
    if not os.path.exists(p):
        req = urllib.request.Request(TILE_URL.format(z=z, x=tx, y=ty), headers={"User-Agent": "flux-footprint-figure/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(p, "wb") as fh:
            fh.write(data)
    return np.asarray(Image.open(p).convert("RGB"))


def mosaic(x0, x1, y0, y1, z):
    """RGB image and its extent in EPSG:3857 covering the box, from z-level tiles."""
    tx0, ty0 = tile_of(x0, y1, z)
    tx1, ty1 = tile_of(x1, y0, z)
    rows = []
    for ty in range(ty0, ty1 + 1):
        rows.append(np.concatenate([fetch(z, tx, ty) for tx in range(tx0, tx1 + 1)], axis=1))
    img = np.concatenate(rows, axis=0)
    bx0, _, _, by1 = tile_bounds(tx0, ty0, z)
    _, bx1, by0, _ = tile_bounds(tx1, ty1, z)
    return img, [bx0, bx1, by0, by1]


def draw_overlay(ax, case, split_name, allow_test, tx, ty, to_merc):
    """fig_generative's first panel on the map: the 80% source-area outline of each of the 80
    CFM samples, of the CFM mean and of the LES target, LES frame -> EPSG:3857."""
    import contourpy
    from matplotlib.lines import Line2D
    from ml_cfm import report_metrics as RM
    from ml_cfm import figstyle as FS
    from ml_cfm.fig_generative import level80
    split = D.load_split(split_name, allow_test=allow_test)
    valid = split.valid_mask.astype(np.float32)
    fields, les, samples = RM.recipe_fields(split, valid)
    i = int(np.where(split.meta["run_id"].astype(str) == case)[0][0])
    xc = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX

    def outline(f, **kw):
        for seg in contourpy.contour_generator(xc, xc, np.asarray(f, np.float64)).lines(level80(f)):
            mx, my = to_merc.transform(tx + seg[:, 0], ty + seg[:, 1])
            ax.plot(mx, my, **kw)

    for s in samples[:, i]:
        outline(s, color=FS.COL["cfm"], lw=0.9, alpha=0.30, zorder=4)
    outline(fields["CFM"][i], color="#ff1493", lw=3.2, zorder=6)
    outline(les[i], color=FS.COL["les"], lw=3.0, zorder=6)
    wd = float(split.wdir_deg[i])
    dt = str(split.meta["datetime"][i]).replace("T", " ")[:16]
    hs = [Line2D([], [], color=FS.COL["cfm"], lw=1.2, alpha=0.6, label=f"80% source area of each of the {samples.shape[0]} CFM samples"),
          Line2D([], [], color="#ff1493", lw=3.2, label="80% source area of the CFM mean"),
          Line2D([], [], color=FS.COL["les"], lw=3.0, label="80% source area of the LES target")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.05), fontsize=13, framealpha=0.92, edgecolor="none",
              title=f"{dt} UTC, wind from {wd:.0f}° ({split.octant[i]})", title_fontsize=13).set_zorder(12)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ml_cfm", "final_recipe", "domain.png"))
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--zoom-inset", type=int, default=19)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--size", type=float, default=15.0, help="figure side [in]")
    ap.add_argument("--overlay-case", default=None, help="run_id whose 80% source-area outlines (samples, CFM mean, LES) to draw")
    ap.add_argument("--split", default="val", help="split of --overlay-case")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--inset", default="upper right", help="inset corner, or 'none'")
    a = ap.parse_args(argv)
    if a.overlay_case and a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from pyproj import Transformer
    import prep_stage6 as P6

    meta = np.load(os.path.join(GRID, "meta.npy"), allow_pickle=True).item()
    topo = np.load(os.path.join(GRID, "topo.npy")) + float(meta["base"])
    nx, dx = int(meta["nx"]), float(meta["dx"])
    x0, y0 = float(meta["x0"]), float(meta["y0"])                     # EPSG:3071 cell-centre origin
    tx, ty = float(meta["tower_x"]), float(meta["tower_y"])
    to_merc = Transformer.from_crs("EPSG:3071", "EPSG:3857", always_xy=True)
    # the domain square and the array rectangle, EPSG:3071 -> 3857
    hx = nx * dx / 2
    cx, cy = x0 + (nx - 1) * dx / 2, y0 + (nx - 1) * dx / 2
    dom = np.array([to_merc.transform(cx + sx * hx, cy + sy * hx) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    ax0, ax1, ay0, ay1 = D.ARRAY_XY
    arr = np.array([to_merc.transform(tx + ex, ty + ny_) for ex, ny_ in ((ax0, ay0), (ax1, ay0), (ax1, ay1), (ax0, ay1))])
    tower = np.array(to_merc.transform(tx, ty))
    # the real terrain: USGS 3DEP 1/3 arc-second (EPSG:4269), a window around the domain, to 3857
    import rasterio
    from rasterio.windows import from_bounds
    to_ll = Transformer.from_crs("EPSG:3071", "EPSG:4269", always_xy=True)
    ll_to_merc = Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True)
    lons, lats = to_ll.transform([x0 - 400, x0 + nx * dx + 400], [y0 - 400, y0 + nx * dx + 400])
    with rasterio.open(os.path.join(REPO, "data", "raw", "output_USGS10m.tif")) as dem:
        win = from_bounds(lons[0], lats[0], lons[1], lats[1], dem.transform)
        Z = dem.read(1, window=win).astype(float)
        Z[Z == dem.nodata] = np.nan
        wt = dem.window_transform(win)
        cols, rows_ = np.meshgrid(np.arange(Z.shape[1]) + 0.5, np.arange(Z.shape[0]) + 0.5)
        LON, LAT = wt * (cols, rows_)
    MX, MY = ll_to_merc.transform(LON, LAT)
    topo = Z

    pad = 250.0
    bx0, bx1 = dom[:, 0].min() - pad, dom[:, 0].max() + pad
    by0, by1 = dom[:, 1].min() - pad, dom[:, 1].max() + pad
    img, ext = mosaic(bx0, bx1, by0, by1, a.zoom)

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(a.size, a.size))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", zorder=1)
    lev = np.arange(np.floor(np.nanmin(topo) / 5) * 5, np.nanmax(topo) + 5, 5)
    cs = ax.contour(MX, MY, topo, levels=lev, colors="w", linewidths=0.7, alpha=0.85, zorder=3)
    ax.clabel(cs, fmt="%.0f m", fontsize=8, colors="w", inline=True, inline_spacing=2)
    ax.add_patch(Polygon(dom, closed=True, fill=False, ec="#ffd400", lw=3.0, zorder=5))
    ax.add_patch(Polygon(arr, closed=True, fill=False, ec="#ff00ff", lw=2.6, zorder=6))
    ax.plot(*tower, marker="*", ms=16, mfc="w", mec="k", mew=0.9, zorder=7)
    ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if a.overlay_case:
        draw_overlay(ax, a.overlay_case, a.split, a.allow_test, tx, ty, to_merc)
    # scale bar and north arrow (Web Mercator scale factor at this latitude)
    lat = P6.TOWER_LAT
    k = 1 / math.cos(math.radians(lat))
    sx, sy = bx0 + 120, by0 + 120
    ax.plot([sx, sx + 1000 * k], [sy, sy], color="w", lw=4, zorder=8); ax.plot([sx, sx + 1000 * k], [sy, sy], color="k", lw=1.5, zorder=9)
    ax.text(sx + 500 * k, sy + 45, "1 km", color="w", ha="center", fontsize=11, weight="bold", zorder=9)
    ax.annotate("N", xy=(bx0 + 160, by1 - 120), xytext=(bx0 + 160, by1 - 440), color="w", ha="center", fontsize=16, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="w", lw=2.5), zorder=9)
    ax.text(bx1 - 40, by0 + 40, "Basemap: Esri World Imagery (Esri, Maxar, Earthstar Geographics, and the GIS User Community)",
            color="w", fontsize=8, ha="right", va="bottom", zorder=9, bbox=dict(fc="black", alpha=0.45, ec="none", pad=3))
    # the inset over the array, placed over the lake (upper right)
    ipad = 60.0
    ix0, ix1 = arr[:, 0].min() - ipad, arr[:, 0].max() + ipad
    iy0, iy1 = arr[:, 1].min() - ipad, arr[:, 1].max() + ipad
    iimg, iext = mosaic(ix0, ix1, iy0, iy1, a.zoom_inset)
    if a.inset == "none":
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        fig.savefig(a.out, dpi=a.dpi)
        print("wrote", a.out)
        return 0
    anchor = {"upper right": (-0.005, -0.03, 1, 1), "upper left": (0.005, -0.03, 1, 1),
              "lower right": (-0.005, 0.045, 1, 1), "lower left": (0.005, 0.045, 1, 1)}[a.inset]
    axins = inset_axes(ax, width="30%", height="50%", loc=a.inset, bbox_to_anchor=anchor, bbox_transform=ax.transAxes, borderpad=0)
    axins.imshow(iimg, extent=iext, origin="upper", interpolation="bilinear", zorder=1)
    axins.add_patch(Polygon(arr, closed=True, fill=False, ec="#ff00ff", lw=3.0, zorder=6))
    axins.plot(*tower, marker="*", ms=30, mfc=(1, 1, 1, 0.45), mec="k", mew=1.2, zorder=7)
    axins.set_xlim(ix0, ix1); axins.set_ylim(iy0, iy1); axins.set_aspect("equal"); axins.set_xticks([]); axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_visible(False)
    # a drop shadow under the inset: its screen box mapped into the main axes' data coordinates,
    # drawn as stacked translucent rectangles offset down-right
    fig.canvas.draw()
    bb = axins.get_window_extent()
    inv = ax.transData.inverted()
    (dx0, dy0), (dx1, dy1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    axins.set_zorder(10)
    px = (dx1 - dx0) / bb.width                          # one screen pixel in data units
    for k in range(12, 0, -1):                           # a soft shadow: spread on all sides, offset down and right
        spread, offx, offy = 1.6 * k * px, 4.0 * px, -6.0 * px
        ax.add_patch(Rectangle((dx0 - spread + offx, dy0 - spread + offy), (dx1 - dx0) + 2 * spread, (dy1 - dy0) + 2 * spread,
                               fc="black", ec="none", alpha=0.07, zorder=9))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
