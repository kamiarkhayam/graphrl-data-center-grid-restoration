from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import V03RestorationModel


@dataclass
class SpatialIndex:
    node_xy: Dict[str, Tuple[float, float]]
    edge_mid_xy: Dict[str, Tuple[float, float]]
    edge_endpoint_xy: Dict[str, Tuple[float, float, float, float]]
    load_xy: Dict[str, Tuple[float, float]]
    load_mw: Dict[str, float]
    load_critical: Dict[str, bool]
    dc_xy_export: List[Tuple[float, float, float]]
    anchor_load_ids: Dict[str, List[str]]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_size: int
    channels: List[str]

    @property
    def raster_shape(self) -> Tuple[int, int, int]:
        return (len(self.channels), self.grid_size, self.grid_size)

    def cell(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        xn = (x - self.x_min) / max(self.x_max - self.x_min, 1e-9)
        yn = (y - self.y_min) / max(self.y_max - self.y_min, 1e-9)
        col = int(np.clip(math.floor(xn * self.grid_size), 0, self.grid_size - 1))
        row = int(np.clip(math.floor(yn * self.grid_size), 0, self.grid_size - 1))
        return row, col

    def norm_xy(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        x, y = xy
        return (
            float((x - self.x_min) / max(self.x_max - self.x_min, 1e-9)),
            float((y - self.y_min) / max(self.y_max - self.y_min, 1e-9)),
        )


def build_spatial_index(
    model: V03RestorationModel, grid_size: int, channels: List[str]
) -> SpatialIndex:
    nodes = model.nodes.copy()
    nodes["x_use"] = pd.to_numeric(nodes.get("x", nodes.get("longitude")), errors="coerce")
    nodes["y_use"] = pd.to_numeric(nodes.get("y", nodes.get("latitude")), errors="coerce")
    valid = nodes.dropna(subset=["x_use", "y_use"]).copy()
    node_xy = {
        str(r.node_id): (float(r.x_use), float(r.y_use))
        for r in valid[["node_id", "x_use", "y_use"]].itertuples(index=False)
    }
    xs = valid["x_use"].to_numpy(dtype=float)
    ys = valid["y_use"].to_numpy(dtype=float)
    pad_x = max((float(np.nanmax(xs)) - float(np.nanmin(xs))) * 0.02, 1e-4)
    pad_y = max((float(np.nanmax(ys)) - float(np.nanmin(ys))) * 0.02, 1e-4)

    edge_mid_xy: Dict[str, Tuple[float, float]] = {}
    edge_endpoint_xy: Dict[str, Tuple[float, float, float, float]] = {}
    for row in model.edges[["edge_id", "from_node", "to_node"]].itertuples(index=False):
        u = node_xy.get(str(row.from_node))
        v = node_xy.get(str(row.to_node))
        if u and v:
            edge_mid_xy[str(row.edge_id)] = ((u[0] + v[0]) / 2.0, (u[1] + v[1]) / 2.0)
            edge_endpoint_xy[str(row.edge_id)] = (u[0], u[1], v[0], v[1])

    load_node = model.loads.set_index("load_id")["node_id"].astype(str).to_dict()
    load_xy = {str(lid): node_xy[node] for lid, node in load_node.items() if node in node_xy}
    load_mw = (model.loads.set_index("load_id")["baseline_p_kw"].astype(float) / 1000.0).to_dict()
    load_critical = (
        model.loads.set_index("load_id")["is_critical"]
        .map(lambda v: str(v).lower() in ["true", "1", "yes"])
        .to_dict()
    )

    dc_xy_export: List[Tuple[float, float, float]] = []
    for _, row in model.dcs.iterrows():
        x = row.get("longitude", np.nan)
        y = row.get("latitude", np.nan)
        cap = row.get("max_export_mw_anchor_mode", 0.0)
        if pd.notna(x) and pd.notna(y):
            dc_xy_export.append((float(x), float(y), float(cap or 0.0)))

    return SpatialIndex(
        node_xy=node_xy,
        edge_mid_xy=edge_mid_xy,
        edge_endpoint_xy=edge_endpoint_xy,
        load_xy=load_xy,
        load_mw={str(k): float(v) for k, v in load_mw.items()},
        load_critical={str(k): bool(v) for k, v in load_critical.items()},
        dc_xy_export=dc_xy_export,
        anchor_load_ids=model.anchor_load_ids,
        x_min=float(np.nanmin(xs)) - pad_x,
        x_max=float(np.nanmax(xs)) + pad_x,
        y_min=float(np.nanmin(ys)) - pad_y,
        y_max=float(np.nanmax(ys)) + pad_y,
        grid_size=int(grid_size),
        channels=list(channels),
    )


def candidate_extra_xy(spatial: SpatialIndex, edge_id: str) -> np.ndarray:
    mid = spatial.edge_mid_xy.get(edge_id)
    endpoints = spatial.edge_endpoint_xy.get(edge_id)
    if mid is None or endpoints is None:
        return np.zeros(6, dtype=np.float32)
    mx, my = spatial.norm_xy(mid)
    ux, uy = spatial.norm_xy((endpoints[0], endpoints[1]))
    vx, vy = spatial.norm_xy((endpoints[2], endpoints[3]))
    return np.asarray([mx, my, ux, uy, vx, vy], dtype=np.float32)


def rasterize_state(
    model: V03RestorationModel,
    spatial: SpatialIndex,
    damaged: List[str],
    repaired: set[str],
    controller: str = "rule_based_anchor",
) -> np.ndarray:
    c, h, w = spatial.raster_shape
    raster = np.zeros((c, h, w), dtype=np.float32)
    channels = {name: i for i, name in enumerate(spatial.channels)}
    unrepaired = [eid for eid in damaged if eid not in repaired]
    for eid in unrepaired:
        xy = spatial.edge_mid_xy.get(str(eid))
        if xy is None:
            continue
        rr, cc = spatial.cell(xy)
        raster[channels["damaged_unrepaired_count"], rr, cc] += 1.0
        info = model.edge_info.get(str(eid), {})
        if (
            str(info.get("hurricane_damage_component_type", info.get("component_type", "")))
            == "overhead_line"
        ):
            raster[channels["damaged_unrepaired_overhead"], rr, cc] += 1.0

    before = model.metrics(set(damaged) - repaired, controller)
    for lid in before.unserved_load_ids:
        xy = spatial.load_xy.get(str(lid))
        if xy is None:
            continue
        rr, cc = spatial.cell(xy)
        mw = float(spatial.load_mw.get(str(lid), 0.0))
        raster[channels["unserved_load_mw"], rr, cc] += mw
        if spatial.load_critical.get(str(lid), False):
            raster[channels["critical_unserved_load_mw"], rr, cc] += mw

    for load_ids in spatial.anchor_load_ids.values():
        for lid in load_ids:
            xy = spatial.load_xy.get(str(lid))
            if xy is None:
                continue
            rr, cc = spatial.cell(xy)
            raster[channels["anchor_zone_load_mw"], rr, cc] += float(
                spatial.load_mw.get(str(lid), 0.0)
            )
    for x, y, cap in spatial.dc_xy_export:
        rr, cc = spatial.cell((x, y))
        raster[channels["dc_anchor_export_mw"], rr, cc] += cap

    # Stable scale factors keep sparse channels numerically comparable.
    raster[channels["damaged_unrepaired_count"]] /= 5.0
    raster[channels["damaged_unrepaired_overhead"]] /= 5.0
    raster[channels["unserved_load_mw"]] /= 5.0
    raster[channels["critical_unserved_load_mw"]] /= 2.0
    raster[channels["anchor_zone_load_mw"]] /= 10.0
    raster[channels["dc_anchor_export_mw"]] /= 10.0
    return raster


class MLPScorer(nn.Module):
    def __init__(self, candidate_dim: int, hidden: int = 192):
        super().__init__()
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden)
        self.cand = nn.Sequential(
            nn.Linear(candidate_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, 1)

    def candidate_embedding(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.cand(candidate_x)

    def state_embedding(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.candidate_embedding(candidate_x, mask, raster)
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return (emb * mask[..., None].float()).sum(dim=1) / denom

    def forward(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.head(self.candidate_embedding(candidate_x, mask, raster)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        raster_channels: int,
        grid_size: int,
        hidden: int = 192,
        conv_channels: int = 24,
        spatial_dim: int = 96,
    ):
        super().__init__()
        self.candidate_dim = int(candidate_dim)
        self.raster_channels = int(raster_channels)
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden)
        self.conv = nn.Sequential(
            nn.Conv2d(raster_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(conv_channels * 2 * 4 * 4, spatial_dim),
            nn.ReLU(),
        )
        self.cand = nn.Sequential(
            nn.Linear(candidate_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + spatial_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, 1)

    def spatial_embedding(self, raster: torch.Tensor) -> torch.Tensor:
        if raster is None:
            raise ValueError("CNN scorer requires raster tensor")
        return self.conv(raster)

    def candidate_embedding(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cand = self.cand(candidate_x)
        spatial = self.spatial_embedding(raster)
        spatial_rep = spatial[:, None, :].expand(-1, candidate_x.shape[1], -1)
        return self.fuse(torch.cat([cand, spatial_rep], dim=-1))

    def state_embedding(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.candidate_embedding(candidate_x, mask, raster)
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return (emb * mask[..., None].float()).sum(dim=1) / denom

    def forward(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.head(self.candidate_embedding(candidate_x, mask, raster)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)


class EncoderActorCritic(nn.Module):
    def __init__(self, actor: nn.Module, hidden_dim: int):
        super().__init__()
        self.actor = actor
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, candidate_x: torch.Tensor, mask: torch.Tensor, raster: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(candidate_x, mask, raster)
        state = self.actor.state_embedding(candidate_x, mask, raster)
        value = self.value(state).squeeze(-1)
        return logits, value

    def act(
        self,
        candidate_x: torch.Tensor,
        mask: torch.Tensor,
        raster: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(candidate_x, mask, raster)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
