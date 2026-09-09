from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dc_restoration.paths import repository_root
from dc_restoration.restoration.environment import V03RestorationModel, score_candidates


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(max(1, layers - 1)):
        mods += [nn.Linear(last, hidden), nn.SiLU()]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class EdgeConditionedMessagePassing(nn.Module):
    """Bidirectional edge-conditioned message passing over the fixed v0.3 graph."""

    def __init__(self, hidden: int):
        super().__init__()
        self.message_mlp = mlp(hidden * 4, hidden, hidden, layers=2)
        self.node_update = mlp(hidden * 3, hidden, hidden, layers=2)
        self.edge_update = mlp(hidden * 4, hidden, hidden, layers=2)
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.Tensor,
        global_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()
        src = node_h[:, src_idx, :]
        dst = node_h[:, dst_idx, :]
        g_edge = global_h[:, None, :].expand(-1, edge_h.shape[1], -1)
        msg_fwd = self.message_mlp(torch.cat([src, dst, edge_h, g_edge], dim=-1))
        msg_rev = self.message_mlp(torch.cat([dst, src, edge_h, g_edge], dim=-1))
        agg = torch.zeros_like(node_h)
        agg.index_add_(1, dst_idx, msg_fwd)
        agg.index_add_(1, src_idx, msg_rev)
        deg = torch.zeros(node_h.shape[1], device=node_h.device, dtype=node_h.dtype)
        one = torch.ones_like(src_idx, dtype=node_h.dtype)
        deg.index_add_(0, dst_idx, one)
        deg.index_add_(0, src_idx, one)
        agg = agg / deg.clamp_min(1.0)[None, :, None]
        node_h = self.node_norm(
            node_h
            + self.node_update(
                torch.cat(
                    [node_h, agg, global_h[:, None, :].expand(-1, node_h.shape[1], -1)], dim=-1
                )
            )
        )
        src2 = node_h[:, src_idx, :]
        dst2 = node_h[:, dst_idx, :]
        edge_h = self.edge_norm(
            edge_h + self.edge_update(torch.cat([src2, dst2, edge_h, g_edge], dim=-1))
        )
        return node_h, edge_h


class TrueGNNActorCritic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        candidate_dim: int,
        hidden: int = 64,
        layers: int = 2,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.global_dim = int(global_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden = int(hidden)
        self.node_encoder = mlp(node_dim, hidden, hidden)
        self.edge_encoder = mlp(edge_dim, hidden, hidden)
        self.global_encoder = mlp(global_dim, hidden, hidden)
        self.candidate_encoder = mlp(candidate_dim, hidden, hidden)
        self.message_layers = nn.ModuleList(
            [EdgeConditionedMessagePassing(hidden) for _ in range(layers)]
        )
        self.actor_head = mlp(hidden * 7, hidden, 1, layers=3)
        self.value_head = mlp(hidden * 3, hidden, 1, layers=3)

    def encode_graph(
        self, batch: Dict[str, torch.Tensor], edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_h = self.node_encoder(batch["node_features"])
        edge_h = self.edge_encoder(batch["edge_features"])
        global_h = self.global_encoder(batch["global_features"])
        for layer in self.message_layers:
            node_h, edge_h = layer(node_h, edge_h, edge_index, global_h)
        graph_h = torch.cat([node_h.mean(dim=1), node_h.max(dim=1).values, global_h], dim=-1)
        return node_h, edge_h, global_h, graph_h

    def forward(
        self, batch: Dict[str, torch.Tensor], edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        node_h, edge_h, global_h, graph_h = self.encode_graph(batch, edge_index)
        bsz, edge_count, hidden = edge_h.shape
        cand_idx = batch["candidate_edge_indices"].clamp_min(0).clamp_max(edge_count - 1)
        gather_e = cand_idx[..., None].expand(-1, -1, hidden)
        cand_edge_h = torch.gather(edge_h, 1, gather_e)
        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()
        cand_src = src_idx[cand_idx]
        cand_dst = dst_idx[cand_idx]
        gather_src = cand_src[..., None].expand(-1, -1, hidden)
        gather_dst = cand_dst[..., None].expand(-1, -1, hidden)
        cand_src_h = torch.gather(node_h, 1, gather_src)
        cand_dst_h = torch.gather(node_h, 1, gather_dst)
        cand_local_h = self.candidate_encoder(batch["candidate_features"])
        g = global_h[:, None, :].expand(-1, cand_idx.shape[1], -1)
        graph_mean = graph_h[:, None, : self.hidden].expand(-1, cand_idx.shape[1], -1)
        logits = self.actor_head(
            torch.cat(
                [
                    cand_src_h,
                    cand_dst_h,
                    cand_edge_h,
                    cand_local_h,
                    g,
                    graph_mean,
                    torch.abs(cand_src_h - cand_dst_h),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        logits = logits.masked_fill(~batch["candidate_mask"], -1e9)
        value = self.value_head(graph_h).squeeze(-1)
        return {"logits": logits, "value": value}

    def act(
        self, batch: Dict[str, torch.Tensor], edge_index: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(batch, edge_index)
        logits = out["logits"]
        value = out["value"]
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class V03GraphAdapter:
    def __init__(self, model: V03RestorationModel):
        self.model = model
        self.node_ids = model.nodes["node_id"].astype(str).tolist()
        self.node_index = {n: i for i, n in enumerate(self.node_ids)}
        self.edge_ids = model.edges["edge_id"].astype(str).tolist()
        self.edge_index_map = {e: i for i, e in enumerate(self.edge_ids)}
        src, dst = [], []
        for eid in self.edge_ids:
            u, v = model.edge_endpoints[eid]
            src.append(self.node_index.get(str(u), 0))
            dst.append(self.node_index.get(str(v), 0))
        self.edge_index = np.asarray([src, dst], dtype=np.int64)
        self.static_node = self._build_static_node_features()
        self.static_edge = self._build_static_edge_features()
        self.node_loads = self._node_load_vectors()

    def _norm_xy(self, s: pd.Series) -> np.ndarray:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        mn, mx = np.nanmin(x), np.nanmax(x)
        return np.nan_to_num((x - mn) / max(mx - mn, 1e-6), nan=0.0).astype(np.float32)

    def _build_static_node_features(self) -> np.ndarray:
        n = self.model.nodes.copy()
        load_kw = np.asarray(
            [self.model.node_load_kw.get(i, 0.0) for i in self.node_ids], dtype=np.float32
        )
        crit_kw = np.asarray(
            [self.model.node_critical_kw.get(i, 0.0) for i in self.node_ids], dtype=np.float32
        )
        voll_kw = np.asarray(
            [self.model.node_voll_kw.get(i, 0.0) for i in self.node_ids], dtype=np.float32
        )
        feats = np.stack(
            [
                pd.to_numeric(n["voltage_kv"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                / 100.0,
                load_kw / 10000.0,
                crit_kw / 10000.0,
                voll_kw / 1e6,
                pd.to_numeric(n["degree"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
                / 10.0,
                n["is_load"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                n["is_substation"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                n["is_transmission"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                n["is_distribution"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                n["is_data_center"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                n["is_critical"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
                .astype(np.float32)
                .to_numpy(),
                self._norm_xy(n["x"]),
                self._norm_xy(n["y"]),
            ],
            axis=1,
        )
        return feats.astype(np.float32)

    def _build_static_edge_features(self) -> np.ndarray:
        feats = []
        for eid in self.edge_ids:
            feats.append(self.model.static_edge_features(eid))
        return np.stack(feats).astype(np.float32)

    def _node_load_vectors(self) -> Dict[str, Tuple[int, float, float]]:
        out = {}
        for lid, info in self.model.load_info.items():
            node = str(info.get("node_id", ""))
            out[str(lid)] = (
                self.node_index.get(node, -1),
                float(info.get("baseline_p_kw", 0.0)) / 1000.0,
                float(info.get("baseline_p_kw", 0.0)) / 1000.0
                if bool(info.get("is_critical", False))
                else 0.0,
            )
        return out

    def candidate_features(
        self, before: Any, score: Dict[str, Any], remaining: int, total: int
    ) -> np.ndarray:
        return np.asarray(
            [
                score["repair_time"] / 24.0,
                score["immediate_voll_gain"] / 1e6,
                score["immediate_critical_gain"] / 10.0,
                score["immediate_unserved_gain"] / 50.0,
                score["voll_gain_per_hour"] / 1e6,
                score["critical_gain_per_hour"] / 10.0,
                before.unserved_mw / 100.0,
                before.critical_unserved_mw / 50.0,
                before.voll_cost / 1e6,
                remaining / 60.0,
                total / 60.0,
                score["immediate_voll_gain"] / max(before.voll_cost, 1.0),
            ],
            dtype=np.float32,
        )

    def build_state(
        self,
        damaged: Sequence[str],
        repair_times: Dict[str, float],
        repaired: set[str],
        k: int,
        provided_scores: Optional[pd.DataFrame] = None,
    ) -> Tuple[Any, List[Dict[str, Any]], Dict[str, np.ndarray]]:
        before, scores = score_candidates(
            self.model, damaged, repaired, repair_times, "rule_based_anchor"
        )
        scores = sorted(
            scores, key=lambda r: (r["immediate_voll_gain"], -r["repair_time"]), reverse=True
        )[:k]
        if provided_scores is not None:
            provided_ids = (
                provided_scores.sort_values("candidate_index")["candidate_edge_id"]
                .astype(str)
                .tolist()[:k]
            )
            by_id = {s["edge_id"]: s for s in scores}
            rebuilt = []
            for eid in provided_ids:
                if eid in by_id:
                    rebuilt.append(by_id[eid])
                else:
                    row = provided_scores.loc[
                        provided_scores["candidate_edge_id"].astype(str).eq(eid)
                    ].iloc[0]
                    rebuilt.append(
                        {
                            "edge_id": eid,
                            "repair_time": float(row["repair_time"]),
                            "immediate_voll_gain": float(row["immediate_voll_gain"]),
                            "immediate_critical_gain": float(row["immediate_critical_gain"]),
                            "immediate_unserved_gain": float(row["immediate_unserved_gain"]),
                            "voll_gain_per_hour": float(row["voll_gain_per_hour"]),
                            "critical_gain_per_hour": float(row["critical_gain_per_hour"]),
                            "unserved_gain_per_hour": float(row["unserved_gain_per_hour"]),
                        }
                    )
            scores = rebuilt[:k]
        node = np.concatenate(
            [self.static_node, np.zeros((len(self.node_ids), 4), dtype=np.float32)], axis=1
        )
        unserved_mw = np.zeros(len(self.node_ids), dtype=np.float32)
        unserved_crit = np.zeros(len(self.node_ids), dtype=np.float32)
        for lid in before.unserved_load_ids:
            idx, mw, cmw = self.node_loads.get(str(lid), (-1, 0.0, 0.0))
            if idx >= 0:
                unserved_mw[idx] += mw
                unserved_crit[idx] += cmw
        node[:, -4] = unserved_mw / 100.0
        node[:, -3] = unserved_crit / 50.0
        edge = np.concatenate(
            [self.static_edge, np.zeros((len(self.edge_ids), 3), dtype=np.float32)], axis=1
        )
        unrepaired = set(map(str, damaged)) - set(map(str, repaired))
        repaired_set = set(map(str, repaired))
        for eid in unrepaired:
            ei = self.edge_index_map.get(eid)
            if ei is not None:
                edge[ei, -3] = 1.0
                u, v = self.model.edge_endpoints.get(eid, ("", ""))
                if u in self.node_index:
                    node[self.node_index[u], -2] += 1.0 / 10.0
                if v in self.node_index:
                    node[self.node_index[v], -2] += 1.0 / 10.0
                edge[ei, -1] = float(repair_times.get(eid, 8.0)) / 24.0
        for eid in repaired_set:
            ei = self.edge_index_map.get(eid)
            if ei is not None:
                edge[ei, -2] = 1.0
                u, v = self.model.edge_endpoints.get(eid, ("", ""))
                if u in self.node_index:
                    node[self.node_index[u], -1] += 1.0 / 10.0
                if v in self.node_index:
                    node[self.node_index[v], -1] += 1.0 / 10.0
        cand_idx = np.zeros(k, dtype=np.int64)
        cand_feat = np.zeros((k, 12), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)
        for i, s in enumerate(scores[:k]):
            cand_idx[i] = self.edge_index_map.get(str(s["edge_id"]), 0)
            cand_feat[i] = self.candidate_features(before, s, len(scores), len(damaged))
            mask[i] = True
        glob = np.asarray(
            [
                before.unserved_mw / 100.0,
                before.critical_unserved_mw / 50.0,
                before.voll_cost / 1e6,
                len(unrepaired) / 60.0,
                len(repaired_set) / 60.0,
                len(damaged) / 60.0,
                before.dc_support_used_mw / 20.0,
                float(before.dc_anchor_active),
                before.outage_islands / 50.0,
            ],
            dtype=np.float32,
        )
        return (
            before,
            scores,
            {
                "node_features": node.astype(np.float32),
                "edge_features": edge.astype(np.float32),
                "global_features": glob,
                "candidate_edge_indices": cand_idx,
                "candidate_features": cand_feat,
                "candidate_mask": mask,
            },
        )


def to_torch_batch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "node_features": torch.as_tensor(
            batch["node_features"], dtype=torch.float32, device=device
        ),
        "edge_features": torch.as_tensor(
            batch["edge_features"], dtype=torch.float32, device=device
        ),
        "global_features": torch.as_tensor(
            batch["global_features"], dtype=torch.float32, device=device
        ),
        "candidate_edge_indices": torch.as_tensor(
            batch["candidate_edge_indices"], dtype=torch.long, device=device
        ),
        "candidate_features": torch.as_tensor(
            batch["candidate_features"], dtype=torch.float32, device=device
        ),
        "candidate_mask": torch.as_tensor(batch["candidate_mask"], dtype=torch.bool, device=device),
    }


def save_true_gnn_checkpoint(
    model: TrueGNNActorCritic,
    edge_index: torch.Tensor,
    path: Path,
    config: Dict[str, Any],
    dims: Dict[str, int],
    step: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "edge_index": edge_index.detach().cpu(),
            "dims": dims,
            "config": config,
            "global_step": int(step),
            "metrics": metrics or {},
            "architecture": "true_dynamic_message_passing_gnn",
        },
        path,
    )


def load_true_gnn_checkpoint(
    path: Path, device: torch.device
) -> Tuple[TrueGNNActorCritic, torch.Tensor, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dims = ckpt["dims"]
    cfg = ckpt.get("config", {})
    hidden = int(cfg.get("true_gnn", {}).get("hidden_dim", dims.get("hidden_dim", 64)))
    layers = int(
        cfg.get("true_gnn", {}).get("message_passing_layers", dims.get("message_passing_layers", 2))
    )
    model = TrueGNNActorCritic(
        dims["node_dim"],
        dims["edge_dim"],
        dims["global_dim"],
        dims["candidate_dim"],
        hidden=hidden,
        layers=layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, torch.as_tensor(ckpt["edge_index"], dtype=torch.long, device=device), ckpt


def compute_gae_arrays(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 or bool(dones[t]) else float(values[t + 1])
        next_nonterminal = 0.0 if bool(dones[t]) else 1.0
        delta = float(rewards[t]) + gamma * next_value * next_nonterminal - float(values[t])
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        adv[t] = last_adv
    return adv, adv + values.astype(np.float32)
