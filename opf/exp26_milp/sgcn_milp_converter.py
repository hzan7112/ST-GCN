"""Self-contained Exp26 ST-GCN to Gurobi MILP converter.

All conversion code for the Exp26 single-step reactive-power optimization lives
in this directory.  The converter embeds the model saved by ``train_exp26.py``:

    V = voltage linear prior + GCN voltage residual
    I = current linear prior + GCN current residual

The public ``embed_sgcn_constraints`` name is kept for compatibility with the
existing RPO script.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None
    GRB = None


DEFAULT_ENGINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "st_gcn_h16_l2_exp26_milp_engine.pt"
)

RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
]


def load_pt(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def npv(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def aff_bound(W, b, lb, ub):
    W, b = np.asarray(W, float), np.asarray(b, float).reshape(-1)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
    return Wp @ lb + Wn @ ub + b, Wp @ ub + Wn @ lb + b


class Exp26MILPOutputs(NamedTuple):
    Vdev_total: object
    Vworst: object
    WorstI: object
    Ploss_total: object
    V_nodes: object


class Exp26STGCNCoreConverter:
    def __init__(
        self,
        engine_path,
        *,
        slack_voltage=1.03,
        big_m_scale=1.0,
        fallback_big_m=1e3,
        weight_tol=1e-8,
        min_big_m=1e-8,
        relu_formulation="big_m",
    ):
        if gp is None or GRB is None:
            raise ModuleNotFoundError("gurobipy is required to build MILP expressions")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)
        if relu_formulation not in {"big_m", "general"}:
            raise ValueError("relu_formulation must be 'big_m' or 'general'")
        if fallback_big_m <= 0.0:
            raise ValueError("fallback_big_m must be positive")
        if big_m_scale <= 0.0:
            raise ValueError("big_m_scale must be positive")

        self.engine = load_pt(engine_path)
        self.sd = self.engine["state_dict"]

        self.nbus = int(self.engine.get("num_nodes", 33))
        self.edge_list = [[int(u), int(v)] for u, v in self.engine.get("edge_list", RADIAL_BRANCHES)]
        self.nedge = len(self.edge_list)

        self.in_features = int(self.engine.get("in_features", 6))
        self.hidden_dim = int(self.engine["hidden_dim"])
        self.num_layers = int(self.engine["num_layers"])
        self.node_relu_dim = int(self.engine["node_relu_dim"])
        self.edge_relu_dim = int(self.engine["edge_relu_dim"])

        self.use_residual = bool(self.engine.get("use_residual", True))
        self.use_initial_anchor = bool(self.engine.get("use_initial_anchor", True))
        self.use_jk = bool(self.engine.get("use_jk", True))
        self.include_input_in_jk = bool(self.engine.get("include_input_in_jk", True))
        self.use_linear_skip = bool(self.engine.get("use_linear_skip", True))

        self.slack_voltage = float(slack_voltage)
        self.big_m_scale = float(big_m_scale)
        self.fallback_big_m = float(fallback_big_m)
        self.weight_tol = float(weight_tol)
        self.min_big_m = float(min_big_m)
        self.relu_formulation = str(relu_formulation)

        if self.nbus != 33 or self.in_features != 6:
            raise ValueError(f"Exp26 converter expects 33 buses and 6 inputs, got {self.nbus}, {self.in_features}")

        self.A = self.mat("frozen_adj_norm")
        self.S_down = self.mat("downstream_matrix")
        self.S_path = self.mat("path_power_matrix")
        self.norm = self.load_norm()
        self.M = self.load_big_m()

        self.uses_vlin = bool(self.engine.get("uses_voltage_linear_prior", False))
        self.W_vlin = npv(self.engine["voltage_linear_prior_W"]).astype(float) if self.uses_vlin else None
        self.b_vlin = npv(self.engine["voltage_linear_prior_b"]).astype(float).reshape(-1) if self.uses_vlin else None

        self.uses_ilin = bool(self.engine.get("uses_current_linear_prior", False))
        self.W_ilin = npv(self.engine["current_linear_prior_W"]).astype(float) if self.uses_ilin else None
        self.b_ilin = npv(self.engine["current_linear_prior_b"]).astype(float).reshape(-1) if self.uses_ilin else None
        self.ilin_dst = (
            npv(self.engine["current_linear_prior_dst_idx"]).astype(int).reshape(-1)
            if self.uses_ilin
            else None
        )

        if not self.uses_vlin:
            raise ValueError("Exp26 engine must include the voltage linear prior")
        if not self.uses_ilin:
            raise ValueError("Exp26 engine must include the current linear prior")
        if self.W_vlin.shape != (32, self.nbus * self.in_features):
            raise ValueError(f"voltage_linear_prior_W shape error: {self.W_vlin.shape}")
        if self.b_vlin.shape != (32,):
            raise ValueError(f"voltage_linear_prior_b shape error: {self.b_vlin.shape}")
        if self.W_ilin.shape != (self.nedge, self.in_features):
            raise ValueError(f"current_linear_prior_W shape error: {self.W_ilin.shape}")
        if self.b_ilin.shape != (self.nedge,):
            raise ValueError(f"current_linear_prior_b shape error: {self.b_ilin.shape}")
        if self.ilin_dst.shape != (self.nedge,):
            raise ValueError(f"current_linear_prior_dst_idx shape error: {self.ilin_dst.shape}")

        self.domain = None
        self.binary_created = 0
        self.fixed_zero = 0
        self.fixed_linear = 0

    def mat(self, key):
        if key not in self.engine or self.engine[key] is None:
            raise KeyError(f"engine missing {key}")
        return npv(self.engine[key]).astype(float)

    def load_norm(self):
        ns = self.engine["norm_stats"]
        return {
            "X_mean": npv(ns["X_mean"]).astype(float).reshape(-1),
            "X_std": npv(ns["X_std"]).astype(float).reshape(-1),
            "YV_mean": npv(ns["YV_mean_wo_slack"]).astype(float).reshape(-1),
            "YV_std": npv(ns["YV_std_wo_slack"]).astype(float).reshape(-1),
            "YI_mean": npv(ns["YI_mean"]).astype(float).reshape(-1),
            "YI_std": npv(ns["YI_std"]).astype(float).reshape(-1),
        }

    def load_big_m(self):
        out = {}
        keys = [
            "M_plus_gcn_layers",
            "M_minus_gcn_layers",
            "M_plus_node",
            "M_minus_node",
            "M_plus_edge",
            "M_minus_edge",
        ]
        for key in keys:
            value = self.engine[key]
            out[key] = [npv(x).astype(float) for x in value] if isinstance(value, list) else npv(value).astype(float)
        return out

    def p(self, name):
        if name not in self.sd:
            raise KeyError(f"state_dict missing parameter: {name}")
        return npv(self.sd[name]).astype(float)

    def p_opt(self, name):
        return npv(self.sd[name]).astype(float) if name in self.sd else None

    def lp(self, k, name):
        return self.p(f"gcn_layers.{k}.{name}")

    def scalar(self, k, name, default=0.0):
        key = f"gcn_layers.{k}.{name}"
        return float(npv(self.sd[key]).reshape(-1)[0]) if key in self.sd else float(default)

    def set_opf_input_bounds(self, X6_lb, X6_ub):
        X6_lb = np.asarray(X6_lb, float)
        X6_ub = np.asarray(X6_ub, float)
        Xn_lb = (X6_lb - self.norm["X_mean"]) / self.norm["X_std"]
        Xn_ub = (X6_ub - self.norm["X_mean"]) / self.norm["X_std"]
        self.domain = self.interval_forward(Xn_lb, Xn_ub)

    def build_numpy_input6(self, Xpq):
        Xpq = np.asarray(Xpq, dtype=float)
        if Xpq.shape != (self.nbus, 2):
            raise ValueError(f"Xpq must have shape ({self.nbus}, 2), got {Xpq.shape}")

        P = Xpq[:, 0]
        Q = Xpq[:, 1]
        X6 = np.stack(
            [
                P,
                Q,
                self.S_down @ P,
                self.S_down @ Q,
                self.S_path @ P,
                self.S_path @ Q,
            ],
            axis=1,
        )
        return X6

    def build_numpy_input6_batch(self, Xpq):
        Xpq = np.asarray(Xpq, dtype=float)
        single = Xpq.ndim == 2
        if single:
            Xpq = Xpq[None, :, :]
        if Xpq.ndim != 3 or Xpq.shape[1:] != (self.nbus, 2):
            raise ValueError(
                f"Xpq must have shape ({self.nbus}, 2) or (batch, {self.nbus}, 2), "
                f"got {Xpq.shape}"
            )

        P = Xpq[:, :, 0]
        Q = Xpq[:, :, 1]
        X6 = np.stack(
            [
                P,
                Q,
                P @ self.S_down.T,
                Q @ self.S_down.T,
                P @ self.S_path.T,
                Q @ self.S_path.T,
            ],
            axis=2,
        )
        return X6[0] if single else X6

    def forward_numpy(self, Xpq, *, return_metrics=True):
        """Evaluate the Exp26 surrogate with numpy for heuristic search."""
        X6 = self.build_numpy_input6_batch(Xpq)
        single = X6.ndim == 2
        if single:
            X6 = X6[None, :, :]
        batch = X6.shape[0]
        Xn = (X6 - self.norm["X_mean"]) / self.norm["X_std"]

        W0, b0 = self.p("input_embed.weight"), self.p("input_embed.bias")
        H0 = np.einsum("bnf,hf->bnh", Xn, W0) + b0

        H = H0
        Hs = []
        for k in range(self.num_layers):
            W, b = self.lp(k, "gcn_linear.weight"), self.lp(k, "gcn_linear.bias")
            Hagg = np.einsum("ij,bjh->bih", self.A, H)
            Z = np.einsum("bni,hi->bnh", Hagg, W) + b
            if self.use_initial_anchor:
                Wa = self.lp(k, "anchor_linear.weight")
                alpha = sigmoid(self.scalar(k, "alpha_raw"))
                Z = Z + alpha * np.einsum("bni,hi->bnh", H0, Wa)

            R = np.maximum(Z, 0.0)
            if self.use_residual:
                beta = sigmoid(self.scalar(k, "beta_raw"))
                H = R + beta * H
            else:
                H = R
            Hs.append(H)

        src = ([H0] + Hs if self.include_input_in_jk else Hs) if self.use_jk else [Hs[-1]]
        Hread = np.concatenate(src, axis=2)

        node_x = Hread
        node_emb = self.p_opt("node_emb.weight")
        if node_emb is not None:
            emb = np.broadcast_to(node_emb[None, :, :], (batch, self.nbus, node_emb.shape[1]))
            node_x = np.concatenate([node_x, emb], axis=2)

        node_z = np.einsum("bni,ri->bnr", node_x, self.p("node_hidden.weight")) + self.p("node_hidden.bias")
        node_r = np.maximum(node_z, 0.0)
        Vres = (
            np.einsum("bni,ri->bnr", node_r, self.p("node_out.weight"))
            + self.p("node_out.bias")
        )[:, :, 0]
        Ws, bs = self.p_opt("node_skip.weight"), self.p_opt("node_skip.bias")
        if Ws is not None:
            Vres = Vres + (np.einsum("bni,ri->bnr", node_x, Ws) + bs)[:, :, 0]

        Vn = np.zeros((batch, self.nbus), dtype=float)
        flat_xn = Xn.reshape(batch, self.nbus * self.in_features)
        Vn[:, 1:] = Vres[:, 1:] + flat_xn @ self.W_vlin.T + self.b_vlin
        V = np.zeros_like(Vn)
        V[:, 0] = self.slack_voltage
        V[:, 1:] = Vn[:, 1:] * self.norm["YV_std"] + self.norm["YV_mean"]

        edges = np.asarray(self.edge_list, dtype=int)
        Hu = Hread[:, edges[:, 0], :]
        Hv = Hread[:, edges[:, 1], :]
        edge_x = np.concatenate([Hu, Hv, Hu - Hv], axis=2)
        edge_emb = self.p_opt("edge_emb.weight")
        if edge_emb is not None:
            emb = np.broadcast_to(edge_emb[None, :, :], (batch, self.nedge, edge_emb.shape[1]))
            edge_x = np.concatenate([edge_x, emb], axis=2)

        edge_z = np.einsum("bei,ri->ber", edge_x, self.p("edge_hidden.weight")) + self.p("edge_hidden.bias")
        edge_r = np.maximum(edge_z, 0.0)
        Ires = (
            np.einsum("bei,ri->ber", edge_r, self.p("edge_out.weight"))
            + self.p("edge_out.bias")
        )[:, :, 0]
        Ws, bs = self.p_opt("edge_skip.weight"), self.p_opt("edge_skip.bias")
        if Ws is not None:
            Ires = Ires + (np.einsum("bei,ri->ber", edge_x, Ws) + bs)[:, :, 0]

        Ilin = np.einsum("bef,ef->be", Xn[:, self.ilin_dst, :], self.W_ilin) + self.b_ilin
        I = (Ires + Ilin) * self.norm["YI_std"] + self.norm["YI_mean"]

        out = {"V": V, "I": I}
        if return_metrics:
            v_lower = float(self.engine.get("config", {}).get("v_lower", 0.95))
            v_upper = float(self.engine.get("config", {}).get("v_upper", 1.05))
            V_wo_slack = V[:, 1:]
            out.update(
                {
                    "Vdev_total": np.sum(np.abs(V_wo_slack - 1.0), axis=1),
                    "Vworst": np.maximum(
                        np.max(V_wo_slack - v_upper, axis=1),
                        np.max(v_lower - V_wo_slack, axis=1),
                    ),
                    "WorstI": np.max(I, axis=1),
                }
            )

        if single:
            return {
                key: (value[0] if isinstance(value, np.ndarray) and value.shape[:1] == (1,) else value)
                for key, value in out.items()
            }
        return out

    def compute_relu_starts(self, Xpq):
        X6 = self.build_numpy_input6(Xpq)
        Xn = (X6 - self.norm["X_mean"]) / self.norm["X_std"]

        W0, b0 = self.p("input_embed.weight"), self.p("input_embed.bias")
        H0 = Xn @ W0.T + b0

        H = H0
        Hs = []
        gcn_z = []

        for k in range(self.num_layers):
            W, b = self.lp(k, "gcn_linear.weight"), self.lp(k, "gcn_linear.bias")
            Hagg = self.A @ H
            Z = Hagg @ W.T + b
            if self.use_initial_anchor:
                Wa = self.lp(k, "anchor_linear.weight")
                alpha = sigmoid(self.scalar(k, "alpha_raw"))
                Z = Z + alpha * (H0 @ Wa.T)

            R = np.maximum(Z, 0.0)
            if self.use_residual:
                beta = sigmoid(self.scalar(k, "beta_raw"))
                H = R + beta * H
            else:
                H = R

            gcn_z.append(Z)
            Hs.append(H)

        src = ([H0] + Hs if self.include_input_in_jk else Hs) if self.use_jk else [Hs[-1]]
        Hread = np.concatenate(src, axis=1)

        node_x = Hread
        node_emb = self.p_opt("node_emb.weight")
        if node_emb is not None:
            node_x = np.concatenate([node_x, node_emb], axis=1)
        node_z = node_x @ self.p("node_hidden.weight").T + self.p("node_hidden.bias")

        edge_emb = self.p_opt("edge_emb.weight")
        edge_z = []
        for e, (u, v) in enumerate(self.edge_list):
            x = np.r_[Hread[u], Hread[v], Hread[u] - Hread[v]]
            if edge_emb is not None:
                x = np.r_[x, edge_emb[e]]
            edge_z.append(x @ self.p("edge_hidden.weight").T + self.p("edge_hidden.bias"))

        return {
            "gcn": gcn_z,
            "node": node_z,
            "edge": np.asarray(edge_z, dtype=float),
        }

    def interval_forward(self, Xn_lb, Xn_ub):
        W0, b0 = self.p("input_embed.weight"), self.p("input_embed.bias")

        H0_lb, H0_ub = [], []
        for i in range(self.nbus):
            lb, ub = aff_bound(W0, b0, Xn_lb[i], Xn_ub[i])
            H0_lb.append(lb)
            H0_ub.append(ub)

        H0_lb, H0_ub = np.array(H0_lb), np.array(H0_ub)
        H_lb, H_ub = H0_lb.copy(), H0_ub.copy()
        Z_lbs, Z_ubs, H_lbs, H_ubs = [], [], [], []

        for k in range(self.num_layers):
            W, b = self.lp(k, "gcn_linear.weight"), self.lp(k, "gcn_linear.bias")
            Wa = self.lp(k, "anchor_linear.weight") if self.use_initial_anchor else None
            alpha = sigmoid(self.scalar(k, "alpha_raw")) if self.use_initial_anchor else 0.0
            beta = sigmoid(self.scalar(k, "beta_raw")) if self.use_residual else 0.0

            Hagg_lb, Hagg_ub = self.A @ H_lb, self.A @ H_ub
            Z_lb, Z_ub = [], []

            for i in range(self.nbus):
                lb, ub = aff_bound(W, b, Hagg_lb[i], Hagg_ub[i])
                if self.use_initial_anchor:
                    al, au = aff_bound(Wa, np.zeros(self.hidden_dim), H0_lb[i], H0_ub[i])
                    lb, ub = lb + alpha * al, ub + alpha * au
                Z_lb.append(lb)
                Z_ub.append(ub)

            Z_lb, Z_ub = np.array(Z_lb), np.array(Z_ub)
            R_lb, R_ub = np.maximum(Z_lb, 0.0), np.maximum(Z_ub, 0.0)
            H_lb, H_ub = (R_lb + beta * H_lb, R_ub + beta * H_ub) if self.use_residual else (R_lb, R_ub)

            Z_lbs.append(Z_lb)
            Z_ubs.append(Z_ub)
            H_lbs.append(H_lb.copy())
            H_ubs.append(H_ub.copy())

        src_lb = ([H0_lb] + H_lbs if self.include_input_in_jk else H_lbs) if self.use_jk else [H_lbs[-1]]
        src_ub = ([H0_ub] + H_ubs if self.include_input_in_jk else H_ubs) if self.use_jk else [H_ubs[-1]]

        return {
            "gcn_lb": Z_lbs,
            "gcn_ub": Z_ubs,
            "Hread_lb": np.concatenate(src_lb, axis=1),
            "Hread_ub": np.concatenate(src_ub, axis=1),
        }

    @staticmethod
    def set_bound(var, lb, ub):
        if isinstance(var, gp.Var):
            if np.isfinite(lb):
                var.setAttr(GRB.Attr.LB, float(lb))
            if np.isfinite(ub):
                var.setAttr(GRB.Attr.UB, float(ub))

    def add_linear(self, m, W, b, x, name):
        W, b = np.asarray(W, float), np.asarray(b, float).reshape(-1)
        if W.shape[1] != len(x):
            raise ValueError(f"{name} input dimension error: expect={W.shape[1]}, actual={len(x)}")

        y = []
        for i in range(W.shape[0]):
            v = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_{i}")
            expr = gp.LinExpr(float(b[i]) if abs(float(b[i])) > self.weight_tol else 0.0)
            for j, c in enumerate(W[i]):
                c = float(c)
                if abs(c) > self.weight_tol:
                    expr += c * x[j]
            m.addConstr(v == expr, name=f"{name}_eq_{i}")
            y.append(v)
        return y

    def add_relu(self, m, z, Mp, Mm, name, dom=None, z_start=None, pri=1):
        y = []
        Mp, Mm = np.asarray(Mp, float).reshape(-1), np.asarray(Mm, float).reshape(-1)

        for d, zi in enumerate(z):
            ub = max(float(Mp[d]) * self.big_m_scale, self.min_big_m)
            lb = -max(float(Mm[d]) * self.big_m_scale, self.min_big_m)

            if dom is not None:
                lb = max(lb, float(dom[0][d]))
                ub = min(ub, float(dom[1][d]))

            if lb > ub:
                lb, ub = -self.fallback_big_m, self.fallback_big_m

            self.set_bound(zi, lb, ub)

            if ub <= 0.0:
                y.append(0.0)
                self.fixed_zero += 1
                continue
            if lb >= 0.0:
                y.append(zi)
                self.fixed_linear += 1
                continue

            yi = m.addVar(lb=0.0, ub=max(ub, self.min_big_m), name=f"{name}_{d}")
            if self.relu_formulation == "general":
                m.addGenConstrMax(yi, [zi], constant=0.0, name=f"{name}_max_{d}")
                y.append(yi)
                continue

            ai = m.addVar(vtype=GRB.BINARY, name=f"{name}_bin_{d}")
            ai.BranchPriority = int(pri)
            if z_start is not None:
                s = 1.0 if float(z_start[d]) >= 0.0 else 0.0
                ai.Start = s
                ai.VarHintVal = s

            self.binary_created += 1
            m.addConstr(yi >= zi, name=f"{name}_lb_{d}")
            m.addConstr(yi <= zi + (-lb) * (1.0 - ai), name=f"{name}_ub1_{d}")
            m.addConstr(yi <= ub * ai, name=f"{name}_ub2_{d}")
            y.append(yi)

        return y

    def build_input6(self, m, Xpq, name):
        Xn = [[None] * 6 for _ in range(self.nbus)]

        for i in range(self.nbus):
            Pd, Qd, Pp, Qp = gp.LinExpr(), gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
            for j in range(self.nbus):
                sd, sp = float(self.S_down[i, j]), float(self.S_path[i, j])
                if sd:
                    Pd += sd * Xpq[j][0]
                    Qd += sd * Xpq[j][1]
                if sp:
                    Pp += sp * Xpq[j][0]
                    Qp += sp * Xpq[j][1]

            raw = [Xpq[i][0], Xpq[i][1], Pd, Qd, Pp, Qp]
            for f in range(6):
                mean, std = self.norm["X_mean"][f], self.norm["X_std"][f]
                x = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_Xn_{i}_{f}")
                m.addConstr(x == raw[f] / std - mean / std, name=f"{name}_Xn_eq_{i}_{f}")
                Xn[i][f] = x

        return Xn

    def embed(self, m, Xpq, name="exp26", relu_starts=None, embed_edge=True):
        self.binary_created = 0
        self.fixed_zero = 0
        self.fixed_linear = 0

        relu_starts = relu_starts or {}
        Xn = self.build_input6(m, Xpq, name)

        W0, b0 = self.p("input_embed.weight"), self.p("input_embed.bias")
        H0 = [self.add_linear(m, W0, b0, Xn[i], f"{name}_in_{i}") for i in range(self.nbus)]

        H, Hs = H0, []
        for k in range(self.num_layers):
            W, b = self.lp(k, "gcn_linear.weight"), self.lp(k, "gcn_linear.bias")
            Wa = self.lp(k, "anchor_linear.weight") if self.use_initial_anchor else None
            alpha = sigmoid(self.scalar(k, "alpha_raw")) if self.use_initial_anchor else 0.0
            beta = sigmoid(self.scalar(k, "beta_raw")) if self.use_residual else 0.0
            Hn = []

            for i in range(self.nbus):
                Hagg = []
                for d in range(self.hidden_dim):
                    expr = gp.LinExpr()
                    for j in range(self.nbus):
                        a = float(self.A[i, j])
                        if a:
                            expr += a * H[j][d]
                    Hagg.append(expr)

                Z = self.add_linear(m, W, b, Hagg, f"{name}_L{k}_lin_{i}")
                if self.use_initial_anchor:
                    A0 = self.add_linear(m, Wa, np.zeros(self.hidden_dim), H0[i], f"{name}_L{k}_anc_{i}")
                    Z2 = []
                    for d in range(self.hidden_dim):
                        z = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_L{k}_z_{i}_{d}")
                        m.addConstr(z == Z[d] + alpha * A0[d], name=f"{name}_L{k}_z_eq_{i}_{d}")
                        Z2.append(z)
                    Z = Z2

                dom = None
                if self.domain is not None:
                    dom = (self.domain["gcn_lb"][k][i], self.domain["gcn_ub"][k][i])

                start = relu_starts.get("gcn", [None] * self.num_layers)[k][i] if "gcn" in relu_starts else None
                R = self.add_relu(
                    m,
                    Z,
                    self.M["M_plus_gcn_layers"][k][i],
                    self.M["M_minus_gcn_layers"][k][i],
                    f"{name}_L{k}_relu_{i}",
                    dom=dom,
                    z_start=start,
                    pri=100 - 20 * k,
                )

                if self.use_residual:
                    Hi = []
                    for d in range(self.hidden_dim):
                        h = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_L{k}_H_{i}_{d}")
                        m.addConstr(h == R[d] + beta * H[i][d], name=f"{name}_L{k}_res_{i}_{d}")
                        Hi.append(h)
                    Hn.append(Hi)
                else:
                    Hn.append(R)

            H = Hn
            Hs.append(H)

        src = ([H0] + Hs if self.include_input_in_jk else Hs) if self.use_jk else [Hs[-1]]
        Hread = [[v for layer in src for v in layer[i]] for i in range(self.nbus)]

        Vres = self.node_head(m, Hread, name, relu_starts.get("node"))
        V = self.denorm_v(m, Xn, Vres, name)

        I = []
        if embed_edge:
            In = self.edge_head(m, Hread, len(Hread[0]), name, relu_starts.get("edge"))
            I = self.denorm_i(m, Xn, In, name)

        aux = {
            "binary_created": self.binary_created,
            "relu_fixed_zero": self.fixed_zero,
            "relu_fixed_linear": self.fixed_linear,
            "binary_theoretical": self.engine.get("binary_count", {}),
        }
        return V, I, aux

    def node_head(self, m, Hread, name, starts=None):
        W1, b1 = self.p("node_hidden.weight"), self.p("node_hidden.bias")
        W2, b2 = self.p("node_out.weight"), self.p("node_out.bias")
        Ws, bs = self.p_opt("node_skip.weight"), self.p_opt("node_skip.bias")
        emb = self.p_opt("node_emb.weight")
        out = []

        for i in range(self.nbus):
            x = list(Hread[i])
            if emb is not None:
                x += [float(v) for v in emb[i]]

            z = self.add_linear(m, W1, b1, x, f"{name}_node_h_{i}")
            dom = None
            if self.domain is not None:
                lbx, ubx = self.domain["Hread_lb"][i], self.domain["Hread_ub"][i]
                if emb is not None:
                    lbx, ubx = np.r_[lbx, emb[i]], np.r_[ubx, emb[i]]
                dom = aff_bound(W1, b1, lbx, ubx)

            r = self.add_relu(
                m,
                z,
                self.M["M_plus_node"][i],
                self.M["M_minus_node"][i],
                f"{name}_node_relu_{i}",
                dom=dom,
                z_start=None if starts is None else starts[i],
                pri=60,
            )
            y = self.add_linear(m, W2, b2, r, f"{name}_node_out_{i}")[0]

            if Ws is not None:
                s = self.add_linear(m, Ws, bs, x, f"{name}_node_skip_{i}")[0]
                v = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_Vres_n_{i}")
                m.addConstr(v == y + s, name=f"{name}_Vres_n_eq_{i}")
                out.append(v)
            else:
                out.append(y)

        return out

    def edge_head(self, m, Hread, dim, name, starts=None):
        W1, b1 = self.p("edge_hidden.weight"), self.p("edge_hidden.bias")
        W2, b2 = self.p("edge_out.weight"), self.p("edge_out.bias")
        Ws, bs = self.p_opt("edge_skip.weight"), self.p_opt("edge_skip.bias")
        emb = self.p_opt("edge_emb.weight")
        out = []

        for e, (u, v) in enumerate(self.edge_list):
            x = list(Hread[u]) + list(Hread[v]) + [Hread[u][d] - Hread[v][d] for d in range(dim)]
            if emb is not None:
                x += [float(a) for a in emb[e]]

            z = self.add_linear(m, W1, b1, x, f"{name}_edge_h_{e}")
            dom = None
            if self.domain is not None:
                hu_lb, hu_ub = self.domain["Hread_lb"][u], self.domain["Hread_ub"][u]
                hv_lb, hv_ub = self.domain["Hread_lb"][v], self.domain["Hread_ub"][v]
                diff_lb, diff_ub = hu_lb - hv_ub, hu_ub - hv_lb
                lbx = np.r_[hu_lb, hv_lb, diff_lb]
                ubx = np.r_[hu_ub, hv_ub, diff_ub]
                if emb is not None:
                    lbx, ubx = np.r_[lbx, emb[e]], np.r_[ubx, emb[e]]
                dom = aff_bound(W1, b1, lbx, ubx)

            r = self.add_relu(
                m,
                z,
                self.M["M_plus_edge"][e],
                self.M["M_minus_edge"][e],
                f"{name}_edge_relu_{e}",
                dom=dom,
                z_start=None if starts is None else starts[e],
                pri=50,
            )
            y = self.add_linear(m, W2, b2, r, f"{name}_edge_out_{e}")[0]

            if Ws is not None:
                s = self.add_linear(m, Ws, bs, x, f"{name}_edge_skip_{e}")[0]
                im = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_Ires_n_{e}")
                m.addConstr(im == y + s, name=f"{name}_Ires_n_eq_{e}")
                out.append(im)
            else:
                out.append(y)

        return out

    def vlin_expr(self, Xn, bus):
        row = bus - 1
        expr = gp.LinExpr(float(self.b_vlin[row]))
        flat = [Xn[i][f] for i in range(self.nbus) for f in range(self.in_features)]
        for j, x in enumerate(flat):
            c = float(self.W_vlin[row, j])
            if abs(c) > self.weight_tol:
                expr += c * x
        return expr

    def ilin_expr(self, Xn, edge):
        dst = int(self.ilin_dst[edge])
        expr = gp.LinExpr(float(self.b_ilin[edge]))
        for f in range(self.in_features):
            c = float(self.W_ilin[edge, f])
            if abs(c) > self.weight_tol:
                expr += c * Xn[dst][f]
        return expr

    def denorm_v(self, m, Xn, Vres, name):
        V = [m.addVar(lb=self.slack_voltage, ub=self.slack_voltage, name=f"{name}_V_0")]
        for i in range(1, self.nbus):
            vn = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_Vn_{i}")
            m.addConstr(vn == Vres[i] + self.vlin_expr(Xn, i), name=f"{name}_Vn_eq_{i}")
            v = m.addVar(lb=0.80, ub=1.20, name=f"{name}_V_{i}")
            m.addConstr(v == vn * self.norm["YV_std"][i - 1] + self.norm["YV_mean"][i - 1], name=f"{name}_V_denorm_{i}")
            V.append(v)
        return V

    def denorm_i(self, m, Xn, In, name):
        I = []
        for e in range(self.nedge):
            inn = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"{name}_In_total_{e}")
            m.addConstr(inn == In[e] + self.ilin_expr(Xn, e), name=f"{name}_In_total_eq_{e}")
            x = m.addVar(lb=-2.0, ub=2.0, name=f"{name}_I_{e}")
            m.addConstr(x == inn * self.norm["YI_std"][e] + self.norm["YI_mean"][e], name=f"{name}_I_denorm_{e}")
            I.append(x)
        return I


class SGCNMILPConverter:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_ENGINE_PATH,
        *,
        relu_formulation: str = "big_m",
        fallback_big_m: float = 1e3,
        big_m_scale: float = 1.0,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Exp26 checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.relu_formulation = relu_formulation
        self.fallback_big_m = float(fallback_big_m)
        self.big_m_scale = float(big_m_scale)
        self.inner = Exp26STGCNCoreConverter(
            str(checkpoint_path),
            big_m_scale=big_m_scale,
            fallback_big_m=fallback_big_m,
            relu_formulation=relu_formulation,
        )

        self.engine = self.inner.engine
        self.payload = self.engine
        self.config = dict(self.engine.get("config", {}))
        self.num_nodes = int(self.inner.nbus)
        self.in_features = int(self.inner.in_features)
        self.edge_list = [[int(u), int(v)] for u, v in self.inner.edge_list]
        self.last_variables = None

    @property
    def binary_count(self) -> int:
        theoretical = self.engine.get("binary_count", {})
        if isinstance(theoretical, dict) and "total_binary" in theoretical:
            return int(theoretical["total_binary"])
        return int(
            self.inner.num_layers * self.inner.nbus * self.inner.hidden_dim
            + self.inner.nbus * self.inner.node_relu_dim
            + self.inner.nedge * self.inner.edge_relu_dim
        )

    def _check_topology(self, topo_mask) -> None:
        if topo_mask is None:
            return
        mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
        edge_count = len(self.edge_list)
        valid = mask.size == edge_count and bool(mask.all())
        if mask.size == edge_count + 5:
            valid = bool(mask[:edge_count].all() and (~mask[edge_count:]).all())
        if not valid:
            raise ValueError("Exp26 uses the frozen radial topology; topo_mask cannot change it")

    def _validate_inputs(self, x_vars: Sequence[Sequence]) -> None:
        if len(x_vars) != self.num_nodes:
            raise ValueError(f"X_vars must have {self.num_nodes} rows, got {len(x_vars)}")
        for i, row in enumerate(x_vars):
            if len(row) != 2:
                raise ValueError(f"X_vars[{i}] must contain [P_net, Q_net]")

    def _add_voltage_metrics(self, model, v_nodes, prefix):
        dev_vars = []
        violation_vars = []
        v_lower = float(self.config.get("v_lower", 0.95))
        v_upper = float(self.config.get("v_upper", 1.05))

        for i in range(1, self.num_nodes):
            dev = model.addVar(lb=0.0, name=f"{prefix}_Vdev_abs_{i}")
            model.addConstr(dev >= v_nodes[i] - 1.0, name=f"{prefix}_Vdev_pos_{i}")
            model.addConstr(dev >= 1.0 - v_nodes[i], name=f"{prefix}_Vdev_neg_{i}")
            dev_vars.append(dev)

            over = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vover_{i}")
            under = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vunder_{i}")
            model.addConstr(over == v_nodes[i] - v_upper, name=f"{prefix}_Vover_constr_{i}")
            model.addConstr(under == v_lower - v_nodes[i], name=f"{prefix}_Vunder_constr_{i}")
            violation_vars.extend([over, under])

        vdev_total = model.addVar(lb=0.0, name=f"{prefix}_Vdev_total")
        model.addConstr(vdev_total == gp.quicksum(dev_vars), name=f"{prefix}_Vdev_total_constr")
        vworst = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_Vworst")
        model.addGenConstrMax(vworst, violation_vars, name=f"{prefix}_Vworst_max")
        return vdev_total, vworst, dev_vars, violation_vars

    def _add_current_metric(self, model, i_edges, prefix):
        if len(i_edges) != len(self.edge_list):
            raise ValueError(f"Expected {len(self.edge_list)} current outputs, got {len(i_edges)}")
        worst_i = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_WorstI")
        model.addGenConstrMax(worst_i, list(i_edges), name=f"{prefix}_WorstI_max")
        return worst_i

    def embed_sgcn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp26",
        relu_starts=None,
    ) -> Exp26MILPOutputs:
        if gp is None or GRB is None:
            raise ModuleNotFoundError("gurobipy is required to embed Exp26 constraints")
        if not isinstance(model, gp.Model):
            raise TypeError("model must be a gurobipy.Model")
        self._validate_inputs(X_vars)
        self._check_topology(topo_mask)
        prefix = str(name_prefix).strip().replace(" ", "_") or "exp26"

        v_nodes, i_edges, aux = self.inner.embed(
            model,
            X_vars,
            name=prefix,
            relu_starts=relu_starts,
            embed_edge=True,
        )
        vdev_total, vworst, vdev_abs, v_violations = self._add_voltage_metrics(model, v_nodes, prefix)
        worst_i = self._add_current_metric(model, i_edges, prefix)
        ploss_placeholder = model.addVar(lb=0.0, ub=0.0, name=f"{prefix}_Ploss_total_placeholder")

        self.last_variables = {
            "V_nodes": v_nodes,
            "I_edges": i_edges,
            "Vdev_abs": vdev_abs,
            "V_violations": v_violations,
            "embed_aux": aux,
        }
        return Exp26MILPOutputs(vdev_total, vworst, worst_i, ploss_placeholder, v_nodes)

    def embed_gnn_constraints(
        self,
        model,
        X_vars: Sequence[Sequence],
        topo_mask=None,
        *,
        name_prefix: str = "exp26",
        relu_starts=None,
    ) -> Exp26MILPOutputs:
        return self.embed_sgcn_constraints(
            model,
            X_vars,
            topo_mask,
            name_prefix=name_prefix,
            relu_starts=relu_starts,
        )


Exp26STGCNMILPConverter = SGCNMILPConverter


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Exp26 ST-GCN MILP converter configuration")
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument("--relu-formulation", choices=("big_m", "general"), default="big_m")
    args = parser.parse_args()
    converter = SGCNMILPConverter(args.checkpoint, relu_formulation=args.relu_formulation)
    print(f"checkpoint: {converter.checkpoint_path}")
    print(
        "shape: "
        f"nodes={converter.num_nodes}, features={converter.in_features}, "
        f"hidden={converter.inner.hidden_dim}, layers={converter.inner.num_layers}, "
        f"node_relu={converter.inner.node_relu_dim}, edge_relu={converter.inner.edge_relu_dim}"
    )
    print(f"ReLU formulation: {converter.relu_formulation}")
    print(f"binary count: {converter.binary_count}")
    print("outputs: Vdev_total, Vworst, WorstI, Ploss placeholder, V_nodes")


if __name__ == "__main__":
    _main()
