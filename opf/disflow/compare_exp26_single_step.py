"""Compare Exp26 ST-GCN-MILP voltage-deviation RPO with an exact DistFlow baseline.

The Exp26 MILP optimizes controllable reactive injections through a learned
surrogate. The controllable set is PV, ESS, and reactive devices available in
the selected operating point. This script evaluates that dispatch with an exact
radial DistFlow calculation, solves a second exact DistFlow-based continuous
RPO problem with the same voltage-deviation objective, and writes voltage and
dispatch comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:
    scipy_minimize = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opf.exp26_milp import single_step_rpo as exp26_rpo


RADIAL_BRANCHES = np.array(
    [
        [0, 1, 0.0922, 0.0470],
        [1, 2, 0.4930, 0.2511],
        [2, 3, 0.3660, 0.1864],
        [3, 4, 0.3811, 0.1941],
        [4, 5, 0.8190, 0.7070],
        [5, 6, 0.1872, 0.6188],
        [6, 7, 0.7114, 0.2351],
        [7, 8, 1.0300, 0.7400],
        [8, 9, 1.0440, 0.7400],
        [9, 10, 0.1966, 0.0650],
        [10, 11, 0.3744, 0.1238],
        [11, 12, 1.4680, 1.1550],
        [12, 13, 0.5416, 0.7129],
        [13, 14, 0.5910, 0.5260],
        [14, 15, 0.7463, 0.5450],
        [15, 16, 1.2890, 1.7210],
        [16, 17, 0.7320, 0.5740],
        [1, 18, 0.1640, 0.1565],
        [18, 19, 1.5042, 1.3554],
        [19, 20, 0.4095, 0.4784],
        [20, 21, 0.7089, 0.9373],
        [2, 22, 0.4512, 0.3083],
        [22, 23, 0.8980, 0.7091],
        [23, 24, 0.8960, 0.7011],
        [5, 25, 0.2030, 0.1034],
        [25, 26, 0.2842, 0.1447],
        [26, 27, 1.0590, 0.9337],
        [27, 28, 0.8042, 0.7006],
        [28, 29, 0.5075, 0.2585],
        [29, 30, 0.9744, 0.9630],
        [30, 31, 0.3105, 0.3619],
        [31, 32, 0.3410, 0.5302],
    ],
    dtype=float,
)


@dataclass
class OperatingPoint:
    x_base: np.ndarray | None
    p_load: np.ndarray
    q_load: np.ndarray
    p_pv: np.ndarray
    pv_q_base: np.ndarray
    topo_mask: np.ndarray
    pv_nodes: np.ndarray
    s_rated: np.ndarray
    ess_nodes: np.ndarray
    ess_s_rated: np.ndarray
    ess_p_base: np.ndarray
    ess_q_base: np.ndarray
    q_device_nodes: np.ndarray
    q_device_min: np.ndarray
    q_device_max: np.ndarray
    q_device_q_base: np.ndarray
    q_device_names: list[str]
    branches: np.ndarray | None
    meta: dict


@dataclass
class DistFlowResult:
    converged: bool
    iterations: int
    voltage_pu: np.ndarray
    branch_p_mw: np.ndarray
    branch_q_mvar: np.ndarray
    branch_current_ka: np.ndarray
    branch_loss_mw: np.ndarray
    total_loss_mw: float
    vdev_total: float
    vworst: float
    worst_i_margin: float


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value, *, dtype=float) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def get_line_max_i_ka(data_path: str | Path, fallback: float) -> float:
    data_path = resolve_repo_path(data_path)
    if not data_path.is_file():
        return float(fallback)
    try:
        data = safe_torch_load(data_path)
    except Exception:
        return float(fallback)
    base_config = data.get("base_config", {}) if isinstance(data, dict) else {}
    return float(base_config.get("line_max_i_ka", fallback))


def load_operating_point(args) -> OperatingPoint:
    ess_nodes = np.array([], dtype=int)
    ess_s_rated = np.array([], dtype=float)
    ess_p_base = np.array([], dtype=float)
    ess_q_base = np.array([], dtype=float)
    q_device_nodes = np.array([], dtype=int)
    q_device_min = np.array([], dtype=float)
    q_device_max = np.array([], dtype=float)
    q_device_q_base = np.array([], dtype=float)
    q_device_names: list[str] = []

    if args.source == "dataset":
        cfg = {"v_lower": args.v_lower, "v_upper": args.v_upper}
        x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta = (
            exp26_rpo.get_dataset_operating_point(
                args.data,
                cfg,
                sample_index=args.sample_index,
            )
        )
        p_load = np.zeros(33, dtype=float)
        q_load = np.zeros(33, dtype=float)

        data = safe_torch_load(resolve_repo_path(args.data))
        sample_idx = int(meta["sample_index"])
        branches = to_numpy(data.get("branch_full")) if "branch_full" in data else None
        ess_nodes = to_numpy(data.get("ess_nodes"), dtype=int).reshape(-1)
        ess_s_rated = to_numpy(data.get("S_ess_mva")).reshape(-1)
        q_device_nodes = to_numpy(data.get("q_device_nodes"), dtype=int).reshape(-1)
        q_device_min = to_numpy(data.get("q_device_min")).reshape(-1)
        q_device_max = to_numpy(data.get("q_device_max")).reshape(-1)
        if "ess_p" in data and ess_nodes.size:
            ess_p_base = to_numpy(data["ess_p"][sample_idx]).reshape(-1)
        if "ess_q" in data and ess_nodes.size:
            ess_q_base = to_numpy(data["ess_q"][sample_idx]).reshape(-1)
        if "qdev_q" in data and q_device_nodes.size:
            q_device_q_base = to_numpy(data["qdev_q"][sample_idx]).reshape(-1)
        if "q_device_names" in data:
            q_device_names = [str(name) for name in data["q_device_names"]]
    else:
        p_load, q_load, p_pv, topo_mask, pv_nodes, s_rated = (
            exp26_rpo.get_default_operating_point(pv_pu=args.pv_pu)
        )
        x_base = None
        pv_q_base = np.zeros(len(pv_nodes), dtype=float)
        branches = None
        meta = {"source": "profile"}

    p_pv_override = exp26_rpo.parse_vector_arg(
        args.p_pv,
        expected=len(pv_nodes),
        name="p_pv",
    )
    if p_pv_override is not None:
        p_pv = p_pv_override

    if args.source == "profile":
        p_load_override = exp26_rpo.parse_vector_arg(
            args.p_load,
            expected=33,
            name="p_load",
        )
        q_load_override = exp26_rpo.parse_vector_arg(
            args.q_load,
            expected=33,
            name="q_load",
        )
        if p_load_override is not None:
            p_load = p_load_override
        if q_load_override is not None:
            q_load = q_load_override

    exp26_rpo.validate_inputs(p_load, q_load, p_pv, topo_mask, pv_nodes, s_rated)
    return OperatingPoint(
        x_base=None if x_base is None else np.asarray(x_base, dtype=float),
        p_load=np.asarray(p_load, dtype=float),
        q_load=np.asarray(q_load, dtype=float),
        p_pv=np.asarray(p_pv, dtype=float),
        pv_q_base=np.asarray(pv_q_base, dtype=float),
        topo_mask=np.asarray(topo_mask, dtype=bool),
        pv_nodes=np.asarray(pv_nodes, dtype=int),
        s_rated=np.asarray(s_rated, dtype=float),
        ess_nodes=ess_nodes,
        ess_s_rated=ess_s_rated,
        ess_p_base=ess_p_base,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
        branches=branches,
        meta=meta,
    )


def q_capacity(p_pv: np.ndarray, s_rated: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(s_rated) ** 2 - np.asarray(p_pv) ** 2, 0.0))


def control_labels(op: OperatingPoint) -> list[str]:
    labels = [f"PV@Bus{int(bus) + 1}" for bus in op.pv_nodes]
    labels.extend(f"ESS@Bus{int(bus) + 1}" for bus in op.ess_nodes)
    for idx, bus in enumerate(op.q_device_nodes):
        name = op.q_device_names[idx] if idx < len(op.q_device_names) else f"QDev{idx + 1}"
        labels.append(f"{name}@Bus{int(bus) + 1}")
    return labels


def control_base(op: OperatingPoint) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(op.pv_q_base, dtype=float).reshape(-1),
            np.asarray(op.ess_q_base, dtype=float).reshape(-1),
            np.asarray(op.q_device_q_base, dtype=float).reshape(-1),
        ]
    )


def control_bounds(op: OperatingPoint) -> tuple[np.ndarray, np.ndarray]:
    pv_caps = q_capacity(op.p_pv, op.s_rated)
    ess_caps = np.sqrt(
        np.maximum(
            np.asarray(op.ess_s_rated, dtype=float) ** 2
            - np.asarray(op.ess_p_base, dtype=float) ** 2,
            0.0,
        )
    )
    lower = np.concatenate([-pv_caps, -ess_caps, np.asarray(op.q_device_min, dtype=float)])
    upper = np.concatenate([pv_caps, ess_caps, np.asarray(op.q_device_max, dtype=float)])
    return lower, upper


def baseline_control(op: OperatingPoint) -> np.ndarray:
    q = control_base(op)
    q[: len(op.pv_nodes)] = 0.0
    lower, upper = control_bounds(op)
    return np.clip(q, lower, upper)


def split_control(op: OperatingPoint, q_control: np.ndarray):
    q = np.asarray(q_control, dtype=float).reshape(-1)
    n_pv = len(op.pv_nodes)
    n_ess = len(op.ess_nodes)
    n_qdev = len(op.q_device_nodes)
    expected = n_pv + n_ess + n_qdev
    if q.size != expected:
        raise ValueError(f"q_control must have length {expected}, got {q.size}")
    q_pv = q[:n_pv]
    q_ess = q[n_pv : n_pv + n_ess]
    q_qdev = q[n_pv + n_ess :]
    return q_pv, q_ess, q_qdev


def build_net_injection(op: OperatingPoint, q_control: np.ndarray) -> np.ndarray:
    q_pv, q_ess, q_qdev = split_control(op, q_control)
    if op.x_base is not None:
        x = op.x_base.copy()
        for idx, bus in enumerate(op.pv_nodes):
            x[int(bus), 1] += -op.pv_q_base[idx] + q_pv[idx]
        for idx, bus in enumerate(op.ess_nodes):
            x[int(bus), 1] += -op.ess_q_base[idx] + q_ess[idx]
        for idx, bus in enumerate(op.q_device_nodes):
            x[int(bus), 1] += -op.q_device_q_base[idx] + q_qdev[idx]
        return x

    x = np.zeros((33, 2), dtype=float)
    x[:, 0] = -op.p_load
    x[:, 1] = -op.q_load
    for idx, bus in enumerate(op.pv_nodes):
        x[int(bus), 0] += op.p_pv[idx]
        x[int(bus), 1] += q_pv[idx]
    return x


class ExactDistFlow:
    def __init__(
        self,
        *,
        branches: np.ndarray = RADIAL_BRANCHES,
        slack_vm_pu: float = 1.03,
        base_kv: float = 12.66,
        base_mva: float = 1.0,
        line_max_i_ka: float = 0.20,
    ):
        self.branches = np.asarray(branches, dtype=float)
        self.num_nodes = 33
        self.slack_vm_pu = float(slack_vm_pu)
        self.base_kv = float(base_kv)
        self.base_mva = float(base_mva)
        self.line_max_i_ka = float(line_max_i_ka)
        self.z_base_ohm = self.base_kv**2 / self.base_mva
        self.i_base_ka = self.base_mva / (np.sqrt(3.0) * self.base_kv)
        self.parent = np.full(self.num_nodes, -1, dtype=int)
        self.parent_branch = np.full(self.num_nodes, -1, dtype=int)
        self.children = [[] for _ in range(self.num_nodes)]
        for branch_idx, (fr, to, _, _) in enumerate(self.branches):
            fr_i = int(fr)
            to_i = int(to)
            self.parent[to_i] = fr_i
            self.parent_branch[to_i] = branch_idx
            self.children[fr_i].append(to_i)
        self.postorder = list(range(self.num_nodes - 1, 0, -1))

    def solve(
        self,
        x_net_mw_mvar: np.ndarray,
        *,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> DistFlowResult:
        x = np.asarray(x_net_mw_mvar, dtype=float)
        if x.shape != (self.num_nodes, 2):
            raise ValueError(f"x_net_mw_mvar must have shape (33, 2), got {x.shape}")

        p_demand = -x[:, 0] / self.base_mva
        q_demand = -x[:, 1] / self.base_mva
        r_pu = self.branches[:, 2] / self.z_base_ohm
        x_pu = self.branches[:, 3] / self.z_base_ohm

        v = np.full(self.num_nodes, self.slack_vm_pu**2, dtype=float)
        ell = np.zeros(len(self.branches), dtype=float)
        p_flow = np.zeros(len(self.branches), dtype=float)
        q_flow = np.zeros(len(self.branches), dtype=float)
        converged = False

        for iteration in range(1, max_iter + 1):
            old_v = v.copy()
            old_ell = ell.copy()

            for node in self.postorder:
                branch_idx = self.parent_branch[node]
                child_branches = [self.parent_branch[c] for c in self.children[node]]
                p_flow[branch_idx] = (
                    p_demand[node]
                    + np.sum(p_flow[child_branches])
                    + r_pu[branch_idx] * ell[branch_idx]
                )
                q_flow[branch_idx] = (
                    q_demand[node]
                    + np.sum(q_flow[child_branches])
                    + x_pu[branch_idx] * ell[branch_idx]
                )

            v[0] = self.slack_vm_pu**2
            for branch_idx, (fr, to, _, _) in enumerate(self.branches):
                fr_i = int(fr)
                to_i = int(to)
                v_parent = max(v[fr_i], 1e-8)
                ell[branch_idx] = (
                    p_flow[branch_idx] ** 2 + q_flow[branch_idx] ** 2
                ) / v_parent
                v[to_i] = (
                    v[fr_i]
                    - 2.0
                    * (
                        r_pu[branch_idx] * p_flow[branch_idx]
                        + x_pu[branch_idx] * q_flow[branch_idx]
                    )
                    + (r_pu[branch_idx] ** 2 + x_pu[branch_idx] ** 2)
                    * ell[branch_idx]
                )

            delta = max(
                float(np.max(np.abs(v - old_v))),
                float(np.max(np.abs(ell - old_ell))),
            )
            if delta < tol and np.all(v > 0.0):
                converged = True
                break

        voltage = np.sqrt(np.maximum(v, 0.0))
        current_ka = np.sqrt(np.maximum(ell, 0.0)) * self.i_base_ka
        branch_loss_mw = 3.0 * current_ka**2 * self.branches[:, 2]
        total_loss_mw = float(np.sum(branch_loss_mw))
        vdev_total = float(np.sum(np.abs(voltage[1:] - 1.0)))
        vworst = float(
            max(
                np.max(voltage[1:] - 1.05),
                np.max(0.95 - voltage[1:]),
            )
        )
        worst_i_margin = float(np.max(current_ka / self.line_max_i_ka - 1.0))

        return DistFlowResult(
            converged=converged,
            iterations=iteration,
            voltage_pu=voltage,
            branch_p_mw=p_flow * self.base_mva,
            branch_q_mvar=q_flow * self.base_mva,
            branch_current_ka=current_ka,
            branch_loss_mw=branch_loss_mw,
            total_loss_mw=total_loss_mw,
            vdev_total=vdev_total,
            vworst=vworst,
            worst_i_margin=worst_i_margin,
        )


def exact_objective(
    metrics: DistFlowResult,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
) -> float:
    if objective == "vdev":
        return metrics.vdev_total
    raise ValueError("Exp26 comparison uses voltage-deviation objective only")


def solve_exact_distflow_rpo(
    op: OperatingPoint,
    distflow: ExactDistFlow,
    *,
    objective: str,
    loss_weight: float,
    voltage_weight: float,
    v_lower: float,
    v_upper: float,
    enforce_current: bool,
    max_iter: int,
) -> dict:
    lower, upper = control_bounds(op)
    q0 = baseline_control(op)
    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    cache: dict[tuple[float, ...], DistFlowResult] = {}

    def eval_q(q_vec):
        key = tuple(np.round(np.asarray(q_vec, dtype=float), 12))
        if key not in cache:
            cache[key] = distflow.solve(build_net_injection(op, np.asarray(q_vec)))
        return cache[key]

    def objective_fun(q_vec):
        result = eval_q(q_vec)
        if not result.converged or not np.all(np.isfinite(result.voltage_pu)):
            return 1e6
        return exact_objective(
            result,
            objective,
            loss_weight=loss_weight,
            voltage_weight=voltage_weight,
        )

    constraints = [
        {"type": "ineq", "fun": lambda q: eval_q(q).voltage_pu[1:] - float(v_lower)},
        {"type": "ineq", "fun": lambda q: float(v_upper) - eval_q(q).voltage_pu[1:]},
    ]
    if enforce_current:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda q: distflow.line_max_i_ka - eval_q(q).branch_current_ka,
            }
        )

    starts = [
        q0,
        np.zeros_like(q0),
        control_base(op),
        0.5 * upper,
        0.5 * lower,
        0.95 * upper,
        0.95 * lower,
    ]

    best = None
    best_metrics = None
    for start in starts:
        start = np.clip(start, lower, upper)
        if scipy_minimize is None:
            opt_result = coordinate_search(
                start,
                lower,
                upper,
                objective_fun,
                lambda q: constraint_violation(
                    eval_q(q),
                    v_lower=v_lower,
                    v_upper=v_upper,
                    line_max_i_ka=distflow.line_max_i_ka,
                    enforce_current=enforce_current,
                ),
                max_iter=max_iter,
            )
        else:
            opt_result = scipy_minimize(
                objective_fun,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": int(max_iter), "ftol": 1e-9, "disp": False},
            )
        metrics = eval_q(opt_result.x)
        max_violation = constraint_violation(
            metrics,
            v_lower=v_lower,
            v_upper=v_upper,
            line_max_i_ka=distflow.line_max_i_ka,
            enforce_current=enforce_current,
        )
        score = objective_fun(opt_result.x) + 1e5 * max(0.0, max_violation)
        if best is None or score < best[0]:
            best = (score, opt_result, max_violation)
            best_metrics = metrics

    assert best is not None and best_metrics is not None
    _, opt_result, max_violation = best
    return {
        "success": bool(opt_result.success and max_violation <= 1e-6),
        "message": str(opt_result.message),
        "objective": float(objective_fun(opt_result.x)),
        "max_constraint_violation": float(max_violation),
        "Q_control_opt": np.asarray(opt_result.x, dtype=float),
        "metrics": best_metrics,
    }


class SimpleOptimizeResult:
    def __init__(self, x, success, message):
        self.x = np.asarray(x, dtype=float)
        self.success = bool(success)
        self.message = str(message)


def coordinate_search(q0, lower, upper, objective_fun, violation_fun, *, max_iter: int):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    span = np.maximum(upper - lower, 0.0)
    steps = np.maximum(0.25 * span, 1e-4)

    def score(q_vec):
        violation = max(0.0, float(violation_fun(q_vec)))
        return float(objective_fun(q_vec)) + 1e5 * violation

    best_score = score(q)
    for _ in range(int(max_iter)):
        improved = False
        for idx in range(q.size):
            for sign in (1.0, -1.0):
                trial = q.copy()
                trial[idx] = np.clip(
                    trial[idx] + sign * steps[idx],
                    lower[idx],
                    upper[idx],
                )
                trial_score = score(trial)
                if trial_score + 1e-12 < best_score:
                    q = trial
                    best_score = trial_score
                    improved = True
        if not improved:
            steps *= 0.5
            if float(np.max(steps)) < 1e-5:
                break

    success = max(0.0, float(violation_fun(q))) <= 1e-6
    backend = "coordinate_search fallback (SciPy unavailable)"
    return SimpleOptimizeResult(q, success, backend)


def constraint_violation(
    metrics: DistFlowResult,
    *,
    v_lower: float,
    v_upper: float,
    line_max_i_ka: float,
    enforce_current: bool,
) -> float:
    violation = max(
        float(np.max(float(v_lower) - metrics.voltage_pu[1:])),
        float(np.max(metrics.voltage_pu[1:] - float(v_upper))),
        0.0,
    )
    if enforce_current:
        violation = max(violation, float(np.max(metrics.branch_current_ka - line_max_i_ka)))
    return violation


def run_milp(op: OperatingPoint, args) -> dict:
    common_kwargs = dict(
        pv_nodes=op.pv_nodes,
        s_rated=op.s_rated,
        engine_path=args.engine,
        relu_formulation=args.relu_formulation,
        big_m_scale=args.big_m_scale,
        fallback_big_m=args.fallback_big_m,
        objective=args.objective,
        loss_objective_model=args.loss_objective_model,
        surrogate_loss_weight=args.surrogate_loss_weight,
        surrogate_loss_consistency_rel=args.surrogate_loss_consistency_rel,
        surrogate_loss_consistency_abs=args.surrogate_loss_consistency_abs,
        loss_weight=args.loss_weight,
        voltage_weight=args.voltage_weight,
        lambda_q=args.lambda_q,
        voltage_margin=args.voltage_margin,
        current_margin=args.current_margin,
        physical_output_bounds=not args.no_physical_output_bounds,
        trust_region_fraction=args.trust_region_fraction,
        trust_region_radius=args.trust_region_radius,
        control_deviation_penalty=args.control_deviation_penalty,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        mip_focus=args.mip_focus,
        method=args.method,
        node_method=args.node_method,
        presolve=args.presolve,
        presparsify=args.presparsify,
        heuristics=args.heuristics,
        cuts=args.cuts,
        var_branch=args.var_branch,
        numeric_focus=args.numeric_focus,
        dual_reductions=args.dual_reductions,
        diagnose_slack=args.diagnose_slack,
        slack_penalty=args.slack_penalty,
        output_flag=0 if args.quiet_milp else 1,
        export_lp=args.export_lp,
        iis_path=args.iis,
        x_base=op.x_base,
        pv_q_base=op.pv_q_base if op.x_base is not None else None,
        ess_nodes=op.ess_nodes,
        ess_s_rated=op.ess_s_rated,
        ess_p_base=op.ess_p_base,
        ess_q_base=op.ess_q_base,
        q_device_nodes=op.q_device_nodes,
        q_device_min=op.q_device_min,
        q_device_max=op.q_device_max,
        q_device_q_base=op.q_device_q_base,
        q_device_names=op.q_device_names,
        warm_start_q=args.warm_start_q,
        sc_integer=not args.no_sc_integer,
        sc_step=args.sc_step,
    )
    if args.strategy == "single":
        return exp26_rpo.run_single_step_rpo(
            op.p_load,
            op.q_load,
            op.p_pv,
            op.topo_mask,
            **common_kwargs,
        )

    return exp26_rpo.run_two_stage_rpo(
        op.p_load,
        op.q_load,
        op.p_pv,
        op.topo_mask,
        heuristic_random_samples=args.heuristic_random_samples,
        heuristic_batch_size=args.heuristic_batch_size,
        heuristic_safety_penalty=args.heuristic_safety_penalty,
        heuristic_seed=args.heuristic_seed,
        local_trust_region_fraction=args.local_trust_region_fraction,
        local_trust_region_radius=args.local_trust_radius,
        torch_warm_starts=args.torch_warm_starts,
        torch_warm_steps=args.torch_warm_steps,
        torch_warm_lr=args.torch_warm_lr,
        torch_q_penalty=args.torch_q_penalty,
        **common_kwargs,
    )


def q_dispatch_dict(op: OperatingPoint, q_control: np.ndarray) -> dict[str, float]:
    labels = control_labels(op)
    values = np.asarray(q_control, dtype=float).reshape(-1)
    if values.size != len(labels):
        raise ValueError(f"q_control must have length {len(labels)}, got {values.size}")
    return {label: float(value) for label, value in zip(labels, values)}


def summarize_metrics(
    name: str,
    q_control: np.ndarray,
    metrics: DistFlowResult,
    op: OperatingPoint,
) -> dict:
    q_pv, q_ess, q_qdev = split_control(op, q_control)
    return {
        "name": name,
        "q_pv": [float(x) for x in np.asarray(q_pv).reshape(-1)],
        "q_ess": [float(x) for x in np.asarray(q_ess).reshape(-1)],
        "q_device": [float(x) for x in np.asarray(q_qdev).reshape(-1)],
        "q_dispatch": q_dispatch_dict(op, q_control),
        "v_min": float(np.min(metrics.voltage_pu[1:])),
        "v_max": float(np.max(metrics.voltage_pu[1:])),
        "vdev_total": float(metrics.vdev_total),
        "vworst": float(metrics.vworst),
        "worst_i_margin": float(metrics.worst_i_margin),
        "total_loss_mw": float(metrics.total_loss_mw),
        "distflow_converged": bool(metrics.converged),
        "distflow_iterations": int(metrics.iterations),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "v_min",
        "v_max",
        "vdev_total",
        "vworst",
        "worst_i_margin",
        "total_loss_mw",
        "distflow_converged",
        "distflow_iterations",
        "q_pv",
        "q_ess",
        "q_device",
        "q_dispatch",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["q_pv"] = json.dumps(out["q_pv"])
            out["q_ess"] = json.dumps(out["q_ess"])
            out["q_device"] = json.dumps(out["q_device"])
            out["q_dispatch"] = json.dumps(out["q_dispatch"])
            writer.writerow(out)


def make_plots(
    out_dir: Path,
    cases: list[tuple[str, np.ndarray, DistFlowResult]],
    *,
    op: OperatingPoint,
    q_caps: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bus = np.arange(1, 34)
    voltage_styles = [
        {"color": "#4C78A8", "linestyle": "-", "marker": "o", "fillstyle": "full"},
        {"color": "#F58518", "linestyle": "--", "marker": "s", "fillstyle": "none"},
        {"color": "#54A24B", "linestyle": ":", "marker": "^", "fillstyle": "none"},
        {"color": "#B279A2", "linestyle": "-.", "marker": "D", "fillstyle": "none"},
        {"color": "#E45756", "linestyle": (0, (3, 1, 1, 1)), "marker": "x", "fillstyle": "full"},
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.25]},
    )
    ref_name, _, ref_metrics = cases[-1]
    for idx, (name, _, metrics) in enumerate(cases):
        style = voltage_styles[idx % len(voltage_styles)]
        axes[0].plot(
            bus,
            metrics.voltage_pu,
            markersize=4.2,
            linewidth=1.9,
            alpha=0.92,
            label=name,
            markeredgewidth=1.2,
            **style,
        )
        delta_mpu = 1000.0 * (metrics.voltage_pu - ref_metrics.voltage_pu)
        axes[1].plot(
            bus,
            delta_mpu,
            markersize=3.6,
            linewidth=1.4,
            alpha=0.9,
            label=name,
            markeredgewidth=1.0,
            **style,
        )
    axes[0].axhline(1.05, color="tab:red", linestyle="--", linewidth=1.0, label="V upper/lower")
    axes[0].axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].set_title("Exact DistFlow Voltage Comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Bus")
    axes[1].set_ylabel("Delta V (1e-3 p.u.)")
    axes[1].set_title(f"Voltage Difference vs {ref_name}")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "voltage_comparison.png")
    plt.close(fig)

    dispatches = [q_dispatch_dict(op, case[1]) for case in cases]
    labels = list(dispatches[0].keys())
    q_values = np.array(
        [[dispatch.get(label, 0.0) for label in labels] for dispatch in dispatches],
        dtype=float,
    )
    x = np.arange(len(labels), dtype=float)
    width = min(0.8 / max(len(cases), 1), 0.26)
    offsets = (np.arange(len(cases)) - (len(cases) - 1) / 2.0) * width
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    for idx, (name, _, _) in enumerate(cases):
        ax.bar(
            x + offsets[idx],
            q_values[idx],
            width=width,
            label=name,
            color=colors[idx % len(colors)],
        )
    pv_count = len(op.pv_nodes)
    if pv_count:
        ax.scatter(
            x[:pv_count],
            q_caps,
            color="black",
            marker="_",
            s=180,
            linewidths=1.4,
            label="PV +Q cap",
        )
        ax.scatter(
            x[:pv_count],
            -q_caps,
            color="black",
            marker="_",
            s=180,
            linewidths=1.4,
            label="PV -Q cap",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("Reactive device")
    ax.set_ylabel("Q output (MVar)")
    ax.set_title("Reactive Device Dispatch Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "q_dispatch_comparison.png")
    plt.close(fig)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, DistFlowResult):
        return {
            "converged": value.converged,
            "iterations": value.iterations,
            "voltage_pu": value.voltage_pu.tolist(),
            "branch_p_mw": value.branch_p_mw.tolist(),
            "branch_q_mvar": value.branch_q_mvar.tolist(),
            "branch_current_ka": value.branch_current_ka.tolist(),
            "branch_loss_mw": value.branch_loss_mw.tolist(),
            "total_loss_mw": value.total_loss_mw,
            "vdev_total": value.vdev_total,
            "vworst": value.vworst,
            "worst_i_margin": value.worst_i_margin,
        }
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Exp26 ST-GCN-MILP RPO, solve exact DistFlow RPO, and write exact "
            "voltage-deviation comparison plots."
        )
    )
    parser.add_argument("--engine", default=str(exp26_rpo.DEFAULT_ENGINE_PATH))
    parser.add_argument(
        "--strategy",
        choices=["two_stage", "single"],
        default=exp26_rpo.DEFAULT_RUN_CONFIG["strategy"],
    )
    parser.add_argument(
        "--source",
        choices=["dataset", "profile"],
        default="dataset",
    )
    parser.add_argument("--data", default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt")
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--pv-pu", type=float, default=0.8)
    parser.add_argument("--p-load", default=None)
    parser.add_argument("--q-load", default=None)
    parser.add_argument("--p-pv", default=None)
    parser.add_argument(
        "--objective",
        choices=["vdev"],
        default="vdev",
        help="Exp26 has no learned Ploss head, so comparison uses Vdev only.",
    )
    parser.add_argument(
        "--loss-objective-model",
        choices=["none"],
        default="none",
    )
    parser.add_argument("--surrogate-loss-weight", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-rel", type=float, default=0.5)
    parser.add_argument("--surrogate-loss-consistency-abs", type=float, default=0.002)
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--voltage-weight",
        type=float,
        default=1e-3,
        help="Unused by the default Exp26 MILP vdev objective.",
    )
    parser.add_argument(
        "--lambda-q",
        type=float,
        default=0.0,
        help="Unused for Exp26 voltage-only objective; must remain 0.",
    )
    parser.add_argument("--v-lower", type=float, default=0.95)
    parser.add_argument("--v-upper", type=float, default=1.05)
    parser.add_argument("--slack-vm-pu", type=float, default=1.03)
    parser.add_argument("--base-kv", type=float, default=12.66)
    parser.add_argument("--base-mva", type=float, default=1.0)
    parser.add_argument("--line-max-i-ka", type=float, default=None)
    parser.add_argument("--default-line-max-i-ka", type=float, default=0.20)
    parser.add_argument(
        "--skip-current-constraints",
        action="store_true",
        help="Do not constrain branch current in the exact DistFlow optimizer.",
    )
    parser.add_argument("--distflow-max-iter", type=int, default=250)
    parser.add_argument("--distflow-pf-max-iter", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        default="opf/disflow/results/exp26_single_step",
    )
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="Only run baseline and exact DistFlow optimization.",
    )
    parser.add_argument(
        "--milp-q",
        default=None,
        help=(
            "JSON list or JSON file containing a precomputed Exp26 MILP full "
            "Q-control dispatch: PV, ESS, then Q devices."
        ),
    )

    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--trust-region-fraction", type=float, default=1.0)
    parser.add_argument("--trust-region-radius", default=None)
    parser.add_argument(
        "--local-trust-region-fraction",
        type=float,
        default=exp26_rpo.DEFAULT_RUN_CONFIG["local_trust_region_fraction"],
    )
    parser.add_argument("--local-trust-radius", default=None)
    parser.add_argument(
        "--heuristic-random-samples",
        type=int,
        default=exp26_rpo.DEFAULT_RUN_CONFIG["heuristic_random_samples"],
    )
    parser.add_argument(
        "--heuristic-batch-size",
        type=int,
        default=exp26_rpo.DEFAULT_RUN_CONFIG["heuristic_batch_size"],
    )
    parser.add_argument(
        "--heuristic-safety-penalty",
        type=float,
        default=exp26_rpo.DEFAULT_RUN_CONFIG["heuristic_safety_penalty"],
    )
    parser.add_argument(
        "--heuristic-seed",
        type=int,
        default=exp26_rpo.DEFAULT_RUN_CONFIG["heuristic_seed"],
    )
    parser.add_argument("--torch-warm-starts", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["torch_warm_starts"])
    parser.add_argument("--torch-warm-steps", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["torch_warm_steps"])
    parser.add_argument("--torch-warm-lr", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["torch_warm_lr"])
    parser.add_argument("--torch-q-penalty", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["torch_q_penalty"])
    parser.add_argument(
        "--no-sc-integer",
        dest="no_sc_integer",
        action="store_true",
        default=not exp26_rpo.DEFAULT_RUN_CONFIG["sc_integer"],
    )
    parser.add_argument("--sc-step", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["sc_step"])
    parser.add_argument("--warm-start-q", default=None)
    parser.add_argument("--control-deviation-penalty", type=float, default=0.0)
    parser.add_argument("--no-physical-output-bounds", action="store_true")
    parser.add_argument(
        "--relu-formulation",
        choices=["big_m", "general"],
        default="big_m",
    )
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--time-limit", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["time_limit"])
    parser.add_argument("--mip-gap", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["mip_gap"])
    parser.add_argument("--threads", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["threads"])
    parser.add_argument("--mip-focus", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["mip_focus"])
    parser.add_argument("--method", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["method"])
    parser.add_argument("--node-method", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["node_method"])
    parser.add_argument("--presolve", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["presolve"])
    parser.add_argument("--presparsify", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["presparsify"])
    parser.add_argument("--heuristics", type=float, default=exp26_rpo.DEFAULT_RUN_CONFIG["heuristics"])
    parser.add_argument("--cuts", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["cuts"])
    parser.add_argument("--var-branch", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["var_branch"])
    parser.add_argument("--numeric-focus", type=int, default=exp26_rpo.DEFAULT_RUN_CONFIG["numeric_focus"])
    parser.add_argument("--dual-reductions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--diagnose-slack", action="store_true")
    parser.add_argument("--slack-penalty", type=float, default=1e4)
    parser.add_argument("--quiet-milp", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    op = load_operating_point(args)
    expected_controls = len(control_labels(op))
    args.trust_region_radius = exp26_rpo.parse_vector_arg(
        args.trust_region_radius,
        expected=expected_controls,
        name="trust_region_radius",
    )
    args.local_trust_radius = exp26_rpo.parse_vector_arg(
        args.local_trust_radius,
        expected=expected_controls,
        name="local_trust_radius",
    )
    args.warm_start_q = exp26_rpo.parse_vector_arg(
        args.warm_start_q,
        expected=expected_controls,
        name="warm_start_q",
    )
    args.milp_q = exp26_rpo.parse_vector_arg(
        args.milp_q,
        expected=expected_controls,
        name="milp_q",
    )
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_max_i_ka = (
        get_line_max_i_ka(args.data, args.default_line_max_i_ka)
        if args.line_max_i_ka is None
        else float(args.line_max_i_ka)
    )
    distflow = ExactDistFlow(
        branches=op.branches if op.branches is not None else RADIAL_BRANCHES,
        slack_vm_pu=args.slack_vm_pu,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        line_max_i_ka=line_max_i_ka,
    )

    q_caps = q_capacity(op.p_pv, op.s_rated)
    baseline_q = baseline_control(op)
    cases: list[tuple[str, np.ndarray, DistFlowResult]] = [
        (
            "Baseline",
            baseline_q,
            distflow.solve(
                build_net_injection(op, baseline_q),
                max_iter=args.distflow_pf_max_iter,
            ),
        )
    ]

    milp_result = None
    milp_solve_time_sec = None
    distflow_solve_time_sec = None
    milp_q = args.milp_q
    if milp_q is None and not args.skip_milp:
        milp_start = time.perf_counter()
        milp_result = run_milp(op, args)
        milp_solve_time_sec = time.perf_counter() - milp_start
        if milp_result.get("Q_control_opt") is None:
            raise RuntimeError(
                "Exp26 MILP did not return a feasible Q_control_opt; rerun with "
                "--diagnose-slack or pass --milp-q with a precomputed dispatch."
            )
        milp_q = np.asarray(milp_result["Q_control_opt"], dtype=float)

    if milp_q is not None:
        cases.append(
            (
                "Exp26 ST-GCN-MILP",
                milp_q,
                distflow.solve(
                    build_net_injection(op, milp_q),
                    max_iter=args.distflow_pf_max_iter,
                ),
            )
        )

    distflow_start = time.perf_counter()
    exact = solve_exact_distflow_rpo(
        op,
        distflow,
        objective=args.objective,
        loss_weight=args.loss_weight,
        voltage_weight=args.voltage_weight,
        v_lower=args.v_lower,
        v_upper=args.v_upper,
        enforce_current=not args.skip_current_constraints,
        max_iter=args.distflow_max_iter,
    )
    distflow_solve_time_sec = time.perf_counter() - distflow_start
    cases.append(("Exact DistFlow", exact["Q_control_opt"], exact["metrics"]))

    make_plots(out_dir, cases, op=op, q_caps=q_caps)
    rows = [summarize_metrics(name, q, metrics, op) for name, q, metrics in cases]
    save_csv(out_dir / "comparison_summary.csv", rows)

    payload = {
        "source": args.source,
        "data": str(resolve_repo_path(args.data)) if args.source == "dataset" else None,
        "sample_index": op.meta.get("sample_index"),
        "sample_reason": op.meta.get("sample_reason"),
        "hour": op.meta.get("hour"),
        "mode": op.meta.get("mode"),
        "objective": args.objective,
        "loss_weight": float(args.loss_weight),
        "voltage_weight": float(args.voltage_weight),
        "line_max_i_ka": float(line_max_i_ka),
        "solve_time_sec": {
            "sgcn_milp": milp_solve_time_sec,
            "exact_distflow_rpo": distflow_solve_time_sec,
        },
        "pv_nodes": op.pv_nodes.tolist(),
        "p_pv": op.p_pv.tolist(),
        "s_rated": op.s_rated.tolist(),
        "control_labels": control_labels(op),
        "baseline_definition": "PV reactive output is set to zero before optimization; ESS and Q devices start from their operating-point values.",
        "ess_nodes": op.ess_nodes.tolist(),
        "ess_s_rated": op.ess_s_rated.tolist(),
        "ess_p_base": op.ess_p_base.tolist(),
        "ess_q_base": op.ess_q_base.tolist(),
        "q_device_nodes": op.q_device_nodes.tolist(),
        "q_device_names": op.q_device_names,
        "q_device_min": op.q_device_min.tolist(),
        "q_device_max": op.q_device_max.tolist(),
        "q_device_q_base": op.q_device_q_base.tolist(),
        "milp_result": milp_result,
        "exact_distflow_solver": {
            "success": exact["success"],
            "message": exact["message"],
            "objective": exact["objective"],
            "max_constraint_violation": exact["max_constraint_violation"],
        },
        "summary": rows,
    }
    with (out_dir / "comparison_result.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)

    print()
    print("Exact voltage-deviation comparison complete.")
    print(f"Output directory: {out_dir}")
    print(f"Voltage plot: {out_dir / 'voltage_comparison.png'}")
    print(f"Q dispatch plot: {out_dir / 'q_dispatch_comparison.png'}")
    print(f"Summary CSV: {out_dir / 'comparison_summary.csv'}")
    print("Solve time:")
    if milp_solve_time_sec is None:
        print("  SGCN-MILP: skipped/precomputed")
    else:
        print(f"  SGCN-MILP: {milp_solve_time_sec:.3f} s")
    print(f"  Exact DistFlow RPO: {distflow_solve_time_sec:.3f} s")
    print()
    for row in rows:
        print(
            f"{row['name']}: "
            f"Vmin={row['v_min']:.6f}, Vmax={row['v_max']:.6f}, "
            f"Vdev={row['vdev_total']:.6f}, "
            f"WorstI={row['worst_i_margin']:.6f}"
        )


if __name__ == "__main__":
    main()





