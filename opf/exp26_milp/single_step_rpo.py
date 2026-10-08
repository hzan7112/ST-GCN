"""Single-step reactive power optimization using the Exp26 ST-GCN MILP.

This script mirrors the older ``rpo_milp/src/RPO_MILP.py`` workflow, but uses
the Exp26 nodal-voltage and branch-current surrogate trained by train_exp26.py:

    V_nodes, Vdev_total, Vworst, WorstI

The decision variables are the reactive injections of PV inverters, ESS
inverters, and discrete/continuous reactive devices present in the selected
operating point. Exp26 has no learned loss head, so the objective uses only
the learned cumulative voltage deviation:

    Vdev_total

The surrogate safety constraints are:

    Vworst <= voltage_margin
    WorstI <= current_margin

where zero means the predicted operating point is exactly at the learned safety
boundary.

The voltage and current heads are learned residuals on top of linear priors.
Ploss_total is intentionally not used because Exp26 did not learn total network
loss and this workflow does not assume exact line impedance parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep numerical libraries quiet before importing numpy/torch/gurobi.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ENGINE_PATH = (
    REPO_ROOT
    / "checkpoints"
    / "st_gcn_h16_l2_exp26_milp_engine.pt"
)

DEFAULT_RUN_CONFIG = {
    "strategy": "two_stage",
    "time_limit": 300.0,
    "mip_gap": 0.05,
    "threads": 8,
    "mip_focus": 2,
    "method": 1,
    "node_method": 1,
    "presolve": 2,
    "presparsify": 1,
    "heuristics": 0.15,
    "cuts": 2,
    "var_branch": 2,
    "numeric_focus": 1,
    "heuristic_random_samples": 20000,
    "heuristic_batch_size": 2048,
    "heuristic_safety_penalty": 500.0,
    "heuristic_seed": 2026,
    "torch_warm_starts": 8,
    "torch_warm_steps": 700,
    "torch_warm_lr": 0.05,
    "torch_q_penalty": 0.0,
    "local_trust_region_fraction": 0.30,
    "sc_integer": True,
    "sc_step": 0.1,
}

gp = None
GRB = None


def require_gurobi():
    global gp, GRB
    if gp is not None and GRB is not None:
        return gp, GRB
    try:
        import gurobipy as gp_mod
        from gurobipy import GRB as grb_mod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gurobipy is required to solve the Exp26 MILP. "
            "Install Gurobi and run this script in the environment where "
            "gurobipy is available."
        ) from exc
    gp = gp_mod
    GRB = grb_mod
    return gp, GRB


def get_converter_class():
    require_gurobi()
    try:
        from opf.exp26_milp.sgcn_milp_converter import Exp26STGCNMILPConverter
    except ModuleNotFoundError:
        from sgcn_milp_converter import Exp26STGCNMILPConverter

    return Exp26STGCNMILPConverter


def prepare_base_profiles():
    """Return IEEE-33 base load and default PV configuration."""
    pload_standard_kw = np.array(
        [
            0, 100, 90, 120, 60, 60, 200, 200, 60, 60,
            45, 60, 60, 120, 60, 60, 60, 90, 90, 90,
            90, 90, 90, 420, 420, 60, 60, 60, 420, 400,
            450, 410, 60,
        ],
        dtype=float,
    )
    qload_standard_kvar = np.array(
        [
            0, 60, 40, 80, 30, 20, 100, 100, 20, 20,
            30, 35, 35, 80, 10, 20, 20, 40, 40, 40,
            40, 40, 50, 200, 200, 25, 25, 20, 70, 600,
            70, 100, 40,
        ],
        dtype=float,
    )

    p_load_mw = pload_standard_kw / 1000.0
    q_load_mvar = qload_standard_kvar / 1000.0
    pv_nodes = np.array([7, 14, 22, 29], dtype=int)
    s_rated_mva = np.array([1.5, 2.0, 3.0, 3.5], dtype=float)
    return p_load_mw, q_load_mvar, pv_nodes, s_rated_mva


def get_standard_radial_topology():
    """Exp26 expects the standard IEEE-33 radial topology."""
    topo_mask = np.zeros(37, dtype=bool)
    topo_mask[:32] = True
    return topo_mask


def get_default_operating_point(pv_pu: float = 0.8):
    p_load, q_load, pv_nodes, s_rated = prepare_base_profiles()
    p_pv_available = float(pv_pu) * s_rated
    topo_mask = get_standard_radial_topology()
    return p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated


def safe_torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(value, *, dtype=float):
    if value is None:
        return np.array([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def sigmoid_scalar(value):
    return 1.0 / (1.0 + np.exp(-float(value)))


def worst_voltage_margin(yv, v_lower=0.95, v_upper=1.05):
    import torch

    over = yv.max(dim=1, keepdim=True).values - float(v_upper)
    under = float(v_lower) - yv.min(dim=1, keepdim=True).values
    return torch.maximum(over, under)


def choose_dataset_sample(data, cfg, requested_index: int):
    import torch

    n = int(data["X"].shape[0])
    if requested_index >= 0:
        if requested_index >= n:
            raise ValueError(f"sample index {requested_index} out of range 0..{n - 1}")
        return int(requested_index), "requested"

    yv_worst = worst_voltage_margin(
        data["Y_V"].float(),
        cfg.get("v_lower", 0.95),
        cfg.get("v_upper", 1.05),
    ).reshape(-1)
    yi_worst = data["Y_I"].float().max(dim=1).values.reshape(-1)
    pv_sum = data["pv_p"].float().sum(dim=1)

    safe = (yv_worst <= 0.0) & (yi_worst <= 0.0)
    with_pv = pv_sum > 1e-6
    candidates = torch.where(safe & with_pv)[0]
    reason = "auto_safe_with_pv"
    if candidates.numel() == 0:
        candidates = torch.where(safe)[0]
        reason = "auto_safe"
    if candidates.numel() == 0:
        candidates = torch.arange(n)
        reason = "auto_any"

    candidate_pv = pv_sum[candidates]
    target = torch.median(candidate_pv)
    local = torch.argmin(torch.abs(candidate_pv - target))
    return int(candidates[local].item()), reason


def get_dataset_operating_point(
    data_path: str | Path,
    cfg: dict,
    *,
    sample_index: int = -1,
):
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    data = safe_torch_load(data_path)
    if not isinstance(data, dict):
        raise TypeError(f"dataset must be a dict: {data_path}")

    required = ["X", "pv_nodes", "S_pv_mva", "pv_p", "pv_q", "Y_V", "Y_I"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset missing required keys for dataset source: {missing}")

    sample_idx, reason = choose_dataset_sample(data, cfg, sample_index)
    x_base = data["X"][sample_idx].detach().cpu().float().numpy()[:, :2]
    pv_nodes = data["pv_nodes"].detach().cpu().numpy().astype(int).reshape(-1)
    s_rated = data["S_pv_mva"].detach().cpu().float().numpy().reshape(-1)
    p_pv = data["pv_p"][sample_idx].detach().cpu().float().numpy().reshape(-1)
    pv_q_base = data["pv_q"][sample_idx].detach().cpu().float().numpy().reshape(-1)
    ess_nodes = to_numpy(data.get("ess_nodes"), dtype=int).reshape(-1)
    ess_s_rated = to_numpy(data.get("S_ess_mva")).reshape(-1)
    ess_p_base = (
        to_numpy(data["ess_p"][sample_idx]).reshape(-1)
        if "ess_p" in data
        else np.array([], dtype=float)
    )
    ess_q_base = (
        to_numpy(data["ess_q"][sample_idx]).reshape(-1)
        if "ess_q" in data
        else np.array([], dtype=float)
    )
    q_device_nodes = to_numpy(data.get("q_device_nodes"), dtype=int).reshape(-1)
    q_device_min = to_numpy(data.get("q_device_min")).reshape(-1)
    q_device_max = to_numpy(data.get("q_device_max")).reshape(-1)
    q_device_q_base = (
        to_numpy(data["qdev_q"][sample_idx]).reshape(-1)
        if "qdev_q" in data
        else np.array([], dtype=float)
    )
    q_device_names = (
        [str(name) for name in data.get("q_device_names", [])]
        if q_device_nodes.size
        else []
    )
    topo_mask = get_standard_radial_topology()

    meta = {
        "data_path": str(data_path),
        "sample_index": sample_idx,
        "sample_reason": reason,
        "hour": int(data["hour"][sample_idx].item()) if "hour" in data else None,
        "mode": int(data["mode"][sample_idx].item()) if "mode" in data else None,
        "Pload_sum": float(data["Pload"][sample_idx].sum().item()) if "Pload" in data else None,
        "Qload_sum": float(data["Qload"][sample_idx].sum().item()) if "Qload" in data else None,
        "X_P_sum": float(x_base[:, 0].sum()),
        "X_Q_sum": float(x_base[:, 1].sum()),
        "pv_q_base": pv_q_base,
        "ess_nodes": ess_nodes,
        "S_ess_mva": ess_s_rated,
        "ess_p_base": ess_p_base,
        "ess_q_base": ess_q_base,
        "q_device_nodes": q_device_nodes,
        "q_device_min": q_device_min,
        "q_device_max": q_device_max,
        "q_device_q_base": q_device_q_base,
        "q_device_names": q_device_names,
    }
    return x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta


def parse_vector_arg(value: str | None, *, expected: int, name: str):
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    else:
        data = json.loads(value)
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size != expected:
        raise ValueError(f"{name} must have length {expected}, got {array.size}")
    return array


def normalize_optional_vector(value, *, dtype=float):
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


def validate_inputs(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    pv_nodes,
    s_rated,
):
    p_load = np.asarray(p_load, dtype=float).reshape(-1)
    q_load = np.asarray(q_load, dtype=float).reshape(-1)
    p_pv_available = np.asarray(p_pv_available, dtype=float).reshape(-1)
    topo_mask = np.asarray(topo_mask, dtype=bool).reshape(-1)
    pv_nodes = np.asarray(pv_nodes, dtype=int).reshape(-1)
    s_rated = np.asarray(s_rated, dtype=float).reshape(-1)

    if p_load.size != 33:
        raise ValueError(f"P_load length must be 33, got {p_load.size}")
    if q_load.size != 33:
        raise ValueError(f"Q_load length must be 33, got {q_load.size}")
    if p_pv_available.size != pv_nodes.size:
        raise ValueError(
            f"P_pv_available length must be {pv_nodes.size}, got {p_pv_available.size}"
        )
    if s_rated.size != pv_nodes.size:
        raise ValueError(f"S_rated length must be {pv_nodes.size}, got {s_rated.size}")
    if topo_mask.size != 37:
        raise ValueError(f"topo_mask length must be 37, got {topo_mask.size}")
    if np.any(p_pv_available < -1e-12):
        raise ValueError("P_pv_available cannot contain negative values")
    if np.any(s_rated <= 0.0):
        raise ValueError("S_rated must be positive")
    if np.any(p_pv_available > s_rated + 1e-9):
        bad = np.where(p_pv_available > s_rated + 1e-9)[0][0]
        raise ValueError(
            f"PV at bus {pv_nodes[bad]} has P={p_pv_available[bad]:.6f} MW "
            f"> S={s_rated[bad]:.6f} MVA"
        )

    return p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated


def build_injection_expressions(model, q_pv, p_load, q_load, p_pv_available, pv_nodes):
    """Build 33x2 [P_net, Q_net] expressions for the Exp26 converter."""
    pv_to_idx = {int(bus): idx for idx, bus in enumerate(pv_nodes)}
    x_vars = [[None, None] for _ in range(33)]

    for bus in range(33):
        p_inj = -float(p_load[bus])
        q_inj = -float(q_load[bus])
        if bus in pv_to_idx:
            idx = pv_to_idx[bus]
            p_inj += float(p_pv_available[idx])
            x_vars[bus][0] = gp.LinExpr(p_inj)
            x_vars[bus][1] = gp.LinExpr(q_inj) + q_pv[idx]
        else:
            x_vars[bus][0] = gp.LinExpr(p_inj)
            x_vars[bus][1] = gp.LinExpr(q_inj)

    return x_vars


def build_dataset_injection_expressions(
    model,
    q_pv,
    q_ess,
    q_device,
    x_base,
    pv_q_base,
    pv_nodes,
    ess_q_base,
    ess_nodes,
    q_device_q_base,
    q_device_nodes,
):
    """Build [P_net, Q_net] expressions around a dataset sample.

    The dataset ``X`` already contains the net injection seen during training,
    including load, PV, ESS, and other reactive devices. We keep fixed loads and
    active power unchanged, subtract the base controllable reactive injections,
    and add the optimized controllable reactive variables.
    """
    x_base = np.asarray(x_base, dtype=float)
    if x_base.shape != (33, 2):
        raise ValueError(f"x_base must have shape (33, 2), got {x_base.shape}")
    pv_q_base = np.asarray(pv_q_base, dtype=float).reshape(-1)
    pv_to_idx = {int(bus): idx for idx, bus in enumerate(pv_nodes)}
    ess_to_idx = {int(bus): idx for idx, bus in enumerate(ess_nodes)}
    qdev_to_idx = {int(bus): idx for idx, bus in enumerate(q_device_nodes)}
    x_vars = [[None, None] for _ in range(33)]

    for bus in range(33):
        x_vars[bus][0] = gp.LinExpr(float(x_base[bus, 0]))
        q_expr = gp.LinExpr(float(x_base[bus, 1]))
        if bus in pv_to_idx:
            idx = pv_to_idx[bus]
            q_expr += -float(pv_q_base[idx]) + q_pv[idx]
        if bus in ess_to_idx:
            idx = ess_to_idx[bus]
            q_expr += -float(ess_q_base[idx]) + q_ess[idx]
        if bus in qdev_to_idx:
            idx = qdev_to_idx[bus]
            q_expr += -float(q_device_q_base[idx]) + q_device[idx]
        x_vars[bus][1] = q_expr

    return x_vars


def add_pv_capacity_constraints(model, q_pv, p_pv_available, pv_nodes, s_rated):
    q_caps = []
    for idx, (bus, p_avail, s_max) in enumerate(
        zip(pv_nodes, p_pv_available, s_rated)
    ):
        q_cap_sq = max(float(s_max) ** 2 - float(p_avail) ** 2, 0.0)
        q_cap = float(np.sqrt(q_cap_sq))
        q_caps.append(q_cap)
        # Bounds help presolve, while the quadratic constraint keeps the
        # apparent-power model explicit for compatibility with older scripts.
        q_pv[idx].LB = -q_cap
        q_pv[idx].UB = q_cap
        model.addQConstr(
            q_pv[idx] * q_pv[idx] <= q_cap_sq,
            name=f"PV_Cap_{int(bus)}",
        )
    return np.asarray(q_caps, dtype=float)


def add_ess_capacity_constraints(model, q_ess, ess_p_base, ess_nodes, ess_s_rated):
    q_caps = []
    for idx, (bus, p_base, s_max) in enumerate(
        zip(ess_nodes, ess_p_base, ess_s_rated)
    ):
        q_cap_sq = max(float(s_max) ** 2 - float(p_base) ** 2, 0.0)
        q_cap = float(np.sqrt(q_cap_sq))
        q_caps.append(q_cap)
        q_ess[idx].LB = -q_cap
        q_ess[idx].UB = q_cap
        model.addQConstr(
            q_ess[idx] * q_ess[idx] <= q_cap_sq,
            name=f"ESS_Cap_{int(bus)}",
        )
    return np.asarray(q_caps, dtype=float)


def add_q_device_bounds(model, q_device, q_device_nodes, q_device_min, q_device_max):
    for idx, bus in enumerate(q_device_nodes):
        q_device[idx].LB = float(q_device_min[idx])
        q_device[idx].UB = float(q_device_max[idx])
    return np.asarray(q_device_min, dtype=float), np.asarray(q_device_max, dtype=float)


def build_control_labels(pv_nodes, ess_nodes, q_device_nodes, q_device_names):
    labels = [f"PV@Bus{int(bus) + 1}" for bus in pv_nodes]
    labels.extend(f"ESS@Bus{int(bus) + 1}" for bus in ess_nodes)
    for idx, bus in enumerate(q_device_nodes):
        name = q_device_names[idx] if idx < len(q_device_names) else f"QDev{idx + 1}"
        labels.append(f"{name}@Bus{int(bus) + 1}")
    return labels


def build_control_base_vector(
    pv_q_base,
    ess_q_base,
    q_device_q_base,
    *,
    n_pv: int,
    n_ess: int,
    n_qdev: int,
):
    pv_ref = np.zeros(n_pv, dtype=float) if pv_q_base is None else np.asarray(pv_q_base, dtype=float).reshape(-1)
    ess_ref = np.asarray(ess_q_base, dtype=float).reshape(-1)
    qdev_ref = np.asarray(q_device_q_base, dtype=float).reshape(-1)
    if pv_ref.size != n_pv:
        raise ValueError(f"PV Q reference length must be {n_pv}, got {pv_ref.size}")
    if ess_ref.size != n_ess:
        raise ValueError(f"ESS Q reference length must be {n_ess}, got {ess_ref.size}")
    if qdev_ref.size != n_qdev:
        raise ValueError(f"Q-device reference length must be {n_qdev}, got {qdev_ref.size}")
    return np.concatenate([pv_ref, ess_ref, qdev_ref])


def control_variable_list(q_pv, q_ess, q_device, n_pv: int, n_ess: int, n_qdev: int):
    return (
        [q_pv[idx] for idx in range(n_pv)]
        + [q_ess[idx] for idx in range(n_ess)]
        + [q_device[idx] for idx in range(n_qdev)]
    )


def build_control_node_vector(pv_nodes, ess_nodes, q_device_nodes):
    return np.concatenate(
        [
            np.asarray(pv_nodes, dtype=int).reshape(-1),
            np.asarray(ess_nodes, dtype=int).reshape(-1),
            np.asarray(q_device_nodes, dtype=int).reshape(-1),
        ]
    )


def build_control_bound_vectors(q_caps, ess_q_caps, qdev_min, qdev_max):
    q_caps = np.asarray(q_caps, dtype=float).reshape(-1)
    ess_q_caps = np.asarray(ess_q_caps, dtype=float).reshape(-1)
    qdev_min = np.asarray(qdev_min, dtype=float).reshape(-1)
    qdev_max = np.asarray(qdev_max, dtype=float).reshape(-1)
    lower = np.concatenate([-q_caps, -ess_q_caps, qdev_min])
    upper = np.concatenate([q_caps, ess_q_caps, qdev_max])
    if lower.shape != upper.shape:
        raise ValueError("Control lower/upper bound vectors have different shapes")
    if np.any(upper < lower):
        bad = int(np.where(upper < lower)[0][0])
        raise ValueError(
            f"Invalid control bounds at index {bad}: lower={lower[bad]}, upper={upper[bad]}"
        )
    return lower, upper


def compute_control_metadata(
    *,
    p_pv_available,
    pv_nodes,
    s_rated,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_s_rated,
    ess_p_base,
    ess_q_base,
    q_device_nodes,
    q_device_min,
    q_device_max,
    q_device_q_base,
    q_device_names,
):
    q_caps = np.sqrt(np.maximum(np.asarray(s_rated, dtype=float) ** 2 - np.asarray(p_pv_available, dtype=float) ** 2, 0.0))
    ess_q_caps = np.sqrt(
        np.maximum(np.asarray(ess_s_rated, dtype=float) ** 2 - np.asarray(ess_p_base, dtype=float) ** 2, 0.0)
    )
    qdev_min = np.asarray(q_device_min, dtype=float).reshape(-1)
    qdev_max = np.asarray(q_device_max, dtype=float).reshape(-1)
    labels = build_control_labels(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
        q_device_names,
    )
    control_nodes = build_control_node_vector(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
    )
    control_lower, control_upper = build_control_bound_vectors(
        q_caps,
        ess_q_caps,
        qdev_min,
        qdev_max,
    )
    control_base = build_control_base_vector(
        pv_q_base if x_base is not None else None,
        ess_q_base,
        q_device_q_base,
        n_pv=len(pv_nodes),
        n_ess=len(ess_nodes),
        n_qdev=len(q_device_nodes),
    )
    return {
        "q_caps": q_caps,
        "ess_q_caps": ess_q_caps,
        "qdev_min": qdev_min,
        "qdev_max": qdev_max,
        "labels": labels,
        "control_nodes": control_nodes,
        "control_lower": control_lower,
        "control_upper": control_upper,
        "control_base": control_base,
    }


def split_control_values(q_control, *, n_pv: int, n_ess: int, n_qdev: int):
    q = np.asarray(q_control, dtype=float).reshape(-1)
    expected = n_pv + n_ess + n_qdev
    if q.size != expected:
        raise ValueError(f"q_control must have length {expected}, got {q.size}")
    return q[:n_pv], q[n_pv : n_pv + n_ess], q[n_pv + n_ess :]


def build_numeric_injection_start(
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    q_control,
):
    q_pv, q_ess, q_qdev = split_control_values(
        q_control,
        n_pv=len(pv_nodes),
        n_ess=len(ess_nodes),
        n_qdev=len(q_device_nodes),
    )

    if x_base is not None:
        x = np.asarray(x_base, dtype=float).copy()
        if x.shape != (33, 2):
            raise ValueError(f"x_base must have shape (33, 2), got {x.shape}")
        for idx, bus in enumerate(pv_nodes):
            x[int(bus), 1] += -float(pv_q_base[idx]) + float(q_pv[idx])
        for idx, bus in enumerate(ess_nodes):
            x[int(bus), 1] += -float(ess_q_base[idx]) + float(q_ess[idx])
        for idx, bus in enumerate(q_device_nodes):
            x[int(bus), 1] += -float(q_device_q_base[idx]) + float(q_qdev[idx])
        return x

    x = np.zeros((33, 2), dtype=float)
    x[:, 0] = -np.asarray(p_load, dtype=float).reshape(33)
    x[:, 1] = -np.asarray(q_load, dtype=float).reshape(33)
    for idx, bus in enumerate(pv_nodes):
        x[int(bus), 0] += float(p_pv_available[idx])
        x[int(bus), 1] += float(q_pv[idx])
    for idx, bus in enumerate(ess_nodes):
        x[int(bus), 1] += float(q_ess[idx])
    for idx, bus in enumerate(q_device_nodes):
        x[int(bus), 1] += float(q_qdev[idx])
    return x


def build_numeric_injection_batch(
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    q_control_batch,
):
    q_batch = np.asarray(q_control_batch, dtype=float)
    if q_batch.ndim == 1:
        q_batch = q_batch[None, :]
    expected = len(pv_nodes) + len(ess_nodes) + len(q_device_nodes)
    if q_batch.ndim != 2 or q_batch.shape[1] != expected:
        raise ValueError(f"q_control_batch must have shape (batch, {expected}), got {q_batch.shape}")

    batch = q_batch.shape[0]
    q_pv = q_batch[:, : len(pv_nodes)]
    q_ess = q_batch[:, len(pv_nodes) : len(pv_nodes) + len(ess_nodes)]
    q_qdev = q_batch[:, len(pv_nodes) + len(ess_nodes) :]

    if x_base is not None:
        x = np.broadcast_to(np.asarray(x_base, dtype=float), (batch, 33, 2)).copy()
        for idx, bus in enumerate(pv_nodes):
            x[:, int(bus), 1] += -float(pv_q_base[idx]) + q_pv[:, idx]
        for idx, bus in enumerate(ess_nodes):
            x[:, int(bus), 1] += -float(ess_q_base[idx]) + q_ess[:, idx]
        for idx, bus in enumerate(q_device_nodes):
            x[:, int(bus), 1] += -float(q_device_q_base[idx]) + q_qdev[:, idx]
        return x

    base = np.zeros((33, 2), dtype=float)
    base[:, 0] = -np.asarray(p_load, dtype=float).reshape(33)
    base[:, 1] = -np.asarray(q_load, dtype=float).reshape(33)
    for idx, bus in enumerate(pv_nodes):
        base[int(bus), 0] += float(p_pv_available[idx])
    x = np.broadcast_to(base, (batch, 33, 2)).copy()
    for idx, bus in enumerate(pv_nodes):
        x[:, int(bus), 1] += q_pv[:, idx]
    for idx, bus in enumerate(ess_nodes):
        x[:, int(bus), 1] += q_ess[:, idx]
    for idx, bus in enumerate(q_device_nodes):
        x[:, int(bus), 1] += q_qdev[:, idx]
    return x


def evaluate_control_batch(
    converter,
    q_batch,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    voltage_margin,
    current_margin,
    safety_penalty,
):
    q_batch = np.asarray(q_batch, dtype=float)
    if q_batch.ndim == 1:
        q_batch = q_batch[None, :]
    X = build_numeric_injection_batch(
        p_load=p_load,
        q_load=q_load,
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_q_base=q_device_q_base,
        q_control_batch=q_batch,
    )
    pred = converter.inner.forward_numpy(X)
    vdev = np.asarray(pred["Vdev_total"], dtype=float).reshape(-1)
    vworst = np.asarray(pred["Vworst"], dtype=float).reshape(-1)
    worst_i = np.asarray(pred["WorstI"], dtype=float).reshape(-1)
    v_violation = np.maximum(vworst - float(voltage_margin), 0.0)
    i_violation = np.maximum(worst_i - float(current_margin), 0.0)
    safety_violation = v_violation + i_violation
    score = vdev + float(safety_penalty) * safety_violation
    return {
        "score": score,
        "Vdev_total": vdev,
        "Vworst": vworst,
        "WorstI": worst_i,
        "safety_violation": safety_violation,
    }


def forward_heuristic_search(
    converter,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    control_lower,
    control_upper,
    control_base,
    initial_q,
    voltage_margin,
    current_margin,
    random_samples,
    local_rounds,
    local_samples,
    batch_size,
    safety_penalty,
    seed,
    output_flag,
):
    lower = np.asarray(control_lower, dtype=float).reshape(-1)
    upper = np.asarray(control_upper, dtype=float).reshape(-1)
    width = np.maximum(upper - lower, 0.0)
    control_base = np.asarray(control_base, dtype=float).reshape(-1)
    initial_q = np.asarray(initial_q, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))

    best = {
        "q": None,
        "score": np.inf,
        "Vdev_total": np.inf,
        "Vworst": np.inf,
        "WorstI": np.inf,
        "safety_violation": np.inf,
        "source": None,
        "evaluations": 0,
    }

    def consider(candidates, source):
        candidates = np.asarray(candidates, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        candidates = np.clip(candidates, lower, upper)
        if candidates.size == 0:
            return
        for start in range(0, candidates.shape[0], int(batch_size)):
            q_chunk = candidates[start : start + int(batch_size)]
            metrics = evaluate_control_batch(
                converter,
                q_chunk,
                p_load=p_load,
                q_load=q_load,
                p_pv_available=p_pv_available,
                pv_nodes=pv_nodes,
                x_base=x_base,
                pv_q_base=pv_q_base,
                ess_nodes=ess_nodes,
                ess_q_base=ess_q_base,
                q_device_nodes=q_device_nodes,
                q_device_q_base=q_device_q_base,
                voltage_margin=voltage_margin,
                current_margin=current_margin,
                safety_penalty=safety_penalty,
            )
            idx = int(np.argmin(metrics["score"]))
            best["evaluations"] += int(q_chunk.shape[0])
            score = float(metrics["score"][idx])
            if score < best["score"]:
                best.update(
                    {
                        "q": q_chunk[idx].copy(),
                        "score": score,
                        "Vdev_total": float(metrics["Vdev_total"][idx]),
                        "Vworst": float(metrics["Vworst"][idx]),
                        "WorstI": float(metrics["WorstI"][idx]),
                        "safety_violation": float(metrics["safety_violation"][idx]),
                        "source": source,
                    }
                )

    seeds = [np.clip(control_base, lower, upper), np.clip(initial_q, lower, upper)]
    if lower.size:
        seeds.append((lower + upper) / 2.0)
    consider(np.unique(np.vstack(seeds), axis=0), "seed")

    if lower.size and int(random_samples) > 0:
        remaining = int(random_samples)
        while remaining > 0:
            n = min(int(batch_size), remaining)
            consider(rng.uniform(lower, upper, size=(n, lower.size)), "global_random")
            remaining -= n

    if lower.size:
        for round_idx in range(int(local_rounds)):
            radius = width * (0.35 * (0.55 ** round_idx))
            if np.all(radius <= 1e-12) or int(local_samples) <= 0:
                break
            lo = np.maximum(lower, best["q"] - radius)
            hi = np.minimum(upper, best["q"] + radius)
            remaining = int(local_samples)
            while remaining > 0:
                n = min(int(batch_size), remaining)
                consider(rng.uniform(lo, hi, size=(n, lower.size)), f"local_round_{round_idx + 1}")
                remaining -= n

    if int(output_flag):
        print()
        print("Forward heuristic search:")
        print(f"  evaluated candidates: {best['evaluations']}")
        print(f"  best source: {best['source']}")
        print(f"  score: {best['score']:.8f}")
        print(f"  Vdev_total: {best['Vdev_total']:.8f}")
        print(f"  Vworst: {best['Vworst']:.8f}")
        print(f"  WorstI: {best['WorstI']:.8f}")
        print(f"  safety violation: {best['safety_violation']:.8f}")

    return best


def torch_surrogate_forward(inner, Xpq, device):
    import torch

    Xpq = Xpq.to(dtype=torch.float32, device=device)
    P = Xpq[:, :, 0]
    Q = Xpq[:, :, 1]
    S_down = torch.as_tensor(inner.S_down, dtype=torch.float32, device=device)
    S_path = torch.as_tensor(inner.S_path, dtype=torch.float32, device=device)
    X6 = torch.stack(
        [
            P,
            Q,
            P @ S_down.T,
            Q @ S_down.T,
            P @ S_path.T,
            Q @ S_path.T,
        ],
        dim=2,
    )

    xm = torch.as_tensor(inner.norm["X_mean"], dtype=torch.float32, device=device)
    xs = torch.as_tensor(inner.norm["X_std"], dtype=torch.float32, device=device)
    Xn = (X6 - xm) / xs

    def tparam(name):
        return torch.as_tensor(inner.p(name), dtype=torch.float32, device=device)

    def tlp(k, name):
        return torch.as_tensor(inner.lp(k, name), dtype=torch.float32, device=device)

    W0, b0 = tparam("input_embed.weight"), tparam("input_embed.bias")
    H0 = Xn @ W0.T + b0
    A = torch.as_tensor(inner.A, dtype=torch.float32, device=device)
    H = H0
    Hs = []
    for k in range(inner.num_layers):
        W, b = tlp(k, "gcn_linear.weight"), tlp(k, "gcn_linear.bias")
        Hagg = torch.einsum("ij,bjh->bih", A, H)
        Z = Hagg @ W.T + b
        if inner.use_initial_anchor:
            Wa = tlp(k, "anchor_linear.weight")
            alpha = float(sigmoid_scalar(inner.scalar(k, "alpha_raw")))
            Z = Z + alpha * (H0 @ Wa.T)
        R = torch.relu(Z)
        if inner.use_residual:
            beta = float(sigmoid_scalar(inner.scalar(k, "beta_raw")))
            H = R + beta * H
        else:
            H = R
        Hs.append(H)

    src = ([H0] + Hs if inner.include_input_in_jk else Hs) if inner.use_jk else [Hs[-1]]
    Hread = torch.cat(src, dim=2)

    node_x = Hread
    node_emb = inner.p_opt("node_emb.weight")
    if node_emb is not None:
        emb = torch.as_tensor(node_emb, dtype=torch.float32, device=device).unsqueeze(0).expand(Xpq.shape[0], -1, -1)
        node_x = torch.cat([node_x, emb], dim=2)
    node_z = node_x @ tparam("node_hidden.weight").T + tparam("node_hidden.bias")
    node_r = torch.relu(node_z)
    Vres = (node_r @ tparam("node_out.weight").T + tparam("node_out.bias"))[:, :, 0]
    Ws = inner.p_opt("node_skip.weight")
    bs = inner.p_opt("node_skip.bias")
    if Ws is not None:
        Vres = Vres + (
            node_x @ torch.as_tensor(Ws, dtype=torch.float32, device=device).T
            + torch.as_tensor(bs, dtype=torch.float32, device=device)
        )[:, :, 0]

    Wv = torch.as_tensor(inner.W_vlin, dtype=torch.float32, device=device)
    bv = torch.as_tensor(inner.b_vlin, dtype=torch.float32, device=device)
    yvm = torch.as_tensor(inner.norm["YV_mean"], dtype=torch.float32, device=device)
    yvs = torch.as_tensor(inner.norm["YV_std"], dtype=torch.float32, device=device)
    Vn = torch.zeros((Xpq.shape[0], inner.nbus), dtype=torch.float32, device=device)
    Vn[:, 1:] = Vres[:, 1:] + Xn.reshape(Xpq.shape[0], -1) @ Wv.T + bv
    V = torch.zeros_like(Vn)
    V[:, 0] = float(inner.slack_voltage)
    V[:, 1:] = Vn[:, 1:] * yvs + yvm

    edges = np.asarray(inner.edge_list, dtype=int)
    edge_u = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    edge_v = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    Hu = Hread.index_select(1, edge_u)
    Hv = Hread.index_select(1, edge_v)
    edge_x = torch.cat([Hu, Hv, Hu - Hv], dim=2)
    edge_emb = inner.p_opt("edge_emb.weight")
    if edge_emb is not None:
        emb = torch.as_tensor(edge_emb, dtype=torch.float32, device=device).unsqueeze(0).expand(Xpq.shape[0], -1, -1)
        edge_x = torch.cat([edge_x, emb], dim=2)

    edge_z = edge_x @ tparam("edge_hidden.weight").T + tparam("edge_hidden.bias")
    edge_r = torch.relu(edge_z)
    Ires = (edge_r @ tparam("edge_out.weight").T + tparam("edge_out.bias"))[:, :, 0]
    Ws = inner.p_opt("edge_skip.weight")
    bs = inner.p_opt("edge_skip.bias")
    if Ws is not None:
        Ires = Ires + (
            edge_x @ torch.as_tensor(Ws, dtype=torch.float32, device=device).T
            + torch.as_tensor(bs, dtype=torch.float32, device=device)
        )[:, :, 0]

    dst = torch.as_tensor(inner.ilin_dst, dtype=torch.long, device=device)
    Wi = torch.as_tensor(inner.W_ilin, dtype=torch.float32, device=device)
    bi = torch.as_tensor(inner.b_ilin, dtype=torch.float32, device=device)
    yim = torch.as_tensor(inner.norm["YI_mean"], dtype=torch.float32, device=device)
    yis = torch.as_tensor(inner.norm["YI_std"], dtype=torch.float32, device=device)
    Ilin = torch.sum(Xn.index_select(1, dst) * Wi.unsqueeze(0), dim=2) + bi
    I = (Ires + Ilin) * yis + yim
    return V, I


def build_torch_injection_batch(
    q,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    device,
):
    import torch

    q = q.to(dtype=torch.float32, device=device).view(-1, q.shape[-1])
    batch = q.shape[0]
    n_pv, n_ess = len(pv_nodes), len(ess_nodes)
    q_pv = q[:, :n_pv]
    q_ess = q[:, n_pv : n_pv + n_ess]
    q_qdev = q[:, n_pv + n_ess :]

    if x_base is not None:
        base = torch.as_tensor(np.asarray(x_base, dtype=np.float32), dtype=torch.float32, device=device)
        X = base.unsqueeze(0).repeat(batch, 1, 1)
        for idx, bus in enumerate(pv_nodes):
            X[:, int(bus), 1] += -float(pv_q_base[idx]) + q_pv[:, idx]
        for idx, bus in enumerate(ess_nodes):
            X[:, int(bus), 1] += -float(ess_q_base[idx]) + q_ess[:, idx]
        for idx, bus in enumerate(q_device_nodes):
            X[:, int(bus), 1] += -float(q_device_q_base[idx]) + q_qdev[:, idx]
        return X

    X = torch.zeros((batch, 33, 2), dtype=torch.float32, device=device)
    X[:, :, 0] = -torch.as_tensor(p_load, dtype=torch.float32, device=device).view(1, 33)
    X[:, :, 1] = -torch.as_tensor(q_load, dtype=torch.float32, device=device).view(1, 33)
    for idx, bus in enumerate(pv_nodes):
        X[:, int(bus), 0] += float(p_pv_available[idx])
        X[:, int(bus), 1] += q_pv[:, idx]
    for idx, bus in enumerate(ess_nodes):
        X[:, int(bus), 1] += q_ess[:, idx]
    for idx, bus in enumerate(q_device_nodes):
        X[:, int(bus), 1] += q_qdev[:, idx]
    return X


def torch_control_score(
    converter,
    q,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    voltage_margin,
    current_margin,
    safety_penalty,
    q_penalty,
    device,
):
    import torch

    X = build_torch_injection_batch(
        q,
        p_load=p_load,
        q_load=q_load,
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_q_base=q_device_q_base,
        device=device,
    )
    V, I = torch_surrogate_forward(converter.inner, X, device)
    config = converter.inner.engine.get("config", {})
    v_lower = float(config.get("v_lower", 0.95)) - float(voltage_margin)
    v_upper = float(config.get("v_upper", 1.05)) + float(voltage_margin)
    vdev = torch.sum(torch.abs(V[:, 1:] - 1.0), dim=1)
    v_violation = torch.relu(v_lower - V[:, 1:]).sum(dim=1) + torch.relu(V[:, 1:] - v_upper).sum(dim=1)
    i_violation = torch.relu(I - float(current_margin)).sum(dim=1)
    score = vdev + float(safety_penalty) * (v_violation + i_violation)
    if float(q_penalty) > 0.0:
        score = score + float(q_penalty) * torch.sum(torch.abs(q), dim=1)
    return score, V, I


def pytorch_multistart_search(
    converter,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    pv_q_base,
    ess_nodes,
    ess_q_base,
    q_device_nodes,
    q_device_q_base,
    q_device_names,
    control_lower,
    control_upper,
    control_base,
    initial_q,
    voltage_margin,
    current_margin,
    random_samples,
    starts,
    steps,
    lr,
    batch_size,
    safety_penalty,
    q_penalty,
    seed,
    sc_integer,
    sc_step,
    output_flag,
):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lower = np.asarray(control_lower, dtype=float).reshape(-1)
    upper = np.asarray(control_upper, dtype=float).reshape(-1)
    control_base = np.clip(np.asarray(control_base, dtype=float).reshape(-1), lower, upper)
    initial_q = np.clip(np.asarray(initial_q, dtype=float).reshape(-1), lower, upper)
    rng = np.random.default_rng(int(seed))

    sc_qdev_idx = find_q_device_index_by_name(q_device_names, "SC") if sc_integer else None
    sc_global_idx = None if sc_qdev_idx is None else len(pv_nodes) + len(ess_nodes) + int(sc_qdev_idx)
    if sc_global_idx is None:
        sc_values = [None]
    else:
        n_min = int(np.ceil(lower[sc_global_idx] / float(sc_step) - 1e-9))
        n_max = int(np.floor(upper[sc_global_idx] / float(sc_step) + 1e-9))
        sc_values = [n * float(sc_step) for n in range(n_min, n_max + 1)]

    best = {
        "q": None,
        "score": np.inf,
        "Vdev_total": np.inf,
        "Vworst": np.inf,
        "WorstI": np.inf,
        "safety_violation": np.inf,
        "source": None,
        "evaluations": 0,
        "device": str(device),
    }

    def update_best(q_np, source):
        metrics = evaluate_control_batch(
            converter,
            q_np,
            p_load=p_load,
            q_load=q_load,
            p_pv_available=p_pv_available,
            pv_nodes=pv_nodes,
            x_base=x_base,
            pv_q_base=pv_q_base,
            ess_nodes=ess_nodes,
            ess_q_base=ess_q_base,
            q_device_nodes=q_device_nodes,
            q_device_q_base=q_device_q_base,
            voltage_margin=voltage_margin,
            current_margin=current_margin,
            safety_penalty=safety_penalty,
        )
        idx = int(np.argmin(metrics["score"]))
        best["evaluations"] += int(np.asarray(q_np).reshape(-1, lower.size).shape[0])
        if float(metrics["score"][idx]) < best["score"]:
            best.update(
                {
                    "q": np.asarray(q_np, dtype=float).reshape(-1, lower.size)[idx].copy(),
                    "score": float(metrics["score"][idx]),
                    "Vdev_total": float(metrics["Vdev_total"][idx]),
                    "Vworst": float(metrics["Vworst"][idx]),
                    "WorstI": float(metrics["WorstI"][idx]),
                    "safety_violation": float(metrics["safety_violation"][idx]),
                    "source": source,
                }
            )

    if int(output_flag):
        print()
        print("Starting PyTorch multi-start warm search...")
        print(f"  device: {device}")
        print(f"  SC enumeration: {'enabled' if sc_global_idx is not None else 'disabled'}")
        print(f"  SC candidates: {len(sc_values)}")
        print(f"  random pool: {int(random_samples)}, starts per SC: {int(starts)}, Adam steps: {int(steps)}")

    for sc_value in sc_values:
        if int(output_flag):
            sc_text = "continuous" if sc_value is None else f"{sc_value:.6f}"
            print(f"  warm search SC={sc_text} ...", flush=True)
        base_candidates = [control_base.copy(), initial_q.copy()]
        if lower.size:
            base_candidates.append((lower + upper) / 2.0)
        fixed_candidates = []
        for cand in base_candidates:
            cand = cand.copy()
            if sc_global_idx is not None:
                cand[sc_global_idx] = sc_value
            fixed_candidates.append(cand)

        random_pool = []
        remaining = max(0, int(random_samples) // max(1, len(sc_values)))
        while remaining > 0:
            n = min(int(batch_size), remaining)
            cand = rng.uniform(lower, upper, size=(n, lower.size))
            if sc_global_idx is not None:
                cand[:, sc_global_idx] = sc_value
            random_pool.append(cand)
            remaining -= n
        pool = np.vstack(fixed_candidates + random_pool)
        update_best(pool, "torch_random_pool")

        with torch.no_grad():
            scores = []
            for start in range(0, pool.shape[0], int(batch_size)):
                q_t = torch.as_tensor(pool[start : start + int(batch_size)], dtype=torch.float32, device=device)
                s_t, _, _ = torch_control_score(
                    converter,
                    q_t,
                    p_load=p_load,
                    q_load=q_load,
                    p_pv_available=p_pv_available,
                    pv_nodes=pv_nodes,
                    x_base=x_base,
                    pv_q_base=pv_q_base,
                    ess_nodes=ess_nodes,
                    ess_q_base=ess_q_base,
                    q_device_nodes=q_device_nodes,
                    q_device_q_base=q_device_q_base,
                    voltage_margin=voltage_margin,
                    current_margin=current_margin,
                    safety_penalty=safety_penalty,
                    q_penalty=q_penalty,
                    device=device,
                )
                scores.append(s_t.detach().cpu())
            scores = torch.cat(scores).numpy()
        top_idx = np.argsort(scores)[: max(1, int(starts))]
        start_q = pool[top_idx]

        free_idx = np.arange(lower.size)
        if sc_global_idx is not None:
            free_idx = free_idx[free_idx != sc_global_idx]
        if free_idx.size == 0:
            continue

        lb_t = torch.as_tensor(lower[free_idx], dtype=torch.float32, device=device)
        ub_t = torch.as_tensor(upper[free_idx], dtype=torch.float32, device=device)
        span_t = torch.clamp(ub_t - lb_t, min=1e-9)
        start_free = np.clip(start_q[:, free_idx], lower[free_idx] + 1e-6, upper[free_idx] - 1e-6)
        ratio = np.clip((start_free - lower[free_idx]) / np.maximum(upper[free_idx] - lower[free_idx], 1e-9), 1e-6, 1.0 - 1e-6)
        u = torch.as_tensor(np.log(ratio / (1.0 - ratio)), dtype=torch.float32, device=device).requires_grad_(True)
        opt = torch.optim.Adam([u], lr=float(lr))
        template = torch.as_tensor(start_q, dtype=torch.float32, device=device)

        for _ in range(int(steps)):
            q_free = lb_t + span_t * torch.sigmoid(u)
            q_full = template.clone()
            q_full[:, free_idx] = q_free
            loss_vec, _, _ = torch_control_score(
                converter,
                q_full,
                p_load=p_load,
                q_load=q_load,
                p_pv_available=p_pv_available,
                pv_nodes=pv_nodes,
                x_base=x_base,
                pv_q_base=pv_q_base,
                ess_nodes=ess_nodes,
                ess_q_base=ess_q_base,
                q_device_nodes=q_device_nodes,
                q_device_q_base=q_device_q_base,
                voltage_margin=voltage_margin,
                current_margin=current_margin,
                safety_penalty=safety_penalty,
                q_penalty=q_penalty,
                device=device,
            )
            opt.zero_grad()
            loss_vec.sum().backward()
            opt.step()

        with torch.no_grad():
            q_free = lb_t + span_t * torch.sigmoid(u)
            q_full = template.clone()
            q_full[:, free_idx] = q_free
        update_best(q_full.detach().cpu().numpy(), "torch_adam")

    if int(output_flag):
        print()
        print("PyTorch multi-start warm search:")
        print(f"  device: {best['device']}")
        print(f"  SC enumeration: {'enabled' if sc_global_idx is not None else 'disabled'}")
        print(f"  evaluated candidates: {best['evaluations']}")
        print(f"  best source: {best['source']}")
        print(f"  score: {best['score']:.8f}")
        print(f"  Vdev_total: {best['Vdev_total']:.8f}")
        print(f"  Vworst: {best['Vworst']:.8f}")
        print(f"  WorstI: {best['WorstI']:.8f}")
        print(f"  safety violation: {best['safety_violation']:.8f}")
        print(f"  Q*: {np.round(best['q'], 6).tolist()}")

    return best


def build_input_bounds_from_control(
    converter,
    *,
    p_load,
    q_load,
    p_pv_available,
    pv_nodes,
    x_base,
    control_nodes,
    control_base,
    control_lower,
    control_upper,
):
    control_nodes = np.asarray(control_nodes, dtype=int).reshape(-1)
    control_base = np.asarray(control_base, dtype=float).reshape(-1)
    control_lower = np.asarray(control_lower, dtype=float).reshape(-1)
    control_upper = np.asarray(control_upper, dtype=float).reshape(-1)

    if not (
        control_nodes.size
        == control_base.size
        == control_lower.size
        == control_upper.size
    ):
        raise ValueError("Control nodes, base values, and bounds must have the same length")

    C = np.zeros((33, control_nodes.size), dtype=float)
    for idx, bus in enumerate(control_nodes):
        C[int(bus), idx] += 1.0

    if x_base is not None:
        x_base = np.asarray(x_base, dtype=float)
        if x_base.shape != (33, 2):
            raise ValueError(f"x_base must have shape (33, 2), got {x_base.shape}")
        p_fixed = x_base[:, 0].astype(float).copy()
        q_fixed = x_base[:, 1].astype(float).copy() - C @ control_base
    else:
        p_fixed = -np.asarray(p_load, dtype=float).reshape(33)
        q_fixed = -np.asarray(q_load, dtype=float).reshape(33)
        for idx, bus in enumerate(pv_nodes):
            p_fixed[int(bus)] += float(p_pv_available[idx])

    q_lb = q_fixed + np.sum(np.where(C >= 0.0, C * control_lower, C * control_upper), axis=1)
    q_ub = q_fixed + np.sum(np.where(C >= 0.0, C * control_upper, C * control_lower), axis=1)

    S_down = converter.inner.S_down
    S_path = converter.inner.S_path

    X6_lb = np.stack(
        [
            p_fixed,
            q_lb,
            S_down @ p_fixed,
            S_down @ q_lb,
            S_path @ p_fixed,
            S_path @ q_lb,
        ],
        axis=1,
    )
    X6_ub = np.stack(
        [
            p_fixed,
            q_ub,
            S_down @ p_fixed,
            S_down @ q_ub,
            S_path @ p_fixed,
            S_path @ q_ub,
        ],
        axis=1,
    )
    return X6_lb, X6_ub


def apply_control_mip_start(control_vars, q_start, labels):
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    if len(control_vars) != q_start.size:
        raise ValueError("Control vars and MIP-start vector must have the same length")
    for idx, var in enumerate(control_vars):
        value = float(q_start[idx])
        var.Start = value
        var.VarHintVal = value
        var.VarHintPri = 10


def add_conditional_qnet_bounds(
    model,
    x_vars,
    *,
    x_base,
    q_load,
    control_nodes,
    control_base,
    control_lower,
    control_upper,
):
    """Constrain controllable-node Q_net to the current operating-point domain.

    The bounds are conditional on the current fixed net injection and the
    current device capability:

        Q_net = Q_fixed_noncontrol + sum(Q_control_at_node)

    For dataset samples, ``x_base[:, 1]`` already includes the sampled
    controllable Q. For profile inputs, the fixed term is ``-q_load``.
    """
    control_nodes = np.asarray(control_nodes, dtype=int).reshape(-1)
    control_base = np.asarray(control_base, dtype=float).reshape(-1)
    control_lower = np.asarray(control_lower, dtype=float).reshape(-1)
    control_upper = np.asarray(control_upper, dtype=float).reshape(-1)

    if not (
        control_nodes.size
        == control_base.size
        == control_lower.size
        == control_upper.size
    ):
        raise ValueError("Control nodes, base values, and bounds must have the same length")

    if control_nodes.size == 0:
        return {
            "enabled": True,
            "nodes": [],
            "lower": [],
            "upper": [],
            "fixed_q": [],
        }

    if np.any((control_nodes < 0) | (control_nodes >= 33)):
        bad = int(control_nodes[np.where((control_nodes < 0) | (control_nodes >= 33))[0][0]])
        raise ValueError(f"Control node index out of range: {bad}")

    if x_base is not None:
        x_base = np.asarray(x_base, dtype=float)
        if x_base.shape != (33, 2):
            raise ValueError(f"x_base must have shape (33, 2), got {x_base.shape}")
        fixed_q = x_base[:, 1].astype(float).copy()
    else:
        q_load = np.asarray(q_load, dtype=float).reshape(-1)
        if q_load.size != 33:
            raise ValueError(f"q_load must have length 33 when x_base is None, got {q_load.size}")
        fixed_q = -q_load.astype(float).copy()

    has_control = np.zeros(33, dtype=bool)
    qnet_lower = fixed_q.copy()
    qnet_upper = fixed_q.copy()

    for idx, bus in enumerate(control_nodes):
        bus = int(bus)
        has_control[bus] = True
        fixed_q[bus] -= float(control_base[idx])
        qnet_lower[bus] -= float(control_base[idx])
        qnet_upper[bus] -= float(control_base[idx])

    for idx, bus in enumerate(control_nodes):
        bus = int(bus)
        qnet_lower[bus] += float(control_lower[idx])
        qnet_upper[bus] += float(control_upper[idx])

    nodes = np.where(has_control)[0]
    for bus in nodes:
        if qnet_upper[bus] < qnet_lower[bus]:
            raise ValueError(
                f"Invalid conditional Q_net bounds at bus {int(bus) + 1}: "
                f"lower={qnet_lower[bus]}, upper={qnet_upper[bus]}"
            )
        model.addConstr(
            x_vars[int(bus)][1] >= float(qnet_lower[bus]),
            name=f"Conditional_Qnet_lb_bus{int(bus) + 1}",
        )
        model.addConstr(
            x_vars[int(bus)][1] <= float(qnet_upper[bus]),
            name=f"Conditional_Qnet_ub_bus{int(bus) + 1}",
        )

    return {
        "enabled": True,
        "nodes": nodes.astype(int).tolist(),
        "lower": qnet_lower[nodes].astype(float).tolist(),
        "upper": qnet_upper[nodes].astype(float).tolist(),
        "fixed_q": fixed_q[nodes].astype(float).tolist(),
    }


def sanitize_name(value: str) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() else "_")
    return "".join(keep).strip("_") or "control"


def add_surrogate_physical_output_bounds(model, outputs, *, enabled: bool):
    if not enabled:
        return {}
    bounds = {
        "Vdev_total_min": 0.0,
        "WorstI_min": -1.0,
        "Vworst_min": -0.05,
    }
    model.addConstr(outputs.Vdev_total >= bounds["Vdev_total_min"], name="Physical_Vdev_total_nonnegative")
    model.addConstr(outputs.WorstI >= bounds["WorstI_min"], name="Physical_WorstI_lower")
    model.addConstr(outputs.Vworst >= bounds["Vworst_min"], name="Physical_Vworst_lower")
    return bounds


def add_control_trust_region(
    model,
    control_vars,
    q_reference,
    lower,
    upper,
    labels,
    *,
    fraction: float,
    radius=None,
):
    q_reference = np.asarray(q_reference, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if not (control_vars and q_reference.size == lower.size == upper.size == len(control_vars)):
        if len(control_vars) == 0 and q_reference.size == lower.size == upper.size == 0:
            return q_reference, np.array([], dtype=float)
        raise ValueError("Control vars, references, and bounds must have the same length")
    if fraction < 0.0:
        raise ValueError("trust_region_fraction must be nonnegative")

    q_ref = np.clip(q_reference, lower, upper)
    width = np.maximum(upper - lower, 0.0)
    if radius is None:
        trust_radius = float(fraction) * width
        if fraction >= 1.0:
            return q_ref, trust_radius
    else:
        trust_radius = np.asarray(radius, dtype=float).reshape(-1)
        if trust_radius.size != q_ref.size:
            raise ValueError(f"trust-region radius must have length {q_ref.size}, got {trust_radius.size}")
        trust_radius = np.maximum(trust_radius, 0.0)

    for idx, var in enumerate(control_vars):
        name = sanitize_name(labels[idx] if idx < len(labels) else f"Q_{idx}")
        model.addConstr(var <= float(q_ref[idx] + trust_radius[idx]), name=f"Trust_ub_{idx}_{name}")
        model.addConstr(var >= float(q_ref[idx] - trust_radius[idx]), name=f"Trust_lb_{idx}_{name}")
    return q_ref, trust_radius


def prepare_control_start(
    warm_start_q,
    control_base,
    lower,
    upper,
    *,
    expected: int,
):
    base = np.asarray(control_base, dtype=float).reshape(-1)
    if base.size != expected:
        raise ValueError(f"control_base must have length {expected}, got {base.size}")
    if warm_start_q is None:
        return np.clip(base, lower, upper), "operating_point"

    start = np.asarray(warm_start_q, dtype=float).reshape(-1)
    if start.size != expected:
        raise ValueError(f"warm_start_q must have length {expected}, got {start.size}")
    return np.clip(start, lower, upper), "user_warm_start"


def snap_to_step(value, lower, upper, step):
    return float(np.clip(round(float(value) / float(step)) * float(step), lower, upper))


def find_q_device_index_by_name(q_device_names, target_name="SC"):
    target = str(target_name).strip().lower()
    for idx, name in enumerate(q_device_names):
        if str(name).strip().lower() == target:
            return idx
    return None


def add_sc_integer_link(
    model,
    q_device,
    q_device_min,
    q_device_max,
    q_device_names,
    *,
    enabled: bool,
    step: float,
    target_name: str = "SC",
):
    if not enabled:
        return None, None
    qdev_idx = find_q_device_index_by_name(q_device_names, target_name)
    if qdev_idx is None:
        return None, None
    step = float(step)
    if step <= 0.0:
        raise ValueError("sc_step must be positive")
    n_min = int(np.ceil(float(q_device_min[qdev_idx]) / step - 1e-9))
    n_max = int(np.floor(float(q_device_max[qdev_idx]) / step + 1e-9))
    sc_n = model.addVar(lb=n_min, ub=n_max, vtype=GRB.INTEGER, name=f"{target_name}_step")
    model.addConstr(q_device[qdev_idx] == step * sc_n, name=f"{target_name}_discrete_link")
    return qdev_idx, sc_n


def default_local_trust_radius(control_lower, control_upper, *, n_pv, n_ess, q_device_names):
    lower = np.asarray(control_lower, dtype=float).reshape(-1)
    upper = np.asarray(control_upper, dtype=float).reshape(-1)
    radius = 0.06 * np.ones_like(lower)
    q_start = int(n_pv) + int(n_ess)
    for idx, name in enumerate(q_device_names):
        global_idx = q_start + idx
        lname = str(name).strip().lower()
        if lname == "sc":
            radius[global_idx] = 0.0
        elif lname == "svc2":
            radius[global_idx] = 0.12
        else:
            radius[global_idx] = 0.08
    return np.minimum(radius, np.maximum(upper - lower, 0.0))


def add_control_deviation_penalty(
    model,
    control_vars,
    q_reference,
    labels,
    *,
    weight: float,
):
    penalty = gp.LinExpr(0.0)
    aux_vars = []
    if weight <= 0.0:
        return penalty, aux_vars
    for idx, var in enumerate(control_vars):
        name = sanitize_name(labels[idx] if idx < len(labels) else f"Q_{idx}")
        dev = model.addVar(lb=0.0, name=f"AbsDev_{idx}_{name}")
        ref = float(q_reference[idx])
        model.addConstr(dev >= var - ref, name=f"AbsDev_pos_{idx}_{name}")
        model.addConstr(dev >= ref - var, name=f"AbsDev_neg_{idx}_{name}")
        penalty += float(weight) * dev
        aux_vars.append(dev)
    return penalty, aux_vars


def build_objective_expression(
    outputs,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_expr=None,
):
    if objective == "vdev":
        return outputs.Vdev_total
    raise ValueError("Exp26 has no learned Ploss head; use objective='vdev'")


def objective_value_from_components(
    output_values,
    objective: str,
    *,
    loss_weight: float,
    voltage_weight: float,
    loss_value=None,
):
    if output_values is None:
        return None
    vdev = float(output_values["Vdev_total"])
    if objective == "vdev":
        return vdev
    raise ValueError("Exp26 has no learned Ploss head; use objective='vdev'")


def select_loss_objective_expression(
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    surrogate_loss_weight: float,
):
    if loss_objective_model in {"none", "disabled"}:
        return None
    raise ValueError("Exp26 has no learned Ploss head; use --loss-objective-model none")


def add_surrogate_loss_consistency_constraints(
    model,
    converter,
    outputs,
    *,
    loss_objective_model: str,
    base_kv: float,
    base_mva: float,
    relative_tol: float,
    absolute_tol: float,
):
    if loss_objective_model not in {"none", "disabled"}:
        raise ValueError("Exp26 loss consistency is unavailable without a learned Ploss head")
    return None


def add_objective(
    model,
    outputs,
    objective: str,
    loss_weight: float,
    voltage_weight: float,
):
    expr = build_objective_expression(
        outputs,
        objective,
        loss_weight=loss_weight,
        voltage_weight=voltage_weight,
    )
    model.setObjective(expr, GRB.MINIMIZE)


def add_surrogate_safety_constraints(
    model,
    outputs,
    *,
    voltage_margin: float,
    current_margin: float,
    diagnose_slack: bool,
    slack_penalty: float,
    base_objective,
):
    if not diagnose_slack:
        model.addConstr(
            outputs.Vworst <= float(voltage_margin),
            name="Surrogate_Vworst_safe",
        )
        model.addConstr(
            outputs.WorstI <= float(current_margin),
            name="Surrogate_WorstI_safe",
        )
        return None, None

    s_v = model.addVar(lb=0.0, name="slack_Vworst_safe")
    s_i = model.addVar(lb=0.0, name="slack_WorstI_safe")
    model.addConstr(
        outputs.Vworst <= float(voltage_margin) + s_v,
        name="Surrogate_Vworst_safe_soft",
    )
    model.addConstr(
        outputs.WorstI <= float(current_margin) + s_i,
        name="Surrogate_WorstI_safe_soft",
    )
    model.setObjective(
        float(slack_penalty) * (s_v + s_i) + base_objective,
        GRB.MINIMIZE,
    )
    return s_v, s_i


def run_single_step_rpo(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    *,
    pv_nodes,
    s_rated,
    engine_path=DEFAULT_ENGINE_PATH,
    relu_formulation="big_m",
    big_m_scale=1.0,
    fallback_big_m=1e3,
    objective="vdev",
    loss_objective_model="none",
    surrogate_loss_weight=0.5,
    surrogate_loss_consistency_rel=0.5,
    surrogate_loss_consistency_abs=0.002,
    loss_weight=0.0,
    voltage_weight=1e-3,
    lambda_q=0.0,
    voltage_margin=0.0,
    current_margin=0.0,
    physical_output_bounds=True,
    trust_region_fraction=1.0,
    trust_region_radius=None,
    control_deviation_penalty=0.0,
    base_kv=12.66,
    base_mva=1.0,
    time_limit=300.0,
    mip_gap=0.01,
    dual_reductions=0,
    threads=8,
    mip_focus=2,
    method=1,
    node_method=1,
    presolve=2,
    presparsify=1,
    heuristics=0.15,
    cuts=2,
    var_branch=2,
    numeric_focus=1,
    diagnose_slack=False,
    slack_penalty=1e4,
    output_flag=1,
    export_lp=None,
    iis_path=None,
    x_base=None,
    pv_q_base=None,
    ess_nodes=None,
    ess_s_rated=None,
    ess_p_base=None,
    ess_q_base=None,
    q_device_nodes=None,
    q_device_min=None,
    q_device_max=None,
    q_device_q_base=None,
    q_device_names=None,
    warm_start_q=None,
    sc_integer=True,
    sc_step=0.1,
    sc_device_name="SC",
):
    gp_mod, grb_mod = require_gurobi()
    converter_class = get_converter_class()

    p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated = validate_inputs(
        p_load,
        q_load,
        p_pv_available,
        topo_mask,
        pv_nodes,
        s_rated,
    )
    ess_nodes = normalize_optional_vector(ess_nodes, dtype=int)
    ess_s_rated = normalize_optional_vector(ess_s_rated)
    ess_p_base = normalize_optional_vector(ess_p_base)
    ess_q_base = normalize_optional_vector(ess_q_base)
    q_device_nodes = normalize_optional_vector(q_device_nodes, dtype=int)
    q_device_min = normalize_optional_vector(q_device_min)
    q_device_max = normalize_optional_vector(q_device_max)
    q_device_q_base = normalize_optional_vector(q_device_q_base)
    q_device_names = [] if q_device_names is None else [str(x) for x in q_device_names]
    if float(lambda_q) != 0.0:
        raise ValueError("Exp26 objective is voltage-only; set lambda_q=0")

    if len(ess_nodes) not in {0, len(ess_s_rated), len(ess_p_base), len(ess_q_base)}:
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(ess_nodes) and not (
        len(ess_s_rated) == len(ess_p_base) == len(ess_q_base) == len(ess_nodes)
    ):
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(q_device_nodes) and not (
        len(q_device_min) == len(q_device_max) == len(q_device_q_base) == len(q_device_nodes)
    ):
        raise ValueError("Q-device node, min, max, and base reactive arrays must have the same length")

    model = gp_mod.Model("Exp26_SingleStep_RPO")
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.OutputFlag = int(output_flag)
    model.Params.NonConvex = 2
    model.Params.DualReductions = int(dual_reductions)
    if int(threads) > 0:
        model.Params.Threads = int(threads)
    model.Params.MIPFocus = int(mip_focus)
    model.Params.Method = int(method)
    model.Params.NodeMethod = int(node_method)
    model.Params.Presolve = int(presolve)
    model.Params.PreSparsify = int(presparsify)
    model.Params.Heuristics = float(heuristics)
    model.Params.Cuts = int(cuts)
    model.Params.VarBranch = int(var_branch)
    model.Params.NumericFocus = int(numeric_focus)

    converter = converter_class(
        engine_path,
        relu_formulation=relu_formulation,
        fallback_big_m=fallback_big_m,
        big_m_scale=big_m_scale,
    )

    q_pv = model.addVars(len(pv_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_pv")
    q_ess = model.addVars(len(ess_nodes), lb=-grb_mod.INFINITY, ub=grb_mod.INFINITY, name="Q_ess")
    q_device = model.addVars(
        len(q_device_nodes),
        lb=-grb_mod.INFINITY,
        ub=grb_mod.INFINITY,
        name="Q_device",
    )
    q_caps = add_pv_capacity_constraints(
        model,
        q_pv,
        p_pv_available,
        pv_nodes,
        s_rated,
    )
    ess_q_caps = add_ess_capacity_constraints(
        model,
        q_ess,
        ess_p_base,
        ess_nodes,
        ess_s_rated,
    )
    qdev_min, qdev_max = add_q_device_bounds(
        model,
        q_device,
        q_device_nodes,
        q_device_min,
        q_device_max,
    )
    sc_qdev_idx, sc_step_var = add_sc_integer_link(
        model,
        q_device,
        qdev_min,
        qdev_max,
        q_device_names,
        enabled=bool(sc_integer),
        step=float(sc_step),
        target_name=str(sc_device_name),
    )
    labels = build_control_labels(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
        q_device_names,
    )
    control_vars = control_variable_list(
        q_pv,
        q_ess,
        q_device,
        len(pv_nodes),
        len(ess_nodes),
        len(q_device_nodes),
    )
    control_nodes = build_control_node_vector(
        pv_nodes,
        ess_nodes,
        q_device_nodes,
    )
    control_lower, control_upper = build_control_bound_vectors(
        q_caps,
        ess_q_caps,
        qdev_min,
        qdev_max,
    )
    control_base = build_control_base_vector(
        pv_q_base if x_base is not None else None,
        ess_q_base,
        q_device_q_base,
        n_pv=len(pv_nodes),
        n_ess=len(ess_nodes),
        n_qdev=len(q_device_nodes),
    )
    sc_global_idx = None
    if sc_qdev_idx is not None:
        sc_global_idx = len(pv_nodes) + len(ess_nodes) + int(sc_qdev_idx)
    q_reference, warm_start_source = prepare_control_start(
        warm_start_q,
        control_base,
        control_lower,
        control_upper,
        expected=len(control_vars),
    )
    if sc_global_idx is not None:
        q_reference[sc_global_idx] = snap_to_step(
            q_reference[sc_global_idx],
            control_lower[sc_global_idx],
            control_upper[sc_global_idx],
            sc_step,
        )
        n_start = int(round(q_reference[sc_global_idx] / float(sc_step)))
        sc_step_var.Start = n_start
        sc_step_var.VarHintVal = n_start
    q_reference, trust_radius = add_control_trust_region(
        model,
        control_vars,
        q_reference,
        control_lower,
        control_upper,
        labels,
        fraction=float(trust_region_fraction),
        radius=trust_region_radius,
    )
    apply_control_mip_start(control_vars, q_reference, labels)

    x_start = build_numeric_injection_start(
        p_load=p_load,
        q_load=q_load,
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_q_base=q_device_q_base,
        q_control=q_reference,
    )
    if x_base is None:
        x_vars = build_injection_expressions(
            model,
            q_pv,
            p_load,
            q_load,
            p_pv_available,
            pv_nodes,
        )
    else:
        if pv_q_base is None:
            raise ValueError("pv_q_base is required when x_base is provided")
        x_vars = build_dataset_injection_expressions(
            model,
            q_pv,
            q_ess,
            q_device,
            x_base,
            pv_q_base,
            pv_nodes,
            ess_q_base,
            ess_nodes,
            q_device_q_base,
            q_device_nodes,
        )

    conditional_qnet_bounds = add_conditional_qnet_bounds(
        model,
        x_vars,
        x_base=x_base,
        q_load=q_load,
        control_nodes=control_nodes,
        control_base=control_base,
        control_lower=control_lower,
        control_upper=control_upper,
    )

    X6_lb, X6_ub = build_input_bounds_from_control(
        converter,
        p_load=p_load,
        q_load=q_load,
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        x_base=x_base,
        control_nodes=control_nodes,
        control_base=control_base,
        control_lower=control_lower,
        control_upper=control_upper,
    )
    converter.inner.set_opf_input_bounds(X6_lb, X6_ub)

    print("Embedding Exp26 ST-GCN surrogate constraints...")
    relu_starts = converter.inner.compute_relu_starts(x_start)
    outputs = converter.embed_sgcn_constraints(
        model,
        x_vars,
        topo_mask=topo_mask,
        name_prefix="exp26_rpo",
        relu_starts=relu_starts,
    )
    embed_aux = converter.last_variables.get("embed_aux", {}) if converter.last_variables else {}

    output_physical_bounds = add_surrogate_physical_output_bounds(
        model,
        outputs,
        enabled=bool(physical_output_bounds),
    )
    loss_objective_expr = select_loss_objective_expression(
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        surrogate_loss_weight=float(surrogate_loss_weight),
    )
    loss_consistency = add_surrogate_loss_consistency_constraints(
        model,
        converter,
        outputs,
        loss_objective_model=str(loss_objective_model),
        base_kv=float(base_kv),
        base_mva=float(base_mva),
        relative_tol=float(surrogate_loss_consistency_rel),
        absolute_tol=float(surrogate_loss_consistency_abs),
    )
    base_objective = build_objective_expression(
        outputs,
        objective,
        loss_weight=loss_weight,
        voltage_weight=voltage_weight,
        loss_expr=loss_objective_expr,
    )
    deviation_penalty, deviation_aux = add_control_deviation_penalty(
        model,
        control_vars,
        q_reference,
        labels,
        weight=float(control_deviation_penalty),
    )
    reactive_flow_penalty = gp_mod.QuadExpr()
    protected_objective = base_objective + deviation_penalty + reactive_flow_penalty
    model.setObjective(protected_objective, grb_mod.MINIMIZE)
    slack_v, slack_i = add_surrogate_safety_constraints(
        model,
        outputs,
        voltage_margin=voltage_margin,
        current_margin=current_margin,
        diagnose_slack=diagnose_slack,
        slack_penalty=slack_penalty,
        base_objective=protected_objective,
    )

    if export_lp:
        export_path = Path(export_lp)
        if not export_path.is_absolute():
            export_path = REPO_ROOT / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(export_path))
        print(f"LP/MPS written before solve: {export_path}")

    print("Starting Gurobi solve...")
    model.optimize()

    result = {
        "status": int(model.status),
        "status_name": status_name(model.status),
        "objective": None,
        "best_bound": None,
        "mip_gap": None,
        "objective_mode": objective,
        "loss_objective_model": str(loss_objective_model),
        "surrogate_loss_weight": float(surrogate_loss_weight),
        "surrogate_loss_consistency": loss_consistency,
        "loss_weight": float(loss_weight),
        "voltage_weight": float(voltage_weight),
        "Q_pv_opt": None,
        "Q_pv_caps": q_caps,
        "Q_ess_opt": None,
        "Q_ess_caps": ess_q_caps,
        "Q_device_opt": None,
        "Q_device_min": qdev_min,
        "Q_device_max": qdev_max,
        "Q_control_opt": None,
        "Q_control_labels": labels,
        "Q_control_reference": q_reference,
        "Q_control_warm_start_source": warm_start_source,
        "Q_control_lower": control_lower,
        "Q_control_upper": control_upper,
        "Q_control_trust_radius": trust_radius,
        "SC_integer": bool(sc_qdev_idx is not None),
        "SC_step": float(sc_step) if sc_qdev_idx is not None else None,
        "conditional_qnet_bounds": conditional_qnet_bounds,
        "trust_region_fraction": float(trust_region_fraction),
        "physical_output_bounds": output_physical_bounds,
        "lambda_q": float(lambda_q),
        "q_flow_regularization": None,
        "q_flow_penalty": None,
        "q_flow_branch_values": None,
        "control_deviation_penalty_weight": float(control_deviation_penalty),
        "control_deviation_abs": None,
        "loss_objective_value": None,
        "objective_components": None,
        "outputs": None,
        "safety_slack": None,
        "binary_count_estimate": converter.binary_count,
        "embed_aux": embed_aux,
        "sol_count": int(model.SolCount),
    }

    if model.SolCount > 0:
        q_solution = np.array([q_pv[idx].X for idx in range(len(pv_nodes))], dtype=float)
        q_ess_solution = np.array(
            [q_ess[idx].X for idx in range(len(ess_nodes))],
            dtype=float,
        )
        q_device_solution = np.array(
            [q_device[idx].X for idx in range(len(q_device_nodes))],
            dtype=float,
        )
        q_control_solution = np.concatenate(
            [q_solution, q_ess_solution, q_device_solution]
        )
        output_values = {
            "Vdev_total": float(outputs.Vdev_total.X),
            "Vworst": float(outputs.Vworst.X),
            "WorstI": float(outputs.WorstI.X),
            "Ploss_total": None,
        }
        loss_objective_value = None
        deviation_values = np.array([dev.X for dev in deviation_aux], dtype=float)
        guarded_objective = objective_value_from_components(
            output_values,
            objective,
            loss_weight=loss_weight,
            voltage_weight=voltage_weight,
            loss_value=loss_objective_value,
        )
        deviation_penalty_value = float(control_deviation_penalty) * float(
            deviation_values.sum()
        )
        q_flow_regularization = None
        q_flow_penalty_value = None
        q_flow_branch_values = None
        objective_components = {
            "guarded_objective": guarded_objective,
            "loss_objective": None,
            "surrogate_ploss": None,
            "q_flow_regularization": q_flow_regularization,
            "q_flow_penalty": q_flow_penalty_value,
            "control_deviation_penalty": deviation_penalty_value,
            "slack_penalty": None,
        }
        slack_values = None
        if slack_v is not None and slack_i is not None:
            slack_values = {
                "Vworst": float(slack_v.X),
                "WorstI": float(slack_i.X),
            }
            objective_components["slack_penalty"] = float(slack_penalty) * (
                slack_values["Vworst"] + slack_values["WorstI"]
            )
        result.update(
            {
                "objective": float(model.ObjVal),
                "best_bound": float(model.ObjBound),
                "mip_gap": float(model.MIPGap),
                "Q_pv_opt": q_solution,
                "Q_ess_opt": q_ess_solution,
                "Q_device_opt": q_device_solution,
                "Q_control_opt": q_control_solution,
                "control_deviation_abs": deviation_values,
                "q_flow_regularization": q_flow_regularization,
                "q_flow_penalty": q_flow_penalty_value,
                "q_flow_branch_values": q_flow_branch_values,
                "loss_objective_value": loss_objective_value,
                "objective_components": objective_components,
                "outputs": output_values,
                "safety_slack": slack_values,
            }
        )
        print_solution(
            result,
            pv_nodes=pv_nodes,
            p_pv_available=p_pv_available,
            ess_nodes=ess_nodes,
            q_device_nodes=q_device_nodes,
            q_device_names=q_device_names,
        )
    elif model.status == grb_mod.INFEASIBLE and iis_path:
        iis_out = Path(iis_path)
        if not iis_out.is_absolute():
            iis_out = REPO_ROOT / iis_out
        iis_out.parent.mkdir(parents=True, exist_ok=True)
        print("Model infeasible; computing IIS...")
        model.computeIIS()
        model.write(str(iis_out))
        print(f"IIS written: {iis_out}")
    else:
        print(f"No feasible solution. Status={model.status} ({status_name(model.status)})")
        if model.status == grb_mod.INF_OR_UNBD and int(dual_reductions) != 0:
            print("Hint: rerun with --dual-reductions 0 to distinguish infeasible vs unbounded.")
        if not diagnose_slack:
            print("Hint: rerun with --diagnose-slack to minimize safety-constraint violations.")

    return result


def run_two_stage_rpo(
    p_load,
    q_load,
    p_pv_available,
    topo_mask,
    *,
    heuristic_random_samples=20000,
    heuristic_batch_size=2048,
    heuristic_safety_penalty=500.0,
    heuristic_seed=2026,
    local_trust_region_fraction=0.30,
    local_trust_region_radius=None,
    torch_warm_starts=8,
    torch_warm_steps=700,
    torch_warm_lr=0.05,
    torch_q_penalty=0.0,
    sc_integer=True,
    sc_step=0.1,
    **rpo_kwargs,
):
    """SC enumeration + PyTorch multi-start search followed by local MILP."""
    gp_mod, grb_mod = require_gurobi()
    del gp_mod, grb_mod

    converter_class = get_converter_class()
    pv_nodes = rpo_kwargs["pv_nodes"]
    s_rated = rpo_kwargs["s_rated"]
    engine_path = rpo_kwargs.get("engine_path", DEFAULT_ENGINE_PATH)

    p_load, q_load, p_pv_available, topo_mask, pv_nodes, s_rated = validate_inputs(
        p_load,
        q_load,
        p_pv_available,
        topo_mask,
        pv_nodes,
        s_rated,
    )
    ess_nodes = normalize_optional_vector(rpo_kwargs.get("ess_nodes"), dtype=int)
    ess_s_rated = normalize_optional_vector(rpo_kwargs.get("ess_s_rated"))
    ess_p_base = normalize_optional_vector(rpo_kwargs.get("ess_p_base"))
    ess_q_base = normalize_optional_vector(rpo_kwargs.get("ess_q_base"))
    q_device_nodes = normalize_optional_vector(rpo_kwargs.get("q_device_nodes"), dtype=int)
    q_device_min = normalize_optional_vector(rpo_kwargs.get("q_device_min"))
    q_device_max = normalize_optional_vector(rpo_kwargs.get("q_device_max"))
    q_device_q_base = normalize_optional_vector(rpo_kwargs.get("q_device_q_base"))
    q_device_names = [] if rpo_kwargs.get("q_device_names") is None else [str(x) for x in rpo_kwargs["q_device_names"]]
    x_base = rpo_kwargs.get("x_base")
    pv_q_base = rpo_kwargs.get("pv_q_base")

    if len(ess_nodes) and not (
        len(ess_s_rated) == len(ess_p_base) == len(ess_q_base) == len(ess_nodes)
    ):
        raise ValueError("ESS node, rating, active-power, and reactive-power arrays must have the same length")
    if len(q_device_nodes) and not (
        len(q_device_min) == len(q_device_max) == len(q_device_q_base) == len(q_device_nodes)
    ):
        raise ValueError("Q-device node, min, max, and base reactive arrays must have the same length")

    metadata = compute_control_metadata(
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        s_rated=s_rated,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_s_rated=ess_s_rated,
        ess_p_base=ess_p_base,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
    )
    q_seed, seed_source = prepare_control_start(
        rpo_kwargs.get("warm_start_q"),
        metadata["control_base"],
        metadata["control_lower"],
        metadata["control_upper"],
        expected=metadata["control_lower"].size,
    )

    converter = converter_class(
        engine_path,
        relu_formulation=rpo_kwargs.get("relu_formulation", "big_m"),
        fallback_big_m=rpo_kwargs.get("fallback_big_m", 1e3),
        big_m_scale=rpo_kwargs.get("big_m_scale", 1.0),
    )
    heuristic = pytorch_multistart_search(
        converter,
        p_load=p_load,
        q_load=q_load,
        p_pv_available=p_pv_available,
        pv_nodes=pv_nodes,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
        control_lower=metadata["control_lower"],
        control_upper=metadata["control_upper"],
        control_base=metadata["control_base"],
        initial_q=q_seed,
        voltage_margin=rpo_kwargs.get("voltage_margin", 0.0),
        current_margin=rpo_kwargs.get("current_margin", 0.0),
        random_samples=heuristic_random_samples,
        starts=torch_warm_starts,
        steps=torch_warm_steps,
        lr=torch_warm_lr,
        batch_size=heuristic_batch_size,
        safety_penalty=heuristic_safety_penalty,
        q_penalty=torch_q_penalty,
        seed=heuristic_seed,
        sc_integer=sc_integer,
        sc_step=sc_step,
        output_flag=rpo_kwargs.get("output_flag", 1),
    )

    if local_trust_region_radius is None:
        local_trust_region_radius = default_local_trust_radius(
            metadata["control_lower"],
            metadata["control_upper"],
            n_pv=len(pv_nodes),
            n_ess=len(ess_nodes),
            q_device_names=q_device_names,
        )
    else:
        local_trust_region_radius = np.asarray(local_trust_region_radius, dtype=float).reshape(-1)

    if int(rpo_kwargs.get("output_flag", 1)):
        print()
        print("Local trust-region MILP:")
        print(f"  warm-start source before heuristic: {seed_source}")
        print(f"  trust-region fraction fallback: {float(local_trust_region_fraction):.4f}")
        print(f"  trust-region radius: {np.round(local_trust_region_radius, 6).tolist()}")

    single_kwargs = dict(rpo_kwargs)
    single_kwargs.update(
        {
            "warm_start_q": heuristic["q"],
            "trust_region_fraction": float(local_trust_region_fraction),
            "trust_region_radius": local_trust_region_radius,
            "sc_integer": sc_integer,
            "sc_step": sc_step,
        }
    )
    result = run_single_step_rpo(
        p_load,
        q_load,
        p_pv_available,
        topo_mask,
        **single_kwargs,
    )
    result["strategy"] = "two_stage"
    result["forward_heuristic"] = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in heuristic.items()
    }
    return result


def status_name(status_code: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status_code, f"STATUS_{status_code}")


def print_solution(
    result,
    *,
    pv_nodes,
    p_pv_available,
    ess_nodes,
    q_device_nodes,
    q_device_names,
):
    print()
    print("Single-step reactive power optimization result")
    print(f"Status: {result['status_name']} ({result['status']})")
    print(f"Objective: {result['objective']:.8f}")
    if result.get("q_flow_regularization") is not None:
        print(
            "Reactive-flow regularizer: "
            f"J_Q_flow={result['q_flow_regularization']:.8f}, "
            f"lambda_Q*J_Q_flow={result['q_flow_penalty']:.8f}"
        )
    print(f"Estimated surrogate ReLU binaries: {result['binary_count_estimate']}")
    embed_aux = result.get("embed_aux") or {}
    if embed_aux:
        print(f"Embedded ReLU binaries: {embed_aux.get('binary_created')}")
        print(f"ReLU fixed zero/linear: {embed_aux.get('relu_fixed_zero')} / {embed_aux.get('relu_fixed_linear')}")
    print()
    print("PV dispatch:")
    q_solution = result["Q_pv_opt"]
    q_caps = result["Q_pv_caps"]
    for idx, bus in enumerate(pv_nodes):
        print(
            f"  bus {int(bus):02d}: P={p_pv_available[idx]:.6f} MW, "
            f"Q={q_solution[idx]: .6f} MVar, |Q|max={q_caps[idx]:.6f}"
        )
    if len(ess_nodes):
        print()
        print("ESS dispatch:")
        q_ess_solution = result["Q_ess_opt"]
        q_ess_caps = result["Q_ess_caps"]
        for idx, bus in enumerate(ess_nodes):
            print(
                f"  bus {int(bus):02d}: "
                f"Q={q_ess_solution[idx]: .6f} MVar, |Q|max={q_ess_caps[idx]:.6f}"
            )
    if len(q_device_nodes):
        print()
        print("Reactive device dispatch:")
        q_device_solution = result["Q_device_opt"]
        q_device_min = result["Q_device_min"]
        q_device_max = result["Q_device_max"]
        for idx, bus in enumerate(q_device_nodes):
            name = q_device_names[idx] if idx < len(q_device_names) else f"QDev{idx + 1}"
            print(
                f"  {name} bus {int(bus):02d}: "
                f"Q={q_device_solution[idx]: .6f} MVar, "
                f"bounds=[{q_device_min[idx]:.6f}, {q_device_max[idx]:.6f}]"
            )
    print()
    print("Surrogate outputs:")
    for key, value in result["outputs"].items():
        if value is None:
            print(f"  {key}: unavailable")
        else:
            print(f"  {key}: {value:.8f}")
    if result.get("safety_slack") is not None:
        print()
        print("Safety slack diagnostics:")
        for key, value in result["safety_slack"].items():
            print(f"  {key}: {value:.8f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-step reactive power optimization with Exp26 ST-GCN MILP."
    )
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE_PATH))
    parser.add_argument(
        "--strategy",
        choices=["two_stage", "single"],
        default=DEFAULT_RUN_CONFIG["strategy"],
        help=(
            "two_stage runs forward heuristic search first, then solves a local "
            "trust-region MILP; single runs the MILP directly."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["dataset", "profile"],
        default="dataset",
        help="dataset uses one sample from the current training dataset; profile uses the old IEEE33 default profile.",
    )
    parser.add_argument(
        "--data",
        default="data/ieee33_nodal_pq_correlated_raw_pool_50k.pt",
        help="Dataset path when --source dataset is used.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="Dataset sample index. Use -1 for deterministic auto selection.",
    )
    parser.add_argument("--pv-pu", type=float, default=0.8)
    parser.add_argument(
        "--p-load",
        default=None,
        help="JSON list or JSON file containing 33 MW loads.",
    )
    parser.add_argument(
        "--q-load",
        default=None,
        help="JSON list or JSON file containing 33 MVar loads.",
    )
    parser.add_argument(
        "--p-pv",
        default=None,
        help="JSON list or JSON file containing active PV output at the PV buses.",
    )
    parser.add_argument(
        "--objective",
        choices=["vdev"],
        default="vdev",
        help=(
            "Exp26 has no learned Ploss head, so the MILP objective is the "
            "learned cumulative voltage deviation."
        ),
    )
    parser.add_argument(
        "--loss-objective-model",
        choices=["none"],
        default="none",
        help=(
            "No loss term is available in Exp26. Kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--surrogate-loss-weight",
        type=float,
        default=0.5,
        help="Unused for Exp26; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--surrogate-loss-consistency-rel",
        type=float,
        default=0.5,
        help=(
            "Unused for Exp26; kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--surrogate-loss-consistency-abs",
        type=float,
        default=0.002,
        help="Unused for Exp26; kept for CLI compatibility.",
    )
    parser.add_argument("--loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--voltage-weight",
        type=float,
        default=1e-3,
        help="Unused for Exp26; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--lambda-q",
        type=float,
        default=0.0,
        help="Unused for Exp26 voltage-only objective; must remain 0.",
    )
    parser.add_argument("--voltage-margin", type=float, default=0.0)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--base-kv", type=float, default=12.66, help="Unused for Exp26.")
    parser.add_argument("--base-mva", type=float, default=1.0, help="Unused for Exp26.")
    parser.add_argument(
        "--trust-region-fraction",
        type=float,
        default=1.0,
        help=(
            "Limit each optimized Q control to this fraction of its full feasible "
            "range around the operating-point Q reference. Use >=1 to disable."
        ),
    )
    parser.add_argument(
        "--trust-region-radius",
        default=None,
        help="JSON list or JSON file containing absolute trust-region radii for --strategy single.",
    )
    parser.add_argument(
        "--local-trust-region-fraction",
        type=float,
        default=DEFAULT_RUN_CONFIG["local_trust_region_fraction"],
        help="Trust-region fraction used by --strategy two_stage around the heuristic Q point.",
    )
    parser.add_argument(
        "--local-trust-radius",
        default=None,
        help=(
            "JSON list or JSON file containing absolute local trust-region radii "
            "for --strategy two_stage. If omitted, a small exp22-style radius is used."
        ),
    )
    parser.add_argument(
        "--heuristic-random-samples",
        type=int,
        default=DEFAULT_RUN_CONFIG["heuristic_random_samples"],
        help="Global random forward-evaluation candidates for --strategy two_stage.",
    )
    parser.add_argument(
        "--heuristic-batch-size",
        type=int,
        default=DEFAULT_RUN_CONFIG["heuristic_batch_size"],
        help="Batch size for numpy forward evaluation in --strategy two_stage.",
    )
    parser.add_argument(
        "--heuristic-safety-penalty",
        type=float,
        default=DEFAULT_RUN_CONFIG["heuristic_safety_penalty"],
        help="Penalty multiplier on surrogate safety violations during forward search.",
    )
    parser.add_argument(
        "--heuristic-seed",
        type=int,
        default=DEFAULT_RUN_CONFIG["heuristic_seed"],
        help="Random seed for --strategy two_stage forward search.",
    )
    parser.add_argument("--torch-warm-starts", type=int, default=DEFAULT_RUN_CONFIG["torch_warm_starts"])
    parser.add_argument("--torch-warm-steps", type=int, default=DEFAULT_RUN_CONFIG["torch_warm_steps"])
    parser.add_argument("--torch-warm-lr", type=float, default=DEFAULT_RUN_CONFIG["torch_warm_lr"])
    parser.add_argument(
        "--torch-q-penalty",
        type=float,
        default=DEFAULT_RUN_CONFIG["torch_q_penalty"],
        help="Optional warm-search |Q| penalty. Keep 0 for the Exp26 voltage-only objective.",
    )
    parser.add_argument(
        "--no-sc-integer",
        dest="no_sc_integer",
        action="store_true",
        default=not DEFAULT_RUN_CONFIG["sc_integer"],
        help="Treat the SC device as continuous instead of enforcing integer tap steps.",
    )
    parser.add_argument("--sc-step", type=float, default=DEFAULT_RUN_CONFIG["sc_step"])
    parser.add_argument(
        "--control-deviation-penalty",
        type=float,
        default=0.0,
        help="Optional linear penalty on absolute Q-control movement from the Q reference.",
    )
    parser.add_argument(
        "--warm-start-q",
        default=None,
        help=(
            "JSON list or JSON file containing a full Q-control warm start: "
            "PV, ESS, then Q devices. If trust-region-fraction < 1, this also "
            "becomes the trust-region center."
        ),
    )
    parser.add_argument(
        "--no-physical-output-bounds",
        action="store_true",
        help="Disable nonnegative/lower physical bounds on direct surrogate outputs.",
    )
    parser.add_argument(
        "--relu-formulation",
        choices=["big_m", "general"],
        default="big_m",
    )
    parser.add_argument("--big-m-scale", type=float, default=1.0)
    parser.add_argument("--fallback-big-m", type=float, default=1e3)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_RUN_CONFIG["time_limit"])
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_RUN_CONFIG["mip_gap"])
    parser.add_argument("--threads", type=int, default=DEFAULT_RUN_CONFIG["threads"])
    parser.add_argument("--mip-focus", type=int, default=DEFAULT_RUN_CONFIG["mip_focus"])
    parser.add_argument("--method", type=int, default=DEFAULT_RUN_CONFIG["method"])
    parser.add_argument("--node-method", type=int, default=DEFAULT_RUN_CONFIG["node_method"])
    parser.add_argument("--presolve", type=int, default=DEFAULT_RUN_CONFIG["presolve"])
    parser.add_argument("--presparsify", type=int, default=DEFAULT_RUN_CONFIG["presparsify"])
    parser.add_argument("--heuristics", type=float, default=DEFAULT_RUN_CONFIG["heuristics"])
    parser.add_argument("--cuts", type=int, default=DEFAULT_RUN_CONFIG["cuts"])
    parser.add_argument("--var-branch", type=int, default=DEFAULT_RUN_CONFIG["var_branch"])
    parser.add_argument("--numeric-focus", type=int, default=DEFAULT_RUN_CONFIG["numeric_focus"])
    parser.add_argument(
        "--dual-reductions",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 0 to distinguish infeasible from unbounded.",
    )
    parser.add_argument(
        "--diagnose-slack",
        action="store_true",
        help="Relax Vworst/WorstI safety constraints with nonnegative slacks.",
    )
    parser.add_argument(
        "--slack-penalty",
        type=float,
        default=1e4,
        help="Penalty on safety slacks when --diagnose-slack is used.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--export-lp", default=None)
    parser.add_argument("--iis", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    x_base = None
    pv_q_base = None
    ess_nodes = np.array([], dtype=int)
    ess_s_rated = np.array([], dtype=float)
    ess_p_base = np.array([], dtype=float)
    ess_q_base = np.array([], dtype=float)
    q_device_nodes = np.array([], dtype=int)
    q_device_min = np.array([], dtype=float)
    q_device_max = np.array([], dtype=float)
    q_device_q_base = np.array([], dtype=float)
    q_device_names = []

    if args.source == "dataset":
        # Read config only for safety limits and auto sample selection. The
        # optimization model itself will load the engine later.
        cfg = {"v_lower": 0.95, "v_upper": 1.05}
        x_base, p_pv, pv_q_base, topo_mask, pv_nodes, s_rated, meta = (
            get_dataset_operating_point(
                args.data,
                cfg,
                sample_index=args.sample_index,
            )
        )
        p_load = np.zeros(33, dtype=float)
        q_load = np.zeros(33, dtype=float)
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from dataset:")
        print(f"  data: {meta['data_path']}")
        print(
            f"  sample_index: {meta['sample_index']} "
            f"({meta['sample_reason']})"
        )
        if meta["hour"] is not None:
            print(f"  hour={meta['hour']}, mode={meta['mode']}")
        if meta["Pload_sum"] is not None:
            print(f"  Pload sum: {meta['Pload_sum']:.6f} MW")
            print(f"  Qload sum: {meta['Qload_sum']:.6f} MVar")
        print(f"  X net P sum: {meta['X_P_sum']:.6f} MW")
        print(f"  X net Q sum: {meta['X_Q_sum']:.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  S_pv: {s_rated.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
        print(f"  baseline pv_q: {pv_q_base.tolist()}")
        ess_nodes = meta["ess_nodes"]
        ess_s_rated = meta["S_ess_mva"]
        ess_p_base = meta["ess_p_base"]
        ess_q_base = meta["ess_q_base"]
        q_device_nodes = meta["q_device_nodes"]
        q_device_min = meta["q_device_min"]
        q_device_max = meta["q_device_max"]
        q_device_q_base = meta["q_device_q_base"]
        q_device_names = meta["q_device_names"]
        if len(ess_nodes):
            print(f"  ESS buses: {ess_nodes.tolist()}")
            print(f"  ESS P/Q base: {ess_p_base.tolist()} / {ess_q_base.tolist()}")
        if len(q_device_nodes):
            print(f"  Q-device buses: {q_device_nodes.tolist()}")
            print(f"  Q-device base: {q_device_q_base.tolist()}")
    else:
        p_load, q_load, p_pv, topo_mask, pv_nodes, s_rated = get_default_operating_point(
            pv_pu=args.pv_pu
        )

        p_load_override = parse_vector_arg(args.p_load, expected=33, name="p_load")
        q_load_override = parse_vector_arg(args.q_load, expected=33, name="q_load")
        p_pv_override = parse_vector_arg(args.p_pv, expected=len(pv_nodes), name="p_pv")
        if p_load_override is not None:
            p_load = p_load_override
        if q_load_override is not None:
            q_load = q_load_override
        if p_pv_override is not None:
            p_pv = p_pv_override

        print("Operating point from profile:")
        print(f"  P_load sum: {float(np.sum(p_load)):.6f} MW")
        print(f"  Q_load sum: {float(np.sum(q_load)):.6f} MVar")
        print(f"  PV buses: {pv_nodes.tolist()}")
        print(f"  P_pv: {p_pv.tolist()}")
    print(f"  closed topology branches: {int(np.sum(topo_mask))} / {len(topo_mask)}")

    warm_start_q = parse_vector_arg(
        args.warm_start_q,
        expected=len(pv_nodes) + len(ess_nodes) + len(q_device_nodes),
        name="warm_start_q",
    )
    expected_controls = len(pv_nodes) + len(ess_nodes) + len(q_device_nodes)
    trust_region_radius = parse_vector_arg(
        args.trust_region_radius,
        expected=expected_controls,
        name="trust_region_radius",
    )
    local_trust_radius = parse_vector_arg(
        args.local_trust_radius,
        expected=expected_controls,
        name="local_trust_radius",
    )
    rpo_kwargs = dict(
        pv_nodes=pv_nodes,
        s_rated=s_rated,
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
        trust_region_radius=trust_region_radius,
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
        output_flag=0 if args.quiet else 1,
        export_lp=args.export_lp,
        iis_path=args.iis,
        x_base=x_base,
        pv_q_base=pv_q_base,
        ess_nodes=ess_nodes,
        ess_s_rated=ess_s_rated,
        ess_p_base=ess_p_base,
        ess_q_base=ess_q_base,
        q_device_nodes=q_device_nodes,
        q_device_min=q_device_min,
        q_device_max=q_device_max,
        q_device_q_base=q_device_q_base,
        q_device_names=q_device_names,
        warm_start_q=warm_start_q,
        sc_integer=not args.no_sc_integer,
        sc_step=args.sc_step,
    )
    if args.strategy == "single":
        run_single_step_rpo(
            p_load,
            q_load,
            p_pv,
            topo_mask,
            **rpo_kwargs,
        )
    else:
        run_two_stage_rpo(
            p_load,
            q_load,
            p_pv,
            topo_mask,
            heuristic_random_samples=args.heuristic_random_samples,
            heuristic_batch_size=args.heuristic_batch_size,
            heuristic_safety_penalty=args.heuristic_safety_penalty,
            heuristic_seed=args.heuristic_seed,
            local_trust_region_fraction=args.local_trust_region_fraction,
            local_trust_region_radius=local_trust_radius,
            torch_warm_starts=args.torch_warm_starts,
            torch_warm_steps=args.torch_warm_steps,
            torch_warm_lr=args.torch_warm_lr,
            torch_q_penalty=args.torch_q_penalty,
            **rpo_kwargs,
        )


if __name__ == "__main__":
    main()



