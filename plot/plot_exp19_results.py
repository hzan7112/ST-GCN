import os
import random
import warnings
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from model.model import StandardGCN


def get_eval_config():
    return {
        "data_path": os.path.join("data", "ieee33_static_vvo_24h_dataset.pt"),
        "out_root": os.path.join("results", "exp24_accuracy_compare"),
        "seed": 42,
        "batch_size": 512,
        "train_ratio": 0.8,
        "v_lower": 0.95,
        "v_upper": 1.05,

        "experiments": [

            {
                "name": "Exp24-PathFeature-VLinRes",
                "engine_path": os.path.join("checkpoints", "st_gcn_h6_l1_pathfeat_vlinres_exp23_milp_engine.pt"),
            },
        ],
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def save_figure(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    tiff_path = os.path.splitext(save_path)[0] + ".tiff"
    try:
        fig.savefig(tiff_path, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def load_pt(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


def build_topology_matrices(edge_list, num_nodes=33, root=0):
    edge_list = sanitize_edge_list(edge_list)

    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)

    parent = [-2] * num_nodes
    children = [[] for _ in range(num_nodes)]
    parent[root] = -1

    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                children[u].append(v)
                q.append(v)

    S_down = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    def dfs(u):
        S_down[u, u] = 1.0
        for v in children[u]:
            dfs(v)
            S_down[u] += S_down[v]

    dfs(root)

    S_path = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        cur = i
        while cur != root and parent[cur] >= 0:
            S_path[i] += S_down[cur]
            cur = parent[cur]

    return S_down, S_path, parent


def augment_downstream_features(X_base, S_down):
    P = X_base[:, :, 0]
    Q = X_base[:, :, 1]
    P_down = P @ S_down.to(X_base.device).T
    Q_down = Q @ S_down.to(X_base.device).T
    return torch.cat([X_base, P_down.unsqueeze(-1), Q_down.unsqueeze(-1)], dim=-1)


def augment_path_features(X_base, S_down, S_path):
    P = X_base[:, :, 0]
    Q = X_base[:, :, 1]
    S_down = S_down.to(X_base.device)
    S_path = S_path.to(X_base.device)

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T
    P_path = P @ S_path.T
    Q_path = Q @ S_path.T

    return torch.cat(
        [
            X_base,
            P_down.unsqueeze(-1),
            Q_down.unsqueeze(-1),
            P_path.unsqueeze(-1),
            Q_path.unsqueeze(-1),
        ],
        dim=-1,
    )


def prepare_input_for_engine(X_all, engine, edge_list):
    expected = int(engine.get("in_features", engine.get("config", {}).get("in_features", 2)))
    raw_features = X_all.shape[-1]
    num_nodes = X_all.shape[1]

    if raw_features == expected:
        return X_all

    if raw_features != 2:
        raise ValueError(f"原始数据 X 维度应为2，当前为 {raw_features}")

    if "downstream_matrix" in engine:
        S_down = torch.as_tensor(engine["downstream_matrix"], dtype=torch.float32)
    else:
        S_down, _, _ = build_topology_matrices(edge_list, num_nodes)

    if expected == 4:
        return augment_downstream_features(X_all, S_down)

    if expected == 6:
        if "path_power_matrix" in engine:
            S_path = torch.as_tensor(engine["path_power_matrix"], dtype=torch.float32)
        else:
            _, S_path, _ = build_topology_matrices(edge_list, num_nodes)

        return augment_path_features(X_all, S_down, S_path)

    raise ValueError(f"暂不支持 engine in_features={expected}")


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    err = y_pred - y_true
    abs_err = np.abs(err)

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12

    return {
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(abs_err / (np.abs(y_true) + 1e-8)) * 100.0),
        "R2": float(1.0 - ss_res / ss_tot),
        "P95": float(np.percentile(abs_err, 95)),
        "P99": float(np.percentile(abs_err, 99)),
        "MaxError": float(np.max(abs_err)),
    }


def compute_margin_safety_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_violate = y_true > 0.0
    pred_violate = y_pred > 0.0

    false_safe = true_violate & (~pred_violate)
    false_violate = (~true_violate) & pred_violate

    true_violate_count = int(np.sum(true_violate))
    true_safe_count = int(np.sum(~true_violate))

    return {
        "TrueViolateCount": true_violate_count,
        "FalseSafeCount": int(np.sum(false_safe)),
        "FalseSafeRate": float(np.sum(false_safe) / true_violate_count) if true_violate_count > 0 else 0.0,
        "FalseViolateCount": int(np.sum(false_violate)),
        "FalseViolateRate": float(np.sum(false_violate) / true_safe_count) if true_safe_count > 0 else 0.0,
    }


def compute_voltage_safety_metrics(v_true, v_pred, v_lower=0.95, v_upper=1.05):
    v_true = np.asarray(v_true)
    v_pred = np.asarray(v_pred)

    true_low = v_true < v_lower
    true_high = v_true > v_upper
    true_violate = true_low | true_high

    pred_low = v_pred < v_lower
    pred_high = v_pred > v_upper
    pred_violate = pred_low | pred_high

    false_safe = true_violate & (~pred_violate)
    false_low_safe = true_low & (~pred_low)
    false_high_safe = true_high & (~pred_high)
    false_violate = (~true_violate) & pred_violate

    true_violate_count = int(np.sum(true_violate))
    true_safe_count = int(np.sum(~true_violate))
    true_low_count = int(np.sum(true_low))
    true_high_count = int(np.sum(true_high))

    return {
        "TrueViolateCount": true_violate_count,
        "FalseSafeCount": int(np.sum(false_safe)),
        "FalseSafeRate": float(np.sum(false_safe) / true_violate_count) if true_violate_count > 0 else 0.0,
        "FalseViolateCount": int(np.sum(false_violate)),
        "FalseViolateRate": float(np.sum(false_violate) / true_safe_count) if true_safe_count > 0 else 0.0,
        "TrueLowCount": true_low_count,
        "FalseLowSafeCount": int(np.sum(false_low_safe)),
        "FalseLowSafeRate": float(np.sum(false_low_safe) / true_low_count) if true_low_count > 0 else 0.0,
        "TrueHighCount": true_high_count,
        "FalseHighSafeCount": int(np.sum(false_high_safe)),
        "FalseHighSafeRate": float(np.sum(false_high_safe) / true_high_count) if true_high_count > 0 else 0.0,
    }


def get_validation_indices(num_samples, train_ratio=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    return perm[int(train_ratio * num_samples):].tolist()


def get_binary_count_text(engine):
    bc = engine.get("binary_count", {})
    if not bc:
        return "N/A"
    if "total_binary" in bc:
        return str(bc["total_binary"])
    if "total_binary_num" in bc:
        return str(bc["total_binary_num"])
    return str(bc)


def strip_base_prefix_if_needed(state_dict):
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("base.") for k in keys):
        return {k.replace("base.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def load_engine_and_model(engine_path, device):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"未找到 engine 文件: {engine_path}")

    engine = load_pt(engine_path, map_location=device)
    cfg = engine.get("config", {})
    edge_list = sanitize_edge_list(engine["edge_list"])

    in_features = int(engine.get("in_features", cfg.get("in_features", 2)))
    num_nodes = int(engine.get("num_nodes", cfg.get("num_nodes", 33)))

    kwargs = {
        "in_features": in_features,
        "hidden_dim": int(engine.get("hidden_dim", cfg.get("hidden_dim", 32))),
        "num_layers": int(engine.get("num_layers", cfg.get("num_layers", 2))),
        "edge_list": edge_list,
        "num_nodes": num_nodes,
        "node_relu_dim": int(engine.get("node_relu_dim", cfg.get("node_relu_dim", 8))),
        "edge_relu_dim": int(engine.get("edge_relu_dim", cfg.get("edge_relu_dim", 8))),
        "node_emb_dim": int(engine.get("node_emb_dim", cfg.get("node_emb_dim", 8))),
        "edge_emb_dim": int(engine.get("edge_emb_dim", cfg.get("edge_emb_dim", 8))),
        "use_residual": bool(engine.get("use_residual", cfg.get("use_residual", True))),
        "use_initial_anchor": bool(engine.get("use_initial_anchor", cfg.get("use_initial_anchor", True))),
        "use_jk": bool(engine.get("use_jk", cfg.get("use_jk", True))),
        "include_input_in_jk": bool(engine.get("include_input_in_jk", cfg.get("include_input_in_jk", True))),
        "use_linear_skip": bool(engine.get("use_linear_skip", cfg.get("use_linear_skip", True))),
    }

    try:
        model = StandardGCN(K=None, **kwargs).to(device)
    except TypeError:
        model = StandardGCN(**kwargs).to(device)

    state_dict = engine.get("state_dict", None)
    if state_dict is None and "wrapper_state_dict" in engine:
        state_dict = strip_base_prefix_if_needed(engine["wrapper_state_dict"])

    state_dict = strip_base_prefix_if_needed(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return engine, model


def load_norm_stats(engine, device):
    ns = engine["norm_stats"]

    X_mean = torch.as_tensor(ns["X_mean"], dtype=torch.float32, device=device)
    X_std = torch.as_tensor(ns["X_std"], dtype=torch.float32, device=device)

    if "YV_mean_wo_slack" in ns:
        YV_mean = torch.as_tensor(ns["YV_mean_wo_slack"], dtype=torch.float32, device=device)
        YV_std = torch.as_tensor(ns["YV_std_wo_slack"], dtype=torch.float32, device=device)
    else:
        YV_mean = torch.as_tensor(ns["YV_mean"], dtype=torch.float32, device=device)
        YV_std = torch.as_tensor(ns["YV_std"], dtype=torch.float32, device=device)

    YI_mean = torch.as_tensor(ns["YI_mean"], dtype=torch.float32, device=device)
    YI_std = torch.as_tensor(ns["YI_std"], dtype=torch.float32, device=device)

    return X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std


def add_voltage_linear_prior_if_needed(engine, X_norm, V_pred_n):
    if not bool(engine.get("uses_voltage_linear_prior", False)):
        return V_pred_n

    W = torch.as_tensor(engine["voltage_linear_prior_W"], dtype=torch.float32, device=X_norm.device)
    b = torch.as_tensor(engine["voltage_linear_prior_b"], dtype=torch.float32, device=X_norm.device)

    V_lin_n = X_norm.reshape(X_norm.shape[0], -1) @ W.T + b

    if V_pred_n.shape[1] == 33:
        V_total = V_pred_n.clone()
        V_total[:, 1:] = V_lin_n + V_pred_n[:, 1:]
        return V_total

    return V_lin_n + V_pred_n


def predict_on_validation_set(model, engine, X_raw, Y_V_raw, Y_I_raw, val_indices, norm_stats, device, batch_size=512):
    X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std = norm_stats

    X_val = X_raw[val_indices].to(device)
    YV_val = Y_V_raw[val_indices].to(device)
    YI_val = Y_I_raw[val_indices].to(device)

    if X_val.shape[-1] != X_mean.shape[-1]:
        raise ValueError(f"输入维度不匹配：X_val={X_val.shape}, X_mean={X_mean.shape}")

    X_norm = (X_val - X_mean) / X_std

    V_list, I_list = [], []

    with torch.no_grad():
        for s in range(0, X_norm.shape[0], batch_size):
            e = min(s + batch_size, X_norm.shape[0])
            out = model(X_norm[s:e])
            V_pred_n, I_pred_n = out[0], out[1]

            V_pred_n = add_voltage_linear_prior_if_needed(engine, X_norm[s:e], V_pred_n)

            if V_pred_n.shape[1] == 33:
                V_phys = V_pred_n[:, 1:] * YV_std + YV_mean
            else:
                V_phys = V_pred_n * YV_std + YV_mean

            I_phys = I_pred_n * YI_std + YI_mean

            V_list.append(V_phys.detach().cpu())
            I_list.append(I_phys.detach().cpu())

    V_pred = torch.cat(V_list, dim=0).numpy()
    I_pred = torch.cat(I_list, dim=0).numpy()

    V_true = YV_val[:, 1:].detach().cpu().numpy()
    I_true = YI_val.detach().cpu().numpy()

    return V_true, V_pred, I_true, I_pred


def plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower=0.95, v_upper=1.05):
    err = V_pred - V_true
    metrics = compute_metrics(V_true, V_pred)
    safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    node_id = 30
    node_col = max(0, min(node_id - 2, V_true.shape[1] - 1))
    show_len = min(300, V_true.shape[0])
    ax.plot(V_true[:show_len, node_col], label="True", linewidth=1.2)
    ax.plot(V_pred[:show_len, node_col], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(v_lower, linestyle="--", linewidth=1.0)
    ax.axhline(v_upper, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage sequence at Bus {node_id}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[0, 1]
    ax.scatter(V_true.reshape(-1), V_pred.reshape(-1), s=6, alpha=0.30, edgecolors="none")
    v_min = min(V_true.min(), V_pred.min())
    v_max = max(V_true.max(), V_pred.max())
    ax.plot([v_min, v_max], [v_min, v_max], linestyle="--", linewidth=1.0)
    ax.axvline(v_lower, linestyle=":", linewidth=0.8)
    ax.axhline(v_lower, linestyle=":", linewidth=0.8)
    ax.axvline(v_upper, linestyle=":", linewidth=0.8)
    ax.axhline(v_upper, linestyle=":", linewidth=0.8)
    ax.set_xlabel("True voltage (p.u.)")
    ax.set_ylabel("Predicted voltage (p.u.)")
    ax.set_title("Voltage mapping")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 0]
    ax.hist(err.reshape(-1), bins=70, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction error (p.u.)")
    ax.set_ylabel("Frequency")
    ax.set_title("Voltage error distribution")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 1]
    text = (
        "Voltage prediction metrics\n"
        f"MAE = {metrics['MAE']:.6f} p.u.\n"
        f"RMSE = {metrics['RMSE']:.6f} p.u.\n"
        f"P95 = {metrics['P95']:.6f} p.u.\n"
        f"P99 = {metrics['P99']:.6f} p.u.\n"
        f"MaxError = {metrics['MaxError']:.6f} p.u.\n"
        f"R2 = {metrics['R2']:.6f}\n\n"
        "Voltage safety metrics\n"
        f"False Safe = {safety['FalseSafeRate'] * 100:.2f}%\n"
        f"False Violate = {safety['FalseViolateRate'] * 100:.2f}%\n"
        f"Low-V False Safe = {safety['FalseLowSafeRate'] * 100:.2f}%\n"
        f"High-V False Safe = {safety['FalseHighSafeRate'] * 100:.2f}%\n"
        f"True violated points = {safety['TrueViolateCount']}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)
    ax.axis("off")

    save_figure(fig, os.path.join(out_dir, "fig1_voltage_accuracy.png"))


def plot_margin_accuracy(I_true, I_pred, out_dir):
    err = I_pred - I_true
    metrics = compute_metrics(I_true, I_pred)
    safety = compute_margin_safety_metrics(I_true, I_pred)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    show_len = min(500, I_true.reshape(-1).shape[0])
    ax.plot(I_true.reshape(-1)[:show_len], label="True", linewidth=1.2)
    ax.plot(I_pred.reshape(-1)[:show_len], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Evaluation point")
    ax.set_ylabel("Current margin")
    ax.set_title("Current-margin sequence")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[0, 1]
    ax.scatter(I_true.reshape(-1), I_pred.reshape(-1), s=6, alpha=0.30, edgecolors="none")
    i_min = min(I_true.min(), I_pred.min())
    i_max = max(I_true.max(), I_pred.max())
    ax.plot([i_min, i_max], [i_min, i_max], linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle=":", linewidth=0.8)
    ax.axhline(0.0, linestyle=":", linewidth=0.8)
    ax.set_xlabel("True margin")
    ax.set_ylabel("Predicted margin")
    ax.set_title("Current-margin mapping")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 0]
    ax.hist(err.reshape(-1), bins=70, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Frequency")
    ax.set_title("Current-margin error distribution")
    ax.grid(alpha=0.12, linewidth=0.4)

    ax = axes[1, 1]
    text = (
        "Current-margin prediction metrics\n"
        f"MAE = {metrics['MAE']:.6f}\n"
        f"RMSE = {metrics['RMSE']:.6f}\n"
        f"P95 = {metrics['P95']:.6f}\n"
        f"P99 = {metrics['P99']:.6f}\n"
        f"MaxError = {metrics['MaxError']:.6f}\n"
        f"R2 = {metrics['R2']:.6f}\n\n"
        "Safety classification metrics\n"
        f"False Safe = {safety['FalseSafeRate'] * 100:.2f}%\n"
        f"False Violate = {safety['FalseViolateRate'] * 100:.2f}%\n"
        f"True violated points = {safety['TrueViolateCount']}\n"
        f"False-safe points = {safety['FalseSafeCount']}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)
    ax.axis("off")

    save_figure(fig, os.path.join(out_dir, "fig2_margin_accuracy.png"))


def plot_nodewise_voltage_error(V_true, V_pred, out_dir):
    abs_err = np.abs(V_pred - V_true)
    mae = abs_err.mean(axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)
    p99 = np.percentile(abs_err, 99, axis=0)
    mx = abs_err.max(axis=0)
    bus_ids = np.arange(2, 34)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(bus_ids, mae, marker="o", markersize=3, linewidth=1.2, label="MAE")
    ax.plot(bus_ids, p95, marker="s", markersize=3, linewidth=1.2, label="P95")
    ax.plot(bus_ids, p99, marker="^", markersize=3, linewidth=1.2, label="P99")
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage absolute error (p.u.)")
    ax.set_title("Node-wise voltage prediction error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)
    save_figure(fig, os.path.join(out_dir, "fig3_nodewise_voltage_error.png"))

    with open(os.path.join(out_dir, "nodewise_voltage_error.csv"), "w", encoding="utf-8") as f:
        f.write("Bus,MAE,P95,P99,MaxError\n")
        for bus, a, b, c, d in zip(bus_ids, mae, p95, p99, mx):
            f.write(f"{bus},{a:.8f},{b:.8f},{c:.8f},{d:.8f}\n")


def plot_branchwise_margin_error(I_true, I_pred, edge_list, out_dir):
    abs_err = np.abs(I_pred - I_true)
    mae = abs_err.mean(axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)
    p99 = np.percentile(abs_err, 99, axis=0)
    mx = abs_err.max(axis=0)
    branch_ids = np.arange(1, len(edge_list) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(branch_ids, mae, marker="o", markersize=3, linewidth=1.2, label="MAE")
    ax.plot(branch_ids, p95, marker="s", markersize=3, linewidth=1.2, label="P95")
    ax.plot(branch_ids, p99, marker="^", markersize=3, linewidth=1.2, label="P99")
    ax.set_xlabel("Branch index")
    ax.set_ylabel("Current-margin absolute error")
    ax.set_title("Branch-wise current-margin prediction error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, linewidth=0.4)
    save_figure(fig, os.path.join(out_dir, "fig4_branchwise_margin_error.png"))

    with open(os.path.join(out_dir, "branchwise_margin_error.csv"), "w", encoding="utf-8") as f:
        f.write("Branch,FromBus,ToBus,MAE,P95,P99,MaxError\n")
        for idx, (a, b, c, d) in enumerate(zip(mae, p95, p99, mx)):
            fb, tb = edge_list[idx]
            f.write(f"{idx + 1},{fb + 1},{tb + 1},{a:.8f},{b:.8f},{c:.8f},{d:.8f}\n")


def save_summary_report(V_true, V_pred, I_true, I_pred, engine, out_dir, v_lower=0.95, v_upper=1.05):
    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)
    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    report_path = os.path.join(out_dir, "accuracy_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("Experiment 19 accuracy evaluation summary\n")
        f.write("============================================================\n\n")
        f.write(f"Model type: {engine.get('model_type', 'N/A')}\n")
        f.write(f"Hidden dim: {engine.get('hidden_dim', 'N/A')}\n")
        f.write(f"Num layers: {engine.get('num_layers', 'N/A')}\n")
        f.write(f"Input features: {engine.get('feature_names', engine.get('input_feature_names', 'N/A'))}\n")
        f.write(f"Uses voltage linear prior: {engine.get('uses_voltage_linear_prior', False)}\n")
        f.write(f"Total ReLU binary variables: {get_binary_count_text(engine)}\n\n")

        f.write("[Voltage prediction]\n")
        for k, v in v_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Voltage safety]\n")
        for k, v in v_safety.items():
            f.write(f"{k}: {v * 100:.4f}%\n" if "Rate" in k else f"{k}: {v}\n")

        f.write("\n[Current-margin prediction]\n")
        for k, v in i_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Current-margin safety]\n")
        for k, v in i_safety.items():
            f.write(f"{k}: {v * 100:.4f}%\n" if "Rate" in k else f"{k}: {v}\n")

    return report_path


def evaluate_one_experiment(exp, data, device, eval_cfg):
    base_dir = os.getcwd()
    engine_path = os.path.join(base_dir, exp["engine_path"])

    if not os.path.exists(engine_path):
        print(f"\n[跳过] 未找到 engine: {engine_path}")
        return None

    engine, model = load_engine_and_model(engine_path, device)
    edge_list = sanitize_edge_list(engine["edge_list"])

    X_base = data["X"].float()
    Y_V = data["Y_V"].float() if "Y_V" in data else data["Y_V_full"].float()
    Y_I = data["Y_I"].float() if "Y_I" in data else data["Y_I_full"].float()

    X_eval = prepare_input_for_engine(X_base, engine, edge_list)
    norm_stats = load_norm_stats(engine, device)

    cfg = engine.get("config", {})
    train_ratio = cfg.get("train_ratio", eval_cfg["train_ratio"])
    seed = cfg.get("seed", eval_cfg["seed"])
    v_lower = cfg.get("v_lower", eval_cfg["v_lower"])
    v_upper = cfg.get("v_upper", eval_cfg["v_upper"])

    val_indices = get_validation_indices(len(X_eval), train_ratio=train_ratio, seed=seed)

    print("\n============================================================")
    print(f"开始评估: {exp['name']}")
    print("============================================================")
    print(f"engine_path = {engine_path}")
    print(f"X_base = {tuple(X_base.shape)}, X_eval = {tuple(X_eval.shape)}")
    print(f"model_type = {engine.get('model_type', 'N/A')}")
    print(f"in_features = {engine.get('in_features', 'N/A')}")
    print(f"uses_voltage_linear_prior = {engine.get('uses_voltage_linear_prior', False)}")
    print(f"binary_count = {get_binary_count_text(engine)}")
    print(f"验证集样本数 = {len(val_indices)}")

    V_true, V_pred, I_true, I_pred = predict_on_validation_set(
        model=model,
        engine=engine,
        X_raw=X_eval,
        Y_V_raw=Y_V,
        Y_I_raw=Y_I,
        val_indices=val_indices,
        norm_stats=norm_stats,
        device=device,
        batch_size=eval_cfg["batch_size"],
    )

    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)
    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    out_dir = os.path.join(base_dir, eval_cfg["out_root"], exp["name"])
    ensure_dir(out_dir)

    plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower, v_upper)
    plot_margin_accuracy(I_true, I_pred, out_dir)
    plot_nodewise_voltage_error(V_true, V_pred, out_dir)
    plot_branchwise_margin_error(I_true, I_pred, edge_list, out_dir)

    report_path = save_summary_report(V_true, V_pred, I_true, I_pred, engine, out_dir, v_lower, v_upper)

    np.savez(
        os.path.join(out_dir, "accuracy_arrays.npz"),
        V_true=V_true,
        V_pred=V_pred,
        I_true=I_true,
        I_pred=I_pred,
        val_indices=np.array(val_indices),
    )

    print("[节点电压预测]")
    print(f"MAE={v_metrics['MAE']:.6f}, RMSE={v_metrics['RMSE']:.6f}, P95={v_metrics['P95']:.6f}, P99={v_metrics['P99']:.6f}, Max={v_metrics['MaxError']:.6f}")
    print(f"V-FS={v_safety['FalseSafeRate'] * 100:.2f}%, V-FV={v_safety['FalseViolateRate'] * 100:.2f}%")

    print("[支路裕度预测]")
    print(f"MAE={i_metrics['MAE']:.6f}, RMSE={i_metrics['RMSE']:.6f}, P95={i_metrics['P95']:.6f}, P99={i_metrics['P99']:.6f}, Max={i_metrics['MaxError']:.6f}")
    print(f"I-FS={i_safety['FalseSafeRate'] * 100:.2f}%, I-FV={i_safety['FalseViolateRate'] * 100:.2f}%")
    print(f"输出目录: {out_dir}")
    print(f"报告: {report_path}")

    return {
        "name": exp["name"],
        "engine_path": engine_path,
        "model_type": engine.get("model_type", "N/A"),
        "in_features": engine.get("in_features", "N/A"),
        "binary": int(get_binary_count_text(engine)) if get_binary_count_text(engine).isdigit() else np.nan,
        "uses_vlin": bool(engine.get("uses_voltage_linear_prior", False)),

        "V_MAE": v_metrics["MAE"],
        "V_RMSE": v_metrics["RMSE"],
        "V_P95": v_metrics["P95"],
        "V_P99": v_metrics["P99"],
        "V_MaxErr": v_metrics["MaxError"],
        "V_R2": v_metrics["R2"],
        "V_FS": v_safety["FalseSafeRate"] * 100,
        "V_FV": v_safety["FalseViolateRate"] * 100,

        "I_MAE": i_metrics["MAE"],
        "I_RMSE": i_metrics["RMSE"],
        "I_P95": i_metrics["P95"],
        "I_P99": i_metrics["P99"],
        "I_MaxErr": i_metrics["MaxError"],
        "I_R2": i_metrics["R2"],
        "I_FS": i_safety["FalseSafeRate"] * 100,
        "I_FV": i_safety["FalseViolateRate"] * 100,
    }


def save_comparison_csv(rows, out_root):
    csv_path = os.path.join(out_root, "exp19_comparison_metrics.csv")
    keys = [
        "name", "binary", "in_features", "uses_vlin",
        "V_MAE", "V_RMSE", "V_P95", "V_P99", "V_MaxErr", "V_FS", "V_FV",
        "I_MAE", "I_RMSE", "I_P95", "I_P99", "I_MaxErr", "I_FS", "I_FV",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    return csv_path


def plot_comparison(rows, out_root):
    if len(rows) == 0:
        return

    names = [r["name"] for r in rows]
    x = np.arange(len(names))

    metrics = [
        ("V_MAE", "Voltage MAE (p.u.)"),
        ("V_RMSE", "Voltage RMSE (p.u.)"),
        ("V_FS", "Voltage false-safe rate (%)"),
        ("I_MAE", "Current-margin MAE"),
        ("I_RMSE", "Current-margin RMSE"),
        ("I_FS", "Current-margin false-safe rate (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.reshape(-1)

    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        ax.bar(x, vals, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.12, linewidth=0.4)

        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)

    save_figure(fig, os.path.join(out_root, "fig0_exp19_comparison.png"))


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    eval_cfg = get_eval_config()
    set_seed(eval_cfg["seed"])
    set_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.getcwd()

    data_path = os.path.join(base_dir, eval_cfg["data_path"])
    out_root = os.path.join(base_dir, eval_cfg["out_root"])
    ensure_dir(out_root)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据集文件: {data_path}")

    data = load_pt(data_path, map_location="cpu")

    print(f"当前验证设备: {device}")
    print(f"数据集: {data_path}")
    print(f"输出目录: {out_root}")

    rows = []
    for exp in eval_cfg["experiments"]:
        row = evaluate_one_experiment(exp, data, device, eval_cfg)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("没有任何实验 engine 被成功评估，请检查 get_eval_config() 中的路径。")

    csv_path = save_comparison_csv(rows, out_root)
    plot_comparison(rows, out_root)

    print("\n============================================================")
    print("实验19绘图与评估完成")
    print("============================================================")
    print(f"对比指标表: {csv_path}")
    print(f"对比图: {os.path.join(out_root, 'fig0_exp19_comparison.png')}")
    print("各实验子目录中包含电压/支路裕度预测图、逐节点/逐支路误差图和文本报告。")


if __name__ == "__main__":
    main()