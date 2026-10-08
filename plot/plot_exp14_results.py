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
        "engine_path": os.path.join(
            "checkpoints",
            "st_gcn_h12_l2_downstream_exp20_milp_engine.pt"
        ),
        "out_dir": os.path.join("results", "st_gcn_h32_l2_downstream_accuracy"),
        "seed": 42,
        "batch_size": 512,
        "train_ratio": 0.8,
        "v_lower": 0.95,
        "v_upper": 1.05,
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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
        fig.savefig(
            tiff_path,
            dpi=300,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception:
        fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


def build_downstream_matrix(edge_list, num_nodes, root=0):
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

    S = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    def dfs(u):
        S[u, u] = 1.0
        for v in children[u]:
            dfs(v)
            S[u] += S[v]

    dfs(root)
    return S, parent


def augment_downstream_features(X_base, downstream_matrix):
    """
    X_base: [N, 33, 2] = [P_net, Q_net]
    return: [N, 33, 4] = [P_net, Q_net, P_down, Q_down]
    """
    if X_base.shape[-1] != 2:
        raise ValueError(f"构造下游功率特征要求原始 X 为 2 维，但当前为 {X_base.shape[-1]} 维。")

    P = X_base[:, :, 0]
    Q = X_base[:, :, 1]

    S = downstream_matrix.to(X_base.device)
    P_down = P @ S.T
    Q_down = Q @ S.T

    return torch.cat([X_base, P_down.unsqueeze(-1), Q_down.unsqueeze(-1)], dim=-1)


def prepare_input_for_engine(X_all, engine, edge_list):
    """
    根据 engine 需要的 in_features 自动准备输入：
    - engine in_features=2: 使用原始 [P,Q]
    - engine in_features=4: 自动扩展为 [P,Q,P_down,Q_down]
    """
    expected_in_features = int(engine.get("in_features", 2))
    raw_in_features = X_all.shape[-1]
    num_nodes = X_all.shape[1]

    if raw_in_features == expected_in_features:
        return X_all, None

    if raw_in_features == 2 and expected_in_features == 4:
        if "downstream_matrix" in engine:
            downstream_matrix = torch.tensor(engine["downstream_matrix"], dtype=torch.float32)
        else:
            downstream_matrix, _ = build_downstream_matrix(edge_list, num_nodes, root=0)

        X_aug = augment_downstream_features(X_all, downstream_matrix)
        return X_aug, downstream_matrix

    raise ValueError(
        f"数据集输入维度与模型不匹配：dataset X has {raw_in_features} features, "
        f"but engine expects {expected_in_features} features."
    )


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(abs_err / (np.abs(y_true) + 1e-8)) * 100.0

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "P95": np.percentile(abs_err, 95),
        "P99": np.percentile(abs_err, 99),
        "MaxError": np.max(abs_err),
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
        "FalseSafeRate": int(np.sum(false_safe)) / true_violate_count if true_violate_count > 0 else 0.0,
        "FalseViolateCount": int(np.sum(false_violate)),
        "FalseViolateRate": int(np.sum(false_violate)) / true_safe_count if true_safe_count > 0 else 0.0,
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

    false_safe_count = int(np.sum(false_safe))
    false_violate_count = int(np.sum(false_violate))
    false_low_safe_count = int(np.sum(false_low_safe))
    false_high_safe_count = int(np.sum(false_high_safe))

    return {
        "TrueViolateCount": true_violate_count,
        "FalseSafeCount": false_safe_count,
        "FalseSafeRate": false_safe_count / true_violate_count if true_violate_count > 0 else 0.0,
        "FalseViolateCount": false_violate_count,
        "FalseViolateRate": false_violate_count / true_safe_count if true_safe_count > 0 else 0.0,

        "TrueLowCount": true_low_count,
        "FalseLowSafeCount": false_low_safe_count,
        "FalseLowSafeRate": false_low_safe_count / true_low_count if true_low_count > 0 else 0.0,

        "TrueHighCount": true_high_count,
        "FalseHighSafeCount": false_high_safe_count,
        "FalseHighSafeRate": false_high_safe_count / true_high_count if true_high_count > 0 else 0.0,
    }


def get_validation_indices(num_samples, train_ratio=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_train = int(train_ratio * num_samples)
    return perm[n_train:].tolist()


def get_binary_count_text(engine):
    binary_count = engine.get("binary_count", {})
    if not binary_count:
        return "N/A"

    if "total_binary" in binary_count:
        return str(binary_count["total_binary"])

    if "total_binary_num" in binary_count:
        return str(binary_count["total_binary_num"])

    return str(binary_count)


def load_engine_and_model(engine_path, device):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"未找到 engine 文件: {engine_path}")

    engine = torch.load(engine_path, map_location=device, weights_only=False)
    cfg = engine.get("config", {})
    edge_list = sanitize_edge_list(engine["edge_list"])

    in_features = int(engine.get("in_features", cfg.get("in_features", 2)))
    num_nodes = int(engine.get("num_nodes", cfg.get("num_nodes", 33)))

    model = StandardGCN(
        in_features=in_features,
        hidden_dim=int(engine.get("hidden_dim", cfg.get("hidden_dim", 96))),
        num_layers=int(engine.get("num_layers", cfg.get("num_layers", 18))),
        K=None,
        edge_list=edge_list,
        num_nodes=num_nodes,
        node_relu_dim=int(engine.get("node_relu_dim", cfg.get("node_relu_dim", 32))),
        edge_relu_dim=int(engine.get("edge_relu_dim", cfg.get("edge_relu_dim", 32))),
        node_emb_dim=int(engine.get("node_emb_dim", cfg.get("node_emb_dim", 16))),
        edge_emb_dim=int(engine.get("edge_emb_dim", cfg.get("edge_emb_dim", 16))),
        use_residual=bool(engine.get("use_residual", cfg.get("use_residual", True))),
        use_initial_anchor=bool(engine.get("use_initial_anchor", cfg.get("use_initial_anchor", True))),
        use_jk=bool(engine.get("use_jk", cfg.get("use_jk", True))),
        include_input_in_jk=bool(engine.get("include_input_in_jk", cfg.get("include_input_in_jk", True))),
        use_linear_skip=bool(engine.get("use_linear_skip", cfg.get("use_linear_skip", True))),
    ).to(device)

    model.load_state_dict(engine["state_dict"], strict=True)
    model.eval()

    return engine, model


def load_norm_stats(engine, device):
    norm_stats = engine["norm_stats"]

    X_mean = torch.tensor(norm_stats["X_mean"], dtype=torch.float32, device=device)
    X_std = torch.tensor(norm_stats["X_std"], dtype=torch.float32, device=device)

    if "YV_mean_wo_slack" in norm_stats:
        YV_mean = torch.tensor(norm_stats["YV_mean_wo_slack"], dtype=torch.float32, device=device)
        YV_std = torch.tensor(norm_stats["YV_std_wo_slack"], dtype=torch.float32, device=device)
    elif "YV_mean" in norm_stats:
        YV_mean = torch.tensor(norm_stats["YV_mean"], dtype=torch.float32, device=device)
        YV_std = torch.tensor(norm_stats["YV_std"], dtype=torch.float32, device=device)
    else:
        raise KeyError("norm_stats 中未找到 YV_mean_wo_slack/YV_std_wo_slack 或 YV_mean/YV_std。")

    YI_mean = torch.tensor(norm_stats["YI_mean"], dtype=torch.float32, device=device)
    YI_std = torch.tensor(norm_stats["YI_std"], dtype=torch.float32, device=device)

    return X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std


def predict_on_validation_set(
    model,
    X_raw,
    Y_V_raw,
    Y_I_raw,
    val_indices,
    norm_stats,
    device,
    batch_size=512,
):
    X_mean, X_std, YV_mean, YV_std, YI_mean, YI_std = norm_stats

    X_val_raw = X_raw[val_indices].to(device)
    YV_val_raw = Y_V_raw[val_indices].to(device)
    YI_val_raw = Y_I_raw[val_indices].to(device)

    if X_val_raw.shape[-1] != X_mean.shape[-1]:
        raise ValueError(
            f"输入维度仍不匹配：X_val_raw={X_val_raw.shape}, X_mean={X_mean.shape}。"
        )

    X_val_norm = (X_val_raw - X_mean) / X_std

    pred_v_list = []
    pred_i_list = []

    with torch.no_grad():
        for start in range(0, X_val_norm.shape[0], batch_size):
            end = min(start + batch_size, X_val_norm.shape[0])

            V_pred_n, I_pred_n, _, _, _ = model(X_val_norm[start:end])

            V_pred_phys = V_pred_n[:, 1:] * YV_std + YV_mean
            I_pred_phys = I_pred_n * YI_std + YI_mean

            pred_v_list.append(V_pred_phys.detach().cpu())
            pred_i_list.append(I_pred_phys.detach().cpu())

    V_pred = torch.cat(pred_v_list, dim=0).numpy()
    I_pred = torch.cat(pred_i_list, dim=0).numpy()

    V_true = YV_val_raw[:, 1:].detach().cpu().numpy()
    I_true = YI_val_raw.detach().cpu().numpy()

    return V_true, V_pred, I_true, I_pred


def plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower=0.95, v_upper=1.05):
    err = V_pred - V_true
    metrics = compute_metrics(V_true, V_pred)
    safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    node_id = 18
    node_col = max(0, min(node_id - 2, V_true.shape[1] - 1))
    show_len = min(300, V_true.shape[0])

    ax.plot(V_true[:show_len, node_col], label="True", linewidth=1.2)
    ax.plot(V_pred[:show_len, node_col], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(v_lower, linestyle="--", linewidth=1.0)
    ax.axhline(v_upper, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Voltage sequence at bus {node_id}")
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

    csv_path = os.path.join(out_dir, "nodewise_voltage_error.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
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

    csv_path = os.path.join(out_dir, "branchwise_margin_error.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Branch,FromBus,ToBus,MAE,P95,P99,MaxError\n")
        for idx, (a, b, c, d) in enumerate(zip(mae, p95, p99, mx)):
            f_bus, t_bus = edge_list[idx]
            f.write(f"{idx + 1},{int(f_bus) + 1},{int(t_bus) + 1},{a:.8f},{b:.8f},{c:.8f},{d:.8f}\n")


def save_summary_report(V_true, V_pred, I_true, I_pred, engine, out_dir, v_lower=0.95, v_upper=1.05):
    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)

    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    report_path = os.path.join(out_dir, "st_gcn_accuracy_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("ST-GCN accuracy evaluation summary\n")
        f.write("============================================================\n\n")

        f.write(f"Model type: {engine.get('model_type', 'ST-GCN')}\n")
        f.write(f"Hidden dim: {engine.get('hidden_dim', 'N/A')}\n")
        f.write(f"Num layers: {engine.get('num_layers', 'N/A')}\n")
        f.write(f"Node head ReLU dim: {engine.get('node_relu_dim', 'N/A')}\n")
        f.write(f"Edge head ReLU dim: {engine.get('edge_relu_dim', 'N/A')}\n")
        f.write(f"Input features: {engine.get('input_feature_names', 'N/A')}\n")
        f.write(f"Total ReLU binary variables: {get_binary_count_text(engine)}\n\n")

        f.write("[Voltage prediction]\n")
        for k, v in v_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Voltage safety]\n")
        for k, v in v_safety.items():
            if "Rate" in k:
                f.write(f"{k}: {v * 100:.4f}%\n")
            else:
                f.write(f"{k}: {v}\n")

        f.write("\n[Current-margin prediction]\n")
        for k, v in i_metrics.items():
            f.write(f"{k}: {v:.8f}\n")

        f.write("\n[Current-margin safety]\n")
        for k, v in i_safety.items():
            if "Rate" in k:
                f.write(f"{k}: {v * 100:.4f}%\n")
            else:
                f.write(f"{k}: {v}\n")

    return report_path


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    eval_cfg = get_eval_config()

    set_seed(eval_cfg["seed"])
    set_plot_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.getcwd()

    data_path = os.path.join(base_dir, eval_cfg["data_path"])
    engine_path = os.path.join(base_dir, eval_cfg["engine_path"])
    out_dir = os.path.join(base_dir, eval_cfg["out_dir"])

    ensure_dir(out_dir)

    print(f"当前验证设备: {device}")
    print("加载数据集与 ST-GCN MILP engine...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据集文件: {data_path}")

    data = torch.load(data_path, map_location="cpu", weights_only=False)

    X_all_base = data["X"].float()
    Y_V_all = data["Y_V"].float()
    Y_I_all = data["Y_I"].float()

    engine, model = load_engine_and_model(engine_path, device)
    norm_stats = load_norm_stats(engine, device)

    cfg = engine.get("config", {})
    edge_list = sanitize_edge_list(engine["edge_list"])

    X_all, downstream_matrix = prepare_input_for_engine(X_all_base, engine, edge_list)

    train_ratio = cfg.get("train_ratio", eval_cfg["train_ratio"])
    seed = cfg.get("seed", eval_cfg["seed"])
    v_lower = cfg.get("v_lower", eval_cfg["v_lower"])
    v_upper = cfg.get("v_upper", eval_cfg["v_upper"])

    print(f"原始数据规模: X={tuple(X_all_base.shape)}, Y_V={tuple(Y_V_all.shape)}, Y_I={tuple(Y_I_all.shape)}")
    print(f"评估输入规模: X_eval={tuple(X_all.shape)}")
    print(
        "模型配置: "
        f"type={engine.get('model_type', 'ST-GCN')}, "
        f"layers={engine.get('num_layers', cfg.get('num_layers', 'N/A'))}, "
        f"hidden={engine.get('hidden_dim', cfg.get('hidden_dim', 'N/A'))}, "
        f"node_relu={engine.get('node_relu_dim', cfg.get('node_relu_dim', 'N/A'))}, "
        f"edge_relu={engine.get('edge_relu_dim', cfg.get('edge_relu_dim', 'N/A'))}, "
        f"node_emb={engine.get('node_emb_dim', cfg.get('node_emb_dim', 'N/A'))}, "
        f"edge_emb={engine.get('edge_emb_dim', cfg.get('edge_emb_dim', 'N/A'))}, "
        f"in_features={engine.get('in_features', cfg.get('in_features', 'N/A'))}"
    )

    print(f"输入特征: {engine.get('input_feature_names', '未保存，按 engine in_features 自动匹配')}")
    print(f"ReLU 二元变量数量: {get_binary_count_text(engine)}")

    val_indices = get_validation_indices(
        num_samples=X_all.shape[0],
        train_ratio=train_ratio,
        seed=seed,
    )

    print(f"验证集样本数: {len(val_indices)}")

    V_true, V_pred, I_true, I_pred = predict_on_validation_set(
        model=model,
        X_raw=X_all,
        Y_V_raw=Y_V_all,
        Y_I_raw=Y_I_all,
        val_indices=val_indices,
        norm_stats=norm_stats,
        device=device,
        batch_size=eval_cfg["batch_size"],
    )

    v_metrics = compute_metrics(V_true, V_pred)
    i_metrics = compute_metrics(I_true, I_pred)

    v_safety = compute_voltage_safety_metrics(V_true, V_pred, v_lower=v_lower, v_upper=v_upper)
    i_safety = compute_margin_safety_metrics(I_true, I_pred)

    print("\n============================================================")
    print("ST-GCN 验证集精度评估结果")
    print("============================================================")

    print("[节点电压预测，剔除平衡节点]")
    print(f"MAE      : {v_metrics['MAE']:.6f} p.u.")
    print(f"RMSE     : {v_metrics['RMSE']:.6f} p.u.")
    print(f"MAPE     : {v_metrics['MAPE']:.4f}%")
    print(f"R2       : {v_metrics['R2']:.6f}")
    print(f"P95      : {v_metrics['P95']:.6f} p.u.")
    print(f"P99      : {v_metrics['P99']:.6f} p.u.")
    print(f"MaxError : {v_metrics['MaxError']:.6f} p.u.")
    print(f"V-FalseSafe      : {v_safety['FalseSafeRate'] * 100:.2f}%")
    print(f"LowV-FalseSafe   : {v_safety['FalseLowSafeRate'] * 100:.2f}%")
    print(f"HighV-FalseSafe  : {v_safety['FalseHighSafeRate'] * 100:.2f}%")
    print(f"V-FalseViolate   : {v_safety['FalseViolateRate'] * 100:.2f}%")
    print(f"True voltage violated points : {v_safety['TrueViolateCount']}")
    print(f"Voltage false-safe points    : {v_safety['FalseSafeCount']}")

    print("\n[支路安全裕度预测]")
    print(f"MAE      : {i_metrics['MAE']:.6f}")
    print(f"RMSE     : {i_metrics['RMSE']:.6f}")
    print(f"MAPE     : {i_metrics['MAPE']:.4f}%")
    print(f"R2       : {i_metrics['R2']:.6f}")
    print(f"P95      : {i_metrics['P95']:.6f}")
    print(f"P99      : {i_metrics['P99']:.6f}")
    print(f"MaxError : {i_metrics['MaxError']:.6f}")
    print(f"I-FalseSafe      : {i_safety['FalseSafeRate'] * 100:.2f}%")
    print(f"I-FalseViolate   : {i_safety['FalseViolateRate'] * 100:.2f}%")
    print(f"True current-margin violated points : {i_safety['TrueViolateCount']}")
    print(f"Current-margin false-safe points    : {i_safety['FalseSafeCount']}")
    print("============================================================")

    plot_voltage_accuracy(V_true, V_pred, out_dir, v_lower=v_lower, v_upper=v_upper)
    plot_margin_accuracy(I_true, I_pred, out_dir)
    plot_nodewise_voltage_error(V_true, V_pred, out_dir)
    plot_branchwise_margin_error(I_true, I_pred, edge_list, out_dir)

    report_path = save_summary_report(
        V_true=V_true,
        V_pred=V_pred,
        I_true=I_true,
        I_pred=I_pred,
        engine=engine,
        out_dir=out_dir,
        v_lower=v_lower,
        v_upper=v_upper,
    )

    np.savez(
        os.path.join(out_dir, "st_gcn_accuracy_arrays.npz"),
        V_true=V_true,
        V_pred=V_pred,
        I_true=I_true,
        I_pred=I_pred,
        val_indices=np.array(val_indices),
    )

    print(f"\n图片与评估报告已保存至: {out_dir}")
    print(f"文本报告: {report_path}")


if __name__ == "__main__":
    main()