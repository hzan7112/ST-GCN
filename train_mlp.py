import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32]
]


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


class MLPNetwork(nn.Module):
    """Predict normalized voltage/current directly from flattened normalized node features."""

    def __init__(
        self,
        input_dim=66,
        hidden_dims=(256, 256, 256),
        output_dim=64,
        dropout=0.0,
        batchnorm=False,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim != 32 + 32:
            raise ValueError(f"output_dim must be 32+32=64, got {output_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.batchnorm = bool(batchnorm)

        dims = [self.input_dim] + self.hidden_dims
        self.hidden_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(self.hidden_dims))
        ])
        self.batchnorm_layers = (
            nn.ModuleList([nn.BatchNorm1d(dim) for dim in self.hidden_dims])
            if self.batchnorm
            else None
        )
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X):
        H = X.reshape(X.size(0), -1)
        if H.size(1) != self.input_dim:
            raise ValueError(
                f"expected flattened input dimension {self.input_dim}, got {H.size(1)}"
            )

        hidden_Z_list = []
        for layer_idx, layer in enumerate(self.hidden_layers):
            Z = layer(H)
            hidden_Z_list.append(Z)
            if self.batchnorm_layers is not None:
                Z = self.batchnorm_layers[layer_idx](Z)
            H = F.relu(Z)
            if self.dropout > 0.0:
                H = F.dropout(H, p=self.dropout, training=self.training)

        out = self.output_layer(H)
        V_pred = out[:, :32]
        I_pred = out[:, 32:]
        return V_pred, I_pred, hidden_Z_list

    def get_binary_count(self):
        per_layer = list(self.hidden_dims)
        return {
            "mlp_hidden_binary_per_layer": per_layer,
            "mlp_binary": sum(per_layer),
            "total_binary": sum(per_layer),
        }


def get_config():
    return {
        "exp_name": "MLP_H448_L3_RawPQ_Direct",
        "data_path": r"data/ieee33_static_vvo_balanced_20k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "mlp_h448_l3_rawpq_direct_best.pt",
        "engine_name": "mlp_h448_l3_rawpq_direct_milp_engine.pt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 800,
        "patience": 180,

        "input_dim": 33 * 2,
        "hidden_dims": [32, 32, 32],
        "output_dim": 32 + 32,
        "activation": "ReLU",
        "dropout": 0.0,
        "batchnorm": False,

        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,

        "voltage_loss_weight": 5.0,
        "edge_loss_weight": 1.5,
        "voltage_violate_mse_weight": 6.0,
        "edge_violate_mse_weight": 4.0,

        "warmup_epochs": 80,
        "penalty_ramp_epochs": 120,

        # 电流符号分类增强：
        # YI > 0 表示支路电流越限；YI <= 0 表示支路电流安全。
        # 对称 sign loss 同时约束假安全（FS）和假越限（FV）。
        "i_false_safe_lambda": 0.0,
        "i_false_safe_margin": 0.008,
        "i_sign_loss_weight": 64.0,
        "i_sign_margin": 0.008,

        # 电压三区域安全分类增强：
        # 低压越限：V < v_lower；安全：v_lower <= V <= v_upper；高压越限：V > v_upper。
        "v_false_safe_lambda": 0.0,
        "v_false_safe_guard": 0.003,
        "v_safety_loss_weight": 1.0,
        "v_safety_margin": 0.002,
        "v_safety_scale": 0.005,
        "v_safety_fs_weight": 1.0,
        "v_safety_fv_weight": 1.0,

        "v_lower": 0.95,
        "v_upper": 1.05,
        "big_m_beta": 1.10,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_tensor(x):
    return x if torch.is_tensor(x) else torch.tensor(x)


def get_edge_list(data):
    return [[int(u), int(v)] for u, v in data["edge_list"]] if "edge_list" in data else RADIAL_BRANCHES


def get_labels(data):
    if "Y_V" in data:
        YV = data["Y_V"]
    elif "Y_V_full" in data:
        YV = data["Y_V_full"]
    else:
        raise KeyError("数据集中找不到 Y_V 或 Y_V_full。")

    if "Y_I" in data:
        YI = data["Y_I"]
    elif "Y_I_full" in data:
        YI = data["Y_I_full"]
    else:
        raise KeyError("数据集中找不到 Y_I 或 Y_I_full。")

    return to_tensor(YV).float(), to_tensor(YI).float()


def normalize_data(X, YV, YI, train_idx):
    X_mean = X[train_idx].mean(dim=(0, 1), keepdim=True)
    X_std = X[train_idx].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    YV_mean = YV[train_idx, 1:].mean(dim=0, keepdim=True)
    YV_std = YV[train_idx, 1:].std(dim=0, keepdim=True).clamp_min(1e-6)

    YI_mean = YI[train_idx].mean(dim=0, keepdim=True)
    YI_std = YI[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)

    Xn = (X - X_mean) / X_std

    YVn = torch.zeros_like(YV)
    YVn[:, 1:] = (YV[:, 1:] - YV_mean) / YV_std

    YIn = (YI - YI_mean) / YI_std

    norm = {
        "X_mean": X_mean.squeeze(0).squeeze(0),
        "X_std": X_std.squeeze(0).squeeze(0),
        "YV_mean_wo_slack": YV_mean.squeeze(0),
        "YV_std_wo_slack": YV_std.squeeze(0),
        "YI_mean": YI_mean.squeeze(0),
        "YI_std": YI_std.squeeze(0),
    }

    return Xn, YVn, YIn, norm


def build_base_model(cfg, edge_list=None):
    return MLPNetwork(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["output_dim"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
    )


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple):
        return out
    raise RuntimeError("model.forward 必须返回 V_pred, I_pred, hidden_Z_list。")


def denorm_outputs(Vn, In, norm):
    device = Vn.device
    B = Vn.shape[0]

    YV_mean = norm["YV_mean_wo_slack"].to(device).view(1, -1)
    YV_std = norm["YV_std_wo_slack"].to(device).view(1, -1)
    YI_mean = norm["YI_mean"].to(device).view(1, -1)
    YI_std = norm["YI_std"].to(device).view(1, -1)

    V = torch.ones((B, 33), dtype=Vn.dtype, device=device)

    if Vn.shape[1] == 33:
        V[:, 1:] = Vn[:, 1:] * YV_std + YV_mean
    else:
        V[:, 1:] = Vn * YV_std + YV_mean

    I = In * YI_std + YI_mean

    return V, I


def masked_mean(x, mask):
    mask = mask.float()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def voltage_safety_classification_loss(v_true, v_pred, cfg):
    """
    电压三区域安全分类损失。

    真实低压/高压越限点分别推到安全边界外侧 margin 处，减少假安全；
    真实安全点约束在 [v_lower, v_upper] 内，减少假越限。
    p.u. 边界距离先除以 v_safety_scale 再平方，避免损失量级过小。
    """
    v_lower = cfg["v_lower"]
    v_upper = cfg["v_upper"]
    margin = cfg["v_safety_margin"]
    scale = max(float(cfg["v_safety_scale"]), 1e-8)

    low_unsafe = v_true < v_lower
    high_unsafe = v_true > v_upper
    safe = (v_true >= v_lower) & (v_true <= v_upper)

    low_fs_loss = masked_mean(
        (torch.relu(v_pred - (v_lower - margin)) / scale) ** 2,
        low_unsafe,
    )
    high_fs_loss = masked_mean(
        (torch.relu((v_upper + margin) - v_pred) / scale) ** 2,
        high_unsafe,
    )
    fv_low_loss = masked_mean(
        (torch.relu(v_lower - v_pred) / scale) ** 2,
        safe,
    )
    fv_high_loss = masked_mean(
        (torch.relu(v_pred - v_upper) / scale) ** 2,
        safe,
    )

    v_fs_loss = low_fs_loss + high_fs_loss
    v_fv_loss = fv_low_loss + fv_high_loss
    v_safety_loss = (
        cfg["v_safety_fs_weight"] * v_fs_loss
        + cfg["v_safety_fv_weight"] * v_fv_loss
    )

    return v_safety_loss, v_fs_loss, v_fv_loss, low_fs_loss, high_fs_loss


def compute_loss(model, batch, norm, cfg, ramp=1.0):
    X, YVn, YIn, YV, YI = batch

    Vn, In, *_ = unpack_forward(model, X)
    V, I = denorm_outputs(Vn, In, norm)

    Vn_use = Vn[:, 1:] if Vn.shape[1] == 33 else Vn
    YVn_use = YVn[:, 1:]

    v_true = YV[:, 1:]
    v_pred = V[:, 1:]

    v_unsafe = (v_true < cfg["v_lower"]) | (v_true > cfg["v_upper"])
    i_unsafe = YI > 0.0

    v_weight = 1.0 + cfg["voltage_violate_mse_weight"] * v_unsafe.float()
    i_weight = 1.0 + cfg["edge_violate_mse_weight"] * i_unsafe.float()

    node_mse = torch.mean(v_weight * (Vn_use - YVn_use) ** 2)
    edge_mse = torch.mean(i_weight * (In - YIn) ** 2)

    base = cfg["voltage_loss_weight"] * node_mse + cfg["edge_loss_weight"] * edge_mse

    # 旧的单侧电流假安全惩罚默认关闭，仅保留用于兼容和对照。
    i_pen = masked_mean(torch.relu(cfg["i_false_safe_margin"] - I) ** 2, i_unsafe)

    # 逐支路电流符号分类增强：越限预测推到 +margin，安全预测推到 -margin。
    i_safe = ~i_unsafe
    i_margin = cfg["i_sign_margin"]
    i_sign_fs_loss = masked_mean(
        torch.relu(i_margin - I) ** 2,
        i_unsafe,
    )
    i_sign_fv_loss = masked_mean(
        torch.relu(I + i_margin) ** 2,
        i_safe,
    )
    i_sign_loss = i_sign_fs_loss + i_sign_fv_loss

    low_mask = v_true < cfg["v_lower"]
    high_mask = v_true > cfg["v_upper"]

    low_pen = masked_mean(torch.relu(v_pred - (cfg["v_lower"] - cfg["v_false_safe_guard"])) ** 2, low_mask)
    high_pen = masked_mean(torch.relu((cfg["v_upper"] + cfg["v_false_safe_guard"]) - v_pred) ** 2, high_mask)

    v_pen = low_pen + high_pen

    v_safety_loss, v_safety_fs_loss, v_safety_fv_loss, v_low_fs_loss, v_high_fs_loss = (
        voltage_safety_classification_loss(v_true, v_pred, cfg)
    )

    total = base + ramp * (
        cfg["i_false_safe_lambda"] * i_pen
        + cfg["i_sign_loss_weight"] * i_sign_loss
        + cfg["v_false_safe_lambda"] * v_pen
        + cfg["v_safety_loss_weight"] * v_safety_loss
    )

    return total, {
        "base": base.detach(),
        "node_mse": node_mse.detach(),
        "edge_mse": edge_mse.detach(),
        "i_pen": i_pen.detach(),
        "i_sign_loss": i_sign_loss.detach(),
        "i_sign_fs_loss": i_sign_fs_loss.detach(),
        "i_sign_fv_loss": i_sign_fv_loss.detach(),
        "v_pen": v_pen.detach(),
        "v_safety_loss": v_safety_loss.detach(),
        "v_safety_fs_loss": v_safety_fs_loss.detach(),
        "v_safety_fv_loss": v_safety_fv_loss.detach(),
        "v_low_fs_loss": v_low_fs_loss.detach(),
        "v_high_fs_loss": v_high_fs_loss.detach(),
    }


@torch.no_grad()
def evaluate(model, loader, norm, cfg, device):
    model.eval()

    V_pred_all, I_pred_all, V_true_all, I_true_all = [], [], [], []
    base_sum = total_sum = node_sum = edge_sum = 0.0
    i_sign_sum = v_safety_sum = 0.0
    n_batch = 0

    for batch in loader:
        batch = [x.to(device) for x in batch]
        loss, info = compute_loss(model, batch, norm, cfg, ramp=1.0)

        X, _, _, YV, YI = batch
        Vn, In, *_ = unpack_forward(model, X)
        V, I = denorm_outputs(Vn, In, norm)

        V_pred_all.append(V.cpu())
        I_pred_all.append(I.cpu())
        V_true_all.append(YV.cpu())
        I_true_all.append(YI.cpu())

        total_sum += float(loss.item())
        base_sum += float(info["base"].item())
        node_sum += float(info["node_mse"].item())
        edge_sum += float(info["edge_mse"].item())
        i_sign_sum += float(info["i_sign_loss"].item())
        v_safety_sum += float(info["v_safety_loss"].item())
        n_batch += 1

    Vp = torch.cat(V_pred_all)
    Ip = torch.cat(I_pred_all)
    Vt = torch.cat(V_true_all)
    It = torch.cat(I_true_all)

    v_err = torch.abs(Vp[:, 1:] - Vt[:, 1:])
    i_err = torch.abs(Ip - It)

    v_true_unsafe = (Vt[:, 1:] < cfg["v_lower"]) | (Vt[:, 1:] > cfg["v_upper"])
    v_pred_unsafe = (Vp[:, 1:] < cfg["v_lower"]) | (Vp[:, 1:] > cfg["v_upper"])

    i_true_unsafe = It > 0.0
    i_pred_unsafe = Ip > 0.0

    v_fs = (v_true_unsafe & (~v_pred_unsafe)).sum().item() / max(v_true_unsafe.sum().item(), 1) * 100
    v_fv = ((~v_true_unsafe) & v_pred_unsafe).sum().item() / max((~v_true_unsafe).sum().item(), 1) * 100

    i_fs = (i_true_unsafe & (~i_pred_unsafe)).sum().item() / max(i_true_unsafe.sum().item(), 1) * 100
    i_fv = ((~i_true_unsafe) & i_pred_unsafe).sum().item() / max((~i_true_unsafe).sum().item(), 1) * 100

    i_tp = (i_true_unsafe & i_pred_unsafe).sum().item()
    i_fn = (i_true_unsafe & (~i_pred_unsafe)).sum().item()
    i_fp = ((~i_true_unsafe) & i_pred_unsafe).sum().item()
    i_tn = ((~i_true_unsafe) & (~i_pred_unsafe)).sum().item()

    i_total = max(i_tp + i_tn + i_fp + i_fn, 1)
    i_unsafe_total = max(i_tp + i_fn, 1)
    i_safe_total = max(i_tn + i_fp, 1)
    i_sign_acc = (i_tp + i_tn) / i_total * 100.0
    i_unsafe_recall = i_tp / i_unsafe_total * 100.0
    i_safe_recall = i_tn / i_safe_total * 100.0
    i_balanced_acc = 0.5 * (i_unsafe_recall + i_safe_recall)

    v_tp = (v_true_unsafe & v_pred_unsafe).sum().item()
    v_fn = (v_true_unsafe & (~v_pred_unsafe)).sum().item()
    v_fp = ((~v_true_unsafe) & v_pred_unsafe).sum().item()
    v_tn = ((~v_true_unsafe) & (~v_pred_unsafe)).sum().item()

    v_total = max(v_tp + v_tn + v_fp + v_fn, 1)
    v_unsafe_total = max(v_tp + v_fn, 1)
    v_safe_total = max(v_tn + v_fp, 1)
    v_safety_acc = (v_tp + v_tn) / v_total * 100.0
    v_unsafe_recall = v_tp / v_unsafe_total * 100.0
    v_safe_recall = v_tn / v_safe_total * 100.0
    v_balanced_acc = 0.5 * (v_unsafe_recall + v_safe_recall)

    return {
        "ValBaseLoss": base_sum / n_batch,
        "ValTotalLoss": total_sum / n_batch,
        "ValNodeMSE": node_sum / n_batch,
        "ValEdgeMSE": edge_sum / n_batch,
        "ValISignLoss": i_sign_sum / n_batch,
        "ValVSafetyLoss": v_safety_sum / n_batch,

        "V_MAE": float(v_err.mean()),
        "V_RMSE": float(torch.sqrt(torch.mean((Vp[:, 1:] - Vt[:, 1:]) ** 2))),
        "V_MaxErr": float(v_err.max()),

        "I_MAE": float(i_err.mean()),
        "I_RMSE": float(torch.sqrt(torch.mean((Ip - It) ** 2))),
        "I_MaxErr": float(i_err.max()),

        "V_FalseSafe": v_fs,
        "V_FalseViolate": v_fv,
        "V_SafetyAcc": v_safety_acc,
        "V_BalancedAcc": v_balanced_acc,
        "V_UnsafeRecall": v_unsafe_recall,
        "V_SafeRecall": v_safe_recall,
        "V_TrueUnsafeCount": int(v_true_unsafe.sum().item()),
        "V_PredUnsafeCount": int(v_pred_unsafe.sum().item()),
        "V_TP": int(v_tp),
        "V_TN": int(v_tn),
        "V_FP": int(v_fp),
        "V_FN": int(v_fn),

        "I_FalseSafe": i_fs,
        "I_FalseViolate": i_fv,
        "I_SignAcc": i_sign_acc,
        "I_BalancedAcc": i_balanced_acc,
        "I_UnsafeRecall": i_unsafe_recall,
        "I_SafeRecall": i_safe_recall,
        "I_TrueUnsafeCount": int(i_true_unsafe.sum().item()),
        "I_PredUnsafeCount": int(i_pred_unsafe.sum().item()),
        "I_TP": int(i_tp),
        "I_TN": int(i_tn),
        "I_FP": int(i_fp),
        "I_FN": int(i_fn),
    }


def ramp_lambda(epoch, cfg):
    if epoch <= cfg["warmup_epochs"]:
        return 0.0

    x = (epoch - cfg["warmup_epochs"]) / max(cfg["penalty_ramp_epochs"], 1)

    return float(min(max(x, 0.0), 1.0))


def selection_metric(m):
    return (
        m["V_MAE"]
        + 0.35 * m["I_MAE"]
        + 0.0030 * m["V_FalseSafe"]
        + 0.0015 * m["V_FalseViolate"]
        + 0.0020 * m["I_FalseSafe"]
        + 0.0015 * m["I_FalseViolate"]
        + 0.0010 * (100.0 - m["V_BalancedAcc"])
        + 0.0008 * (100.0 - m["I_BalancedAcc"])
    )


@torch.no_grad()
def extract_big_m(model, Xn, cfg, device):
    model.eval()
    batch_size = cfg["batch_size"]

    hidden_z_all = [[] for _ in cfg["hidden_dims"]]

    for i in range(0, len(Xn), batch_size):
        xb = Xn[i:i + batch_size].to(device)
        out = unpack_forward(model, xb)

        if len(out) < 3:
            raise RuntimeError("模型 forward 需要返回 hidden_Z_list 用于 Big-M 提取。")

        hidden_Z_list = out[2]
        if len(hidden_Z_list) != len(cfg["hidden_dims"]):
            raise RuntimeError(
                "hidden_Z_list 层数与 config.hidden_dims 不一致。"
            )

        for layer_idx, Z in enumerate(hidden_Z_list):
            hidden_z_all[layer_idx].append(Z.detach().cpu())

    beta = cfg["big_m_beta"]
    M_plus_mlp, M_minus_mlp = [], []
    for layer_values in hidden_z_all:
        Z = torch.cat(layer_values, dim=0)
        M_plus_mlp.append(torch.clamp(Z.max(dim=0).values, min=0.0) * beta)
        M_minus_mlp.append(torch.clamp((-Z).max(dim=0).values, min=0.0) * beta)

    return {
        "M_plus_mlp_layers": M_plus_mlp,
        "M_minus_mlp_layers": M_minus_mlp,
    }


def binary_count(model, cfg):
    base = model

    if hasattr(base, "get_binary_count"):
        return base.get_binary_count()

    mlp_binary = sum(cfg["hidden_dims"])
    return {
        "mlp_hidden_binary_per_layer": list(cfg["hidden_dims"]),
        "mlp_binary": mlp_binary,
        "total_binary": mlp_binary,
    }


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)

    X_raw = to_tensor(data["X"]).float()
    YV, YI = get_labels(data)

    X_raw = X_raw[:, :, :2].contiguous()

    n = len(X_raw)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    Xn, YVn, YIn, norm = normalize_data(X_raw, YV, YI, train_idx)

    train_ds = TensorDataset(Xn[train_idx], YVn[train_idx], YIn[train_idx], YV[train_idx], YI[train_idx])
    val_ds = TensorDataset(Xn[val_idx], YVn[val_idx], YIn[val_idx], YV[val_idx], YI[val_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)

    model = build_base_model(cfg, edge_list).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25
    )

    best_metric = float("inf")
    best_state = None
    best_metrics = None
    wait = 0
    t0 = time.time()

    print("\n================ 实验配置 ================")
    print(f"实验: {cfg['exp_name']}")
    print("输入特征: [P_net,Q_net]")
    print("模型输入: flatten(X2), 33*2=66")
    print("电压输出: V_pred(非 slack 32 节点) = MLP_voltage")
    print("支路输出: I_pred = MLP_current")
    print("训练目标增强: 逐支路对称电流符号分类 + 电压三区域安全分类")
    print(
        f"线性电压先验 Val: "
        "disabled"
    )
    print(
        f"线性电流先验 Val: "
        "disabled"
    )
    print(
        f"input_dim={cfg['input_dim']}, hidden_dims={cfg['hidden_dims']}, "
        f"output_dim={cfg['output_dim']}, activation={cfg['activation']}, "
        f"dropout={cfg['dropout']}, batchnorm={cfg['batchnorm']}"
    )
    print(f"二元变量估计: {binary_count(model, cfg)}")
    print("==========================================\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_train = 0.0
        base_train = 0.0
        lam = ramp_lambda(epoch, cfg)

        for batch in train_loader:
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad()
            loss, info = compute_loss(model, batch, norm, cfg, ramp=lam)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            total_train += float(loss.item())
            base_train += float(info["base"].item())

        metrics = evaluate(model, val_loader, norm, cfg, device)
        sel = selection_metric(metrics)
        scheduler.step(sel)

        train_base = base_train / len(train_loader)
        train_total = total_train / len(train_loader)

        if sel < best_metric:
            best_metric = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()

            torch.save(
                {
                    "model_state_dict": best_state,
                    "base_state_dict": model.state_dict(),
                    "config": cfg,
                    "norm_stats": {k: v.cpu() for k, v in norm.items()},
                    "edge_list": edge_list,
                },
                os.path.join(cfg["save_dir"], cfg["best_model_name"])
            )

            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:03d}/{cfg['epochs']}] | LR={lr:.2e} | "
                f"TrainBase={train_base:.4f} | TrainTotal={train_total:.4f} | "
                f"ValBase={metrics['ValBaseLoss']:.4f} | ValTotal={metrics['ValTotalLoss']:.4f} | "
                f"NodeMSE={metrics['ValNodeMSE']:.4f} | EdgeMSE={metrics['ValEdgeMSE']:.4f} | "
                f"ISign={metrics['ValISignLoss']:.4f} | VSafety={metrics['ValVSafetyLoss']:.4f} | "
                f"V_MAE={metrics['V_MAE']:.6f} | I_MAE={metrics['I_MAE']:.6f} | "
                f"V-FS={metrics['V_FalseSafe']:.2f}% | V-FV={metrics['V_FalseViolate']:.2f}% | "
                f"V-BAcc={metrics['V_BalancedAcc']:.2f}% | "
                f"I-FS={metrics['I_FalseSafe']:.2f}% | I-FV={metrics['I_FalseViolate']:.2f}% | "
                f"I-BAcc={metrics['I_BalancedAcc']:.2f}% | "
                f"λI-sign={cfg['i_sign_loss_weight'] * lam:.3f} | "
                f"λV-safe={cfg['v_safety_loss_weight'] * lam:.3f}"
            )

        if wait >= cfg["patience"]:
            print(f"\n早停触发: epoch={epoch}, best_metric={best_metric:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, norm, cfg, device)
    M = extract_big_m(model, Xn, cfg, device)

    engine = {
        "model_type": cfg["exp_name"],

        "state_dict": model.state_dict(),
        "wrapper_state_dict": model.state_dict(),

        "uses_voltage_linear_prior": False,
        "uses_current_linear_prior": False,

        "config": cfg,
        "input_shape": [33, 2],
        "input_dim": cfg["input_dim"],
        "feature_names": ["P_net", "Q_net"],

        "hidden_dims": list(cfg["hidden_dims"]),
        "output_dim": cfg["output_dim"],
        "voltage_output_dim": 32,
        "current_output_dim": 32,
        "activation": cfg["activation"],
        "dropout": cfg["dropout"],
        "batchnorm": cfg["batchnorm"],

        "edge_list": edge_list,
        "norm_stats": {k: v.cpu() for k, v in norm.items()},
        "binary_count": binary_count(model, cfg),

        "safety_settings": {
            "v_lower": cfg["v_lower"],
            "v_upper": cfg["v_upper"],
            "i_limit_margin": 0.0,
            "i_false_safe_margin": cfg["i_false_safe_margin"],
            "i_sign_margin": cfg["i_sign_margin"],
            "i_sign_loss_weight": cfg["i_sign_loss_weight"],
            "v_false_safe_guard": cfg["v_false_safe_guard"],
            "v_safety_margin": cfg["v_safety_margin"],
            "v_safety_scale": cfg["v_safety_scale"],
            "v_safety_loss_weight": cfg["v_safety_loss_weight"],
            "v_safety_fs_weight": cfg["v_safety_fs_weight"],
            "v_safety_fv_weight": cfg["v_safety_fv_weight"],
        },

        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "elapsed_sec": float(time.time() - t0),

        **M,
    }

    torch.save(engine, os.path.join(cfg["save_dir"], cfg["engine_name"]))

    print("\n================ 最终验证结果 ================")
    print(f"SelectionMetric: {best_metric:.6f}")

    print("\n[MLP direct model]")
    for k, v in final_metrics.items():
        if "False" in k or "Acc" in k or "Recall" in k:
            print(f"{k}: {v:.2f}%")
        elif "Count" in k or k.endswith(("_TP", "_TN", "_FP", "_FN")):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")

    print(f"\n二元变量估计: {binary_count(model, cfg)}")
    print(f"best model: {os.path.join(cfg['save_dir'], cfg['best_model_name'])}")
    print(f"MILP engine: {os.path.join(cfg['save_dir'], cfg['engine_name'])}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
