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


class FirstOrderNonlinearGCNLayer(nn.Module):
    def __init__(self, hidden_dim, use_initial_anchor=True, use_residual=True):
        super().__init__()
        self.use_initial_anchor = use_initial_anchor
        self.use_residual = use_residual
        self.gcn_linear = nn.Linear(hidden_dim, hidden_dim)

        if use_initial_anchor:
            self.anchor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.anchor_linear = None
            self.alpha_raw = None

        if use_residual:
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.beta_raw = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gcn_linear.weight)
        nn.init.zeros_(self.gcn_linear.bias)
        if self.anchor_linear is not None:
            nn.init.xavier_uniform_(self.anchor_linear.weight)

    def forward(self, H, H0, A_norm):
        Z = self.gcn_linear(torch.matmul(A_norm, H))

        if self.use_initial_anchor:
            Z = Z + torch.sigmoid(self.alpha_raw) * self.anchor_linear(H0)

        H_out = F.relu(Z)

        if self.use_residual:
            H_out = H_out + torch.sigmoid(self.beta_raw) * H

        return H_out, Z


class DeepFirstOrderGCN(nn.Module):
    def __init__(
        self,
        in_features=6,
        hidden_dim=16,
        num_layers=2,
        K=None,
        edge_list=None,
        num_nodes=33,
        node_relu_dim=4,
        edge_relu_dim=4,
        node_emb_dim=4,
        edge_emb_dim=8,
        use_residual=True,
        use_initial_anchor=True,
        use_jk=True,
        include_input_in_jk=True,
        use_linear_skip=True,
    ):
        super().__init__()

        edge_list = sanitize_edge_list(edge_list if edge_list is not None else RADIAL_BRANCHES)

        if K is not None:
            try:
                num_layers = max(num_layers, int(K))
            except Exception:
                pass

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = 1
        self.requested_K = K
        self.num_nodes = num_nodes
        self.edge_list = edge_list
        self.num_edges = len(edge_list)

        self.node_relu_dim = node_relu_dim
        self.edge_relu_dim = edge_relu_dim
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim

        self.use_residual = use_residual
        self.use_initial_anchor = use_initial_anchor
        self.use_jk = use_jk
        self.include_input_in_jk = include_input_in_jk
        self.use_linear_skip = use_linear_skip

        self.src_indices = [e[0] for e in edge_list]
        self.dst_indices = [e[1] for e in edge_list]

        self.register_buffer("src_indices_tensor", torch.tensor(self.src_indices, dtype=torch.long))
        self.register_buffer("dst_indices_tensor", torch.tensor(self.dst_indices, dtype=torch.long))
        self.register_buffer("A_norm", self._build_adj(edge_list, num_nodes))

        self.input_embed = nn.Linear(in_features, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            FirstOrderNonlinearGCNLayer(
                hidden_dim=hidden_dim,
                use_initial_anchor=use_initial_anchor,
                use_residual=use_residual,
            )
            for _ in range(num_layers)
        ])

        if use_jk:
            jk_blocks = num_layers + 1 if include_input_in_jk else num_layers
            self.readout_dim = hidden_dim * jk_blocks
        else:
            self.readout_dim = hidden_dim

        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(self.num_edges, edge_emb_dim)

        node_input_dim = self.readout_dim + node_emb_dim
        edge_input_dim = self.readout_dim * 3 + edge_emb_dim

        self.node_hidden = nn.Linear(node_input_dim, node_relu_dim)
        self.node_out = nn.Linear(node_relu_dim, 1)
        self.node_skip = nn.Linear(node_input_dim, 1) if use_linear_skip else None

        self.edge_hidden = nn.Linear(edge_input_dim, edge_relu_dim)
        self.edge_out = nn.Linear(edge_relu_dim, 1)
        self.edge_skip = nn.Linear(edge_input_dim, 1) if use_linear_skip else None

        self.reset_parameters()

    @staticmethod
    def _build_adj(edge_list, num_nodes):
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        for u, v in edge_list:
            A[u, v] = 1.0
            A[v, u] = 1.0

        A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
        deg = A_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)

        return torch.diag(deg_inv_sqrt) @ A_hat @ torch.diag(deg_inv_sqrt)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_embed.weight)
        nn.init.zeros_(self.input_embed.bias)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.edge_emb.weight, mean=0.0, std=0.02)

        for layer in [self.node_hidden, self.node_out, self.edge_hidden, self.edge_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.node_skip is not None:
            nn.init.xavier_uniform_(self.node_skip.weight)
            nn.init.zeros_(self.node_skip.bias)

        if self.edge_skip is not None:
            nn.init.xavier_uniform_(self.edge_skip.weight)
            nn.init.zeros_(self.edge_skip.bias)

    def build_jk_feature(self, H0, H_list):
        if not self.use_jk:
            return H_list[-1]
        if self.include_input_in_jk:
            return torch.cat([H0] + H_list, dim=-1)
        return torch.cat(H_list, dim=-1)

    def forward(self, X):
        batch_size = X.size(0)
        device = X.device
        A_norm = self.A_norm.to(device)

        H0 = self.input_embed(X)
        H = H0
        H_list, gcn_Z_list = [], []

        for layer in self.gcn_layers:
            H, Z = layer(H, H0, A_norm)
            H_list.append(H)
            gcn_Z_list.append(Z)

        H_readout = self.build_jk_feature(H0, H_list)

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_emb(node_ids).unsqueeze(0).expand(batch_size, -1, -1)
        H_node = torch.cat([H_readout, node_emb], dim=-1)

        Z_node = self.node_hidden(H_node)
        V_pred = self.node_out(F.relu(Z_node)).squeeze(-1)

        if self.node_skip is not None:
            V_pred = V_pred + self.node_skip(H_node).squeeze(-1)

        H_src = H_readout[:, self.src_indices, :]
        H_dst = H_readout[:, self.dst_indices, :]
        H_diff = H_src - H_dst

        edge_ids = torch.arange(self.num_edges, device=device)
        edge_emb = self.edge_emb(edge_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_edge = torch.cat([H_src, H_dst, H_diff, edge_emb], dim=-1)

        Z_edge = self.edge_hidden(H_edge)
        I_pred = self.edge_out(F.relu(Z_edge)).squeeze(-1)

        if self.edge_skip is not None:
            I_pred = I_pred + self.edge_skip(H_edge).squeeze(-1)

        return V_pred, I_pred, gcn_Z_list, Z_node, Z_edge

    def get_frozen_adj_norm(self):
        return self.A_norm.detach().cpu().numpy()

    def get_frozen_adj_powers(self):
        return self.A_norm.detach().unsqueeze(0).cpu().numpy()

    def get_frozen_node_emb(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ids = torch.arange(self.num_nodes, device=self.node_emb.weight.device)
            out = self.node_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_frozen_edge_emb(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ids = torch.arange(self.num_edges, device=self.edge_emb.weight.device)
            out = self.edge_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_binary_count(self):
        gcn_binary = self.num_layers * self.num_nodes * self.hidden_dim
        node_head_binary = self.num_nodes * self.node_relu_dim
        edge_head_binary = self.num_edges * self.edge_relu_dim

        return {
            "gcn_binary": gcn_binary,
            "node_head_binary": node_head_binary,
            "edge_head_binary": edge_head_binary,
            "total_binary": gcn_binary + node_head_binary + edge_head_binary,
        }


StandardGCN = DeepFirstOrderGCN


def get_config():
    return {
        "exp_name": "Exp25_H16_L2",
        "data_path": r"data/ieee33_static_vvo_balanced_20k.pt",
        "save_dir": r"checkpoints",
        "best_model_name": "st_gcn_h16_l2_exp25_best.pt",
        "engine_name": "st_gcn_h16_l2_exp25_milp_engine.pt",

        "seed": 42,
        "train_ratio": 0.8,
        "batch_size": 256,
        "epochs": 800,
        "patience": 180,

        "hidden_dim": 16,
        "num_layers": 2,
        "node_relu_dim": 4,
        "edge_relu_dim": 4,
        "node_emb_dim": 4,
        "edge_emb_dim": 8,

        "use_residual": True,
        "use_initial_anchor": True,
        "use_jk": True,
        "include_input_in_jk": True,
        "use_linear_skip": True,

        "ridge_alpha_v": 1e-3,
        "ridge_alpha_i": 1e-3,
        "edge_linear_prior_mode": "dst_node_X6",

        "zero_init_voltage_residual_head": True,
        "zero_init_current_residual_head": True,

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


def build_topology_matrices(edge_list, num_nodes=33):
    adj = [[] for _ in range(num_nodes)]

    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)

    parent = np.full(num_nodes, -2, dtype=int)
    children = [[] for _ in range(num_nodes)]
    parent[0] = -1
    queue = [0]

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                children[u].append(v)
                queue.append(v)

    S_down = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    def dfs(u):
        S_down[u, u] = 1.0
        for v in children[u]:
            dfs(v)
            S_down[u] += S_down[v]

    dfs(0)

    S_path = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        cur = i
        while cur != 0 and parent[cur] >= 0:
            S_path[i] += S_down[cur]
            cur = parent[cur]

    return S_down, S_path, parent


def augment_path_power_features(X, S_down, S_path):
    P = X[:, :, 0]
    Q = X[:, :, 1]

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T
    P_path = P @ S_path.T
    Q_path = Q @ S_path.T

    return np.stack([P, Q, P_down, Q_down, P_path, Q_path], axis=2).astype(np.float32)


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


def fit_voltage_linear_prior(Xn, YVn, train_idx, alpha=1e-3):
    Xmat = Xn[train_idx].reshape(len(train_idx), -1).double().numpy()
    Ymat = YVn[train_idx, 1:].double().numpy()

    ones = np.ones((Xmat.shape[0], 1), dtype=np.float64)
    Xa = np.concatenate([Xmat, ones], axis=1)

    reg = float(alpha) * np.eye(Xa.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0

    coef = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Ymat)

    W = coef[:-1].T.astype(np.float32)
    b = coef[-1].astype(np.float32)

    return torch.tensor(W), torch.tensor(b)


def fit_edge_current_linear_prior(Xn, YIn, train_idx, edge_list, alpha=1e-3):
    edge_list = sanitize_edge_list(edge_list)
    dst_idx = torch.tensor([v for _, v in edge_list], dtype=torch.long)

    W = []
    b = []

    for e, dst in enumerate(dst_idx.tolist()):
        Xmat = Xn[train_idx, dst, :].double().numpy()
        y = YIn[train_idx, e].double().numpy().reshape(-1, 1)

        ones = np.ones((Xmat.shape[0], 1), dtype=np.float64)
        Xa = np.concatenate([Xmat, ones], axis=1)

        reg = float(alpha) * np.eye(Xa.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0

        coef = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y).reshape(-1)

        W.append(coef[:-1].astype(np.float32))
        b.append(np.float32(coef[-1]))

    return torch.tensor(np.stack(W, axis=0)), torch.tensor(np.array(b, dtype=np.float32)), dst_idx


@torch.no_grad()
def eval_voltage_prior(Xn, YV, W, b, norm, idx):
    Xflat = Xn[idx].reshape(len(idx), -1)
    Vn = Xflat @ W.T + b

    V = torch.ones((len(idx), 33), dtype=torch.float32)
    V[:, 1:] = Vn * norm["YV_std_wo_slack"].view(1, -1) + norm["YV_mean_wo_slack"].view(1, -1)

    err = torch.abs(V[:, 1:] - YV[idx, 1:])

    return {
        "V_MAE": float(err.mean()),
        "V_RMSE": float(torch.sqrt(torch.mean((V[:, 1:] - YV[idx, 1:]) ** 2))),
        "V_MaxErr": float(err.max()),
    }


@torch.no_grad()
def eval_edge_current_prior(Xn, YI, W, b, dst_idx, norm, idx):
    X_edge = Xn[idx][:, dst_idx, :]
    In = (X_edge * W.view(1, W.size(0), W.size(1))).sum(dim=-1) + b.view(1, -1)
    I = In * norm["YI_std"].view(1, -1) + norm["YI_mean"].view(1, -1)

    err = torch.abs(I - YI[idx])

    true_unsafe = YI[idx] > 0.0
    pred_unsafe = I > 0.0

    fs = (true_unsafe & (~pred_unsafe)).sum().item() / max(true_unsafe.sum().item(), 1) * 100
    fv = ((~true_unsafe) & pred_unsafe).sum().item() / max((~true_unsafe).sum().item(), 1) * 100

    return {
        "I_MAE": float(err.mean()),
        "I_RMSE": float(torch.sqrt(torch.mean((I - YI[idx]) ** 2))),
        "I_MaxErr": float(err.max()),
        "I_FalseSafe": fs,
        "I_FalseViolate": fv,
    }


def build_base_model(cfg, edge_list):
    return StandardGCN(
        in_features=6,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        edge_list=edge_list,
        num_nodes=33,
        node_relu_dim=cfg["node_relu_dim"],
        edge_relu_dim=cfg["edge_relu_dim"],
        node_emb_dim=cfg["node_emb_dim"],
        edge_emb_dim=cfg["edge_emb_dim"],
        use_residual=cfg["use_residual"],
        use_initial_anchor=cfg["use_initial_anchor"],
        use_jk=cfg["use_jk"],
        include_input_in_jk=cfg["include_input_in_jk"],
        use_linear_skip=cfg["use_linear_skip"],
    )


def zero_init_voltage_head(model):
    for name in ["node_out", "node_skip"]:
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def zero_init_current_head(model):
    for name in ["edge_out", "edge_skip"]:
        layer = getattr(model, name, None)
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class VoltageCurrentLinearResidualModel(nn.Module):
    def __init__(self, base_model, W_vlin, b_vlin, W_ilin, b_ilin, dst_idx):
        super().__init__()

        self.base = base_model

        self.register_buffer("W_vlin", W_vlin.float())
        self.register_buffer("b_vlin", b_vlin.float())

        self.register_buffer("W_ilin", W_ilin.float())
        self.register_buffer("b_ilin", b_ilin.float())
        self.register_buffer("edge_linear_dst_idx", dst_idx.long())

    def current_linear_prior(self, X):
        X_edge = X[:, self.edge_linear_dst_idx, :]
        return (X_edge * self.W_ilin.view(1, self.W_ilin.size(0), self.W_ilin.size(1))).sum(dim=-1) + self.b_ilin.view(1, -1)

    def forward(self, X):
        out = self.base(X)

        if not isinstance(out, tuple) or len(out) < 5:
            raise RuntimeError("base model forward 必须返回 V_res, I_res, gcn_Z_list, Z_node, Z_edge。")

        V_res, I_res, gcn_Z_list, Z_node, Z_edge = out[:5]

        V_lin = X.reshape(X.size(0), -1) @ self.W_vlin.T + self.b_vlin
        I_lin = self.current_linear_prior(X)

        if V_res.shape[1] == 33:
            V_total = V_res.clone()
            V_total[:, 1:] = V_lin + V_res[:, 1:]
        else:
            V_total = V_lin + V_res

        I_total = I_lin + I_res

        return V_total, I_total, gcn_Z_list, Z_node, Z_edge

    def get_frozen_adj_norm(self):
        if hasattr(self.base, "get_frozen_adj_norm"):
            return self.base.get_frozen_adj_norm()
        return None

    def get_binary_count(self):
        if hasattr(self.base, "get_binary_count"):
            return self.base.get_binary_count()
        return None


def unpack_forward(model, X):
    out = model(X)
    if isinstance(out, tuple):
        return out
    raise RuntimeError("model.forward 必须返回 V_pred, I_pred, gcn_Z_list, Z_node, Z_edge。")


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

    gcn_z_all = [[] for _ in range(cfg["num_layers"])]
    node_z_all, edge_z_all = [], []

    for i in range(0, len(Xn), batch_size):
        xb = Xn[i:i + batch_size].to(device)
        out = unpack_forward(model, xb)

        if len(out) < 5:
            raise RuntimeError("模型 forward 需要返回 gcn_Z_list, Z_node, Z_edge 用于 Big-M 提取。")

        gcn_Z_list, Z_node, Z_edge = out[2], out[3], out[4]

        for k, z in enumerate(gcn_Z_list):
            gcn_z_all[k].append(z.detach().cpu())

        node_z_all.append(Z_node.detach().cpu())
        edge_z_all.append(Z_edge.detach().cpu())

    beta = cfg["big_m_beta"]

    M_plus_gcn, M_minus_gcn = [], []

    for k in range(cfg["num_layers"]):
        Z = torch.cat(gcn_z_all[k], dim=0)
        M_plus_gcn.append(torch.clamp(Z.max(dim=0).values, min=0.0) * beta)
        M_minus_gcn.append(torch.clamp((-Z).max(dim=0).values, min=0.0) * beta)

    Z_node = torch.cat(node_z_all, dim=0)
    Z_edge = torch.cat(edge_z_all, dim=0)

    return {
        "M_plus_gcn_layers": M_plus_gcn,
        "M_minus_gcn_layers": M_minus_gcn,
        "M_plus_node": torch.clamp(Z_node.max(dim=0).values, min=0.0) * beta,
        "M_minus_node": torch.clamp((-Z_node).max(dim=0).values, min=0.0) * beta,
        "M_plus_edge": torch.clamp(Z_edge.max(dim=0).values, min=0.0) * beta,
        "M_minus_edge": torch.clamp((-Z_edge).max(dim=0).values, min=0.0) * beta,
    }


def binary_count(model, cfg):
    base = model.base if isinstance(model, VoltageCurrentLinearResidualModel) else model

    if hasattr(base, "get_binary_count"):
        return base.get_binary_count()

    gcn_binary = cfg["num_layers"] * 33 * cfg["hidden_dim"]
    node_binary = 33 * cfg["node_relu_dim"]
    edge_binary = 32 * cfg["edge_relu_dim"]

    return {
        "gcn_binary": gcn_binary,
        "node_head_binary": node_binary,
        "edge_head_binary": edge_binary,
        "total_binary": gcn_binary + node_binary + edge_binary,
    }


def main():
    cfg = get_config()
    set_seed(cfg["seed"])

    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    data = load_pt(cfg["data_path"])
    edge_list = get_edge_list(data)

    X_raw = to_tensor(data["X"]).float().numpy()
    YV, YI = get_labels(data)

    S_down, S_path, parent = build_topology_matrices(edge_list, 33)
    X_aug = torch.tensor(augment_path_power_features(X_raw, S_down, S_path), dtype=torch.float32)

    n = len(X_aug)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    Xn, YVn, YIn, norm = normalize_data(X_aug, YV, YI, train_idx)

    W_vlin, b_vlin = fit_voltage_linear_prior(Xn, YVn, train_idx, alpha=cfg["ridge_alpha_v"])
    vlin_metrics = eval_voltage_prior(Xn, YV, W_vlin, b_vlin, norm, val_idx)

    W_ilin, b_ilin, edge_linear_dst_idx = fit_edge_current_linear_prior(
        Xn, YIn, train_idx, edge_list, alpha=cfg["ridge_alpha_i"]
    )
    ilin_metrics = eval_edge_current_prior(Xn, YI, W_ilin, b_ilin, edge_linear_dst_idx, norm, val_idx)

    train_ds = TensorDataset(Xn[train_idx], YVn[train_idx], YIn[train_idx], YV[train_idx], YI[train_idx])
    val_ds = TensorDataset(Xn[val_idx], YVn[val_idx], YIn[val_idx], YV[val_idx], YI[val_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)

    base_model = build_base_model(cfg, edge_list)

    if cfg["zero_init_voltage_residual_head"]:
        zero_init_voltage_head(base_model)

    if cfg["zero_init_current_residual_head"]:
        zero_init_current_head(base_model)

    model = VoltageCurrentLinearResidualModel(
        base_model=base_model,
        W_vlin=W_vlin,
        b_vlin=b_vlin,
        W_ilin=W_ilin,
        b_ilin=b_ilin,
        dst_idx=edge_linear_dst_idx,
    ).to(device)

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
    print("输入特征: [P_net,Q_net,P_down,Q_down,P_path,Q_path]")
    print("电压输出: V_pred = V_linear_prior + GCN_voltage_residual")
    print("支路输出: I_pred = I_linear_prior(dst_node_X6) + GCN_current_residual")
    print("训练目标增强: 逐支路对称电流符号分类 + 电压三区域安全分类")
    print(
        f"线性电压先验 Val: "
        f"MAE={vlin_metrics['V_MAE']:.6f}, "
        f"RMSE={vlin_metrics['V_RMSE']:.6f}, "
        f"MaxErr={vlin_metrics['V_MaxErr']:.6f}"
    )
    print(
        f"线性电流先验 Val: "
        f"MAE={ilin_metrics['I_MAE']:.6f}, "
        f"RMSE={ilin_metrics['I_RMSE']:.6f}, "
        f"MaxErr={ilin_metrics['I_MaxErr']:.6f}, "
        f"I-FS={ilin_metrics['I_FalseSafe']:.2f}%, "
        f"I-FV={ilin_metrics['I_FalseViolate']:.2f}%"
    )
    print(
        f"hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}, "
        f"node_head={cfg['node_relu_dim']}, edge_head={cfg['edge_relu_dim']}"
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
                    "base_state_dict": model.base.state_dict(),
                    "voltage_linear_prior_W": model.W_vlin.detach().cpu(),
                    "voltage_linear_prior_b": model.b_vlin.detach().cpu(),
                    "current_linear_prior_W": model.W_ilin.detach().cpu(),
                    "current_linear_prior_b": model.b_ilin.detach().cpu(),
                    "current_linear_prior_dst_idx": model.edge_linear_dst_idx.detach().cpu(),
                    "config": cfg,
                    "norm_stats": {k: v.cpu() for k, v in norm.items()},
                    "edge_list": edge_list,
                    "downstream_matrix": torch.tensor(S_down, dtype=torch.float32),
                    "path_power_matrix": torch.tensor(S_path, dtype=torch.float32),
                    "parent_array": torch.tensor(parent, dtype=torch.long),
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

    frozen_adj = model.get_frozen_adj_norm()

    if frozen_adj is not None:
        frozen_adj = torch.tensor(frozen_adj, dtype=torch.float32)

    engine = {
        "model_type": cfg["exp_name"],

        "state_dict": model.base.state_dict(),
        "wrapper_state_dict": model.state_dict(),

        "uses_voltage_linear_prior": True,
        "uses_current_linear_prior": True,

        "voltage_linear_prior_input": "normalized_flattened_X6",
        "voltage_linear_prior_target": "normalized_voltage_without_slack",
        "voltage_linear_prior_W": model.W_vlin.detach().cpu(),
        "voltage_linear_prior_b": model.b_vlin.detach().cpu(),

        "current_linear_prior_input": "normalized_X6_at_branch_dst_node",
        "current_linear_prior_target": "normalized_branch_current_margin",
        "current_linear_prior_W": model.W_ilin.detach().cpu(),
        "current_linear_prior_b": model.b_ilin.detach().cpu(),
        "current_linear_prior_dst_idx": model.edge_linear_dst_idx.detach().cpu(),

        "config": cfg,
        "in_features": 6,
        "feature_names": ["P_net", "Q_net", "P_down", "Q_down", "P_path", "Q_path"],

        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "node_relu_dim": cfg["node_relu_dim"],
        "edge_relu_dim": cfg["edge_relu_dim"],
        "node_emb_dim": cfg["node_emb_dim"],
        "edge_emb_dim": cfg["edge_emb_dim"],

        "use_residual": cfg["use_residual"],
        "use_initial_anchor": cfg["use_initial_anchor"],
        "use_jk": cfg["use_jk"],
        "include_input_in_jk": cfg["include_input_in_jk"],
        "use_linear_skip": cfg["use_linear_skip"],

        "edge_list": edge_list,
        "downstream_matrix": torch.tensor(S_down, dtype=torch.float32),
        "path_power_matrix": torch.tensor(S_path, dtype=torch.float32),
        "parent_array": torch.tensor(parent, dtype=torch.long),
        "frozen_adj_norm": frozen_adj,

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

        "linear_prior_metrics": {
            "voltage": vlin_metrics,
            "current": ilin_metrics,
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

    print("\n[Linear voltage prior]")
    print(f"V_MAE: {vlin_metrics['V_MAE']:.6f}")
    print(f"V_RMSE: {vlin_metrics['V_RMSE']:.6f}")
    print(f"V_MaxErr: {vlin_metrics['V_MaxErr']:.6f}")

    print("\n[Linear current prior]")
    print(f"I_MAE: {ilin_metrics['I_MAE']:.6f}")
    print(f"I_RMSE: {ilin_metrics['I_RMSE']:.6f}")
    print(f"I_MaxErr: {ilin_metrics['I_MaxErr']:.6f}")
    print(f"I_FalseSafe: {ilin_metrics['I_FalseSafe']:.2f}%")
    print(f"I_FalseViolate: {ilin_metrics['I_FalseViolate']:.2f}%")

    print("\n[GCN residual model]")
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
