from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
np.seterr(divide="ignore", invalid="ignore", over="ignore")

import torch
import pandapower as pp
from tqdm import tqdm


# =========================================================
# 0. 全局配置
# =========================================================
SEED = 42

PROJECT_ROOT = r"D:\pythonproject\ST-GCN"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAVE_NAME = "ieee33_nodal_pq_correlated_raw_pool_50k.pt"
CSV_NAME = "ieee33_nodal_pq_correlated_classification_50k.csv"

POOL_SAMPLES = 50000
CANDIDATE_MULTIPLIER = 1.20

SLACK_VM_PU = 1.03
UNIFORM_MAX_I_KA = 0.20

LOAD_PROFILE_PEAK = 1.30
# 负荷采用低秩相关扰动，而非 32 个负荷节点独立大幅随机变化
# 综合效果约为 P -18%~+20%，Q 在 P 基础上再发生合理功率因数变化
LOAD_GLOBAL_JITTER = 0.10
LOAD_GROUP_JITTER = 0.06
LOAD_LOCAL_JITTER = 0.03
PF_GLOBAL_JITTER = 0.04
PF_GROUP_JITTER = 0.04

# PV 采用公共天气因子 + 较小站点差异
PV_GLOBAL_JITTER = 0.20
PV_SITE_JITTER = 0.08
ESS_P_MAX_FRACTION = 0.60

# 实际 MILP 安全约束，仅用于分类统计，不用于生成阶段过滤
V_MIN_LIMIT = 0.95
V_MAX_LIMIT = 1.05
I_MARGIN_LIMIT = 0.0


# =========================================================
# 1. IEEE33 静态径向拓扑
# =========================================================
RADIAL_BRANCHES = [
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
]


# =========================================================
# 2. 负荷曲线与唯一可控节点设备布置
# =========================================================
def prepare_profiles_and_devices():
    # 标准 MATPOWER case33bw 基础负荷
    pload_kw = np.array(
        [
            0, 100, 90, 120, 60, 60, 200, 200, 60, 60, 45,
            60, 60, 120, 60, 60, 60, 90, 90, 90, 90, 90,
            90, 420, 420, 60, 60, 60, 120, 200, 150, 210, 60
        ],
        dtype=np.float64,
    )

    qload_kvar = np.array(
        [
            0, 60, 40, 80, 30, 20, 100, 100, 20, 20, 30,
            35, 35, 80, 10, 20, 20, 40, 40, 40, 40, 40,
            50, 200, 200, 25, 25, 20, 70, 600, 70, 100, 40
        ],
        dtype=np.float64,
    )

    assert abs(pload_kw.sum() / 1000.0 - 3.715) < 1e-12
    assert abs(qload_kvar.sum() / 1000.0 - 2.300) < 1e-12

    pload_ratio = np.array(
        [
            2.15, 2.3, 1.2, 2.35, 2.35, 2.6, 3.0, 2.25,
            2.7, 1.8, 1.35, 1.2, 1.15, 1.1, 1.35, 1.45,
            1.5, 1.65, 1.9, 2.0, 1.2, 1.8, 1.85, 1.8
        ],
        dtype=np.float64,
    )

    qload_ratio = np.array(
        [
            1.15, 1.3, 0.8, 1.35, 1.35, 1.6, 2.0, 1.25,
            2.1, 1.4, 1.15, 1.0, 0.9, 1.0, 1.2, 1.25,
            1.3, 1.45, 1.2, 1.0, 1.0, 1.4, 1.55, 1.4
        ],
        dtype=np.float64,
    )

    pload_ratio = pload_ratio / pload_ratio.max() * LOAD_PROFILE_PEAK
    qload_ratio = qload_ratio / qload_ratio.max() * LOAD_PROFILE_PEAK

    # 24h 曲线只作为各时刻的中心运行点。
    # 空间差异由后续“全网 + 电气区域 + 小幅节点空间模态”相关扰动产生，
    # 不再人为给 32 个负荷节点各自设置独立的大幅日曲线。
    Pload_24h = (pload_kw[:, None] * pload_ratio[None, :]) / 1000.0
    Qload_24h = (qload_kvar[:, None] * qload_ratio[None, :]) / 1000.0

    pv_shape_raw = np.array(
        [
            0, 0, 0, 0, 0, 0.25, 0.49, 0.52, 0.75, 0.9, 1.0, 1.25,
            1.1, 0.9, 0.79, 0.76, 0.75, 0.55, 0.35, 0.25, 0, 0, 0, 0
        ],
        dtype=np.float64,
    )
    pv_shape = pv_shape_raw / pv_shape_raw.max()

    # 每个可控节点最多一个无功设备。
    # PV 节点保持不变；ESS 从原来的 PV 重合节点移到 Bus12/Bus27；
    # SVC1 从原来的 PV@Bus8 重合节点移到 Bus11；其余 Q 设备位置保留。
    devices = {
        "pv_nodes": np.array([7, 14, 22, 29], dtype=np.int64),       # Bus 8,15,23,30
        "S_pv_mva": np.array([0.5, 0.8, 1.0, 1.2], dtype=np.float64),

        "ess_nodes": np.array([11, 26], dtype=np.int64),             # Bus 12,27
        "S_ess_mva": np.array([0.8, 0.8], dtype=np.float64),

        "q_device_names": ["SVC1", "SVG1", "SVG2", "SVC2", "SC"],
        "q_device_nodes": np.array([10, 3, 16, 20, 1], dtype=np.int64),  # Bus 11,4,17,21,2
        "q_device_min": np.array([-1.0, -0.5, -0.8, -2.0, 0.0], dtype=np.float64),
        "q_device_max": np.array([1.0, 0.5, 0.8, 2.0, 0.4], dtype=np.float64),
    }

    all_ctrl = np.concatenate(
        [
            devices["pv_nodes"],
            devices["ess_nodes"],
            devices["q_device_nodes"],
        ]
    )
    if len(np.unique(all_ctrl)) != len(all_ctrl):
        raise RuntimeError("可控设备节点存在重合，违反“一节点最多一个无功设备”的设定。")

    Ppv_24h = pv_shape[:, None] * (0.8 * devices["S_pv_mva"])[None, :]

    ctrl_nodes = all_ctrl
    ctrl_names = (
        [f"PV{i+1}" for i in range(len(devices["pv_nodes"]))]
        + [f"ESS{i+1}" for i in range(len(devices["ess_nodes"]))]
        + devices["q_device_names"]
    )
    ctrl_types = (
        ["PV"] * len(devices["pv_nodes"])
        + ["ESS"] * len(devices["ess_nodes"])
        + ["QDEV"] * len(devices["q_device_nodes"])
    )

    devices["ctrl_nodes"] = ctrl_nodes
    devices["ctrl_names"] = ctrl_names
    devices["ctrl_types"] = ctrl_types

    # 依据 IEEE33 径向拓扑划分 4 个电气区域。
    # 这里只表示空间相关区域，不赋予“住宅/商业/工业”等无数据依据的语义标签。
    load_group_id = np.zeros(33, dtype=np.int64)
    load_group_id[1:18] = 0      # Bus 2~18，主干
    load_group_id[18:22] = 1     # Bus 19~22，支路
    load_group_id[22:25] = 2     # Bus 23~25，支路
    load_group_id[25:33] = 3     # Bus 26~33，支路

    # 每个区域内部只保留一个很小的空间模态，使同一区域节点并非完全同比例。
    # 权重固定在 [-1,1]，每个样本只随机一个区域模态系数。
    load_local_weight = np.zeros(33, dtype=np.float64)
    for g in range(4):
        idx_g = np.where(load_group_id == g)[0]
        idx_g = idx_g[idx_g != 0]  # Bus1 无负荷
        if len(idx_g) == 1:
            load_local_weight[idx_g] = 0.0
        elif len(idx_g) > 1:
            load_local_weight[idx_g] = np.linspace(-1.0, 1.0, len(idx_g))

    devices["load_group_id"] = load_group_id
    devices["load_local_weight"] = load_local_weight
    devices["n_load_groups"] = 4

    return Pload_24h, Qload_24h, Ppv_24h, devices


# =========================================================
# 3. pandapower 网络
# =========================================================
def setup_static_ieee33_env():
    net = pp.create_empty_network()

    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i + 1}")

    pp.create_ext_grid(net, bus=0, vm_pu=SLACK_VM_PU)

    for f, t, r, x in RADIAL_BRANCHES:
        pp.create_line_from_parameters(
            net,
            from_bus=int(f),
            to_bus=int(t),
            length_km=1.0,
            r_ohm_per_km=float(r),
            x_ohm_per_km=float(x),
            c_nf_per_km=0.0,
            max_i_ka=UNIFORM_MAX_I_KA,
        )

    for i in range(33):
        pp.create_load(net, bus=i, p_mw=0.0, q_mvar=0.0)

    return net


def add_static_devices(net, devices):
    pv_idx = [
        pp.create_sgen(net, bus=int(bus), p_mw=0.0, q_mvar=0.0, name=f"PV@{bus+1}")
        for bus in devices["pv_nodes"]
    ]

    ess_idx = [
        pp.create_sgen(net, bus=int(bus), p_mw=0.0, q_mvar=0.0, name=f"ESS@{bus+1}")
        for bus in devices["ess_nodes"]
    ]

    qdev_idx = [
        pp.create_sgen(net, bus=int(bus), p_mw=0.0, q_mvar=0.0, name=f"{name}@{bus+1}")
        for name, bus in zip(devices["q_device_names"], devices["q_device_nodes"])
    ]

    return {
        "pv_sgen_idx": np.asarray(pv_idx, dtype=np.int64),
        "ess_sgen_idx": np.asarray(ess_idx, dtype=np.int64),
        "qdev_sgen_idx": np.asarray(qdev_idx, dtype=np.int64),
    }


# =========================================================
# 4. 潮流求解
# =========================================================
def run_powerflow_with_fallback(net):
    strategies = [
        dict(algorithm="bfsw", init="flat", tolerance_mva=1e-7, max_iteration=100),
        dict(algorithm="nr", init="flat", tolerance_mva=1e-7, max_iteration=40),
        dict(algorithm="nr", init="auto", tolerance_mva=1e-7, max_iteration=40),
    ]

    for kw in strategies:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                pp.runpp(
                    net,
                    enforce_q_lims=False,
                    calculate_voltage_angles=False,
                    numba=False,
                    **kw,
                )

            if not bool(net.converged):
                continue

            vm = net.res_bus.vm_pu.values
            loading = net.res_line.loading_percent.values

            if np.all(np.isfinite(vm)) and np.all(np.isfinite(loading)):
                return True

        except Exception:
            try:
                pp.reset_results(net)
            except Exception:
                pass

    return False


# =========================================================
# 5. 每小时 Latin Hypercube 采样计划
# =========================================================
# LHS 随机维度：
# 负荷  = 1 全网P + 4区域P + 4区域小空间模态 + 1全网PF + 4区域PF = 14
# PV    = 1 公共天气 + 4站点偏差 + 4个Q可行域坐标 = 9
# ESS   = 2个P + 2个Q = 4
# Q设备 = 5个Q = 5
# 合计 32 维；相比原先 83 维独立扰动显著降低源荷随机空间维数。
N_LHS_DIM = 14 + 9 + 4 + 5


def lhs_matrix(rng, n, d):
    """
    Latin Hypercube:
    每一维把 [0,1] 分成 n 个等宽小区间，每个区间恰取一个点，
    然后各维独立打乱。这样每个边际维度都被均匀覆盖。
    """
    if n <= 0:
        return np.empty((0, d), dtype=np.float64)

    out = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        values = (np.arange(n, dtype=np.float64) + rng.random(n)) / n
        rng.shuffle(values)
        out[:, j] = values
    return out


def build_candidate_plan(rng, n_candidates):
    hours = np.arange(n_candidates, dtype=np.int64) % 24
    U = np.empty((n_candidates, N_LHS_DIM), dtype=np.float64)

    for h in range(24):
        idx = np.where(hours == h)[0]
        U[idx] = lhs_matrix(rng, len(idx), N_LHS_DIM)

    return hours, U


# =========================================================
# 6. 直接在节点净注入 P-Q 可行域内采样
# =========================================================
def sample_nodal_pq_point(hour, u, Pload_24h, Qload_24h, Ppv_24h, devices):
    c = 0

    # -----------------------------------------------------
    # 负荷：低秩相关扰动
    # P = 24h中心 × 全网公共因子 × 区域公共因子 × 小幅区域内空间模态
    # Q = P的相关倍率 × 全网/区域功率因数扰动
    # -----------------------------------------------------
    z_global_p = 2.0 * u[c] - 1.0
    c += 1

    z_group_p = 2.0 * u[c:c+4] - 1.0
    c += 4

    z_local = 2.0 * u[c:c+4] - 1.0
    c += 4

    z_global_pf = 2.0 * u[c] - 1.0
    c += 1

    z_group_pf = 2.0 * u[c:c+4] - 1.0
    c += 4

    group_id = devices["load_group_id"]
    local_weight = devices["load_local_weight"]

    global_p_factor = 1.0 + LOAD_GLOBAL_JITTER * z_global_p
    group_p_factor = 1.0 + LOAD_GROUP_JITTER * z_group_p[group_id]
    local_factor = (
        1.0
        + LOAD_LOCAL_JITTER
        * z_local[group_id]
        * local_weight
    )

    p_scale = global_p_factor * group_p_factor * local_factor

    # Q 与 P 高度相关，但允许功率因数发生独立的小幅变化。
    pf_factor = (
        (1.0 + PF_GLOBAL_JITTER * z_global_pf)
        * (1.0 + PF_GROUP_JITTER * z_group_pf[group_id])
    )
    q_scale = p_scale * pf_factor

    Pload = Pload_24h[:, hour] * p_scale
    Qload = Qload_24h[:, hour] * q_scale

    P_net = -Pload.copy()
    Q_net = -Qload.copy()

    n_ctrl = len(devices["ctrl_nodes"])
    ctrl_p = np.zeros(n_ctrl, dtype=np.float64)
    ctrl_q = np.zeros(n_ctrl, dtype=np.float64)
    ctrl_p_min = np.zeros(n_ctrl, dtype=np.float64)
    ctrl_p_max = np.zeros(n_ctrl, dtype=np.float64)
    ctrl_q_min = np.zeros(n_ctrl, dtype=np.float64)
    ctrl_q_max = np.zeros(n_ctrl, dtype=np.float64)

    # -----------------------------------------------------
    # PV 节点
    # PV 有功属于源侧运行条件：24h 中心 × 公共天气 × 小幅站点差异。
    # 不再让 4 个 PV 各自完全独立 ±25%。
    # 给定实际 PV 有功后，只在其逆变器容量圆对应的节点净 Q 可行域内采样。
    # -----------------------------------------------------
    z_pv_global = 2.0 * u[c] - 1.0
    c += 1

    z_pv_site = 2.0 * u[c:c+4] - 1.0
    c += 4

    u_pv_q = u[c:c+4]
    c += 4

    pv_p = np.zeros(4, dtype=np.float64)
    pv_q = np.zeros(4, dtype=np.float64)

    pv_global_factor = 1.0 + PV_GLOBAL_JITTER * z_pv_global

    for k, bus in enumerate(devices["pv_nodes"]):
        S = devices["S_pv_mva"][k]
        p0 = Ppv_24h[hour, k]

        pv_site_factor = 1.0 + PV_SITE_JITTER * z_pv_site[k]
        pdev = np.clip(
            p0 * pv_global_factor * pv_site_factor,
            0.0,
            0.98 * S,
        )

        # 当前样本的 PV 有功是已知运行条件，因此 Pnet 是一个点，不是决策区间。
        pnet = -Pload[bus] + pdev
        qcap = np.sqrt(max(S * S - pdev * pdev, 0.0))

        qnet_low = -Qload[bus] - qcap
        qnet_high = -Qload[bus] + qcap
        qnet = qnet_low + u_pv_q[k] * (qnet_high - qnet_low)
        qdev = qnet + Qload[bus]

        P_net[bus] = pnet
        Q_net[bus] = qnet
        pv_p[k] = pdev
        pv_q[k] = qdev

        pos = k
        ctrl_p[pos] = pnet
        ctrl_q[pos] = qnet
        ctrl_p_min[pos] = pnet
        ctrl_p_max[pos] = pnet
        ctrl_q_min[pos] = qnet_low
        ctrl_q_max[pos] = qnet_high

    # -----------------------------------------------------
    # ESS 节点
    # 直接在 ESS 容量圆对应的节点净注入可行域内分层采样
    # -----------------------------------------------------
    u_ess_p = u[c:c+2]
    c += 2
    u_ess_q = u[c:c+2]
    c += 2

    ess_p = np.zeros(2, dtype=np.float64)
    ess_q = np.zeros(2, dtype=np.float64)

    offset = len(devices["pv_nodes"])

    for k, bus in enumerate(devices["ess_nodes"]):
        S = devices["S_ess_mva"][k]
        p_dev_lim = ESS_P_MAX_FRACTION * S

        pnet_low = -Pload[bus] - p_dev_lim
        pnet_high = -Pload[bus] + p_dev_lim
        pnet = pnet_low + u_ess_p[k] * (pnet_high - pnet_low)

        pdev = pnet + Pload[bus]
        qcap = np.sqrt(max(S * S - pdev * pdev, 0.0))

        qnet_low = -Qload[bus] - qcap
        qnet_high = -Qload[bus] + qcap
        qnet = qnet_low + u_ess_q[k] * (qnet_high - qnet_low)

        qdev = qnet + Qload[bus]

        P_net[bus] = pnet
        Q_net[bus] = qnet
        ess_p[k] = pdev
        ess_q[k] = qdev

        pos = offset + k
        ctrl_p[pos] = pnet
        ctrl_q[pos] = qnet
        ctrl_p_min[pos] = pnet_low
        ctrl_p_max[pos] = pnet_high
        ctrl_q_min[pos] = qnet_low
        ctrl_q_max[pos] = qnet_high

    # -----------------------------------------------------
    # SVC/SVG/SC 节点
    # P_net 由当地负荷决定，Q_net 在完整可行区间内分层铺开
    # -----------------------------------------------------
    u_qdev = u[c:c+5]
    c += 5

    qdev_q = np.zeros(5, dtype=np.float64)
    offset += len(devices["ess_nodes"])

    for k, bus in enumerate(devices["q_device_nodes"]):
        qmin = devices["q_device_min"][k]
        qmax = devices["q_device_max"][k]

        pnet = -Pload[bus]
        qnet_low = -Qload[bus] + qmin
        qnet_high = -Qload[bus] + qmax
        qnet = qnet_low + u_qdev[k] * (qnet_high - qnet_low)

        qdev = qnet + Qload[bus]

        P_net[bus] = pnet
        Q_net[bus] = qnet
        qdev_q[k] = qdev

        pos = offset + k
        ctrl_p[pos] = pnet
        ctrl_q[pos] = qnet
        ctrl_p_min[pos] = pnet
        ctrl_p_max[pos] = pnet
        ctrl_q_min[pos] = qnet_low
        ctrl_q_max[pos] = qnet_high

    if c != N_LHS_DIM:
        raise RuntimeError(f"LHS 维度使用错误: used={c}, expected={N_LHS_DIM}")

    X = np.stack([P_net, Q_net], axis=1).astype(np.float32)

    return {
        "X": X,
        "Pload": Pload.astype(np.float32),
        "Qload": Qload.astype(np.float32),
        "pv_p": pv_p.astype(np.float32),
        "pv_q": pv_q.astype(np.float32),
        "ess_p": ess_p.astype(np.float32),
        "ess_q": ess_q.astype(np.float32),
        "qdev_q": qdev_q.astype(np.float32),
        "ctrl_pnet": ctrl_p.astype(np.float32),
        "ctrl_qnet": ctrl_q.astype(np.float32),
        "ctrl_pnet_min": ctrl_p_min.astype(np.float32),
        "ctrl_pnet_max": ctrl_p_max.astype(np.float32),
        "ctrl_qnet_min": ctrl_q_min.astype(np.float32),
        "ctrl_qnet_max": ctrl_q_max.astype(np.float32),
    }


# =========================================================
# 7. 样本施加
# =========================================================
def apply_point(net, idx, point):
    net.load.loc[:, "p_mw"] = point["Pload"]
    net.load.loc[:, "q_mvar"] = point["Qload"]

    net.sgen.loc[idx["pv_sgen_idx"], "p_mw"] = point["pv_p"]
    net.sgen.loc[idx["pv_sgen_idx"], "q_mvar"] = point["pv_q"]

    net.sgen.loc[idx["ess_sgen_idx"], "p_mw"] = point["ess_p"]
    net.sgen.loc[idx["ess_sgen_idx"], "q_mvar"] = point["ess_q"]

    net.sgen.loc[idx["qdev_sgen_idx"], "p_mw"] = 0.0
    net.sgen.loc[idx["qdev_sgen_idx"], "q_mvar"] = point["qdev_q"]

    net.line.loc[:, "in_service"] = True


# =========================================================
# 8. 原始 50k 样本池生成
# =========================================================
def generate_pool(net, idx, Pload_24h, Qload_24h, Ppv_24h, devices, rng):
    n_candidates = int(np.ceil(POOL_SAMPLES * CANDIDATE_MULTIPLIER))
    candidate_hours, U = build_candidate_plan(rng, n_candidates)

    fields = [
        "X", "Pload", "Qload", "pv_p", "pv_q", "ess_p", "ess_q", "qdev_q",
        "ctrl_pnet", "ctrl_qnet",
        "ctrl_pnet_min", "ctrl_pnet_max",
        "ctrl_qnet_min", "ctrl_qnet_max",
    ]
    saved = {k: [] for k in fields}
    yv_list = []
    yi_list = []
    hour_list = []

    attempt = 0
    pf_fail = 0
    nonfinite_fail = 0
    hour_attempt = np.zeros(24, dtype=np.int64)
    hour_accept = np.zeros(24, dtype=np.int64)
    hour_pf_fail = np.zeros(24, dtype=np.int64)

    t0 = time.perf_counter()
    pbar = tqdm(total=POOL_SAMPLES, desc="生成节点净PQ可行域样本池", unit="样本")

    for j in range(n_candidates):
        if len(yv_list) >= POOL_SAMPLES:
            break

        hour = int(candidate_hours[j])
        attempt += 1
        hour_attempt[hour] += 1

        point = sample_nodal_pq_point(
            hour, U[j], Pload_24h, Qload_24h, Ppv_24h, devices
        )
        apply_point(net, idx, point)

        if not run_powerflow_with_fallback(net):
            pf_fail += 1
            hour_pf_fail[hour] += 1
            continue

        YV = net.res_bus.vm_pu.values.astype(np.float32)
        loading = net.res_line.loading_percent.values.astype(np.float32)
        YI = loading / 100.0 - 1.0

        if not np.all(np.isfinite(YV)) or not np.all(np.isfinite(YI)):
            nonfinite_fail += 1
            continue

        for k in fields:
            saved[k].append(point[k])

        yv_list.append(YV)
        yi_list.append(YI)
        hour_list.append(hour)
        hour_accept[hour] += 1
        pbar.update(1)

    pbar.close()

    if len(yv_list) < POOL_SAMPLES:
        raise RuntimeError(
            f"候选样本不足以得到 {POOL_SAMPLES} 个收敛样本："
            f"仅得到 {len(yv_list)}。可提高 CANDIDATE_MULTIPLIER。"
        )

    stats = {
        "attempt_count": int(attempt),
        "success_count": int(len(yv_list)),
        "powerflow_fail_count": int(pf_fail),
        "nonfinite_fail_count": int(nonfinite_fail),
        "wall_time_sec": float(time.perf_counter() - t0),
        "hour_attempt_count": hour_attempt.tolist(),
        "hour_accept_count": hour_accept.tolist(),
        "hour_pf_fail_count": hour_pf_fail.tolist(),
    }

    print("\n每小时潮流统计:")
    for h in range(24):
        rate = (
            100.0 * hour_accept[h] / hour_attempt[h]
            if hour_attempt[h] > 0 else 0.0
        )
        print(
            f"h={h:02d}: attempt={hour_attempt[h]:5d}, "
            f"accept={hour_accept[h]:5d}, "
            f"pf_fail={hour_pf_fail[h]:4d}, rate={rate:6.2f}%"
        )

    out = {k: np.stack(v) for k, v in saved.items()}
    out["Y_V"] = np.stack(yv_list)
    out["Y_I"] = np.stack(yi_list)
    out["hour"] = np.asarray(hour_list, dtype=np.int64)
    out["stats"] = stats
    return out


# =========================================================
# 9. 分类统计
# =========================================================
def classify_pool(pool):
    YV = pool["Y_V"]
    YI = pool["Y_I"]

    vmin = YV.min(axis=1)
    vmax = YV.max(axis=1)
    imax = YI.max(axis=1)
    loading = (imax + 1.0) * 100.0

    v_unsafe = (vmin < V_MIN_LIMIT) | (vmax > V_MAX_LIMIT)
    i_unsafe = imax > I_MARGIN_LIMIT

    joint_class = v_unsafe.astype(np.int64) + 2 * i_unsafe.astype(np.int64)

    v_margin = np.minimum(vmin - V_MIN_LIMIT, V_MAX_LIMIT - vmax)
    i_safe_margin = -imax

    near_v = np.abs(v_margin) <= 0.01
    near_i = np.abs(imax) <= 0.10

    vmin_bins = np.array(
        [-np.inf, 0.88, 0.90, 0.93, 0.95, 0.97, 1.00, np.inf]
    )
    vmax_bins = np.array(
        [-np.inf, 1.00, 1.03, 1.05, 1.07, 1.10, 1.12, np.inf]
    )
    loading_bins = np.array(
        [-np.inf, 80, 90, 100, 110, 130, 150, 180, np.inf]
    )

    cls = {
        "vmin": vmin.astype(np.float32),
        "vmax": vmax.astype(np.float32),
        "imax": imax.astype(np.float32),
        "max_loading_percent": loading.astype(np.float32),
        "joint_class": joint_class.astype(np.int64),
        "v_margin": v_margin.astype(np.float32),
        "i_safe_margin": i_safe_margin.astype(np.float32),
        "near_v_boundary": near_v.astype(np.bool_),
        "near_i_boundary": near_i.astype(np.bool_),
        "vmin_bin": np.digitize(vmin, vmin_bins[1:-1]).astype(np.int64),
        "vmax_bin": np.digitize(vmax, vmax_bins[1:-1]).astype(np.int64),
        "loading_bin": np.digitize(loading, loading_bins[1:-1]).astype(np.int64),
    }

    legends = {
        "joint_class": [
            "V_safe_I_safe",
            "V_unsafe_I_safe",
            "V_safe_I_unsafe",
            "V_unsafe_I_unsafe",
        ],
        "vmin_bin": [
            "(-inf,0.88)", "[0.88,0.90)", "[0.90,0.93)", "[0.93,0.95)",
            "[0.95,0.97)", "[0.97,1.00)", "[1.00,+inf)",
        ],
        "vmax_bin": [
            "(-inf,1.00)", "[1.00,1.03)", "[1.03,1.05)", "[1.05,1.07)",
            "[1.07,1.10)", "[1.10,1.12)", "[1.12,+inf)",
        ],
        "loading_bin": [
            "(-inf,80%)", "[80%,90%)", "[90%,100%)", "[100%,110%)",
            "[110%,130%)", "[130%,150%)", "[150%,180%)", "[180%,+inf)",
        ],
    }
    return cls, legends


def print_classification(pool, cls, legends, devices):
    n = len(pool["Y_V"])

    print("\n================ 节点净PQ原始池统计 ================")
    print("唯一可控节点:")
    for name, typ, bus in zip(
        devices["ctrl_names"], devices["ctrl_types"], devices["ctrl_nodes"]
    ):
        print(f"  {name:5s}  type={typ:4s}  Bus={int(bus)+1}")

    print(f"\n样本总数: {n}")
    print(f"电压范围: [{cls['vmin'].min():.4f}, {cls['vmax'].max():.4f}] p.u.")
    print(
        f"最大支路 loading 范围: "
        f"[{cls['max_loading_percent'].min():.2f}%, "
        f"{cls['max_loading_percent'].max():.2f}%]"
    )

    print("\n[联合安全状态]")
    for i, label in enumerate(legends["joint_class"]):
        c = int(np.sum(cls["joint_class"] == i))
        print(f"{label:24s}: {c:6d} ({100.0*c/n:6.2f}%)")

    for key, title in [
        ("vmin_bin", "[Vmin 分段]"),
        ("vmax_bin", "[Vmax 分段]"),
        ("loading_bin", "[最大支路 loading 分段]"),
    ]:
        print(f"\n{title}")
        for i, label in enumerate(legends[key]):
            c = int(np.sum(cls[key] == i))
            print(f"{label:18s}: {c:6d} ({100.0*c/n:6.2f}%)")

    nv = int(cls["near_v_boundary"].sum())
    ni = int(cls["near_i_boundary"].sum())
    nb = int((cls["near_v_boundary"] & cls["near_i_boundary"]).sum())

    print("\n[边界邻域]")
    print(f"|V安全裕度| <= 0.01 p.u.: {nv:6d} ({100.0*nv/n:6.2f}%)")
    print(f"|I裕度| <= 0.10       : {ni:6d} ({100.0*ni/n:6.2f}%)")
    print(f"同时接近 V/I 边界      : {nb:6d} ({100.0*nb/n:6.2f}%)")

    print("\n[24h 样本]")
    hour_count = np.bincount(pool["hour"], minlength=24)
    for h, c in enumerate(hour_count):
        print(f"hour={h:02d}: {int(c):5d} ({100.0*c/n:6.2f}%)")

    print("====================================================\n")


# =========================================================
# 10. 保存
# =========================================================
def save_dataset(pool, cls, legends, Pload_24h, Qload_24h, Ppv_24h, devices):
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, SAVE_NAME)

    dataset = {
        "X": torch.tensor(pool["X"], dtype=torch.float32),
        "Y_V": torch.tensor(pool["Y_V"], dtype=torch.float32),
        "Y_I": torch.tensor(pool["Y_I"], dtype=torch.float32),
        "hour": torch.tensor(pool["hour"], dtype=torch.long),

        "Pload": torch.tensor(pool["Pload"], dtype=torch.float32),
        "Qload": torch.tensor(pool["Qload"], dtype=torch.float32),
        "pv_p": torch.tensor(pool["pv_p"], dtype=torch.float32),
        "pv_q": torch.tensor(pool["pv_q"], dtype=torch.float32),
        "ess_p": torch.tensor(pool["ess_p"], dtype=torch.float32),
        "ess_q": torch.tensor(pool["ess_q"], dtype=torch.float32),
        "qdev_q": torch.tensor(pool["qdev_q"], dtype=torch.float32),

        # 可控节点净注入及其当样本上下文下的可行域边界
        "ctrl_pnet": torch.tensor(pool["ctrl_pnet"], dtype=torch.float32),
        "ctrl_qnet": torch.tensor(pool["ctrl_qnet"], dtype=torch.float32),
        "ctrl_pnet_min": torch.tensor(pool["ctrl_pnet_min"], dtype=torch.float32),
        "ctrl_pnet_max": torch.tensor(pool["ctrl_pnet_max"], dtype=torch.float32),
        "ctrl_qnet_min": torch.tensor(pool["ctrl_qnet_min"], dtype=torch.float32),
        "ctrl_qnet_max": torch.tensor(pool["ctrl_qnet_max"], dtype=torch.float32),

        "Pload_24h": torch.tensor(Pload_24h, dtype=torch.float32),
        "Qload_24h": torch.tensor(Qload_24h, dtype=torch.float32),
        "Ppv_24h": torch.tensor(Ppv_24h, dtype=torch.float32),

        "edge_list": [[int(r[0]), int(r[1])] for r in RADIAL_BRANCHES],
        "branch_full": [
            [int(r[0]), int(r[1]), float(r[2]), float(r[3])]
            for r in RADIAL_BRANCHES
        ],

        "pv_nodes": torch.tensor(devices["pv_nodes"], dtype=torch.long),
        "S_pv_mva": torch.tensor(devices["S_pv_mva"], dtype=torch.float32),
        "ess_nodes": torch.tensor(devices["ess_nodes"], dtype=torch.long),
        "S_ess_mva": torch.tensor(devices["S_ess_mva"], dtype=torch.float32),
        "q_device_names": devices["q_device_names"],
        "q_device_nodes": torch.tensor(devices["q_device_nodes"], dtype=torch.long),
        "q_device_min": torch.tensor(devices["q_device_min"], dtype=torch.float32),
        "q_device_max": torch.tensor(devices["q_device_max"], dtype=torch.float32),

        "ctrl_nodes": torch.tensor(devices["ctrl_nodes"], dtype=torch.long),
        "ctrl_names": devices["ctrl_names"],
        "ctrl_types": devices["ctrl_types"],

        "load_group_id": torch.tensor(devices["load_group_id"], dtype=torch.long),
        "load_local_weight": torch.tensor(
            devices["load_local_weight"], dtype=torch.float32
        ),

        "classification": {
            k: torch.tensor(v)
            for k, v in cls.items()
        },
        "classification_legends": legends,

        "stats": pool["stats"],
        "base_config": {
            "slack_vm_pu": SLACK_VM_PU,
            "line_max_i_ka": UNIFORM_MAX_I_KA,
            "load_profile_peak": LOAD_PROFILE_PEAK,
            "load_global_jitter": LOAD_GLOBAL_JITTER,
            "load_group_jitter": LOAD_GROUP_JITTER,
            "load_local_jitter": LOAD_LOCAL_JITTER,
            "pf_global_jitter": PF_GLOBAL_JITTER,
            "pf_group_jitter": PF_GROUP_JITTER,
            "pv_global_jitter": PV_GLOBAL_JITTER,
            "pv_site_jitter": PV_SITE_JITTER,
            "ess_p_max_fraction": ESS_P_MAX_FRACTION,
            "sampling": (
                "per-hour Latin Hypercube with low-rank correlated load/PV uncertainty "
                "and nodal net P-Q control-region sampling"
            ),
            "one_reactive_device_per_controllable_node": True,
        },
    }

    torch.save(dataset, save_path)
    return save_path


def save_csv(pool, cls, legends):
    import csv

    csv_path = Path(DATA_DIR) / CSV_NAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index", "hour", "vmin", "vmax", "max_loading_percent",
                "v_margin", "i_safe_margin", "joint_class",
                "near_v_boundary", "near_i_boundary",
            ]
        )

        for i in range(len(pool["Y_V"])):
            writer.writerow(
                [
                    i,
                    int(pool["hour"][i]),
                    float(cls["vmin"][i]),
                    float(cls["vmax"][i]),
                    float(cls["max_loading_percent"][i]),
                    float(cls["v_margin"][i]),
                    float(cls["i_safe_margin"][i]),
                    legends["joint_class"][int(cls["joint_class"][i])],
                    int(cls["near_v_boundary"][i]),
                    int(cls["near_i_boundary"][i]),
                ]
            )

    return str(csv_path)


# =========================================================
# 11. 主程序
# =========================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    Pload_24h, Qload_24h, Ppv_24h, devices = prepare_profiles_and_devices()
    net = setup_static_ieee33_env()
    idx = add_static_devices(net, devices)

    pool = generate_pool(
        net,
        idx,
        Pload_24h,
        Qload_24h,
        Ppv_24h,
        devices,
        rng,
    )

    cls, legends = classify_pool(pool)
    print_classification(pool, cls, legends, devices)

    save_path = save_dataset(
        pool,
        cls,
        legends,
        Pload_24h,
        Qload_24h,
        Ppv_24h,
        devices,
    )
    csv_path = save_csv(pool, cls, legends)

    print("==============================================")
    print("节点净注入 P-Q 可行域 50k 母池生成完成")
    print(f"PT:  {save_path}")
    print(f"CSV: {csv_path}")
    print("本脚本未按电压/电流安全状态过滤任何已收敛样本。")
    print("后续应重新统计四类样本数量，再决定最终20k分类抽样比例。")
    print("==============================================")
