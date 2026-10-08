import os
import time
import warnings

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
SAVE_NAME = "ieee33_static_vvo_raw_classified_pool_50k.pt"

POOL_SAMPLES = 50000
TARGET_SAMPLES = 20000  # 仅保留为后续筛选目标，本脚本不执行筛选
BATCH_SIZE = 500
MAX_ATTEMPT_MULTIPLIER = 40

# 数据生成护栏应宽于实际优化电压约束，用于保留边界和较强非线性运行点
V_GUARD_MIN = 0.88
V_GUARD_MAX = 1.12
I_MARGIN_MAX_KEEP = 0.80

# case33bw 原始数据没有给出支路热稳额定值；0.20 kA 仅作为本文电流裕度标签的研究设定。
# 它不再与过大的 2~3 倍基础负荷组合使用。
UNIFORM_MAX_I_KA = 0.20

# 保留你的 VVO 研究设定。标准 case33bw 的平衡节点电压为 1.0 p.u.；
# 若需要完全复现标准基准，可改成 1.00。
SLACK_VM_PU = 1.03

# 24h 原始曲线只保留“形状”，统一归一化后将峰值控制在 1.30 倍标准 IEEE33 基础负荷。
LOAD_PROFILE_PEAK = 1.30

# 运行点覆盖增强：P/Q 独立扰动，避免节点功率因数被锁死
P_LOAD_JITTER = 0.15
Q_LOAD_JITTER = 0.20

# 每个 PV 在同一小时独立受云量扰动
PV_P_JITTER = 0.25

# ESS 同时采样有功和无功，并严格满足 P^2 + Q^2 <= S^2
SAMPLE_ESS_ACTIVE = True
ESS_P_MAX_FRACTION = 0.60

# 节点级平滑时序差异幅度。IEEE33 本身没有住宅/商业/工业负荷类型标签，
# 因此采用确定性的节点级时序差异来打破所有节点共享同一条 24h 曲线的问题
NODE_PROFILE_DIVERSITY = 0.12


# =========================================================
# 1. IEEE33 静态径向拓扑，0-based 编号
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
# 2. 负荷、PV、ESS、SVG/SVC/SC 配置
# =========================================================
def prepare_profiles_and_devices():
    # 标准 IEEE 33 节点（Baran-Wu / MATPOWER case33bw）基础负荷。
    # Bus1 为平衡节点且基础负荷为 0；Bus33 为 60 kW / 40 kvar。
    pload_kw = np.array(
        [
            0, 100, 90, 120, 60, 60, 200, 200, 60, 60, 45,
            60, 60, 120, 60, 60, 60, 90, 90, 90, 90, 90,
            90, 420, 420, 60, 60, 60, 120, 200, 150, 210, 60
        ],
        dtype=np.float64
    )

    qload_kvar = np.array(
        [
            0, 60, 40, 80, 30, 20, 100, 100, 20, 20, 30,
            35, 35, 80, 10, 20, 20, 40, 40, 40, 40, 40,
            50, 200, 200, 25, 25, 20, 70, 600, 70, 100, 40
        ],
        dtype=np.float64
    )

    # 标准 case33bw 基础总负荷校验。
    assert abs(pload_kw.sum() / 1000.0 - 3.715) < 1e-12
    assert abs(qload_kvar.sum() / 1000.0 - 2.300) < 1e-12

    pload_ratio = np.array(
        [
            2.15, 2.3, 1.2, 2.35, 2.35, 2.6, 3.0, 2.25,
            2.7, 1.8, 1.35, 1.2, 1.15, 1.1, 1.35, 1.45,
            1.5, 1.65, 1.9, 2.0, 1.2, 1.8, 1.85, 1.8
        ],
        dtype=np.float64
    )

    qload_ratio = np.array(
        [
            1.15, 1.3, 0.8, 1.35, 1.35, 1.6, 2.0, 1.25,
            2.1, 1.4, 1.15, 1.0, 0.9, 1.0, 1.2, 1.25,
            1.3, 1.45, 1.2, 1.0, 1.0, 1.4, 1.55, 1.4
        ],
        dtype=np.float64
    )

    # 原始倍率会把标准 IEEE33 基础负荷放大到最高 11.145 MW。
    # 保留 P/Q 日曲线形状，分别归一化到相同的 1.30 倍峰值。
    pload_ratio = pload_ratio / pload_ratio.max() * LOAD_PROFILE_PEAK
    qload_ratio = qload_ratio / qload_ratio.max() * LOAD_PROFILE_PEAK

    # 进一步打破“所有节点共享同一条日曲线”的低维结构。
    # 每个节点使用固定的平滑幅值/相位差异，不引入虚假的负荷类型标签。
    hours = np.arange(24, dtype=np.float64)[None, :]
    nodes = np.arange(33, dtype=np.float64)[:, None]

    p_amp = 0.04 + NODE_PROFILE_DIVERSITY * ((nodes * 7.0) % 17.0) / 16.0
    q_amp = 0.04 + NODE_PROFILE_DIVERSITY * ((nodes * 11.0) % 19.0) / 18.0
    p_phase = ((nodes * 5.0) % 9.0) - 4.0
    q_phase = ((nodes * 7.0) % 11.0) - 5.0

    p_profile_mod = 1.0 + p_amp * np.sin(2.0 * np.pi * (hours - p_phase) / 24.0)
    q_profile_mod = 1.0 + q_amp * np.sin(2.0 * np.pi * (hours - q_phase) / 24.0)

    Pload_24h = (
        pload_kw[:, None] * pload_ratio[None, :] * p_profile_mod
    ) / 1000.0
    Qload_24h = (
        qload_kvar[:, None] * qload_ratio[None, :] * q_profile_mod
    ) / 1000.0

    pv_shape_raw = np.array(
        [
            0, 0, 0, 0, 0, 0.25, 0.49, 0.52, 0.75, 0.9, 1.0, 1.25,
            1.1, 0.9, 0.79, 0.76, 0.75, 0.55, 0.35, 0.25, 0, 0, 0, 0
        ],
        dtype=np.float64
    )
    pv_shape = pv_shape_raw / pv_shape_raw.max()

    devices = {
        # 图片配置：PV 节点 8,15,23,30；容量 0.5,0.8,1.0,1.2 MVA
        "pv_nodes": np.array([7, 14, 22, 29], dtype=np.int64),
        "S_pv_mva": np.array([0.5, 0.8, 1.0, 1.2], dtype=np.float64),

        # ESS 节点 15,30；容量 0.8 MVA
        "ess_nodes": np.array([14, 29], dtype=np.int64),
        "S_ess_mva": np.array([0.8, 0.8], dtype=np.float64),

        # 顺序严格对齐 MATLAB: mpc.svg.nodes = [8,4,17,21,2]
        # 其中节点 8 为 SVC1，节点 4 为 SVG1，节点 17 为 SVG2，节点 21 为 SVC2，节点 2 为 SC
        "q_device_names": ["SVC1", "SVG1", "SVG2", "SVC2", "SC"],
        "q_device_nodes": np.array([7, 3, 16, 20, 1], dtype=np.int64),
        "q_device_min": np.array([-1.0, -0.5, -0.8, -2.0, 0.0], dtype=np.float64),
        "q_device_max": np.array([1.0, 0.5, 0.8, 2.0, 0.4], dtype=np.float64),
    }

    # 为适配后续无功优化，PV 有功按 24h 轨迹固定到各自容量的 80% 峰值附近
    # 若后续 MATLAB 中仍使用 [1100,1100,1100,1100] kVA，应先将 MATLAB 改为 [500,800,1000,1200] kVA
    Ppv_24h = pv_shape[:, None] * (0.8 * devices["S_pv_mva"])[None, :]

    return Pload_24h, Qload_24h, Ppv_24h, devices


# =========================================================
# 3. pandapower 静态 IEEE33 网络
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
        pp.create_sgen(
            net,
            bus=int(bus),
            p_mw=0.0,
            q_mvar=0.0,
            name=f"PV@{bus + 1}",
        )
        for bus in devices["pv_nodes"]
    ]

    ess_idx = [
        pp.create_sgen(
            net,
            bus=int(bus),
            p_mw=0.0,
            q_mvar=0.0,
            name=f"ESS@{bus + 1}",
        )
        for bus in devices["ess_nodes"]
    ]

    qdev_idx = [
        pp.create_sgen(
            net,
            bus=int(bus),
            p_mw=0.0,
            q_mvar=0.0,
            name=f"{name}@{bus + 1}",
        )
        for name, bus in zip(devices["q_device_names"], devices["q_device_nodes"])
    ]

    return {
        "pv_sgen_idx": np.array(pv_idx, dtype=np.int64),
        "ess_sgen_idx": np.array(ess_idx, dtype=np.int64),
        "qdev_sgen_idx": np.array(qdev_idx, dtype=np.int64),
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

            if not np.all(np.isfinite(vm)):
                continue
            if not np.all(np.isfinite(loading)):
                continue

            return True

        except Exception:
            try:
                pp.reset_results(net)
            except Exception:
                pass

    return False


# =========================================================
# 5. 采样函数
# =========================================================
def sample_by_mode(rng, low, high, mode):
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    if mode == 0:
        return rng.uniform(low, high)

    if mode == 1:
        # 偏向无功注入，制造高电压样本
        return rng.uniform(0.50 * high, high)

    if mode == 2:
        # 偏向无功吸收，制造低电压样本
        return rng.uniform(low, 0.50 * low)

    if mode == 3:
        # 中间运行区域
        center = 0.5 * (low + high)
        width = 0.25 * (high - low)
        return np.clip(center + rng.uniform(-width, width), low, high)

    if mode == 4:
        # 设备边界区域
        side = rng.integers(0, 2, size=len(low))
        val_low = rng.uniform(low, 0.70 * low)
        val_high = rng.uniform(0.70 * high, high)
        return np.where(side == 0, val_low, val_high)

    return rng.uniform(low, high)


def sample_operating_point(hour, rng, Pload_24h, Qload_24h, Ppv_24h, devices):
    # 0随机；1高电压倾向；2低电压倾向；3中间；4设备边界
    mode = int(rng.choice([0, 1, 2, 3, 4], p=[0.38, 0.20, 0.20, 0.10, 0.12]))

    # P/Q 独立节点扰动，允许功率因数和空间功率分布发生变化
    p_scale = rng.uniform(1.0 - P_LOAD_JITTER, 1.0 + P_LOAD_JITTER, size=33)
    q_scale = rng.uniform(1.0 - Q_LOAD_JITTER, 1.0 + Q_LOAD_JITTER, size=33)

    if mode == 1:
        p_scale *= rng.uniform(0.88, 1.00)
        q_scale *= rng.uniform(0.88, 1.00)
    elif mode == 2:
        p_scale *= rng.uniform(1.05, 1.18)
        q_scale *= rng.uniform(1.05, 1.18)
    elif mode == 4:
        p_scale *= rng.uniform(0.88, 1.18)
        q_scale *= rng.uniform(0.88, 1.18)

    Pload = Pload_24h[:, hour] * p_scale
    Qload = Qload_24h[:, hour] * q_scale

    # 4 个 PV 在相同太阳时序中心附近独立扰动
    pv_p = Ppv_24h[hour].copy()
    pv_p *= rng.uniform(
        1.0 - PV_P_JITTER,
        1.0 + PV_P_JITTER,
        size=len(pv_p),
    )

    if mode == 1:
        pv_p *= rng.uniform(1.00, 1.08, size=len(pv_p))
    elif mode == 2:
        pv_p *= rng.uniform(0.90, 1.00, size=len(pv_p))

    pv_p = np.clip(pv_p, 0.0, 0.98 * devices["S_pv_mva"])

    pv_q_max = np.sqrt(np.maximum(devices["S_pv_mva"] ** 2 - pv_p ** 2, 0.0))
    pv_q = sample_by_mode(rng, -pv_q_max, pv_q_max, mode)

    if SAMPLE_ESS_ACTIVE:
        ess_p = rng.uniform(
            -ESS_P_MAX_FRACTION,
            ESS_P_MAX_FRACTION,
            size=len(devices["ess_nodes"]),
        ) * devices["S_ess_mva"]
    else:
        ess_p = np.zeros(len(devices["ess_nodes"]), dtype=np.float64)

    ess_q_max = np.sqrt(np.maximum(devices["S_ess_mva"] ** 2 - ess_p ** 2, 0.0))
    ess_q = sample_by_mode(rng, -ess_q_max, ess_q_max, mode)

    qdev_q = sample_by_mode(
        rng,
        devices["q_device_min"],
        devices["q_device_max"],
        mode,
    )

    return mode, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q


# =========================================================
# 6. 样本施加与 X 构造
# =========================================================
def apply_operating_point(net, idx, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q):
    net.load.loc[:, "p_mw"] = Pload
    net.load.loc[:, "q_mvar"] = Qload

    net.sgen.loc[idx["pv_sgen_idx"], "p_mw"] = pv_p
    net.sgen.loc[idx["pv_sgen_idx"], "q_mvar"] = pv_q

    net.sgen.loc[idx["ess_sgen_idx"], "p_mw"] = ess_p
    net.sgen.loc[idx["ess_sgen_idx"], "q_mvar"] = ess_q

    net.sgen.loc[idx["qdev_sgen_idx"], "p_mw"] = 0.0
    net.sgen.loc[idx["qdev_sgen_idx"], "q_mvar"] = qdev_q

    net.line.loc[:, "in_service"] = True


def build_net_injection(Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q, devices):
    P_net = -Pload.copy()
    Q_net = -Qload.copy()

    for k, bus in enumerate(devices["pv_nodes"]):
        P_net[bus] += pv_p[k]
        Q_net[bus] += pv_q[k]

    for k, bus in enumerate(devices["ess_nodes"]):
        P_net[bus] += ess_p[k]
        Q_net[bus] += ess_q[k]

    for k, bus in enumerate(devices["q_device_nodes"]):
        Q_net[bus] += qdev_q[k]

    X = np.zeros((33, 2), dtype=np.float32)
    X[:, 0] = P_net.astype(np.float32)
    X[:, 1] = Q_net.astype(np.float32)

    return X


# =========================================================
# 7. 数据池生成
# =========================================================
def generate_pool(net, idx, Pload_24h, Qload_24h, Ppv_24h, devices, rng, pool_samples):
    """
    生成未经电压/电流安全筛选的原始潮流样本池。

    仅排除：
    1) AC 潮流不收敛；
    2) 潮流结果出现 NaN/Inf。

    不再依据 Vmin/Vmax 或线路 loading 删除任何已收敛样本。
    """
    X_list, YV_list, YI_list = [], [], []
    hour_list, mode_list = [], []
    pload_list, qload_list = [], []
    pv_p_list, pv_q_list = [], []
    ess_p_list, ess_q_list = [], []
    qdev_q_list = []

    attempt = 0
    pf_fail = 0
    nonfinite_fail = 0
    max_attempt = pool_samples * MAX_ATTEMPT_MULTIPLIER

    hour_attempt = np.zeros(24, dtype=np.int64)
    hour_accept = np.zeros(24, dtype=np.int64)
    hour_pf_fail = np.zeros(24, dtype=np.int64)
    hour_nonfinite = np.zeros(24, dtype=np.int64)

    pbar = tqdm(total=pool_samples, desc="生成未筛选AC潮流样本池", unit="样本")
    t0 = time.perf_counter()

    while len(X_list) < pool_samples and attempt < max_attempt:
        for _ in range(BATCH_SIZE):
            if len(X_list) >= pool_samples or attempt >= max_attempt:
                break

            hour = int(attempt % 24)
            attempt += 1
            hour_attempt[hour] += 1

            mode, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q = sample_operating_point(
                hour, rng, Pload_24h, Qload_24h, Ppv_24h, devices
            )

            apply_operating_point(
                net, idx, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q
            )

            if not run_powerflow_with_fallback(net):
                pf_fail += 1
                hour_pf_fail[hour] += 1
                continue

            YV = net.res_bus.vm_pu.values.astype(np.float32)
            loading = net.res_line.loading_percent.values.astype(np.float32)
            YI = loading / 100.0 - 1.0

            if not np.all(np.isfinite(YV)) or not np.all(np.isfinite(YI)):
                nonfinite_fail += 1
                hour_nonfinite[hour] += 1
                continue

            X_list.append(
                build_net_injection(
                    Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q, devices
                )
            )
            YV_list.append(YV)
            YI_list.append(YI)

            hour_list.append(hour)
            mode_list.append(mode)
            pload_list.append(Pload.astype(np.float32))
            qload_list.append(Qload.astype(np.float32))
            pv_p_list.append(pv_p.astype(np.float32))
            pv_q_list.append(pv_q.astype(np.float32))
            ess_p_list.append(ess_p.astype(np.float32))
            ess_q_list.append(ess_q.astype(np.float32))
            qdev_q_list.append(qdev_q.astype(np.float32))
            hour_accept[hour] += 1
            pbar.update(1)

        if attempt % 2400 == 0:
            pbar.set_postfix(
                {
                    "attempt": attempt,
                    "pf_fail": pf_fail,
                    "nonfinite": nonfinite_fail,
                }
            )

    pbar.close()

    if len(X_list) == 0:
        raise RuntimeError("没有生成任何有效AC潮流样本。")

    if len(X_list) < pool_samples:
        print(f"[警告] 收敛样本不足：目标 {pool_samples}，实际 {len(X_list)}。")

    hour_accept_rate = np.divide(
        hour_accept, np.maximum(hour_attempt, 1), dtype=np.float64
    )

    stats = {
        "attempt_count": int(attempt),
        "success_count": int(len(X_list)),
        "powerflow_fail_count": int(pf_fail),
        "nonfinite_fail_count": int(nonfinite_fail),
        "wall_time_sec": float(time.perf_counter() - t0),
        "hour_attempt_count": hour_attempt.tolist(),
        "hour_accept_count": hour_accept.tolist(),
        "hour_pf_fail_count": hour_pf_fail.tolist(),
        "hour_nonfinite_count": hour_nonfinite.tolist(),
        "hour_accept_rate": hour_accept_rate.tolist(),
    }

    print()
    print("每小时AC潮流收敛统计:")
    for h in range(24):
        print(
            f"  h={h:02d}: attempt={hour_attempt[h]:5d}, "
            f"accept={hour_accept[h]:5d}, "
            f"pf_fail={hour_pf_fail[h]:4d}, "
            f"nonfinite={hour_nonfinite[h]:3d}, "
            f"rate={100.0 * hour_accept_rate[h]:6.2f}%"
        )

    return {
        "X": np.stack(X_list),
        "Y_V": np.stack(YV_list),
        "Y_I": np.stack(YI_list),
        "hour": np.array(hour_list, dtype=np.int64),
        "mode": np.array(mode_list, dtype=np.int64),
        "Pload": np.stack(pload_list),
        "Qload": np.stack(qload_list),
        "pv_p": np.stack(pv_p_list),
        "pv_q": np.stack(pv_q_list),
        "ess_p": np.stack(ess_p_list),
        "ess_q": np.stack(ess_q_list),
        "qdev_q": np.stack(qdev_q_list),
        "stats": stats,
    }

# =========================================================
# 8. 电压均匀 + 电流越限/非越限均衡重采样
# =========================================================
def _quantile_bins(values, n_bins=4):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.allclose(values, values[0]):
        return np.zeros(len(values), dtype=np.int64)

    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int64)

    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def _coverage_weights(indices, descriptors, current_sign, modes):
    if len(indices) == 0:
        return np.empty(0, dtype=np.float64)

    weights = np.ones(len(indices), dtype=np.float64)

    # 对每一个边际维度分别提高稀有区间权重，避免高维交叉分箱过度稀疏
    categorical = [current_sign, modes]
    for arr in categorical:
        local = arr[indices]
        unique, counts = np.unique(local, return_counts=True)
        count_map = dict(zip(unique.tolist(), counts.tolist()))
        weights += np.array([1.0 / count_map[v] for v in local]) * len(indices)

    for desc in descriptors:
        local = desc[indices]
        b = _quantile_bins(local, n_bins=4)
        unique, counts = np.unique(b, return_counts=True)
        count_map = dict(zip(unique.tolist(), counts.tolist()))
        weights += np.array([1.0 / count_map[v] for v in b]) * len(indices)

    weights = np.maximum(weights, 1e-12)
    return weights / weights.sum()


def balanced_resample(pool, target_size, rng):
    YV = pool["Y_V"]
    YI = pool["Y_I"]
    hours = pool["hour"]
    modes = pool["mode"]

    v_min = YV.min(axis=1)
    v_max = YV.max(axis=1)
    v_dev = np.maximum(np.abs(v_min - 1.0), np.abs(v_max - 1.0))
    i_max = YI.max(axis=1)
    current_sign = (i_max > 0).astype(np.int64)

    total_p_load = pool["Pload"].sum(axis=1)
    total_q_load = pool["Qload"].sum(axis=1)
    total_pv = pool["pv_p"].sum(axis=1)
    pv_penetration = total_pv / np.maximum(total_p_load, 1e-6)

    # 节点净无功活跃度，能够反映无功设备和负荷组合的空间变化
    q_activity = np.abs(pool["X"][:, :, 1]).sum(axis=1)

    descriptors = [
        total_p_load,
        pv_penetration,
        total_q_load,
        q_activity,
        i_max,
        v_dev,
    ]

    # 最终 20000 样本按 24 小时近似等额分配
    base_quota = target_size // 24
    remainder = target_size % 24
    hour_quota = np.full(24, base_quota, dtype=np.int64)
    hour_quota[:remainder] += 1

    selected = []

    for h in range(24):
        cand = np.where(hours == h)[0]
        quota = int(hour_quota[h])

        if len(cand) == 0:
            raise RuntimeError(
                f"hour={h} 数据池中没有有效样本。请先根据每小时接受率检查物理配置或 guard。"
            )

        prob = _coverage_weights(
            cand,
            descriptors,
            current_sign,
            modes,
        )

        if len(cand) >= quota:
            chosen = rng.choice(cand, size=quota, replace=False, p=prob)
        else:
            print(
                f"[警告] hour={h:02d} 数据池仅 {len(cand)} 个样本，"
                f"低于最终配额 {quota}，需要少量有放回补采。"
            )
            chosen = cand.tolist()
            need = quota - len(cand)
            extra = rng.choice(cand, size=need, replace=True, p=prob)
            chosen = np.concatenate([np.asarray(chosen, dtype=np.int64), extra])

        selected.extend(np.asarray(chosen, dtype=np.int64).tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)

    out = {}
    for k, v in pool.items():
        if isinstance(v, np.ndarray) and len(v) == len(YV):
            out[k] = v[selected]
        else:
            out[k] = v

    out_vmin = out["Y_V"].min(axis=1)
    out_vmax = out["Y_V"].max(axis=1)
    out_vdev = np.maximum(np.abs(out_vmin - 1.0), np.abs(out_vmax - 1.0))
    out_imax = out["Y_I"].max(axis=1)

    out["balance_info"] = {
        "target_size": int(target_size),
        "strategy": (
            "24h near-equal quota + marginal inverse-frequency coverage weighting "
            "over load, PV penetration, Q level, Q activity, max current and voltage deviation"
        ),
        "selected_pos_margin_count": int((out_imax > 0).sum()),
        "selected_nonpos_margin_count": int((out_imax <= 0).sum()),
        "hour_count": np.bincount(out["hour"], minlength=24).tolist(),
        "total_p_load_range_mw": [
            float(out["Pload"].sum(axis=1).min()),
            float(out["Pload"].sum(axis=1).max()),
        ],
        "pv_penetration_range": [
            float(
                (
                    out["pv_p"].sum(axis=1)
                    / np.maximum(out["Pload"].sum(axis=1), 1e-6)
                ).min()
            ),
            float(
                (
                    out["pv_p"].sum(axis=1)
                    / np.maximum(out["Pload"].sum(axis=1), 1e-6)
                ).max()
            ),
        ],
        "voltage_deviation_range": [
            float(out_vdev.min()),
            float(out_vdev.max()),
        ],
        "max_current_margin_range": [
            float(out_imax.min()),
            float(out_imax.max()),
        ],
    }

    return out



# =========================================================
# 8.5 原始样本分类统计（只统计，不筛选）
# =========================================================
def classify_raw_pool(pool):
    YV = pool["Y_V"]
    YI = pool["Y_I"]

    vmin = YV.min(axis=1)
    vmax = YV.max(axis=1)
    imax = YI.max(axis=1)
    loading_max = (imax + 1.0) * 100.0

    v_low_violate = vmin < 0.95
    v_high_violate = vmax > 1.05
    i_violate = imax > 0.0

    voltage_class = (
        v_low_violate.astype(np.int64)
        + 2 * v_high_violate.astype(np.int64)
    )

    voltage_unsafe = v_low_violate | v_high_violate
    joint_class = (
        voltage_unsafe.astype(np.int64)
        + 2 * i_violate.astype(np.int64)
    )

    v_margin_low = vmin - 0.95
    v_margin_high = 1.05 - vmax
    v_margin = np.minimum(v_margin_low, v_margin_high)
    i_safe_margin = -imax

    near_v_boundary = np.abs(v_margin) <= 0.01
    near_i_boundary = np.abs(imax) <= 0.10

    vmin_bins = np.array(
        [-np.inf, 0.88, 0.90, 0.93, 0.95, 0.97, 1.00, np.inf]
    )
    vmax_bins = np.array(
        [-np.inf, 1.00, 1.03, 1.05, 1.07, 1.10, 1.12, np.inf]
    )
    loading_bins = np.array(
        [-np.inf, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 180.0, np.inf]
    )

    vmin_bin = np.digitize(vmin, vmin_bins[1:-1], right=False)
    vmax_bin = np.digitize(vmax, vmax_bins[1:-1], right=False)
    loading_bin = np.digitize(loading_max, loading_bins[1:-1], right=False)

    classification = {
        "vmin": vmin.astype(np.float32),
        "vmax": vmax.astype(np.float32),
        "imax_margin": imax.astype(np.float32),
        "max_loading_percent": loading_max.astype(np.float32),
        "v_margin": v_margin.astype(np.float32),
        "i_safe_margin": i_safe_margin.astype(np.float32),
        "voltage_class": voltage_class.astype(np.int64),
        "joint_class": joint_class.astype(np.int64),
        "near_v_boundary": near_v_boundary.astype(np.bool_),
        "near_i_boundary": near_i_boundary.astype(np.bool_),
        "vmin_bin": vmin_bin.astype(np.int64),
        "vmax_bin": vmax_bin.astype(np.int64),
        "loading_bin": loading_bin.astype(np.int64),
    }

    legends = {
        "voltage_class": {
            0: "V_safe",
            1: "low_voltage_only",
            2: "high_voltage_only",
            3: "low_and_high_voltage",
        },
        "joint_class": {
            0: "V_safe_I_safe",
            1: "V_unsafe_I_safe",
            2: "V_safe_I_unsafe",
            3: "V_unsafe_I_unsafe",
        },
        "vmin_bin": [
            "(-inf,0.88)",
            "[0.88,0.90)",
            "[0.90,0.93)",
            "[0.93,0.95)",
            "[0.95,0.97)",
            "[0.97,1.00)",
            "[1.00,+inf)",
        ],
        "vmax_bin": [
            "(-inf,1.00)",
            "[1.00,1.03)",
            "[1.03,1.05)",
            "[1.05,1.07)",
            "[1.07,1.10)",
            "[1.10,1.12)",
            "[1.12,+inf)",
        ],
        "loading_bin": [
            "(-inf,80%)",
            "[80%,90%)",
            "[90%,100%)",
            "[100%,110%)",
            "[110%,130%)",
            "[130%,150%)",
            "[150%,180%)",
            "[180%,+inf)",
        ],
    }

    return classification, legends


def _count_rows(values, labels, total):
    rows = []
    for i, label in enumerate(labels):
        count = int(np.sum(values == i))
        rows.append((label, count, 100.0 * count / total))
    return rows


def print_classification_stats(pool, cls, legends):
    n = len(pool["Y_V"])

    print()
    print("================ 原始样本分类统计 ================")
    print(f"AC潮流收敛样本总数: {n}")
    print(f"电压总体范围: [{cls['vmin'].min():.4f}, {cls['vmax'].max():.4f}] p.u.")
    print(
        f"最大线路负载率范围: "
        f"[{cls['max_loading_percent'].min():.2f}%, "
        f"{cls['max_loading_percent'].max():.2f}%]"
    )

    print()
    print("[联合安全状态]")
    joint_labels = [legends["joint_class"][i] for i in range(4)]
    for label, count, ratio in _count_rows(cls["joint_class"], joint_labels, n):
        print(f"{label:24s}: {count:6d} ({ratio:6.2f}%)")

    print()
    print("[电压状态]")
    voltage_labels = [legends["voltage_class"][i] for i in range(4)]
    for label, count, ratio in _count_rows(cls["voltage_class"], voltage_labels, n):
        print(f"{label:24s}: {count:6d} ({ratio:6.2f}%)")

    print()
    print("[Vmin 分段]")
    for label, count, ratio in _count_rows(cls["vmin_bin"], legends["vmin_bin"], n):
        print(f"{label:18s}: {count:6d} ({ratio:6.2f}%)")

    print()
    print("[Vmax 分段]")
    for label, count, ratio in _count_rows(cls["vmax_bin"], legends["vmax_bin"], n):
        print(f"{label:18s}: {count:6d} ({ratio:6.2f}%)")

    print()
    print("[最大支路 loading 分段]")
    for label, count, ratio in _count_rows(
        cls["loading_bin"], legends["loading_bin"], n
    ):
        print(f"{label:18s}: {count:6d} ({ratio:6.2f}%)")

    nv = int(cls["near_v_boundary"].sum())
    ni = int(cls["near_i_boundary"].sum())
    nboth = int((cls["near_v_boundary"] & cls["near_i_boundary"]).sum())

    print()
    print("[边界邻域，仅统计]")
    print(f"|V安全裕度| <= 0.01 p.u.: {nv:6d} ({100.0*nv/n:6.2f}%)")
    print(f"|I裕度| <= 0.10       : {ni:6d} ({100.0*ni/n:6.2f}%)")
    print(f"同时接近V/I边界        : {nboth:6d} ({100.0*nboth/n:6.2f}%)")

    hour_count = np.bincount(pool["hour"], minlength=24)
    print()
    print("[24h 收敛样本数]")
    for h, c in enumerate(hour_count):
        print(f"hour={h:02d}: {int(c):5d} ({100.0*c/n:6.2f}%)")

    mode_count = np.bincount(pool["mode"], minlength=5)
    print()
    print("[采样 mode]")
    for m, c in enumerate(mode_count):
        print(f"mode={m}: {int(c):6d} ({100.0*c/n:6.2f}%)")

    print("====================================================")
    print()


def save_classification_csv(pool, cls, legends, out_csv):
    import csv

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    joint_names = legends["joint_class"]
    voltage_names = legends["voltage_class"]

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "hour",
                "mode",
                "vmin",
                "vmax",
                "max_loading_percent",
                "v_margin",
                "i_safe_margin",
                "voltage_class",
                "joint_class",
                "near_v_boundary",
                "near_i_boundary",
                "vmin_bin",
                "vmax_bin",
                "loading_bin",
            ]
        )

        for i in range(len(pool["Y_V"])):
            writer.writerow(
                [
                    i,
                    int(pool["hour"][i]),
                    int(pool["mode"][i]),
                    float(cls["vmin"][i]),
                    float(cls["vmax"][i]),
                    float(cls["max_loading_percent"][i]),
                    float(cls["v_margin"][i]),
                    float(cls["i_safe_margin"][i]),
                    voltage_names[int(cls["voltage_class"][i])],
                    joint_names[int(cls["joint_class"][i])],
                    int(cls["near_v_boundary"][i]),
                    int(cls["near_i_boundary"][i]),
                    legends["vmin_bin"][int(cls["vmin_bin"][i])],
                    legends["vmax_bin"][int(cls["vmax_bin"][i])],
                    legends["loading_bin"][int(cls["loading_bin"][i])],
                ]
            )

# =========================================================
# 9. 统计输出
# =========================================================
def print_dataset_stats(data, title):
    YV = data["Y_V"]
    YI = data["Y_I"]

    v_min = YV.min(axis=1)
    v_max = YV.max(axis=1)
    i_max = YI.max(axis=1)
    v_char = np.where((v_max - 1.0) >= (1.0 - v_min), v_max, v_min)

    print(f"\n================ {title} ================")
    print(f"样本数: {len(YV)}")
    print(f"节点电压总体范围: [{YV.min():.4f}, {YV.max():.4f}]")
    print(f"最小电压范围: [{v_min.min():.4f}, {v_min.max():.4f}]")
    print(f"最大电压范围: [{v_max.min():.4f}, {v_max.max():.4f}]")
    print(f"电压均匀化特征 v_char 范围: [{v_char.min():.4f}, {v_char.max():.4f}]")
    print(f"支路裕度总体范围: [{YI.min():.4f}, {YI.max():.4f}]")
    print(f"最大支路裕度范围: [{i_max.min():.4f}, {i_max.max():.4f}]")
    print(f"i_max > 0 样本比例: {(i_max > 0).mean() * 100:.2f}%")
    print(f"i_max <= 0 样本比例: {(i_max <= 0).mean() * 100:.2f}%")

    if "hour" in data:
        hour_count = np.bincount(data["hour"], minlength=24)
        print("24h 样本数:", hour_count.tolist())
        if np.any(hour_count == 0):
            print("[警告] 仍存在 0 样本小时:", np.where(hour_count == 0)[0].tolist())

    bins = np.linspace(V_GUARD_MIN, V_GUARD_MAX, 11)
    hist, _ = np.histogram(np.clip(v_char, V_GUARD_MIN, V_GUARD_MAX), bins=bins)
    print("v_char 十等分统计:", hist.tolist())

    if "stats" in data:
        print("生成统计:", data["stats"])

    if "balance_info" in data:
        print("均衡统计:", data["balance_info"])

    print("============================================================\n")


# =========================================================
# 10. 保存数据集
# =========================================================
def to_torch_dataset(data, Pload_24h, Qload_24h, Ppv_24h, devices, save_path, classification=None, legends=None):
    edge_list = [[int(r[0]), int(r[1])] for r in RADIAL_BRANCHES]
    branch_full = [[int(r[0]), int(r[1]), float(r[2]), float(r[3])] for r in RADIAL_BRANCHES]

    dataset = {
        "X": torch.tensor(data["X"], dtype=torch.float32),
        "Y_V": torch.tensor(data["Y_V"], dtype=torch.float32),
        "Y_I": torch.tensor(data["Y_I"], dtype=torch.float32),

        "hour": torch.tensor(data["hour"], dtype=torch.long),
        "mode": torch.tensor(data["mode"], dtype=torch.long),

        "Pload": torch.tensor(data["Pload"], dtype=torch.float32),
        "Qload": torch.tensor(data["Qload"], dtype=torch.float32),

        "pv_p": torch.tensor(data["pv_p"], dtype=torch.float32),
        "pv_q": torch.tensor(data["pv_q"], dtype=torch.float32),
        "ess_p": torch.tensor(data["ess_p"], dtype=torch.float32),
        "ess_q": torch.tensor(data["ess_q"], dtype=torch.float32),
        "qdev_q": torch.tensor(data["qdev_q"], dtype=torch.float32),

        "Pload_24h": torch.tensor(Pload_24h, dtype=torch.float32),
        "Qload_24h": torch.tensor(Qload_24h, dtype=torch.float32),
        "Ppv_24h": torch.tensor(Ppv_24h, dtype=torch.float32),

        "edge_list": edge_list,
        "branch_full": branch_full,

        "pv_nodes": torch.tensor(devices["pv_nodes"], dtype=torch.long),
        "S_pv_mva": torch.tensor(devices["S_pv_mva"], dtype=torch.float32),

        "ess_nodes": torch.tensor(devices["ess_nodes"], dtype=torch.long),
        "S_ess_mva": torch.tensor(devices["S_ess_mva"], dtype=torch.float32),

        "q_device_names": devices["q_device_names"],
        "q_device_nodes": torch.tensor(devices["q_device_nodes"], dtype=torch.long),
        "q_device_min": torch.tensor(devices["q_device_min"], dtype=torch.float32),
        "q_device_max": torch.tensor(devices["q_device_max"], dtype=torch.float32),

        "base_config": {
            "slack_vm_pu": SLACK_VM_PU,
            "line_max_i_ka": UNIFORM_MAX_I_KA,
            "v_guard_min": V_GUARD_MIN,
            "v_guard_max": V_GUARD_MAX,
            "i_margin_max_keep": I_MARGIN_MAX_KEEP,
            "standard_ieee33_base_load": True,
            "load_profile_peak": LOAD_PROFILE_PEAK,
            "p_load_jitter": P_LOAD_JITTER,
            "q_load_jitter": Q_LOAD_JITTER,
            "node_profile_diversity": NODE_PROFILE_DIVERSITY,
            "pv_p_jitter": PV_P_JITTER,
            "sample_ess_active": SAMPLE_ESS_ACTIVE,
            "ess_p_max_fraction": ESS_P_MAX_FRACTION,
            "pool_samples": POOL_SAMPLES,
            "target_samples": TARGET_SAMPLES,
            "pv_config_note": "Research modification on canonical IEEE33: PV nodes [8,15,23,30], S=[0.5,0.8,1.0,1.2] MVA.",
            "q_device_order_note": "q devices aligned with MATLAB mpc.svg.nodes=[8,4,17,21,2].",
        },

        "stats": data.get("stats", {}),
        "balance_info": data.get("balance_info", {}),
    }

    if classification is not None:
        dataset["classification"] = {
            k: torch.tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in classification.items()
        }

    if legends is not None:
        dataset["classification_legends"] = legends

    torch.save(dataset, save_path)


# =========================================================
# 11. 主程序
# =========================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, SAVE_NAME)
    stats_csv = os.path.join(DATA_DIR, "ieee33_raw_classification_50k.csv")

    Pload_24h, Qload_24h, Ppv_24h, devices = prepare_profiles_and_devices()

    net = setup_static_ieee33_env()
    idx = add_static_devices(net, devices)

    pool = generate_pool(
        net=net,
        idx=idx,
        Pload_24h=Pload_24h,
        Qload_24h=Qload_24h,
        Ppv_24h=Ppv_24h,
        devices=devices,
        rng=rng,
        pool_samples=POOL_SAMPLES,
    )

    classification, legends = classify_raw_pool(pool)
    print_classification_stats(pool, classification, legends)

    to_torch_dataset(
        data=pool,
        Pload_24h=Pload_24h,
        Qload_24h=Qload_24h,
        Ppv_24h=Ppv_24h,
        devices=devices,
        save_path=save_path,
        classification=classification,
        legends=legends,
    )

    save_classification_csv(
        pool=pool,
        cls=classification,
        legends=legends,
        out_csv=stats_csv,
    )

    print("==============================================")
    print("原始增强扰动样本池生成 + 分类统计完成")
    print(f"AC潮流收敛样本数: {len(pool['Y_V'])}")
    print(f"原始数据保存: {save_path}")
    print(f"逐样本分类CSV: {stats_csv}")
    print("说明: 未按电压或电流状态删除任何已收敛样本。")
    print("下一步依据本次实际类别分布，再决定最终20000训练样本的抽样比例。")
    print("==============================================")

