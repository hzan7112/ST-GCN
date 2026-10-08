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
SAVE_NAME = "ieee33_static_vvo_24h_dataset.pt"

POOL_SAMPLES = 30000
TARGET_SAMPLES = 12000
BATCH_SIZE = 500
MAX_ATTEMPT_MULTIPLIER = 30

V_GUARD_MIN = 0.90
V_GUARD_MAX = 1.10
I_MARGIN_MAX_KEEP = 0.80

UNIFORM_MAX_I_KA = 0.20
SLACK_VM_PU = 1.03

# 与你当前 MATLAB IEEE33_Prediction.m 的负荷向量顺序完全对齐
USE_MATLAB_LOAD_VECTOR_DIRECT = True

# 围绕 24h 固定运行点做小扰动；模式采样中会额外增强边界工况
LOAD_JITTER = 0.03
PV_P_JITTER = 0.03

# 当前暂按无功优化：ESS 有功固定为 0，仅采样 ESS 无功
# 如果后续 ESS 有功充放电也进入优化变量，可改为 True
SAMPLE_ESS_ACTIVE = False


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
    [16, 17, 0.3720, 0.5740],
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
    [31, 32, 0.3410, 0.5362],
]


# =========================================================
# 2. 负荷、PV、ESS、SVG/SVC/SC 配置
# =========================================================
def prepare_profiles_and_devices():
    if USE_MATLAB_LOAD_VECTOR_DIRECT:
        pload_kw = np.array(
            [
                100, 90, 120, 60, 60, 200, 200, 60, 60, 45, 60,
                60, 120, 60, 60, 60, 90, 90, 90, 90, 90, 90,
                420, 420, 60, 60, 60, 420, 400, 450, 410, 60, 0
            ],
            dtype=np.float64
        )

        qload_kvar = np.array(
            [
                60, 40, 80, 30, 20, 100, 100, 20, 20, 30, 35,
                35, 80, 10, 20, 20, 40, 40, 40, 40, 40, 50,
                200, 200, 25, 25, 20, 70, 600, 70, 100, 40, 0
            ],
            dtype=np.float64
        )
    else:
        pload_kw = np.array(
            [
                0, 100, 90, 120, 60, 60, 200, 200, 60, 60, 45,
                60, 60, 120, 60, 60, 60, 90, 90, 90, 90, 90,
                90, 420, 420, 60, 60, 60, 420, 400, 450, 410, 60
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

    Pload_24h = (pload_kw[:, None] * pload_ratio[None, :]) / 1000.0
    Qload_24h = (qload_kvar[:, None] * qload_ratio[None, :]) / 1000.0

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
    # 0随机；1高电压倾向；2低电压倾向；3中间；4边界
    mode = int(rng.choice([0, 1, 2, 3, 4], p=[0.38, 0.20, 0.20, 0.10, 0.12]))

    load_scale = rng.uniform(1.0 - LOAD_JITTER, 1.0 + LOAD_JITTER, size=33)

    if mode == 1:
        load_scale *= rng.uniform(0.90, 1.00)
    elif mode == 2:
        load_scale *= rng.uniform(1.03, 1.12)
    elif mode == 4:
        load_scale *= rng.uniform(0.92, 1.15)

    Pload = Pload_24h[:, hour] * load_scale
    Qload = Qload_24h[:, hour] * load_scale

    pv_p = Ppv_24h[hour].copy()
    pv_p *= rng.uniform(1.0 - PV_P_JITTER, 1.0 + PV_P_JITTER, size=len(pv_p))

    if mode == 1:
        pv_p *= rng.uniform(1.00, 1.05, size=len(pv_p))
    elif mode == 2:
        pv_p *= rng.uniform(0.95, 1.00, size=len(pv_p))

    pv_p = np.clip(pv_p, 0.0, 0.98 * devices["S_pv_mva"])

    pv_q_max = np.sqrt(np.maximum(devices["S_pv_mva"] ** 2 - pv_p ** 2, 0.0))
    pv_q = sample_by_mode(rng, -pv_q_max, pv_q_max, mode)

    ess_p = np.zeros(len(devices["ess_nodes"]), dtype=np.float64)

    if SAMPLE_ESS_ACTIVE:
        ess_p = rng.uniform(-0.5, 0.5, size=len(devices["ess_nodes"])) * devices["S_ess_mva"]

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
    X_list, YV_list, YI_list = [], [], []
    hour_list, mode_list = [], []
    pv_p_list, pv_q_list = [], []
    ess_p_list, ess_q_list = [], []
    qdev_q_list = []

    attempt = 0
    pf_fail = 0
    guard_reject = 0
    max_attempt = pool_samples * MAX_ATTEMPT_MULTIPLIER

    pbar = tqdm(total=pool_samples, desc="生成24h静态拓扑无功优化可行域数据池", unit="样本")
    t0 = time.perf_counter()

    while len(X_list) < pool_samples and attempt < max_attempt:
        for _ in range(BATCH_SIZE):
            if len(X_list) >= pool_samples or attempt >= max_attempt:
                break

            attempt += 1
            hour = int(rng.integers(0, 24))

            mode, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q = sample_operating_point(
                hour,
                rng,
                Pload_24h,
                Qload_24h,
                Ppv_24h,
                devices,
            )

            apply_operating_point(net, idx, Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q)

            if not run_powerflow_with_fallback(net):
                pf_fail += 1
                continue

            YV = net.res_bus.vm_pu.values.astype(np.float32)
            loading = net.res_line.loading_percent.values.astype(np.float32)
            YI = loading / 100.0 - 1.0

            if (
                not np.all(np.isfinite(YV))
                or not np.all(np.isfinite(YI))
                or YV.min() < V_GUARD_MIN
                or YV.max() > V_GUARD_MAX
                or YI.max() > I_MARGIN_MAX_KEEP
            ):
                guard_reject += 1
                continue

            X_list.append(build_net_injection(Pload, Qload, pv_p, pv_q, ess_p, ess_q, qdev_q, devices))
            YV_list.append(YV)
            YI_list.append(YI)

            hour_list.append(hour)
            mode_list.append(mode)
            pv_p_list.append(pv_p.astype(np.float32))
            pv_q_list.append(pv_q.astype(np.float32))
            ess_p_list.append(ess_p.astype(np.float32))
            ess_q_list.append(ess_q.astype(np.float32))
            qdev_q_list.append(qdev_q.astype(np.float32))

            pbar.update(1)

        if attempt % 2000 == 0 and YI_list:
            pos_ratio = 100.0 * np.mean([y.max() > 0 for y in YI_list])
            pbar.set_postfix(
                {
                    "attempt": attempt,
                    "pf_fail": pf_fail,
                    "reject": guard_reject,
                    "pos%": f"{pos_ratio:.1f}",
                }
            )

    pbar.close()

    if len(X_list) == 0:
        raise RuntimeError("未生成任何有效样本，请检查负荷、设备容量、线路限流或电压护栏设置。")

    if len(X_list) < pool_samples:
        print(f"[警告] 有效样本不足：目标 {pool_samples}，实际 {len(X_list)}。后续重采样会使用现有样本。")

    stats = {
        "attempt_count": int(attempt),
        "success_count": int(len(X_list)),
        "powerflow_fail_count": int(pf_fail),
        "guard_reject_count": int(guard_reject),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {
        "X": np.stack(X_list),
        "Y_V": np.stack(YV_list),
        "Y_I": np.stack(YI_list),
        "hour": np.array(hour_list, dtype=np.int64),
        "mode": np.array(mode_list, dtype=np.int64),
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
def balanced_resample(pool, target_size, rng, n_v_bins=20):
    YV = pool["Y_V"]
    YI = pool["Y_I"]

    v_min = YV.min(axis=1)
    v_max = YV.max(axis=1)
    i_max = YI.max(axis=1)

    # 用样本中最偏离 1.0 的电压作为电压均匀化指标
    v_char = np.where((v_max - 1.0) >= (1.0 - v_min), v_max, v_min)
    v_char = np.clip(v_char, V_GUARD_MIN, V_GUARD_MAX)

    current_sign = (i_max > 0).astype(np.int64)

    v_bins = np.linspace(V_GUARD_MIN, V_GUARD_MAX, n_v_bins + 1)
    v_idx = np.clip(np.digitize(v_char, v_bins) - 1, 0, n_v_bins - 1)

    target_pos = target_size // 2
    target_neg = target_size - target_pos

    def sample_one_sign(sign, sign_target):
        chosen = []
        all_sign = np.where(current_sign == sign)[0]

        if len(all_sign) == 0:
            return chosen

        per_bin = int(np.ceil(sign_target / n_v_bins))

        for b in range(n_v_bins):
            cand = np.where((current_sign == sign) & (v_idx == b))[0]
            if len(cand) == 0:
                continue
            take = min(per_bin, len(cand))
            chosen.extend(rng.choice(cand, size=take, replace=False).tolist())

        if len(chosen) < sign_target:
            chosen_set = set(chosen)
            rest = np.array([i for i in all_sign if i not in chosen_set], dtype=np.int64)
            need = sign_target - len(chosen)

            if len(rest) >= need:
                chosen.extend(rng.choice(rest, size=need, replace=False).tolist())
            elif len(rest) > 0:
                chosen.extend(rest.tolist())
                need = sign_target - len(chosen)
                chosen.extend(rng.choice(all_sign, size=need, replace=True).tolist())
            else:
                chosen.extend(rng.choice(all_sign, size=need, replace=True).tolist())

        elif len(chosen) > sign_target:
            chosen = rng.choice(np.array(chosen), size=sign_target, replace=False).tolist()

        return chosen

    neg_selected = sample_one_sign(0, target_neg)
    pos_selected = sample_one_sign(1, target_pos)

    if len(neg_selected) < target_neg:
        print(f"[警告] 非越限样本不足，目标 {target_neg}，实际可采 {len(neg_selected)}。")

    if len(pos_selected) < target_pos:
        print(f"[警告] 越限样本不足，目标 {target_pos}，实际可采 {len(pos_selected)}。")

    selected = neg_selected + pos_selected

    if len(selected) < target_size:
        need = target_size - len(selected)
        selected.extend(rng.choice(np.arange(len(YV)), size=need, replace=True).tolist())

    selected = np.array(selected[:target_size], dtype=np.int64)
    rng.shuffle(selected)

    out = {}
    for k, v in pool.items():
        if isinstance(v, np.ndarray) and len(v) == len(YV):
            out[k] = v[selected]
        else:
            out[k] = v

    out["balance_info"] = {
        "target_size": int(target_size),
        "selected_pos_margin_count": int((out["Y_I"].max(axis=1) > 0).sum()),
        "selected_nonpos_margin_count": int((out["Y_I"].max(axis=1) <= 0).sum()),
        "voltage_char_min": float(v_char[selected].min()),
        "voltage_char_max": float(v_char[selected].max()),
    }

    return out


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
def to_torch_dataset(data, Pload_24h, Qload_24h, Ppv_24h, devices, save_path):
    edge_list = [[int(r[0]), int(r[1])] for r in RADIAL_BRANCHES]
    branch_full = [[int(r[0]), int(r[1]), float(r[2]), float(r[3])] for r in RADIAL_BRANCHES]

    dataset = {
        "X": torch.tensor(data["X"], dtype=torch.float32),
        "Y_V": torch.tensor(data["Y_V"], dtype=torch.float32),
        "Y_I": torch.tensor(data["Y_I"], dtype=torch.float32),

        "hour": torch.tensor(data["hour"], dtype=torch.long),
        "mode": torch.tensor(data["mode"], dtype=torch.long),

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
            "use_matlab_load_vector_direct": USE_MATLAB_LOAD_VECTOR_DIRECT,
            "load_jitter": LOAD_JITTER,
            "pv_p_jitter": PV_P_JITTER,
            "sample_ess_active": SAMPLE_ESS_ACTIVE,
            "pool_samples": POOL_SAMPLES,
            "target_samples": TARGET_SAMPLES,
            "pv_config_note": "PV uses image configuration: nodes [8,15,23,30], S=[0.5,0.8,1.0,1.2] MVA.",
            "q_device_order_note": "q devices aligned with MATLAB mpc.svg.nodes=[8,4,17,21,2].",
        },

        "stats": data.get("stats", {}),
        "balance_info": data.get("balance_info", {}),
    }

    torch.save(dataset, save_path)


# =========================================================
# 11. 主程序
# =========================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, SAVE_NAME)

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

    print_dataset_stats(pool, "重采样前数据池统计")

    data = balanced_resample(
        pool=pool,
        target_size=TARGET_SAMPLES,
        rng=rng,
        n_v_bins=20,
    )

    print_dataset_stats(data, "最终均衡数据集统计")

    to_torch_dataset(
        data=data,
        Pload_24h=Pload_24h,
        Qload_24h=Qload_24h,
        Ppv_24h=Ppv_24h,
        devices=devices,
        save_path=save_path,
    )

    print("==============================================")
    print("静态拓扑 ST-GCN / ST-MGCN / ST-SGCN 统一数据集生成完毕")
    print(f"保存路径: {save_path}")
    print("说明: X=[P_net,Q_net], Y_V=节点电压, Y_I=支路电流裕度")
    print("设备配置: PV=[8,15,23,30], ESS=[15,30], Q设备=[8,4,17,21,2]")
    print("==============================================")