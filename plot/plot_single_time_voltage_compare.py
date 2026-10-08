import os
import warnings
import numpy as np
import torch
import pandapower as pp
import matplotlib.pyplot as plt

from model.st_gcn_milp_converter import STGCNMILPConverter

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"D:\pythonproject\ST-GCN"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ieee33_static_vvo_24h_dataset.pt")
ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "st_gcn_h64_l9_downstream_exp15_milp_engine.pt")

TIME_INDEX = 11

SAVE_DIR = os.path.join(PROJECT_ROOT, "results")
SAVE_NAME = f"voltage_compare_t{TIME_INDEX + 1}.tiff"

# 若后续你把优化结果保存成 npz，可改为 True 并设置 RESULT_PATH
USE_SAVED_RESULT = True
RESULT_PATH = os.path.join(SAVE_DIR, f"single_time_stgcn_vvo_t{TIME_INDEX + 1}.npz")

# 这次日志中的最终局部仿射优化结果
Q_PV_OPT = np.array([-0.111528, 0.0, -0.6, 0.719818], dtype=np.float64)
Q_ESS_OPT = np.array([0.270503, 0.8], dtype=np.float64)
Q_DEV_OPT = np.array([0.0, -0.5, 0.210992, -1.731401, 0.0], dtype=np.float64)


def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_time_data(data, t):
    return {
        "Pload": npy(data["Pload_24h"])[:, t].astype(np.float64),
        "Qload": npy(data["Qload_24h"])[:, t].astype(np.float64),
        "Ppv": npy(data["Ppv_24h"])[t].astype(np.float64),

        "pv_nodes": npy(data["pv_nodes"]).astype(int),
        "ess_nodes": npy(data["ess_nodes"]).astype(int),

        "qdev_names": data["q_device_names"],
        "qdev_nodes": npy(data["q_device_nodes"]).astype(int),
    }


def load_optimized_q():
    if USE_SAVED_RESULT and os.path.exists(RESULT_PATH):
        r = np.load(RESULT_PATH)
        return r["Q_pv"], r["Q_ess"], r["Q_dev"]

    return Q_PV_OPT, Q_ESS_OPT, Q_DEV_OPT


def build_x_numeric(d, converter, q_pv, q_ess, q_dev):
    n = converter.num_nodes
    S_down = converter.get_downstream_matrix()

    P_net = -d["Pload"].copy()
    Q_net = -d["Qload"].copy()

    for k, bus in enumerate(d["pv_nodes"]):
        P_net[bus] += d["Ppv"][k]
        Q_net[bus] += q_pv[k]

    for k, bus in enumerate(d["ess_nodes"]):
        Q_net[bus] += q_ess[k]

    for k, bus in enumerate(d["qdev_nodes"]):
        Q_net[bus] += q_dev[k]

    X = np.zeros((n, 4), dtype=np.float64)
    X[:, 0] = P_net
    X[:, 1] = Q_net
    X[:, 2] = S_down @ P_net
    X[:, 3] = S_down @ Q_net

    return X


def build_pp_net(data):
    base = data.get("base_config", {})
    slack_vm = float(base.get("slack_vm_pu", 1.03))
    max_i = float(base.get("line_max_i_ka", 0.20))

    net = pp.create_empty_network()

    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i + 1}")

    pp.create_ext_grid(net, bus=0, vm_pu=slack_vm)

    for f, t, r, x in data["branch_full"]:
        pp.create_line_from_parameters(
            net,
            from_bus=int(f),
            to_bus=int(t),
            length_km=1.0,
            r_ohm_per_km=float(r),
            x_ohm_per_km=float(x),
            c_nf_per_km=0.0,
            max_i_ka=max_i,
        )

    for i in range(33):
        pp.create_load(net, bus=i, p_mw=0.0, q_mvar=0.0)

    pv_idx = [
        pp.create_sgen(net, bus=int(b), p_mw=0.0, q_mvar=0.0, name=f"PV@{b + 1}")
        for b in npy(data["pv_nodes"]).astype(int)
    ]

    ess_idx = [
        pp.create_sgen(net, bus=int(b), p_mw=0.0, q_mvar=0.0, name=f"ESS@{b + 1}")
        for b in npy(data["ess_nodes"]).astype(int)
    ]

    qdev_idx = [
        pp.create_sgen(net, bus=int(b), p_mw=0.0, q_mvar=0.0, name=f"{name}@{b + 1}")
        for name, b in zip(data["q_device_names"], npy(data["q_device_nodes"]).astype(int))
    ]

    return net, np.array(pv_idx), np.array(ess_idx), np.array(qdev_idx)


def run_pp_voltage(data, d, q_pv, q_ess, q_dev):
    net, pv_idx, ess_idx, qdev_idx = build_pp_net(data)

    net.load.loc[:, "p_mw"] = d["Pload"]
    net.load.loc[:, "q_mvar"] = d["Qload"]

    net.sgen.loc[pv_idx, "p_mw"] = d["Ppv"]
    net.sgen.loc[pv_idx, "q_mvar"] = q_pv

    net.sgen.loc[ess_idx, "p_mw"] = 0.0
    net.sgen.loc[ess_idx, "q_mvar"] = q_ess

    net.sgen.loc[qdev_idx, "p_mw"] = 0.0
    net.sgen.loc[qdev_idx, "q_mvar"] = q_dev

    for kw in [
        dict(algorithm="bfsw", init="flat", tolerance_mva=1e-7, max_iteration=100),
        dict(algorithm="nr", init="flat", tolerance_mva=1e-7, max_iteration=50),
        dict(algorithm="nr", init="auto", tolerance_mva=1e-7, max_iteration=50),
    ]:
        try:
            pp.runpp(net, enforce_q_lims=False, calculate_voltage_angles=False, numba=False, **kw)

            if bool(net.converged):
                return net.res_bus.vm_pu.values.copy()

        except Exception:
            pass

    raise RuntimeError("pandapower 潮流未收敛。")


def plot_voltage_compare(v_base, v_gcn, v_pp, save_path):
    nodes = np.arange(1, 34)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.plot(
        nodes,
        v_base,
        marker="o",
        markersize=3.8,
        linewidth=1.2,
        label="Before VVO (Pandapower)",
    )

    ax.plot(
        nodes,
        v_gcn,
        marker="s",
        markersize=3.6,
        linewidth=1.2,
        label="GCN-based VVO (GCN prediction)",
    )

    ax.plot(
        nodes,
        v_pp,
        marker="^",
        markersize=3.6,
        linewidth=1.2,
        label="GCN-based VVO (Pandapower check)",
    )

    ax.axhline(1.05, linestyle="--", linewidth=1.0, label="Upper limit")
    ax.axhline(0.95, linestyle="--", linewidth=1.0, label="Lower limit")
    ax.axhline(1.00, linestyle=":", linewidth=0.9, label="Reference")

    ax.set_xlim(1, 33)
    ax.set_xticks(np.arange(1, 34, 2))
    ax.set_xlabel("Bus index", fontsize=12)
    ax.set_ylabel("Voltage magnitude (p.u.)", fontsize=12)

    ax.tick_params(axis="both", direction="in", labelsize=10)
    ax.grid(True, alpha=0.12, linewidth=0.4)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(save_path.replace(".tiff", ".png"), dpi=300, bbox_inches="tight")
    plt.show()


def print_voltage_stats(name, v):
    print(f"{name}:")
    print(f"  V_min       = {v[1:].min():.6f}")
    print(f"  V_max       = {v[1:].max():.6f}")
    print(f"  mean|V-1|   = {np.mean(np.abs(v[1:] - 1.0)):.8f}")
    print(f"  safe[0.95,1.05] = {bool((v[1:] >= 0.95).all() and (v[1:] <= 1.05).all())}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    data = load_pt(DATA_PATH)
    d = get_time_data(data, TIME_INDEX)
    converter = STGCNMILPConverter(ENGINE_PATH)

    q_pv_opt, q_ess_opt, q_dev_opt = load_optimized_q()

    q_pv_zero = np.zeros(4, dtype=np.float64)
    q_ess_zero = np.zeros(2, dtype=np.float64)
    q_dev_zero = np.zeros(5, dtype=np.float64)

    v_base_pp = run_pp_voltage(data, d, q_pv_zero, q_ess_zero, q_dev_zero)

    x_opt = build_x_numeric(d, converter, q_pv_opt, q_ess_opt, q_dev_opt)
    v_gcn_opt = converter.forward_numpy(x_opt)["V"]

    v_pp_opt = run_pp_voltage(data, d, q_pv_opt, q_ess_opt, q_dev_opt)

    print("\n================ 节点电压统计 ================")
    print_voltage_stats("未进行无功优化 Pandapower", v_base_pp)
    print_voltage_stats("GCN 潮流替代无功优化 GCN预测", v_gcn_opt)
    print_voltage_stats("GCN 潮流替代无功优化 Pandapower校核", v_pp_opt)
    print("=============================================\n")

    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    plot_voltage_compare(v_base_pp, v_gcn_opt, v_pp_opt, save_path)

    print(f"电压对比图已保存:")
    print(save_path)
    print(save_path.replace(".tiff", ".png"))


if __name__ == "__main__":
    main()