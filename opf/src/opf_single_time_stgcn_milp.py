import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

from model.st_gcn_milp_converter import STGCNPathVLinResBigMConverter


BRANCHES = [
    [0, 1, 0.0922, 0.0470], [1, 2, 0.4930, 0.2511], [2, 3, 0.3660, 0.1864],
    [3, 4, 0.3811, 0.1941], [4, 5, 0.8190, 0.7070], [5, 6, 0.1872, 0.6188],
    [6, 7, 0.7114, 0.2351], [7, 8, 1.0300, 0.7400], [8, 9, 1.0440, 0.7400],
    [9, 10, 0.1966, 0.0650], [10, 11, 0.3744, 0.1238], [11, 12, 1.4680, 1.1550],
    [12, 13, 0.5416, 0.7129], [13, 14, 0.5910, 0.5260], [14, 15, 0.7463, 0.5450],
    [15, 16, 1.2890, 1.7210], [16, 17, 0.3720, 0.5740], [1, 18, 0.1640, 0.1565],
    [18, 19, 1.5042, 1.3554], [19, 20, 0.4095, 0.4784], [20, 21, 0.7089, 0.9373],
    [2, 22, 0.4512, 0.3083], [22, 23, 0.8980, 0.7091], [23, 24, 0.8960, 0.7011],
    [5, 25, 0.2030, 0.1034], [25, 26, 0.2842, 0.1447], [26, 27, 1.0590, 0.9337],
    [27, 28, 0.8042, 0.7006], [28, 29, 0.5075, 0.2585], [29, 30, 0.9744, 0.9630],
    [30, 31, 0.3105, 0.3619], [31, 32, 0.3410, 0.5362],
]


def cfg():
    return {
        "engine_path": r"checkpoints/st_gcn_h16_l2_pathfeat_vlinres_exp20_milp_engine.pt",
        "time_index": 11,
        "sign_mode": "generation_minus_load",

        "slack_vm_pu": 1.03,
        "line_max_i_ka": 0.20,

        "v_lower": 0.9515,
        "v_upper": 1.0485,
        "i_upper": -0.015,

        "w_q": 1e-3,

        "threads": 8,
        "time_limit": 36000,
        "mip_gap": 0.01,

        "sc_index": 10,
        "sc_step": 0.1,

        "save_npz": r"results/pure_gcn_milp_result.npz",
        "save_fig": r"results/pure_gcn_milp_voltage_compare.png",
        "log_file": r"results/pure_gcn_milp_gurobi.log",
    }


def profiles(t):
    pkw = np.array([100,90,120,60,60,200,200,60,60,45,60,60,120,60,60,60,90,90,90,90,90,90,420,420,60,60,60,420,400,450,410,60,0], float)
    qkv = np.array([60,40,80,30,20,100,100,20,20,30,35,35,80,10,20,20,40,40,40,40,40,50,200,200,25,25,20,70,600,70,100,40,0], float)
    pr = np.array([2.15,2.3,1.2,2.35,2.35,2.6,3.0,2.25,2.7,1.8,1.35,1.2,1.15,1.1,1.35,1.45,1.5,1.65,1.9,2.0,1.2,1.8,1.85,1.8], float)
    qr = np.array([1.15,1.3,0.8,1.35,1.35,1.6,2.0,1.25,2.1,1.4,1.15,1.0,0.9,1.0,1.2,1.25,1.3,1.45,1.2,1.0,1.0,1.4,1.55,1.4], float)

    pv_shape = np.array([0,0,0,0,0,0.25,0.49,0.52,0.75,0.9,1.0,1.25,1.1,0.9,0.79,0.76,0.75,0.55,0.35,0.25,0,0,0,0], float)
    pv_shape /= pv_shape.max()

    S_pv = np.array([0.5, 0.8, 1.0, 1.2])
    S_ess = np.array([0.8, 0.8])

    return {
        "P_load": pkw * pr[t] / 1000,
        "Q_load": qkv * qr[t] / 1000,

        "pv_nodes": np.array([7, 14, 22, 29]),
        "S_pv": S_pv,
        "P_pv": pv_shape[t] * (0.8 * S_pv),

        "ess_nodes": np.array([14, 29]),
        "S_ess": S_ess,
        "P_ess": np.zeros(2),

        "qdev_names": ["SVC1", "SVG1", "SVG2", "SVC2", "SC"],
        "qdev_nodes": np.array([7, 3, 16, 20, 1]),
        "qdev_min": np.array([-1.0, -0.5, -0.8, -2.0, 0.0]),
        "qdev_max": np.array([1.0, 0.5, 0.8, 2.0, 0.4]),
    }


def q_bounds(s):
    pv = np.sqrt(np.maximum(s["S_pv"] ** 2 - s["P_pv"] ** 2, 0))
    ess = np.sqrt(np.maximum(s["S_ess"] ** 2 - s["P_ess"] ** 2, 0))
    return np.r_[-pv, -ess, s["qdev_min"]], np.r_[pv, ess, s["qdev_max"]], pv, ess


def split_q(q):
    return q[:4], q[4:6], q[6:]


def q_map(s):
    C = np.zeros((33, 11))
    for k, b in enumerate(s["pv_nodes"]):
        C[b, k] = 1.0
    for k, b in enumerate(s["ess_nodes"]):
        C[b, 4 + k] = 1.0
    for k, b in enumerate(s["qdev_nodes"]):
        C[b, 6 + k] += 1.0
    return C


def x6_bounds(s, conv, lb, ub, sign):
    C = q_map(s)
    P0, Q0 = -s["P_load"].copy(), -s["Q_load"].copy()

    for k, b in enumerate(s["pv_nodes"]):
        P0[b] += s["P_pv"][k]
    for k, b in enumerate(s["ess_nodes"]):
        P0[b] += s["P_ess"][k]

    if sign != "generation_minus_load":
        P0, Q0, C = -P0, -Q0, -C

    base = [P0, Q0, conv.S_down @ P0, conv.S_down @ Q0, conv.S_path @ P0, conv.S_path @ Q0]
    coef = [np.zeros((33, 11)), C, np.zeros((33, 11)), conv.S_down @ C, np.zeros((33, 11)), conv.S_path @ C]

    lo, hi = np.zeros((33, 6)), np.zeros((33, 6))

    for f in range(6):
        W = coef[f]
        lo[:, f] = base[f] + np.sum(np.where(W >= 0, W * lb, W * ub), axis=1)
        hi[:, f] = base[f] + np.sum(np.where(W >= 0, W * ub, W * lb), axis=1)

    return lo, hi


def x_expr(s, Q, sign):
    qpv, qess, qdev = split_q(list(Q))
    pv = {int(b): k for k, b in enumerate(s["pv_nodes"])}
    es = {int(b): k for k, b in enumerate(s["ess_nodes"])}
    qd = {int(b): k for k, b in enumerate(s["qdev_nodes"])}

    X = []

    for i in range(33):
        p = gp.LinExpr(-float(s["P_load"][i]))
        q = gp.LinExpr(-float(s["Q_load"][i]))

        if i in pv:
            p += float(s["P_pv"][pv[i]])
            q += qpv[pv[i]]
        if i in es:
            p += float(s["P_ess"][es[i]])
            q += qess[es[i]]
        if i in qd:
            q += qdev[qd[i]]
        if sign != "generation_minus_load":
            p, q = -p, -q

        X.append([p, q])

    return X


def absvar(m, x, name, ub=10.0):
    y = m.addVar(lb=0.0, ub=float(ub), name=name)
    m.addConstr(y >= x)
    m.addConstr(y >= -x)
    return y


def create_model(c):
    os.makedirs(os.path.dirname(c["log_file"]), exist_ok=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("Threads", c["threads"])
    env.start()

    m = gp.Model("pure_gcn_milp_vvo", env=env)

    m.Params.OutputFlag = 1
    m.Params.LogFile = c["log_file"]
    m.Params.Threads = c["threads"]
    m.Params.TimeLimit = c["time_limit"]
    m.Params.MIPGap = c["mip_gap"]

    m.Params.Method = 1
    m.Params.NodeMethod = 1
    m.Params.Presolve = 2
    m.Params.PreSparsify = 1
    m.Params.MIPFocus = 3
    m.Params.Heuristics = 0.25
    m.Params.Cuts = 2
    m.Params.VarBranch = 2
    m.Params.NumericFocus = 1
    m.Params.PumpPasses = 10
    m.Params.StartNodeLimit = 10000

    return env, m


def add_q_vars(m, c, lb, ub):
    Q = []
    sc_step_var = None

    for k in range(11):
        if k == c["sc_index"]:
            step = float(c["sc_step"])
            n_min = int(round(lb[k] / step))
            n_max = int(round(ub[k] / step))

            sc_step_var = m.addVar(lb=n_min, ub=n_max, vtype=GRB.INTEGER, name="SC_step")
            q = m.addVar(lb=float(lb[k]), ub=float(ub[k]), name=f"Q_{k}_SC")
            m.addConstr(q == step * sc_step_var, name="SC_discrete_link")

            sc_step_var.Start = 0
            sc_step_var.VarHintVal = 0
            q.Start = 0.0
            q.VarHintVal = 0.0
        else:
            q_start = float(np.clip(0.0, lb[k], ub[k]))
            q = m.addVar(lb=float(lb[k]), ub=float(ub[k]), name=f"Q_{k}")
            q.Start = q_start
            q.VarHintVal = q_start

        Q.append(q)

    return Q, sc_step_var


def runpp(s, q, c, title):
    try:
        import pandapower as pp
    except Exception:
        print("未安装 pandapower，跳过真实潮流校验。")
        return None

    qpv, qess, qdev = split_q(q)
    net = pp.create_empty_network()

    for _ in range(33):
        pp.create_bus(net, vn_kv=12.66)

    pp.create_ext_grid(net, bus=0, vm_pu=c["slack_vm_pu"])

    for f, t, r, x in BRANCHES:
        pp.create_line_from_parameters(net, int(f), int(t), 1.0, float(r), float(x), 0.0, c["line_max_i_ka"])

    for i in range(33):
        pp.create_load(net, i, p_mw=s["P_load"][i], q_mvar=s["Q_load"][i])

    for k, b in enumerate(s["pv_nodes"]):
        pp.create_sgen(net, int(b), p_mw=s["P_pv"][k], q_mvar=qpv[k])
    for k, b in enumerate(s["ess_nodes"]):
        pp.create_sgen(net, int(b), p_mw=s["P_ess"][k], q_mvar=qess[k])
    for k, b in enumerate(s["qdev_nodes"]):
        pp.create_sgen(net, int(b), p_mw=0.0, q_mvar=qdev[k])

    pp.runpp(
        net,
        algorithm="bfsw",
        init="flat",
        tolerance_mva=1e-7,
        max_iteration=100,
        enforce_q_lims=False,
        calculate_voltage_angles=False,
        numba=False,
    )

    V = net.res_bus.vm_pu.values
    I = net.res_line.loading_percent.values / 100.0 - 1.0

    print(f"\n【{title} pandapower】")
    print(f"电压范围: {V[1:].min():.6f} ~ {V[1:].max():.6f}")
    print(f"最大支路裕度: {I.max():.6f}")
    print(f"线路最大 loading_percent: {net.res_line.loading_percent.values.max():.2f}%")

    return {"V": V, "I": I, "loading": net.res_line.loading_percent.values}


def save_result_and_plot(c, s, result, pp_no, pp_opt):
    os.makedirs(os.path.dirname(c["save_npz"]), exist_ok=True)
    os.makedirs(os.path.dirname(c["save_fig"]), exist_ok=True)

    np.savez(
        c["save_npz"],
        Q=result["q"],
        V_pred=result["V_pred"],
        I_pred=result["I_pred"],
        obj=result["obj"],
        bound=result["bound"],
        gap=result["gap"],
        runtime=result["runtime"],
        V_no_opt=None if pp_no is None else pp_no["V"],
        V_pp_opt=None if pp_opt is None else pp_opt["V"],
        I_no_opt=None if pp_no is None else pp_no["I"],
        I_pp_opt=None if pp_opt is None else pp_opt["I"],
        P_load=s["P_load"],
        Q_load=s["Q_load"],
        P_pv=s["P_pv"],
    )

    bus = np.arange(1, 34)

    plt.figure(figsize=(9, 4.8))

    if pp_no is not None:
        plt.plot(bus, pp_no["V"], marker="o", linewidth=1.3, markersize=3.5, label="No VVO - pandapower")

    plt.plot(bus, result["V_pred"], marker="s", linewidth=1.4, markersize=3.5, label="GCN-MILP predicted after VVO")

    if pp_opt is not None:
        plt.plot(bus, pp_opt["V"], marker="^", linewidth=1.3, markersize=3.5, label="After VVO - pandapower")

    plt.axhline(1.05, linestyle="--", linewidth=1.0)
    plt.axhline(0.95, linestyle="--", linewidth=1.0)
    plt.axhline(1.00, linestyle=":", linewidth=1.0)

    plt.xlim(1, 33)
    plt.xticks(np.arange(1, 34, 2))
    plt.xlabel("Bus")
    plt.ylabel("Voltage (p.u.)")
    plt.grid(True, linestyle=":", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(c["save_fig"], dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n结果已保存: {c['save_npz']}")
    print(f"电压对比图已保存: {c['save_fig']}")
    print(f"Gurobi日志已保存: {c['log_file']}")


def solve():
    c = cfg()
    s = profiles(c["time_index"])
    lb, ub, pv_cap, ess_cap = q_bounds(s)

    pp_no = runpp(s, np.zeros(11), c, "未优化")

    conv0 = STGCNPathVLinResBigMConverter(c["engine_path"], slack_voltage=c["slack_vm_pu"])
    xlb, xub = x6_bounds(s, conv0, lb, ub, c["sign_mode"])

    env, m = create_model(c)

    try:
        Q, sc_step_var = add_q_vars(m, c, lb, ub)

        conv = STGCNPathVLinResBigMConverter(c["engine_path"], slack_voltage=c["slack_vm_pu"])
        conv.set_opf_input_bounds(xlb, xub)

        V, I, aux = conv.embed(
            m,
            x_expr(s, Q, c["sign_mode"]),
            name="pure",
            relu_starts=None,
            embed_edge=True,
        )

        for i in range(1, 33):
            m.addConstr(V[i] >= c["v_lower"], name=f"Vmin_{i + 1}")
            m.addConstr(V[i] <= c["v_upper"], name=f"Vmax_{i + 1}")

        for e in range(len(I)):
            m.addConstr(I[e] <= c["i_upper"], name=f"Imax_{e + 1}")

        obj = gp.LinExpr()

        for i in range(1, 33):
            obj += absvar(m, V[i] - 1.0, f"absV_{i}", 0.3)

        for k in range(11):
            obj += c["w_q"] * absvar(m, Q[k], f"absQ_{k}", max(abs(lb[k]), abs(ub[k])))

        m.setObjective(obj, GRB.MINIMIZE)
        m.update()

        print("\n================ Pure GCN-MILP 模型规模 ================")
        print(f"变量数: {m.NumVars}")
        print(f"约束数: {m.NumConstrs}")
        print(f"二元变量数: {m.NumBinVars}")
        print("ReLU二元变量数:", aux["binary_created"])
        print("ReLU固定为0:", aux["relu_fixed_zero"])
        print("ReLU固定为线性:", aux["relu_fixed_linear"])
        print("SC离散档位变量:", "启用" if sc_step_var is not None else "未启用")
        print("engine理论二元变量:", aux["binary_theoretical"])
        print("======================================================\n")

        m.optimize()

        if m.SolCount <= 0:
            print(f"未得到可行解，status={m.status}")
            return

        q = np.array([x.X for x in Q])
        Vp = np.array([x.X for x in V])
        Ip = np.array([x.X for x in I])

        pp_opt = runpp(s, q, c, "优化后")

        qpv, qess, qdev = split_q(q)

        print("\n================ Pure GCN-MILP 最终无功调度结果 ================")
        for k, b in enumerate(s["pv_nodes"]):
            print(f"PV  Bus {b + 1:02d} | Q={qpv[k]: .6f} | Qcap={pv_cap[k]:.6f}")
        for k, b in enumerate(s["ess_nodes"]):
            print(f"ESS Bus {b + 1:02d} | Q={qess[k]: .6f} | Qcap={ess_cap[k]:.6f}")
        for k, b in enumerate(s["qdev_nodes"]):
            print(f"{s['qdev_names'][k]:4s} Bus {b + 1:02d} | Q={qdev[k]: .6f}")

        print("\n【Pure GCN-MILP 预测状态】")
        print(f"Obj={m.ObjVal:.8f}, Bound={m.ObjBound:.8f}, Gap={m.MIPGap * 100:.2f}%, Runtime={m.Runtime:.2f}s")
        print(f"预测电压范围: {Vp[1:].min():.6f} ~ {Vp[1:].max():.6f}")
        print(f"预测最大支路裕度: {Ip.max():.6f}")

        result = {
            "q": q,
            "V_pred": Vp,
            "I_pred": Ip,
            "obj": float(m.ObjVal),
            "bound": float(m.ObjBound),
            "gap": float(m.MIPGap),
            "runtime": float(m.Runtime),
        }

        save_result_and_plot(c, s, result, pp_no, pp_opt)

    finally:
        m.dispose()
        env.dispose()


if __name__ == "__main__":
    solve()