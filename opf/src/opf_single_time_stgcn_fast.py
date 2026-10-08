import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

from model.model import StandardGCN
from model.st_gcn_milp_converter import STGCNPathVLinResBigMConverter, load_pt


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
        "time_limit": 300,
        "mip_gap": 0.05,

        "sc_index": 10,
        "sc_step": 0.1,

        "random_each_sc": 20000,
        "topk_each_sc": 32,
        "torch_steps": 1200,
        "torch_lr": 0.05,

        "trust_radius": [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.08, 0.08, 0.08, 0.12, 0.00],
        "obj_cutoff_margin": 1e-3,

        "save_npz": r"results/exp22_enum_trust_result.npz",
        "save_fig": r"results/exp22_enum_trust_voltage_compare.png",
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
    pv = np.sqrt(np.maximum(s["S_pv"] ** 2 - s["P_pv"] ** 2, 0.0))
    ess = np.sqrt(np.maximum(s["S_ess"] ** 2 - s["P_ess"] ** 2, 0.0))
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


def load_model(engine_path, device):
    e = load_pt(engine_path)
    cc = e.get("config", {})

    model = StandardGCN(
        in_features=int(e.get("in_features", 6)),
        hidden_dim=int(e["hidden_dim"]),
        num_layers=int(e["num_layers"]),
        edge_list=[[int(u), int(v)] for u, v in e["edge_list"]],
        num_nodes=33,
        node_relu_dim=int(e["node_relu_dim"]),
        edge_relu_dim=int(e["edge_relu_dim"]),
        node_emb_dim=int(e["node_emb_dim"]),
        edge_emb_dim=int(e["edge_emb_dim"]),
        use_residual=bool(e.get("use_residual", cc.get("use_residual", True))),
        use_initial_anchor=bool(e.get("use_initial_anchor", cc.get("use_initial_anchor", True))),
        use_jk=bool(e.get("use_jk", cc.get("use_jk", True))),
        include_input_in_jk=bool(e.get("include_input_in_jk", cc.get("include_input_in_jk", True))),
        use_linear_skip=bool(e.get("use_linear_skip", cc.get("use_linear_skip", True))),
    ).to(device)

    model.load_state_dict(e["state_dict"])
    model.eval()

    return e, model


def torch_x6(s, q, conv, device, sign):
    q = q.view(-1, 11)
    B = q.shape[0]

    P = -torch.tensor(s["P_load"], dtype=torch.float32, device=device).view(1, 33).repeat(B, 1)
    Q = -torch.tensor(s["Q_load"], dtype=torch.float32, device=device).view(1, 33).repeat(B, 1)

    for k, b in enumerate(s["pv_nodes"]):
        P[:, b] += float(s["P_pv"][k])
        Q[:, b] += q[:, k]
    for k, b in enumerate(s["ess_nodes"]):
        P[:, b] += float(s["P_ess"][k])
        Q[:, b] += q[:, 4 + k]
    for k, b in enumerate(s["qdev_nodes"]):
        Q[:, b] += q[:, 6 + k]

    if sign != "generation_minus_load":
        P, Q = -P, -Q

    Sd = torch.tensor(conv.S_down, dtype=torch.float32, device=device)
    Sp = torch.tensor(conv.S_path, dtype=torch.float32, device=device)

    return torch.stack([P, Q, P @ Sd.T, Q @ Sd.T, P @ Sp.T, Q @ Sp.T], dim=2)


def torch_forward(engine, model, Xn):
    Vres, In, gZ, Zn, Ze = model(Xn)

    if engine.get("uses_voltage_linear_prior", False):
        W = torch.as_tensor(engine["voltage_linear_prior_W"], dtype=torch.float32, device=Xn.device)
        b = torch.as_tensor(engine["voltage_linear_prior_b"], dtype=torch.float32, device=Xn.device)
        Vn = Vres.clone()
        Vn[:, 1:] = Vn[:, 1:] + Xn.reshape(Xn.shape[0], -1) @ W.T + b
    else:
        Vn = Vres

    ns = engine["norm_stats"]
    ym = torch.as_tensor(ns["YV_mean_wo_slack"], dtype=torch.float32, device=Xn.device)
    ys = torch.as_tensor(ns["YV_std_wo_slack"], dtype=torch.float32, device=Xn.device)
    im = torch.as_tensor(ns["YI_mean"], dtype=torch.float32, device=Xn.device)
    istd = torch.as_tensor(ns["YI_std"], dtype=torch.float32, device=Xn.device)

    V = torch.ones((Xn.shape[0], 33), dtype=torch.float32, device=Xn.device)
    V[:, 1:] = Vn[:, 1:] * ys + ym
    I = In * istd + im

    return V, I, gZ, Zn, Ze


def set_sc(q, sc_idx, sc_value):
    q = q.clone()
    q[:, sc_idx] = float(sc_value)
    return q


def torch_score(s, c, conv, engine, model, q, xm, xs):
    Xn = (torch_x6(s, q, conv, q.device, c["sign_mode"]) - xm) / xs
    V, I, *_ = torch_forward(engine, model, Xn)

    y = torch.sum(torch.abs(V[:, 1:] - 1.0), dim=1)
    y += c["w_q"] * torch.sum(torch.abs(q), dim=1)
    y += 200.0 * torch.relu(c["v_lower"] - V[:, 1:]).sum(dim=1)
    y += 200.0 * torch.relu(V[:, 1:] - c["v_upper"]).sum(dim=1)
    y += 200.0 * torch.relu(I - c["i_upper"]).sum(dim=1)

    return y


def eval_q(s, c, q_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine, model = load_model(c["engine_path"], device)
    conv = STGCNPathVLinResBigMConverter(c["engine_path"])

    ns = engine["norm_stats"]
    xm = torch.as_tensor(ns["X_mean"], dtype=torch.float32, device=device)
    xs = torch.as_tensor(ns["X_std"], dtype=torch.float32, device=device)

    q = torch.tensor(q_np, dtype=torch.float32, device=device).view(1, 11)
    Xn = (torch_x6(s, q, conv, device, c["sign_mode"]) - xm) / xs

    with torch.no_grad():
        V, I, gZ, Zn, Ze = torch_forward(engine, model, Xn)
        obj = torch.sum(torch.abs(V[:, 1:] - 1.0)).item() + c["w_q"] * torch.sum(torch.abs(q)).item()

    starts = {
        "gcn": [z[0].detach().cpu().numpy() for z in gZ],
        "node": Zn[0].detach().cpu().numpy(),
        "edge": Ze[0].detach().cpu().numpy(),
    }

    return obj, V[0].cpu().numpy(), I[0].cpu().numpy(), starts


def pytorch_sc_enum_multistart(s, c):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine, model = load_model(c["engine_path"], device)
    conv = STGCNPathVLinResBigMConverter(c["engine_path"])

    lb, ub, _, _ = q_bounds(s)
    sc_idx, step = c["sc_index"], c["sc_step"]
    sc_values = np.arange(lb[sc_idx], ub[sc_idx] + 0.5 * step, step)
    sc_values = np.round(sc_values / step) * step

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device)

    ns = engine["norm_stats"]
    xm = torch.as_tensor(ns["X_mean"], dtype=torch.float32, device=device)
    xs = torch.as_tensor(ns["X_std"], dtype=torch.float32, device=device)

    rng = np.random.default_rng(42)
    best = {"obj": np.inf, "q": None, "sc": None}

    print("\n================ SC枚举 + PyTorch多起点连续优化 ================")

    for sc in sc_values:
        q_rand = rng.uniform(lb, ub, size=(c["random_each_sc"], 11))
        q_zero = np.zeros((1, 11))
        q_all = np.vstack([q_zero, q_rand])
        q_all[:, sc_idx] = sc

        q_all_t = torch.tensor(q_all, dtype=torch.float32, device=device)

        with torch.no_grad():
            score0 = torch_score(s, c, conv, engine, model, q_all_t, xm, xs)
            top_idx = torch.topk(-score0, k=min(c["topk_each_sc"], len(q_all_t))).indices
            q_init = q_all_t[top_idx].clone()

        ratio = (q_init - lb_t + 1e-6) / (ub_t - q_init + 1e-6)
        u = torch.log(torch.clamp(ratio, 1e-6, 1e6)).detach().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=c["torch_lr"])

        for _ in range(c["torch_steps"]):
            q = lb_t + (ub_t - lb_t) * torch.sigmoid(u)
            q = set_sc(q, sc_idx, sc)
            loss = torch_score(s, c, conv, engine, model, q, xm, xs).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            q = lb_t + (ub_t - lb_t) * torch.sigmoid(u)
            q = set_sc(q, sc_idx, sc)
            score = torch_score(s, c, conv, engine, model, q, xm, xs)
            j = torch.argmin(score).item()
            q_best = q[j].detach().cpu().numpy()
            obj, V, I, _ = eval_q(s, c, q_best)

        print(f"SC={sc:.1f} | Obj={obj:.6f} | V=[{V[1:].min():.6f},{V[1:].max():.6f}] | Imax={I.max():.6f}")

        if obj < best["obj"]:
            best = {"obj": obj, "q": q_best, "sc": sc}

    obj, V, I, starts = eval_q(s, c, best["q"])

    print("\n================ PyTorch最终中心点 ================")
    print(f"best SC = {best['sc']:.1f}")
    print(f"Q* = {np.round(best['q'], 6)}")
    print(f"Obj = {obj:.6f}")
    print(f"V range = [{V[1:].min():.6f}, {V[1:].max():.6f}]")
    print(f"I max = {I.max():.6f}")
    print("====================================================\n")

    return best["q"], obj, starts


def trust_bounds(c, lb0, ub0, center):
    r = np.asarray(c["trust_radius"], float)
    lb = np.maximum(lb0, center - r)
    ub = np.minimum(ub0, center + r)

    idx, step = c["sc_index"], c["sc_step"]
    sc = round(center[idx] / step) * step
    lb[idx] = sc
    ub[idx] = sc

    return lb, ub


def create_model(c):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("Threads", c["threads"])
    env.start()

    m = gp.Model("enum_sc_small_trust_milp", env=env)

    m.Params.OutputFlag = 1
    m.Params.Threads = c["threads"]
    m.Params.TimeLimit = c["time_limit"]
    m.Params.MIPGap = c["mip_gap"]
    m.Params.Method = 1
    m.Params.NodeMethod = 1
    m.Params.Presolve = 2
    m.Params.PreSparsify = 1
    m.Params.MIPFocus = 2
    m.Params.Heuristics = 0.15
    m.Params.Cuts = 2
    m.Params.VarBranch = 2
    m.Params.NumericFocus = 1

    return env, m

def add_q_vars(m, c, lb, ub, q0):
    Q = []
    sc_fixed = None

    for k in range(11):
        if k == c["sc_index"]:
            step = float(c["sc_step"])
            sc_value = round(float(q0[k]) / step) * step
            sc_value = min(max(sc_value, float(lb[k])), float(ub[k]))

            q = m.addVar(
                lb=sc_value,
                ub=sc_value,
                name=f"Q_{k}_SC_fixed"
            )

            q.Start = sc_value
            q.VarHintVal = sc_value
            sc_fixed = sc_value

        else:
            q_start = float(np.clip(q0[k], lb[k], ub[k]))

            q = m.addVar(
                lb=float(lb[k]),
                ub=float(ub[k]),
                name=f"Q_{k}"
            )

            q.Start = q_start
            q.VarHintVal = q_start

        Q.append(q)

    return Q, sc_fixed


def solve_milp(s, c, q_center, center_obj, starts):
    lb0, ub0, pv_cap, ess_cap = q_bounds(s)
    lb, ub = trust_bounds(c, lb0, ub0, q_center)

    print("\n================ 小信赖域边界 ================")
    for k in range(11):
        print(f"Q{k:02d}: [{lb[k]: .6f}, {ub[k]: .6f}] | center={q_center[k]: .6f}")
    print("=============================================\n")

    conv0 = STGCNPathVLinResBigMConverter(c["engine_path"], slack_voltage=c["slack_vm_pu"])
    xlb, xub = x6_bounds(s, conv0, lb, ub, c["sign_mode"])

    env, m = create_model(c)

    try:
        Q, sc_n = add_q_vars(m, c, lb, ub, q_center)

        conv = STGCNPathVLinResBigMConverter(c["engine_path"], slack_voltage=c["slack_vm_pu"])
        conv.set_opf_input_bounds(xlb, xub)

        V, I, aux = conv.embed(
            m,
            x_expr(s, Q, c["sign_mode"]),
            name="enum_trust",
            relu_starts=starts,
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

        m.addConstr(obj <= center_obj + c["obj_cutoff_margin"], name="objective_cutoff")
        m.setObjective(obj, GRB.MINIMIZE)
        m.update()

        print("\n================ Gurobi 模型规模 ================")
        print(f"变量数: {m.NumVars}")
        print(f"约束数: {m.NumConstrs}")
        print(f"二元变量数: {m.NumBinVars}")
        print("ReLU二元变量数:", aux["binary_created"])
        print("ReLU固定为0:", aux["relu_fixed_zero"])
        print("ReLU固定为线性:", aux["relu_fixed_linear"])
        print("SC整数档位变量:", "启用" if sc_n is not None else "未启用")
        print("engine理论二元变量:", aux["binary_theoretical"])
        print("================================================\n")

        m.optimize()

        if m.SolCount <= 0:
            print(f"未得到可行解，status={m.status}")
            return None

        q = np.array([x.X for x in Q])
        Vp = np.array([x.X for x in V])
        Ip = np.array([x.X for x in I])

        result = {
            "q": q,
            "V_pred": Vp,
            "I_pred": Ip,
            "obj": float(m.ObjVal),
            "bound": float(m.ObjBound),
            "gap": float(m.MIPGap),
            "lb": lb,
            "ub": ub,
            "aux": aux,
            "pv_cap": pv_cap,
            "ess_cap": ess_cap,
        }

        print("\n【MILP结果】")
        print(f"Obj={result['obj']:.8f}, Bound={result['bound']:.8f}, Gap={result['gap'] * 100:.2f}%")
        print(f"预测电压范围: {Vp[1:].min():.6f} ~ {Vp[1:].max():.6f}")
        print(f"预测最大支路裕度: {Ip.max():.6f}")

        return result

    finally:
        m.dispose()
        env.dispose()


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

    pp.runpp(net, algorithm="bfsw", init="flat", tolerance_mva=1e-7, max_iteration=100,
             enforce_q_lims=False, calculate_voltage_angles=False, numba=False)

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
        trust_lb=result["lb"],
        trust_ub=result["ub"],
        V_pred=result["V_pred"],
        I_pred=result["I_pred"],
        obj=result["obj"],
        bound=result["bound"],
        gap=result["gap"],
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


def solve():
    c = cfg()
    s = profiles(c["time_index"])

    q_center, center_obj, starts = pytorch_sc_enum_multistart(s, c)
    pp_no = runpp(s, np.zeros(11), c, "未优化")

    result = solve_milp(s, c, q_center, center_obj, starts)

    if result is None:
        return

    pp_opt = runpp(s, result["q"], c, "优化后")

    qpv, qess, qdev = split_q(result["q"])
    pv_cap, ess_cap = result["pv_cap"], result["ess_cap"]

    print("\n================ 最终无功调度结果 ================")
    for k, b in enumerate(s["pv_nodes"]):
        print(f"PV  Bus {b + 1:02d} | Q={qpv[k]: .6f} | Qcap={pv_cap[k]:.6f}")
    for k, b in enumerate(s["ess_nodes"]):
        print(f"ESS Bus {b + 1:02d} | Q={qess[k]: .6f} | Qcap={ess_cap[k]:.6f}")
    for k, b in enumerate(s["qdev_nodes"]):
        print(f"{s['qdev_names'][k]:4s} Bus {b + 1:02d} | Q={qdev[k]: .6f}")

    save_result_and_plot(c, s, result, pp_no, pp_opt)


if __name__ == "__main__":
    solve()