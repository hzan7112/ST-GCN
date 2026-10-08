# analyze_relu_unstable.py
# 统计 ST-GCN MILP engine 中 ReLU 的 fixed_zero / fixed_linear / unstable 分布
# 适配字段：
#   M_plus_gcn_layers, M_minus_gcn_layers
#   M_plus_node, M_minus_node
#   M_plus_edge, M_minus_edge

import os
import csv
import torch
import numpy as np
from collections import defaultdict


ENGINE_PATH = r"D:\pythonproject\ST-GCN\checkpoints\st_gcn_h16_l2_pathfeat_vlinres_exp20_milp_engine.pt"
OUT_DIR = r"D:\pythonproject\ST-GCN\results"
TOL = 1e-8


def load_engine(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)


def status_of(lb, ub, tol=TOL):
    if ub <= tol:
        return "fixed_zero"
    if lb >= -tol:
        return "fixed_linear"
    return "unstable"


def add_records(records, group, layer, lb_arr, ub_arr):
    lb_arr = to_np(lb_arr)
    ub_arr = to_np(ub_arr)

    if lb_arr.shape != ub_arr.shape:
        raise ValueError(f"{group} layer {layer}: lb shape {lb_arr.shape} != ub shape {ub_arr.shape}")

    lb_flat = lb_arr.reshape(-1)
    ub_flat = ub_arr.reshape(-1)

    for idx, (lb, ub) in enumerate(zip(lb_flat, ub_flat)):
        records.append({
            "group": group,
            "layer": layer,
            "index": idx,
            "lb": float(lb),
            "ub": float(ub),
            "status": status_of(lb, ub),
            "width": float(ub - lb),
            "min_abs_bound": float(min(abs(lb), abs(ub))),
        })


def collect_relu_bounds(engine):
    records = []

    # GCN 主体层
    if "M_plus_gcn_layers" in engine and "M_minus_gcn_layers" in engine:
        mps = engine["M_plus_gcn_layers"]
        mms = engine["M_minus_gcn_layers"]

        if len(mps) != len(mms):
            raise ValueError("M_plus_gcn_layers 和 M_minus_gcn_layers 层数不一致")

        for layer, (mp, mm) in enumerate(zip(mps, mms)):
            lb = -to_np(mm)
            ub = to_np(mp)
            add_records(records, "gcn", layer, lb, ub)

    # node head
    if "M_plus_node" in engine and "M_minus_node" in engine:
        lb = -to_np(engine["M_minus_node"])
        ub = to_np(engine["M_plus_node"])
        add_records(records, "node_head", 0, lb, ub)

    # edge head
    if "M_plus_edge" in engine and "M_minus_edge" in engine:
        lb = -to_np(engine["M_minus_edge"])
        ub = to_np(engine["M_plus_edge"])
        add_records(records, "edge_head", 0, lb, ub)

    return records


def summarize(records):
    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    total = len(records)
    fixed_zero = sum(r["status"] == "fixed_zero" for r in records)
    fixed_linear = sum(r["status"] == "fixed_linear" for r in records)
    unstable = sum(r["status"] == "unstable" for r in records)

    print("\n================ ReLU unstable 总统计 ================")
    print(f"ReLU 总数:      {total}")
    print(f"固定为 0:       {fixed_zero:6d}  ({pct(fixed_zero, total):6.2f}%)")
    print(f"固定为线性:     {fixed_linear:6d}  ({pct(fixed_linear, total):6.2f}%)")
    print(f"unstable:       {unstable:6d}  ({pct(unstable, total):6.2f}%)")
    print("======================================================\n")

    groups = defaultdict(list)
    for r in records:
        groups[(r["group"], r["layer"])].append(r)

    print("按模块/层统计：")
    print(
        f"{'group':<14} {'layer':>5} {'total':>8} "
        f"{'fixed0':>8} {'linear':>8} {'unstable':>10} {'unstable%':>10} "
        f"{'avg_width':>12} {'median_width':>14} {'max_width':>12}"
    )

    for (group, layer), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(items)
        f0 = sum(r["status"] == "fixed_zero" for r in items)
        fl = sum(r["status"] == "fixed_linear" for r in items)
        un_items = [r for r in items if r["status"] == "unstable"]
        un = len(un_items)

        widths = np.array([r["width"] for r in un_items], dtype=float)
        avg_w = float(widths.mean()) if widths.size else 0.0
        med_w = float(np.median(widths)) if widths.size else 0.0
        max_w = float(widths.max()) if widths.size else 0.0

        print(
            f"{group:<14} {layer:>5} {n:>8} "
            f"{f0:>8} {fl:>8} {un:>10} {pct(un, n):>9.2f}% "
            f"{avg_w:>12.6g} {med_w:>14.6g} {max_w:>12.6g}"
        )

    print("\nunstable width 最大的前 30 个 ReLU：")
    top = sorted(
        [r for r in records if r["status"] == "unstable"],
        key=lambda x: x["width"],
        reverse=True
    )[:30]

    print(f"{'rank':>4} {'group':<14} {'layer':>5} {'index':>8} {'lb':>12} {'ub':>12} {'width':>12}")
    for k, r in enumerate(top, 1):
        print(
            f"{k:>4} {r['group']:<14} {r['layer']:>5} {r['index']:>8} "
            f"{r['lb']:>12.6g} {r['ub']:>12.6g} {r['width']:>12.6g}"
        )


def save_csv(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, "relu_unstable_detail.csv")
    summary_path = os.path.join(out_dir, "relu_unstable_summary.csv")

    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "layer", "index", "lb", "ub", "status", "width", "min_abs_bound"]
        )
        writer.writeheader()
        writer.writerows(records)

    groups = defaultdict(list)
    for r in records:
        groups[(r["group"], r["layer"])].append(r)

    summary_rows = []
    for (group, layer), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(items)
        f0 = sum(r["status"] == "fixed_zero" for r in items)
        fl = sum(r["status"] == "fixed_linear" for r in items)
        un_items = [r for r in items if r["status"] == "unstable"]
        widths = np.array([r["width"] for r in un_items], dtype=float)

        summary_rows.append({
            "group": group,
            "layer": layer,
            "total": n,
            "fixed_zero": f0,
            "fixed_linear": fl,
            "unstable": len(un_items),
            "unstable_pct": 100.0 * len(un_items) / n if n else 0.0,
            "avg_unstable_width": float(widths.mean()) if widths.size else 0.0,
            "median_unstable_width": float(np.median(widths)) if widths.size else 0.0,
            "max_unstable_width": float(widths.max()) if widths.size else 0.0,
        })

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group", "layer", "total",
                "fixed_zero", "fixed_linear", "unstable", "unstable_pct",
                "avg_unstable_width", "median_unstable_width", "max_unstable_width"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n已保存详细统计: {detail_path}")
    print(f"已保存汇总统计: {summary_path}")


def main():
    print(f"读取 engine: {ENGINE_PATH}")

    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"找不到 engine 文件: {ENGINE_PATH}")

    engine = load_engine(ENGINE_PATH)

    print("\ncheckpoint 顶层结构：")
    for k, v in engine.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor{tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k}: list({len(v)})")
        elif isinstance(v, dict):
            print(f"  {k}: dict({len(v)})")
        else:
            print(f"  {k}: {type(v).__name__}")

    records = collect_relu_bounds(engine)

    if not records:
        raise RuntimeError("没有找到 M_plus/M_minus ReLU bound 字段")

    summarize(records)
    save_csv(records, OUT_DIR)

    print("\n判断标准：")
    print("  fixed_zero   : ub <= 0，ReLU 永远关闭，不需要二元变量")
    print("  fixed_linear : lb >= 0，ReLU 永远线性激活，不需要二元变量")
    print("  unstable     : lb < 0 < ub，ReLU 激活状态不确定，需要二元变量")


if __name__ == "__main__":
    main()