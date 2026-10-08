from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SEED = 42

DEFAULT_POOL = r"D:\pythonproject\ST-GCN\data\ieee33_nodal_pq_correlated_raw_pool_50k.pt"
DEFAULT_OUT = r"D:\pythonproject\ST-GCN\data\ieee33_nodal_pq_correlated_balanced_20k.pt"

TARGET_SIZE = 20000

# 基于最新相关扰动母池重新设定：
# safe_inner 27.5%, boundary 47.5%, moderate 22.5%, extreme 2.5%
TARGET_QUOTAS = {
    "safe_inner": 5500,
    "boundary": 9500,
    "moderate": 4500,
    "extreme": 500,
}

# 类别不足时优先从物理相邻类别借样本，保持样本唯一。
# 只有整个母池唯一样本都不足时才会触发最终有放回兜底。
SHORTAGE_STRATEGY = "borrow"

# boundary 内部最低保留比例，仅在候选数量足够时执行。
# 一个样本可同时属于多个边界子类型，因此这里只用于加权抽样，不是硬配额。
BOUNDARY_WEIGHTS = {
    "low_v": 1.0,
    "high_v": 1.0,   # 高压边界天然较少，适度提高权重
    "current": 1.0,
    "multi": 1.0,    # 同时接近多个约束边界的样本更有价值
}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pool(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["Y_V", "Y_I"]:
        if key not in data:
            raise KeyError(f"母池缺少字段: {key}")
    return data


def classify_four_classes(pool):
    YV = to_numpy(pool["Y_V"])
    YI = to_numpy(pool["Y_I"])

    vmin = YV.min(axis=1)
    vmax = YV.max(axis=1)
    imax = YI.max(axis=1)
    loading = (imax + 1.0) * 100.0

    # 极端区域优先级最高
    extreme = (
        (vmin < 0.88)
        | (vmax > 1.12)
        | (loading > 180.0)
    )

    # 实际 MILP 约束附近
    low_v_boundary = (vmin >= 0.94) & (vmin <= 0.96)
    high_v_boundary = (vmax >= 1.04) & (vmax <= 1.06)
    current_boundary = (loading >= 90.0) & (loading <= 110.0)

    boundary = (
        (~extreme)
        & (low_v_boundary | high_v_boundary | current_boundary)
    )

    # 明确位于安全内部
    safe_inner = (
        (~extreme)
        & (~boundary)
        & (vmin >= 0.96)
        & (vmax <= 1.04)
        & (loading <= 90.0)
    )

    # 其余非极端样本均归入中度区域
    moderate = ~(extreme | boundary | safe_inner)

    names = np.full(len(YV), "", dtype=object)
    names[safe_inner] = "safe_inner"
    names[boundary] = "boundary"
    names[moderate] = "moderate"
    names[extreme] = "extreme"

    if np.any(names == ""):
        raise RuntimeError("存在未分类样本。")

    boundary_flags = {
        "low_v": low_v_boundary & boundary,
        "high_v": high_v_boundary & boundary,
        "current": current_boundary & boundary,
    }
    multi_boundary = (
        boundary_flags["low_v"].astype(np.int64)
        + boundary_flags["high_v"].astype(np.int64)
        + boundary_flags["current"].astype(np.int64)
    ) >= 2
    boundary_flags["multi"] = multi_boundary

    metrics = {
        "vmin": vmin.astype(np.float32),
        "vmax": vmax.astype(np.float32),
        "imax": imax.astype(np.float32),
        "max_loading_percent": loading.astype(np.float32),
    }

    return names, boundary_flags, metrics


def print_pool_stats(class_name, boundary_flags):
    n = len(class_name)

    print("\n================ 新母池四类互斥统计 ================")
    for name, quota in TARGET_QUOTAS.items():
        c = int(np.sum(class_name == name))
        print(
            f"{name:12s}: {c:6d} "
            f"({100.0*c/n:6.2f}%)  target={quota:5d}"
        )

    print("\n[boundary 子类型，允许重叠]")
    for key in ["low_v", "high_v", "current", "multi"]:
        c = int(boundary_flags[key].sum())
        print(f"{key:10s}: {c:6d} ({100.0*c/n:6.2f}% of pool)")

    print("====================================================\n")


def weighted_unique_sample(candidates, quota, rng, weights=None):
    candidates = np.asarray(candidates, dtype=np.int64)
    quota = min(int(quota), len(candidates))

    if quota <= 0:
        return np.empty(0, dtype=np.int64)

    if weights is None:
        return rng.choice(candidates, size=quota, replace=False)

    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(candidates):
        raise ValueError("weights 与 candidates 长度不一致。")

    w = np.maximum(w, 1e-12)
    w = w / w.sum()

    return rng.choice(
        candidates,
        size=quota,
        replace=False,
        p=w,
    )


def boundary_sampling_weights(candidates, boundary_flags):
    w = np.ones(len(candidates), dtype=np.float64)

    for key, factor in BOUNDARY_WEIGHTS.items():
        mask = boundary_flags[key][candidates]
        w[mask] *= factor

    return w


BORROW_ORDER = {
    "safe_inner": ["boundary", "moderate", "extreme"],
    "boundary": ["moderate", "safe_inner", "extreme"],
    "moderate": ["boundary", "safe_inner", "extreme"],
    "extreme": ["moderate", "boundary", "safe_inner"],
}


def sample_target_classes(class_name, boundary_flags, rng):
    used = np.zeros(len(class_name), dtype=bool)
    selected_by_target = {}
    shortage_log = []

    # 第一轮：全部在本类中无放回抽样
    for target_class, quota in TARGET_QUOTAS.items():
        candidates = np.where(
            (class_name == target_class) & (~used)
        )[0]

        if target_class == "boundary":
            weights = boundary_sampling_weights(
                candidates, boundary_flags
            )
        else:
            weights = None

        chosen = weighted_unique_sample(
            candidates,
            quota,
            rng,
            weights=weights,
        )

        used[chosen] = True
        selected_by_target[target_class] = chosen.tolist()

        shortage = quota - len(chosen)
        if shortage > 0:
            shortage_log.append(
                {
                    "target_class": target_class,
                    "available": int(len(candidates)),
                    "shortage": int(shortage),
                    "action": "",
                }
            )

    # 第二轮：不足类别从相邻类别借唯一样本
    for log in shortage_log:
        target_class = log["target_class"]
        need = (
            TARGET_QUOTAS[target_class]
            - len(selected_by_target[target_class])
        )

        borrowed_detail = {}

        for donor_class in BORROW_ORDER[target_class]:
            if need <= 0:
                break

            donor = np.where(
                (class_name == donor_class) & (~used)
            )[0]

            if len(donor) == 0:
                continue

            take = min(need, len(donor))

            if donor_class == "boundary":
                weights = boundary_sampling_weights(
                    donor, boundary_flags
                )
            else:
                weights = None

            extra = weighted_unique_sample(
                donor,
                take,
                rng,
                weights=weights,
            )

            used[extra] = True
            selected_by_target[target_class].extend(extra.tolist())
            borrowed_detail[donor_class] = len(extra)
            need -= len(extra)

        # 最后一层兜底，仅理论上母池唯一样本总数不足才会触发
        if need > 0:
            fallback = rng.choice(
                np.arange(len(class_name)),
                size=need,
                replace=True,
            )
            selected_by_target[target_class].extend(
                fallback.tolist()
            )
            borrowed_detail["fallback_replace"] = need

        log["action"] = "borrow:" + ",".join(
            f"{k}={v}" for k, v in borrowed_detail.items()
        )

    selected = []
    target_label = []

    for target_class in TARGET_QUOTAS:
        idx = selected_by_target[target_class]
        selected.extend(idx)
        target_label.extend([target_class] * len(idx))

    selected = np.asarray(selected, dtype=np.int64)
    target_label = np.asarray(target_label, dtype=object)

    if len(selected) != TARGET_SIZE:
        raise RuntimeError(
            f"最终抽样数量错误: {len(selected)} != {TARGET_SIZE}"
        )

    order = rng.permutation(len(selected))
    return selected[order], target_label[order], shortage_log


def subset_dataset(pool, selected):
    n_pool = len(pool["Y_V"])
    out = {}

    for k, v in pool.items():
        if (
            isinstance(v, torch.Tensor)
            and v.ndim >= 1
            and len(v) == n_pool
        ):
            out[k] = v[selected]
        else:
            out[k] = v

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    pool = load_pool(args.pool)
    class_name, boundary_flags, metrics = classify_four_classes(pool)

    print_pool_stats(class_name, boundary_flags)

    selected, target_label, shortage_log = sample_target_classes(
        class_name,
        boundary_flags,
        rng,
    )

    final = subset_dataset(pool, selected)

    actual_class = class_name[selected]
    actual_boundary = {
        key: flags[selected]
        for key, flags in boundary_flags.items()
    }

    final["selected_pool_indices"] = torch.tensor(
        selected,
        dtype=torch.long,
    )
    final["selected_target_class"] = target_label.tolist()
    final["selected_actual_class"] = actual_class.tolist()

    final["selection_metrics"] = {
        "vmin": torch.tensor(
            metrics["vmin"][selected],
            dtype=torch.float32,
        ),
        "vmax": torch.tensor(
            metrics["vmax"][selected],
            dtype=torch.float32,
        ),
        "max_loading_percent": torch.tensor(
            metrics["max_loading_percent"][selected],
            dtype=torch.float32,
        ),
    }

    final["boundary_flags"] = {
        key: torch.tensor(value, dtype=torch.bool)
        for key, value in actual_boundary.items()
    }

    target_counts = Counter(target_label.tolist())
    actual_counts = Counter(actual_class.tolist())

    unique_count = len(np.unique(selected))
    duplicate_count = len(selected) - unique_count

    boundary_counts = {
        key: int(value.sum())
        for key, value in actual_boundary.items()
    }

    final["selection_info"] = {
        "seed": SEED,
        "pool_path": args.pool,
        "target_size": TARGET_SIZE,
        "target_quotas": TARGET_QUOTAS,
        "shortage_strategy": SHORTAGE_STRATEGY,
        "shortage_log": shortage_log,
        "target_class_counts": dict(target_counts),
        "actual_class_counts": dict(actual_counts),
        "boundary_subtype_counts": boundary_counts,
        "unique_sample_count": int(unique_count),
        "duplicate_sample_count": int(duplicate_count),
        "classification_rule": {
            "safe_inner": (
                "not extreme/boundary and "
                "Vmin>=0.96, Vmax<=1.04, max_loading<=90%"
            ),
            "boundary": (
                "not extreme and any of "
                "0.94<=Vmin<=0.96, "
                "1.04<=Vmax<=1.06, "
                "90%<=max_loading<=110%"
            ),
            "moderate": (
                "all remaining non-extreme samples"
            ),
            "extreme": (
                "Vmin<0.88 or Vmax>1.12 or max_loading>180%"
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, out_path)

    print("\n================ 最终20k四类抽样统计 ================")

    print("\n[目标类别]")
    for name in TARGET_QUOTAS:
        print(
            f"{name:12s}: "
            f"{target_counts.get(name, 0):5d} / "
            f"{TARGET_QUOTAS[name]:5d}"
        )

    print("\n[实际来源类别]")
    for name in TARGET_QUOTAS:
        c = actual_counts.get(name, 0)
        print(
            f"{name:12s}: {c:5d} "
            f"({100.0*c/TARGET_SIZE:6.2f}%)"
        )

    print("\n[最终 boundary 子类型，允许重叠]")
    for key in ["low_v", "high_v", "current", "multi"]:
        c = boundary_counts[key]
        print(
            f"{key:10s}: {c:5d} "
            f"({100.0*c/TARGET_SIZE:6.2f}% of final)"
        )

    print("\n[类别不足处理]")
    if shortage_log:
        for log in shortage_log:
            print(log)
    else:
        print("所有类别数量充足，没有触发借样或重复采样。")

    print("\n[重复样本]")
    print(f"unique={unique_count}")
    print(f"duplicate={duplicate_count}")

    print("\n输出文件:")
    print(out_path)
    print("====================================================")


if __name__ == "__main__":
    main()
