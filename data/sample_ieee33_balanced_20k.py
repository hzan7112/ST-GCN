from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SEED = 42

DEFAULT_POOL = r"D:\pythonproject\ST-GCN\data\ieee33_static_vvo_raw_classified_pool_50k.pt"
DEFAULT_OUT = r"D:\pythonproject\ST-GCN\data\ieee33_static_vvo_balanced_20k.pt"

TARGET_SIZE = 20000

TARGET_QUOTAS = {
    "safe_inner": 5000,
    "boundary": 9000,
    "moderate": 4000,
    "extreme": 2000,
}

SHORTAGE_STRATEGY = "borrow"
MIN_SAMPLES_PER_HOUR = 400

BORROW_ORDER = {
    "safe_inner": ["boundary", "moderate", "extreme"],
    "boundary": ["moderate", "safe_inner", "extreme"],
    "moderate": ["boundary", "extreme", "safe_inner"],
    "extreme": ["moderate", "boundary", "safe_inner"],
}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pool(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["Y_V", "Y_I", "hour"]:
        if key not in data:
            raise KeyError(f"母池缺少字段: {key}")
    return data


def classify_mutually_exclusive(pool):
    YV = to_numpy(pool["Y_V"])
    YI = to_numpy(pool["Y_I"])

    vmin = YV.min(axis=1)
    vmax = YV.max(axis=1)
    imax = YI.max(axis=1)
    loading = (imax + 1.0) * 100.0

    extreme = (
        (vmin < 0.88)
        | (vmax > 1.12)
        | (loading > 180.0)
    )

    boundary = (
        (~extreme)
        & (
            ((vmin >= 0.94) & (vmin <= 0.96))
            | ((vmax >= 1.04) & (vmax <= 1.06))
            | ((loading >= 90.0) & (loading <= 110.0))
        )
    )

    safe_inner = (
        (~extreme)
        & (~boundary)
        & (vmin >= 0.96)
        & (vmax <= 1.04)
        & (loading <= 90.0)
    )

    moderate = ~(extreme | boundary | safe_inner)

    names = np.full(len(YV), "", dtype=object)
    names[safe_inner] = "safe_inner"
    names[boundary] = "boundary"
    names[moderate] = "moderate"
    names[extreme] = "extreme"

    if np.any(names == ""):
        raise RuntimeError("存在未分类样本。")

    metrics = {
        "vmin": vmin.astype(np.float32),
        "vmax": vmax.astype(np.float32),
        "max_loading_percent": loading.astype(np.float32),
    }
    return names, metrics


def print_pool_stats(class_name, hours):
    n = len(class_name)
    print("\n================ 母池四类互斥统计 ================")
    for name in TARGET_QUOTAS:
        c = int(np.sum(class_name == name))
        print(
            f"{name:12s}: {c:6d} "
            f"({100.0*c/n:6.2f}%)  target={TARGET_QUOTAS[name]:5d}"
        )

    print("\n各类别24h最小/最大样本数:")
    for name in TARGET_QUOTAS:
        idx = np.where(class_name == name)[0]
        counts = np.bincount(hours[idx], minlength=24)
        print(
            f"{name:12s}: min={counts.min():4d}, "
            f"max={counts.max():4d}, total={counts.sum():6d}"
        )
    print("====================================================\n")


def hour_balanced_unique_sample(candidates, quota, hours, rng):
    candidates = np.asarray(candidates, dtype=np.int64)
    if quota <= 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64)

    quota = min(quota, len(candidates))
    by_hour = {
        h: candidates[hours[candidates] == h]
        for h in range(24)
    }
    active = [h for h in range(24) if len(by_hour[h]) > 0]

    base = quota // len(active)
    rem = quota % len(active)

    extra_hours = set(
        rng.choice(
            np.asarray(active),
            size=min(rem, len(active)),
            replace=False,
        ).tolist()
    )

    selected = []
    for h in active:
        qh = base + (1 if h in extra_hours else 0)
        take = min(qh, len(by_hour[h]))
        if take > 0:
            selected.extend(
                rng.choice(by_hour[h], size=take, replace=False).tolist()
            )

    selected = np.asarray(selected, dtype=np.int64)

    if len(selected) < quota:
        used = set(selected.tolist())
        rest = np.asarray(
            [i for i in candidates if i not in used],
            dtype=np.int64,
        )
        need = quota - len(selected)
        if need > 0:
            extra = rng.choice(rest, size=need, replace=False)
            selected = np.concatenate([selected, extra])

    return selected


def initial_select(class_name, hours, rng, strategy):
    selected_by_target = {}
    used = np.zeros(len(class_name), dtype=bool)
    logs = []

    for target_class, quota in TARGET_QUOTAS.items():
        candidates = np.where(
            (class_name == target_class) & (~used)
        )[0]

        take = min(quota, len(candidates))
        chosen = hour_balanced_unique_sample(
            candidates, take, hours, rng
        )

        used[chosen] = True
        selected_by_target[target_class] = chosen.tolist()

        shortage = quota - len(chosen)
        if shortage > 0:
            logs.append(
                {
                    "target_class": target_class,
                    "available": len(candidates),
                    "shortage": shortage,
                    "action": "",
                }
            )

    for log in logs:
        target_class = log["target_class"]
        need = TARGET_QUOTAS[target_class] - len(
            selected_by_target[target_class]
        )

        if strategy == "replace":
            own_all = np.where(class_name == target_class)[0]
            if len(own_all) == 0:
                raise RuntimeError(
                    f"{target_class} 数量为0，无法本类有放回补采。"
                )
            extra = rng.choice(own_all, size=need, replace=True)
            selected_by_target[target_class].extend(extra.tolist())
            log["action"] = f"replace_within_class={need}"
            continue

        borrowed_detail = {}

        for donor_class in BORROW_ORDER[target_class]:
            if need <= 0:
                break

            donor_candidates = np.where(
                (class_name == donor_class) & (~used)
            )[0]

            if len(donor_candidates) == 0:
                continue

            take = min(need, len(donor_candidates))
            extra = hour_balanced_unique_sample(
                donor_candidates, take, hours, rng
            )

            used[extra] = True
            selected_by_target[target_class].extend(extra.tolist())
            borrowed_detail[donor_class] = len(extra)
            need -= len(extra)

        if need > 0:
            all_idx = np.arange(len(class_name))
            extra = rng.choice(all_idx, size=need, replace=True)
            selected_by_target[target_class].extend(extra.tolist())
            borrowed_detail["fallback_replace"] = need

        log["action"] = "borrow:" + ",".join(
            f"{k}={v}" for k, v in borrowed_detail.items()
        )

    return selected_by_target, logs


def flatten_selection(selected_by_target):
    selected = []
    target_label = []

    for target_class in TARGET_QUOTAS:
        idx = selected_by_target[target_class]
        selected.extend(idx)
        target_label.extend([target_class] * len(idx))

    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(target_label, dtype=object),
    )


def repair_hour_minimum(
    selected,
    target_label,
    actual_class_name,
    hours,
    rng,
    min_per_hour,
):
    selected = selected.copy()
    selected_set = set(selected.tolist())
    logs = []

    for h in range(24):
        while np.sum(hours[selected] == h) < min_per_hour:
            deficit = min_per_hour - int(
                np.sum(hours[selected] == h)
            )
            changed = False

            for target_class in TARGET_QUOTAS:
                positions = np.where(target_label == target_class)[0]

                candidates = np.where(
                    (hours == h)
                    & (actual_class_name == target_class)
                )[0]
                candidates = np.asarray(
                    [i for i in candidates if i not in selected_set],
                    dtype=np.int64,
                )

                if len(candidates) == 0:
                    continue

                final_hour_counts = np.bincount(
                    hours[selected], minlength=24
                )

                donor_positions = []
                for pos in positions:
                    donor_h = int(hours[selected[pos]])
                    if (
                        donor_h != h
                        and final_hour_counts[donor_h] > min_per_hour
                    ):
                        donor_positions.append(pos)

                if not donor_positions:
                    continue

                swap_n = min(
                    deficit,
                    len(candidates),
                    len(donor_positions),
                )

                in_idx = rng.choice(
                    candidates, size=swap_n, replace=False
                )
                out_pos = rng.choice(
                    np.asarray(donor_positions),
                    size=swap_n,
                    replace=False,
                )

                for pos, new_idx in zip(out_pos, in_idx):
                    old_idx = int(selected[pos])
                    selected_set.discard(old_idx)
                    selected[pos] = int(new_idx)
                    selected_set.add(int(new_idx))

                logs.append(
                    f"hour={h:02d}: swap {swap_n} within {target_class}"
                )
                changed = True
                break

            if not changed:
                logs.append(
                    f"WARNING hour={h:02d}: "
                    f"current={np.sum(hours[selected] == h)} "
                    f"< minimum={min_per_hour}"
                )
                break

    return selected, logs


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
    parser.add_argument(
        "--shortage-strategy",
        choices=["borrow", "replace"],
        default=SHORTAGE_STRATEGY,
    )
    parser.add_argument(
        "--min-per-hour",
        type=int,
        default=MIN_SAMPLES_PER_HOUR,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    pool = load_pool(args.pool)
    hours = to_numpy(pool["hour"]).astype(np.int64)

    class_name, metrics = classify_mutually_exclusive(pool)
    print_pool_stats(class_name, hours)

    print(f"Shortage strategy: {args.shortage_strategy}")
    print(f"Minimum samples per hour: {args.min_per_hour}")

    selected_by_target, shortage_log = initial_select(
        class_name, hours, rng, args.shortage_strategy
    )

    selected, target_label = flatten_selection(
        selected_by_target
    )

    if len(selected) != TARGET_SIZE:
        raise RuntimeError(
            f"初始抽样数量错误: {len(selected)} != {TARGET_SIZE}"
        )

    selected, repair_log = repair_hour_minimum(
        selected,
        target_label,
        class_name,
        hours,
        rng,
        args.min_per_hour,
    )

    final = subset_dataset(pool, selected)

    actual_selected_class = class_name[selected]

    final["selected_pool_indices"] = torch.tensor(
        selected, dtype=torch.long
    )
    final["selected_target_class"] = target_label.tolist()
    final["selected_actual_class"] = actual_selected_class.tolist()
    final["selection_metrics"] = {
        "vmin": torch.tensor(
            metrics["vmin"][selected], dtype=torch.float32
        ),
        "vmax": torch.tensor(
            metrics["vmax"][selected], dtype=torch.float32
        ),
        "max_loading_percent": torch.tensor(
            metrics["max_loading_percent"][selected],
            dtype=torch.float32,
        ),
    }

    target_counts = Counter(target_label.tolist())
    actual_counts = Counter(actual_selected_class.tolist())
    hour_counts = np.bincount(hours[selected], minlength=24)

    unique_count = len(np.unique(selected))
    duplicate_count = len(selected) - unique_count

    final["selection_info"] = {
        "seed": SEED,
        "pool_path": args.pool,
        "target_size": TARGET_SIZE,
        "target_quotas": TARGET_QUOTAS,
        "shortage_strategy": args.shortage_strategy,
        "shortage_log": shortage_log,
        "hour_repair_log": repair_log,
        "min_samples_per_hour": args.min_per_hour,
        "target_class_counts": dict(target_counts),
        "actual_class_counts": dict(actual_counts),
        "hour_counts": hour_counts.tolist(),
        "unique_sample_count": int(unique_count),
        "duplicate_sample_count": int(duplicate_count),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, out_path)

    print("\n================ 最终20k抽样统计 ================")

    print("\n[目标类别数量]")
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

    print("\n[类别不足处理]")
    if shortage_log:
        for log in shortage_log:
            print(log)
    else:
        print("所有类别均充足，没有触发借样或有放回补采。")

    print("\n[24h最终数量]")
    for h, c in enumerate(hour_counts):
        print(
            f"hour={h:02d}: {int(c):4d} "
            f"({100.0*c/TARGET_SIZE:5.2f}%)"
        )

    print("\n[重复样本]")
    print(f"unique={unique_count}")
    print(f"duplicate={duplicate_count}")

    if repair_log:
        print("\n[小时覆盖修复]")
        for line in repair_log:
            print(line)

    print("\n输出文件:")
    print(out_path)
    print("====================================================")


if __name__ == "__main__":
    main()
