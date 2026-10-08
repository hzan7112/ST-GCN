import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import train_exp25 as exp25


def parse_args():
    cfg = exp25.get_config()
    default_checkpoint = os.path.join(cfg["save_dir"], cfg["best_model_name"])

    parser = argparse.ArgumentParser(
        description="Evaluate train_exp25.py checkpoint on the test split."
    )
    parser.add_argument("--data-path", default=None, help="Dataset .pt path.")
    parser.add_argument("--checkpoint", default=default_checkpoint, help="Checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Optional directory for regression_metrics.csv and classification_metrics.csv.",
    )
    return parser.parse_args()


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_float_tensor_dict(stats):
    return {k: exp25.to_tensor(v).float() for k, v in stats.items()}


def pick_test_indices(data, cfg, n):
    for key in ("test_idx", "test_indices", "idx_test", "test_index"):
        if isinstance(data, dict) and key in data:
            return exp25.to_tensor(data[key]).long(), f"dataset[{key}]"

    if isinstance(data, dict) and "splits" in data and isinstance(data["splits"], dict):
        splits = data["splits"]
        for key in ("test", "test_idx", "test_indices"):
            if key in splits:
                return exp25.to_tensor(splits[key]).long(), f"dataset[splits][{key}]"

    idx = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    n_train = int(n * cfg["train_ratio"])
    return idx[n_train:], (
        f"reproduced holdout split: seed={cfg['seed']}, "
        f"train_ratio={cfg['train_ratio']}, using tail {n - n_train} samples"
    )


def build_features(data, checkpoint, edge_list):
    X_raw = exp25.to_tensor(data["X"]).float().numpy()

    if "downstream_matrix" in checkpoint and "path_power_matrix" in checkpoint:
        S_down = exp25.to_tensor(checkpoint["downstream_matrix"]).float().numpy()
        S_path = exp25.to_tensor(checkpoint["path_power_matrix"]).float().numpy()
    else:
        S_down, S_path, _ = exp25.build_topology_matrices(edge_list, 33)

    X_aug = torch.tensor(
        exp25.augment_path_power_features(X_raw, S_down, S_path),
        dtype=torch.float32,
    )
    return X_aug


def normalize_with_stats(X, YV, YI, norm):
    X_mean = norm["X_mean"].view(1, 1, -1)
    X_std = norm["X_std"].view(1, 1, -1).clamp_min(1e-6)
    Xn = (X - X_mean) / X_std

    YVn = torch.zeros_like(YV)
    YVn[:, 1:] = (
        (YV[:, 1:] - norm["YV_mean_wo_slack"].view(1, -1))
        / norm["YV_std_wo_slack"].view(1, -1).clamp_min(1e-6)
    )
    YIn = (YI - norm["YI_mean"].view(1, -1)) / norm["YI_std"].view(1, -1).clamp_min(1e-6)
    return Xn, YVn, YIn


def build_model(cfg, checkpoint, edge_list, device):
    W_vlin = checkpoint.get("voltage_linear_prior_W")
    b_vlin = checkpoint.get("voltage_linear_prior_b")
    W_ilin = checkpoint.get("current_linear_prior_W")
    b_ilin = checkpoint.get("current_linear_prior_b")
    dst_idx = checkpoint.get("current_linear_prior_dst_idx")

    missing = [
        name
        for name, value in (
            ("voltage_linear_prior_W", W_vlin),
            ("voltage_linear_prior_b", b_vlin),
            ("current_linear_prior_W", W_ilin),
            ("current_linear_prior_b", b_ilin),
            ("current_linear_prior_dst_idx", dst_idx),
        )
        if value is None
    ]
    if missing:
        raise KeyError(f"checkpoint is missing required exp25 prior tensors: {missing}")

    base_model = exp25.build_base_model(cfg, edge_list)
    model = exp25.VoltageCurrentLinearResidualModel(
        base_model=base_model,
        W_vlin=exp25.to_tensor(W_vlin).float(),
        b_vlin=exp25.to_tensor(b_vlin).float(),
        W_ilin=exp25.to_tensor(W_ilin).float(),
        b_ilin=exp25.to_tensor(b_ilin).float(),
        dst_idx=exp25.to_tensor(dst_idx).long(),
    ).to(device)

    wrapper_state = checkpoint.get("model_state_dict") or checkpoint.get("wrapper_state_dict")
    base_state = checkpoint.get("base_state_dict") or checkpoint.get("state_dict")

    if wrapper_state is not None:
        model.load_state_dict(wrapper_state)
    elif base_state is not None:
        model.base.load_state_dict(base_state)
    else:
        raise KeyError("checkpoint does not contain model_state_dict/base_state_dict")

    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, norm, device):
    V_pred_all, I_pred_all, V_true_all, I_true_all = [], [], [], []

    for X, _, _, YV, YI in loader:
        X = X.to(device)
        Vn, In, *_ = exp25.unpack_forward(model, X)
        V, I = exp25.denorm_outputs(Vn, In, norm)

        V_pred_all.append(V.cpu())
        I_pred_all.append(I.cpu())
        V_true_all.append(YV.cpu())
        I_true_all.append(YI.cpu())

    return (
        torch.cat(V_pred_all, dim=0),
        torch.cat(I_pred_all, dim=0),
        torch.cat(V_true_all, dim=0),
        torch.cat(I_true_all, dim=0),
    )


def regression_metrics(pred, true, eps=1e-8):
    err = pred - true
    abs_err = err.abs()
    rmse = torch.sqrt(torch.mean(err ** 2))
    value_range = (true.max() - true.min()).abs().clamp_min(eps)
    mean_abs = true.abs().mean().clamp_min(eps)

    return {
        "MAE": float(abs_err.mean()),
        "MAPE(%)": float((abs_err / true.abs().clamp_min(eps)).mean() * 100.0),
        "RMSE": float(rmse),
        "NRMSE_range(%)": float(rmse / value_range * 100.0),
        "NRMSE_mean(%)": float(rmse / mean_abs * 100.0),
        "MaxAE": float(abs_err.max()),
    }


def binary_classification_metrics(true_unsafe, pred_unsafe):
    tp = int((true_unsafe & pred_unsafe).sum().item())
    fn = int((true_unsafe & (~pred_unsafe)).sum().item())
    fp = int(((~true_unsafe) & pred_unsafe).sum().item())
    tn = int(((~true_unsafe) & (~pred_unsafe)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    unsafe_total = max(tp + fn, 1)
    safe_total = max(tn + fp, 1)

    unsafe_recall = tp / unsafe_total * 100.0
    safe_recall = tn / safe_total * 100.0

    return {
        "FalseSafe(%)": fn / unsafe_total * 100.0,
        "FalseViolate(%)": fp / safe_total * 100.0,
        "SafetyAcc(%)": (tp + tn) / total * 100.0,
        "BalancedAcc(%)": 0.5 * (unsafe_recall + safe_recall),
        "TrueUnsafe": tp + fn,
        "PredUnsafe": tp + fp,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def format_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def print_markdown_table(title, rows, columns):
    print(f"\n{title}")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(format_value(row[col]) for col in columns) + " |")


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    checkpoint = load_pt(args.checkpoint)
    cfg = exp25.get_config()
    cfg.update(checkpoint.get("config", {}))

    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    data = load_pt(cfg["data_path"])
    edge_list = checkpoint.get("edge_list") or exp25.get_edge_list(data)
    edge_list = exp25.sanitize_edge_list(edge_list)

    X_aug = build_features(data, checkpoint, edge_list)
    YV, YI = exp25.get_labels(data)

    test_idx, split_note = pick_test_indices(data, cfg, len(X_aug))

    if "norm_stats" in checkpoint:
        norm = as_float_tensor_dict(checkpoint["norm_stats"])
        Xn, YVn, YIn = normalize_with_stats(X_aug, YV, YI, norm)
    else:
        idx = torch.randperm(len(X_aug), generator=torch.Generator().manual_seed(cfg["seed"]))
        n_train = int(len(X_aug) * cfg["train_ratio"])
        Xn, YVn, YIn, norm = exp25.normalize_data(X_aug, YV, YI, idx[:n_train])
        norm = as_float_tensor_dict(norm)

    test_ds = TensorDataset(Xn[test_idx], YVn[test_idx], YIn[test_idx], YV[test_idx], YI[test_idx])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    model = build_model(cfg, checkpoint, edge_list, device)
    Vp, Ip, Vt, It = collect_predictions(model, test_loader, norm, device)

    regression_rows = []
    for name, pred, true in (
        ("Voltage", Vp[:, 1:], Vt[:, 1:]),
        ("Current", Ip, It),
    ):
        row = {"Target": name}
        row.update(regression_metrics(pred, true))
        regression_rows.append(row)

    v_true_unsafe = (Vt[:, 1:] < cfg["v_lower"]) | (Vt[:, 1:] > cfg["v_upper"])
    v_pred_unsafe = (Vp[:, 1:] < cfg["v_lower"]) | (Vp[:, 1:] > cfg["v_upper"])
    i_true_unsafe = It > 0.0
    i_pred_unsafe = Ip > 0.0

    classification_rows = []
    for name, true_unsafe, pred_unsafe in (
        ("Voltage", v_true_unsafe, v_pred_unsafe),
        ("Current", i_true_unsafe, i_pred_unsafe),
    ):
        row = {"Target": name}
        row.update(binary_classification_metrics(true_unsafe, pred_unsafe))
        classification_rows.append(row)

    regression_columns = [
        "Target",
        "MAE",
        "MAPE(%)",
        "RMSE",
        "NRMSE_range(%)",
        "NRMSE_mean(%)",
        "MaxAE",
    ]
    classification_columns = [
        "Target",
        "FalseSafe(%)",
        "FalseViolate(%)",
        "SafetyAcc(%)",
        "BalancedAcc(%)",
        "TrueUnsafe",
        "PredUnsafe",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    print("============== Exp25 Test Evaluation ==============")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {cfg['data_path']}")
    print(f"Test split: {split_note}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Voltage unsafe rule: V < {cfg['v_lower']} or V > {cfg['v_upper']}")
    print("Current unsafe rule: I > 0")

    print_markdown_table("Regression Metrics", regression_rows, regression_columns)
    print_markdown_table("Classification Metrics", classification_rows, classification_columns)
    print("====================================================")

    if args.csv_dir is not None:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        write_csv(csv_dir / "regression_metrics.csv", regression_rows, regression_columns)
        write_csv(csv_dir / "classification_metrics.csv", classification_rows, classification_columns)
        print(f"CSV saved to: {csv_dir}")


if __name__ == "__main__":
    main()
