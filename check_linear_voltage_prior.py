from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

DEFAULT_DATA = r"D:\pythonproject\ST-GCN\data\ieee33_static_vvo_24h_dataset.pt"
DEFAULT_OUT = r"D:\pythonproject\ST-GCN\diagnostics\linear_voltage_diagnostic"

CURRENT_BINS = [
    (0.0, 50.0),
    (50.0, 80.0),
    (80.0, 100.0),
    (100.0, 130.0),
    (130.0, 180.0),
]

VMIN_BINS = [
    (0.90, 0.94),
    (0.94, 0.97),
    (0.97, 1.00),
    (1.00, 1.03),
    (1.03, 1.06),
    (1.06, 1.10),
]


def build_topology_matrices(edge_list, n_bus):
    children = [[] for _ in range(n_bus)]
    parent = [-1] * n_bus

    for f, t in edge_list:
        children[int(f)].append(int(t))
        parent[int(t)] = int(f)

    descendants = [None] * n_bus

    def collect_desc(i):
        out = [i]
        for c in children[i]:
            out.extend(collect_desc(c))
        descendants[i] = out
        return out

    collect_desc(0)

    S_down = np.zeros((n_bus, n_bus), dtype=np.float32)
    S_path = np.zeros((n_bus, n_bus), dtype=np.float32)

    for i in range(n_bus):
        S_down[i, descendants[i]] = 1.0

        cur = i
        while cur >= 0:
            S_path[i, cur] = 1.0
            cur = parent[cur]

    return S_down, S_path


def augment_x6(X, edge_list):
    X = np.asarray(X, dtype=np.float32)
    n_bus = X.shape[1]
    S_down, S_path = build_topology_matrices(edge_list, n_bus)

    P = X[:, :, 0]
    Q = X[:, :, 1]

    P_down = P @ S_down.T
    Q_down = Q @ S_down.T
    P_path = P @ S_path.T
    Q_path = Q @ S_path.T

    return np.stack(
        [P, Q, P_down, Q_down, P_path, Q_path],
        axis=2,
    ).astype(np.float32)


def make_split(n, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


class LinearVoltageModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean,
        std,
    )


def train_linear(train_x, train_y, val_x, val_y, epochs, batch_size, lr, device):
    torch.manual_seed(SEED)

    model = LinearVoltageModel(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)

    best_state = None
    best_val = float("inf")
    patience = 100
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mae = torch.mean(torch.abs(model(val_x_t) - val_y_t)).item()

        if val_mae < best_val - 1e-8:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch={epoch:4d}  val_mae={val_mae:.8f}  best={best_val:.8f}")

        if stale >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return pred


def mae_rows(pred, true):
    return np.mean(np.abs(pred - true), axis=1)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bin_stats(values, sample_mae, bins, label_name):
    rows = []
    for i, (lo, hi) in enumerate(bins):
        if i == len(bins) - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)

        count = int(mask.sum())
        rows.append(
            {
                "bin": f"[{lo:.4f},{hi:.4f}{']' if i == len(bins)-1 else ')'}",
                "count": count,
                "ratio_percent": 100.0 * count / len(values),
                "voltage_mae": float(sample_mae[mask].mean()) if count else np.nan,
                label_name + "_mean": float(values[mask].mean()) if count else np.nan,
                label_name + "_min": float(values[mask].min()) if count else np.nan,
                label_name + "_max": float(values[mask].max()) if count else np.nan,
            }
        )
    return rows


def hour_stats(hours, sample_mae, max_loading, vmin):
    rows = []
    total = len(hours)
    for h in range(24):
        mask = hours == h
        count = int(mask.sum())
        rows.append(
            {
                "hour": h,
                "count": count,
                "ratio_percent": 100.0 * count / total,
                "voltage_mae": float(sample_mae[mask].mean()) if count else np.nan,
                "max_loading_percent_mean": float(max_loading[mask].mean()) if count else np.nan,
                "vmin_mean": float(vmin[mask].mean()) if count else np.nan,
            }
        )
    return rows


def mode_stats(modes, sample_mae, max_loading, vmin):
    rows = []
    total = len(modes)
    for m in sorted(np.unique(modes).tolist()):
        mask = modes == m
        count = int(mask.sum())
        rows.append(
            {
                "mode": int(m),
                "count": count,
                "ratio_percent": 100.0 * count / total,
                "voltage_mae": float(sample_mae[mask].mean()) if count else np.nan,
                "max_loading_percent_mean": float(max_loading[mask].mean()) if count else np.nan,
                "vmin_mean": float(vmin[mask].mean()) if count else np.nan,
            }
        )
    return rows


def correlation_report(sample_mae, max_loading, vmin, vdev):
    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "corr_mae_vs_max_loading": corr(sample_mae, max_loading),
        "corr_mae_vs_vmin": corr(sample_mae, vmin),
        "corr_mae_vs_voltage_deviation": corr(sample_mae, vdev),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether the IEEE33 voltage dataset is too close to a linear operating manifold."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--features", choices=["x2", "x6"], default="x6")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    data = torch.load(args.data, map_location="cpu", weights_only=False)

    X2 = data["X"].cpu().numpy().astype(np.float32)
    YV = data["Y_V"].cpu().numpy().astype(np.float32)
    YI = data["Y_I"].cpu().numpy().astype(np.float32)
    hours = data["hour"].cpu().numpy().astype(np.int64)
    modes = data["mode"].cpu().numpy().astype(np.int64)
    edge_list = data["edge_list"]

    if args.features == "x6":
        X = augment_x6(X2, edge_list)
    else:
        X = X2

    X = X.reshape(len(X), -1)

    train_idx, val_idx, test_idx = make_split(len(X))

    train_x = X[train_idx]
    val_x = X[val_idx]
    test_x = X[test_idx]

    train_y = YV[train_idx]
    val_y = YV[val_idx]
    test_y = YV[test_idx]

    train_x, val_x, test_x, x_mean, x_std = standardize(train_x, val_x, test_x)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Feature set: {args.features}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Device: {device}")

    model, best_val = train_linear(
        train_x,
        train_y,
        val_x,
        val_y,
        args.epochs,
        args.batch_size,
        args.lr,
        device,
    )

    test_pred = predict(model, test_x, device)
    sample_mae = mae_rows(test_pred, test_y)

    global_mae = float(sample_mae.mean())
    global_rmse = float(np.sqrt(np.mean((test_pred - test_y) ** 2)))
    global_maxerr = float(np.max(np.abs(test_pred - test_y)))

    test_yi = YI[test_idx]
    max_loading = (test_yi.max(axis=1) + 1.0) * 100.0
    vmin = test_y.min(axis=1)
    vmax = test_y.max(axis=1)
    vdev = np.maximum(np.abs(vmin - 1.0), np.abs(vmax - 1.0))
    test_hours = hours[test_idx]
    test_modes = modes[test_idx]

    current_rows = bin_stats(
        max_loading,
        sample_mae,
        CURRENT_BINS,
        "max_loading_percent",
    )
    vmin_rows = bin_stats(
        vmin,
        sample_mae,
        VMIN_BINS,
        "vmin",
    )
    hour_rows = hour_stats(
        test_hours,
        sample_mae,
        max_loading,
        vmin,
    )
    mode_rows = mode_stats(
        test_modes,
        sample_mae,
        max_loading,
        vmin,
    )

    corr = correlation_report(sample_mae, max_loading, vmin, vdev)

    write_csv(
        out_dir / "mae_by_current_loading.csv",
        current_rows,
        list(current_rows[0].keys()),
    )
    write_csv(
        out_dir / "mae_by_min_voltage.csv",
        vmin_rows,
        list(vmin_rows[0].keys()),
    )
    write_csv(
        out_dir / "mae_by_hour.csv",
        hour_rows,
        list(hour_rows[0].keys()),
    )
    write_csv(
        out_dir / "mae_by_mode.csv",
        mode_rows,
        list(mode_rows[0].keys()),
    )

    test_detail_rows = []
    for k, idx0 in enumerate(test_idx):
        test_detail_rows.append(
            {
                "dataset_index": int(idx0),
                "hour": int(test_hours[k]),
                "mode": int(test_modes[k]),
                "sample_voltage_mae": float(sample_mae[k]),
                "max_loading_percent": float(max_loading[k]),
                "vmin": float(vmin[k]),
                "vmax": float(vmax[k]),
                "max_voltage_deviation": float(vdev[k]),
            }
        )

    write_csv(
        out_dir / "test_sample_detail.csv",
        test_detail_rows,
        list(test_detail_rows[0].keys()),
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "features": args.features,
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
        "best_val_mae": best_val,
        "test_mae": global_mae,
    }
    torch.save(checkpoint, out_dir / "linear_voltage_diagnostic_model.pt")

    summary_lines = [
        "================ Linear Voltage Diagnostic ================",
        f"dataset: {args.data}",
        f"features: {args.features}",
        f"train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        f"best_val_mae: {best_val:.8f}",
        f"test_mae: {global_mae:.8f}",
        f"test_rmse: {global_rmse:.8f}",
        f"test_maxerr: {global_maxerr:.8f}",
        "",
        "Correlation:",
        f"MAE vs max loading: {corr['corr_mae_vs_max_loading']:.6f}",
        f"MAE vs vmin: {corr['corr_mae_vs_vmin']:.6f}",
        f"MAE vs max voltage deviation: {corr['corr_mae_vs_voltage_deviation']:.6f}",
        "",
        "MAE by max branch loading:",
    ]

    for row in current_rows:
        summary_lines.append(
            f"{row['bin']:>18s}  count={row['count']:4d}  "
            f"ratio={row['ratio_percent']:6.2f}%  MAE={row['voltage_mae']:.8f}"
        )

    summary_lines.append("")
    summary_lines.append("MAE by minimum voltage:")
    for row in vmin_rows:
        summary_lines.append(
            f"{row['bin']:>18s}  count={row['count']:4d}  "
            f"ratio={row['ratio_percent']:6.2f}%  MAE={row['voltage_mae']:.8f}"
        )

    summary_lines.append("")
    summary_lines.append("Final dataset hour distribution:")
    full_hour_counts = np.bincount(hours, minlength=24)
    for h, c in enumerate(full_hour_counts):
        summary_lines.append(
            f"hour={h:02d}  count={int(c):4d}  ratio={100.0*c/len(hours):6.2f}%"
        )

    summary_lines.append("")
    summary_lines.append("Interpretation guide:")
    summary_lines.append(
        "1) If linear-model MAE rises strongly with max branch loading or voltage deviation, "
        "the AC nonlinearity exists but the dataset may be dominated by mild operating points."
    )
    summary_lines.append(
        "2) If MAE remains nearly constant and very small across all loading/voltage bins, "
        "the sampled fixed-topology operating domain itself is highly close to linear."
    )
    summary_lines.append(
        "3) If high-load hours have much lower final-sample ratios than other hours, "
        "the generation guard/current filter is biasing the nominal 24h sampling distribution."
    )

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    (out_dir / "diagnostic_summary.txt").write_text(summary, encoding="utf-8")

    print("\nSaved:")
    for name in [
        "diagnostic_summary.txt",
        "mae_by_current_loading.csv",
        "mae_by_min_voltage.csv",
        "mae_by_hour.csv",
        "mae_by_mode.csv",
        "test_sample_detail.csv",
        "linear_voltage_diagnostic_model.pt",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
