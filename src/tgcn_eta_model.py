import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightgbm import LGBMRegressor
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tgcn_eta")


GCN_HIDDEN_DIM: Final[int]   = 64
GCN_OUTPUT_DIM: Final[int]   = 32
GCN_LAYERS: Final[int]       = 2
SAGE_HOPS: Final[int]        = 2

GRU_SPAN: Final[int]         = 5

ETA_ACCURACY_THRESHOLD: Final[float] = 0.15

TARGET: Final[str] = "segment_actual_time"


@dataclass
class ETAConfig:
    trip_path:  Path  = Path("/mnt/user-data/outputs/delivery_processed.csv")
    node_path:  Path  = Path("/mnt/user-data/outputs/hub_audit_metrics.csv")
    edge_path:  Path  = Path("/mnt/user-data/outputs/corridor_audit_metrics.csv")
    output_dir: Path  = Path("/mnt/user-data/outputs")
    gcn_hidden: int   = GCN_HIDDEN_DIM
    gcn_out:    int   = GCN_OUTPUT_DIM
    gcn_layers: int   = GCN_LAYERS
    sage_hops:  int   = SAGE_HOPS
    random_seed: int  = 42
    n_jobs:     int   = -1

    baseline_features: list[str] = field(default_factory=lambda: [
        "segment_osrm_time",
        "segment_osrm_distance",
        "actual_distance_to_destination",
        "factor",
        "lead_time_min",
        "hour_of_day",
        "is_weekend",
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "month_sin", "month_cos",
        "route_type_id",
    ])

    graph_extra_features: list[str] = field(default_factory=lambda: [
        "od_mean_delay",
        "od_median_delay",
        "od_p90_delay",
        "od_std_delay",
        "od_reliability",
        "od_flow_weight",
        "src_node_src_outbound_trips",
        "src_node_src_mean_delay",
        "src_node_src_reliability",
        "dst_node_dst_inbound_trips",
        "dst_node_dst_mean_delay",
        "rolling_delay_3t",
        "rolling_delay_7t",
        "ewm_delay",
        "corridor_trip_rank",
    ])


def load_data(cfg: ETAConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading datasets …")
    trip_df = pd.read_csv(cfg.trip_path, low_memory=False)
    node_df = pd.read_csv(cfg.node_path)
    edge_df = pd.read_csv(cfg.edge_path)

    lag_cols = ["rolling_delay_3t", "rolling_delay_7t", "rolling_delay_14t", "ewm_delay"]
    for col in lag_cols:
        if col in trip_df.columns:
            trip_df[col] = trip_df[col].fillna(trip_df["od_median_delay"])

    log.info(
        "Loaded  trips=%d  hubs=%d  corridors=%d",
        len(trip_df), len(node_df), len(edge_df),
    )
    return trip_df, node_df, edge_df


def build_graph_objects(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> tuple[dict[str, int], np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    log.info("Building graph objects (N=%d nodes, E=%d edges) …", len(node_df), len(edge_df))

    node_ids  = node_df["node_id"].tolist()
    node_map  = {nid: i for i, nid in enumerate(node_ids)}
    N         = len(node_ids)

    node_feat_cols = [
        "in_degree_norm", "out_degree_norm", "betweenness_norm",
        "closeness_norm", "clustering_norm",
        "sla_breach_rate", "importance_score",
    ]
    X_raw = node_df[node_feat_cols].values.astype(np.float32)
    col_min, col_max = X_raw.min(0), X_raw.max(0)
    col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    X_nodes = (X_raw - col_min) / col_range

    rows, cols, weights = [], [], []
    for _, row in edge_df.iterrows():
        s = node_map.get(row["source_center"])
        d = node_map.get(row["destination_center"])
        if s is None or d is None:
            continue
        w = float(row["median_delay_ratio"])
        rows += [s, d]; cols += [d, s]; weights += [w, w]

    A = sp.csr_matrix(
        (weights, (rows, cols)), shape=(N, N), dtype=np.float32
    )

    A_tilde  = A + sp.eye(N, dtype=np.float32)
    deg      = np.asarray(A_tilde.sum(1)).flatten()
    D_inv_sq = sp.diags(np.where(deg > 0, deg ** -0.5, 0.0))
    A_hat    = D_inv_sq @ A_tilde @ D_inv_sq

    log.info("Graph: X_nodes %s | A_hat nnz=%d", X_nodes.shape, A_hat.nnz)
    return node_map, X_nodes, A_hat, A


def gcn_propagate(
    X: np.ndarray,
    A_hat: sp.csr_matrix,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    seed: int = 42,
) -> np.ndarray:
    rng     = np.random.default_rng(seed)
    P       = X.shape[1]
    dims    = [P] + [hidden_dim] * (n_layers - 1) + [out_dim]

    H = X.copy()
    for i in range(n_layers):
        fan_in, fan_out = dims[i], dims[i + 1]
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        W     = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32)
        H = np.maximum(0, (A_hat @ H) @ W)

    return H


def graphsage_aggregate(
    X: np.ndarray,
    A: sp.csr_matrix,
    n_hops: int,
) -> np.ndarray:
    deg     = np.asarray(A.sum(1)).flatten()
    D_inv   = sp.diags(np.where(deg > 0, 1.0 / deg, 0.0))
    A_norm  = D_inv @ A

    H = X.copy()
    embs = [H]
    for _ in range(n_hops):
        H_neigh = A_norm @ H
        H       = (H + H_neigh) / 2.0
        embs.append(H)

    return np.concatenate(embs, axis=1)


def gru_temporal_features(trip_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing GRU-style temporal corridor states …")

    df = trip_df.sort_values(["source_center", "destination_center", "od_start_time"]).copy()
    corridor = ["source_center", "destination_center"]

    df["gru_h_delay"] = (
        df.groupby(corridor, observed=True)["delay_ratio"]
        .transform(lambda x: x.shift(1).ewm(span=GRU_SPAN, min_periods=1).mean())
        .fillna(df["od_median_delay"])
    )

    df["gru_h_speed"] = (
        df.groupby(corridor, observed=True)["speed_deviation"]
        .transform(lambda x: x.shift(1).ewm(span=GRU_SPAN, min_periods=1).mean())
        .fillna(0.0)
    )

    return df


def attach_graph_embeddings(
    trip_df: pd.DataFrame,
    node_map: dict[str, int],
    gcn_emb: np.ndarray,
    sage_emb: np.ndarray,
) -> pd.DataFrame:
    log.info("Attaching graph embeddings to %d trips …", len(trip_df))

    gcn_dim  = gcn_emb.shape[1]
    sage_dim = sage_emb.shape[1]

    gcn_src  = np.zeros((len(trip_df), gcn_dim),  dtype=np.float32)
    gcn_dst  = np.zeros((len(trip_df), gcn_dim),  dtype=np.float32)
    sage_src = np.zeros((len(trip_df), sage_dim), dtype=np.float32)
    sage_dst = np.zeros((len(trip_df), sage_dim), dtype=np.float32)

    for i, (src, dst) in enumerate(
        zip(trip_df["source_center"], trip_df["destination_center"])
    ):
        s_idx = node_map.get(src)
        d_idx = node_map.get(dst)
        if s_idx is not None:
            gcn_src[i]  = gcn_emb[s_idx]
            sage_src[i] = sage_emb[s_idx]
        if d_idx is not None:
            gcn_dst[i]  = gcn_emb[d_idx]
            sage_dst[i] = sage_emb[d_idx]

    gcn_src_cols  = [f"gcn_src_{j}"  for j in range(gcn_dim)]
    gcn_dst_cols  = [f"gcn_dst_{j}"  for j in range(gcn_dim)]
    sage_src_cols = [f"sage_src_{j}" for j in range(sage_dim)]
    sage_dst_cols = [f"sage_dst_{j}" for j in range(sage_dim)]

    emb_df = pd.DataFrame(
        np.concatenate([gcn_src, gcn_dst, sage_src, sage_dst], axis=1),
        columns=gcn_src_cols + gcn_dst_cols + sage_src_cols + sage_dst_cols,
        index=trip_df.index,
    )

    log.info(
        "Graph embedding matrix: %d trips × %d graph features",
        len(emb_df), emb_df.shape[1],
    )
    return emb_df


def split(
    trip_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    cfg: ETAConfig,
) -> tuple:
    train_mask = trip_df["data"] == "training"
    test_mask  = ~train_mask

    def _safe_get(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        missing = set(cols) - set(present)
        if missing:
            log.warning("  Feature columns missing: %s", missing)
        return df[present]

    X_base      = _safe_get(trip_df, cfg.baseline_features).values.astype(np.float32)
    graph_extra = _safe_get(trip_df, cfg.graph_extra_features + ["gru_h_delay", "gru_h_speed"])
    X_graph_tab = np.concatenate([X_base, graph_extra.values.astype(np.float32)], axis=1)
    X_graph_full= np.concatenate([X_graph_tab, emb_df.values], axis=1)

    y = trip_df[TARGET].values.astype(np.float32)

    return (
        X_base[train_mask],      X_base[test_mask],
        X_graph_full[train_mask],X_graph_full[test_mask],
        y[train_mask],           y[test_mask],
        trip_df[test_mask].copy(),
    )


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ETAConfig,
) -> XGBRegressor:
    log.info("Training BASELINE (XGBoost, %d features) …", X_train.shape[1])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = XGBRegressor(
        n_estimators   = 300,
        max_depth        = 6,
        learning_rate  = 0.05,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        objective       = "reg:squarederror",
        reg_lambda     = 1.0,
        reg_alpha      = 0.1,
        random_state     = cfg.random_seed,
        n_jobs         = cfg.n_jobs,
        verbosity      = 0,
    )
    model.fit(X_train_s, y_train)
    model._scaler = scaler
    return model


def train_graph_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ETAConfig,
) -> LGBMRegressor:
    log.info("Training GRAPH MODEL (LightGBM, %d features) …", X_train.shape[1])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LGBMRegressor(
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.03,
        num_leaves       = 127,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        objective        = "regression_l1",
        reg_lambda       = 1.0,
        reg_alpha        = 0.1,
        random_state     = cfg.random_seed,
        n_jobs           = cfg.n_jobs,
        verbose          = -1,
    )
    model.fit(X_train_s, y_train)
    model._scaler = scaler
    return model


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict:
    X_test_s = model._scaler.transform(X_test)
    preds    = np.maximum(0, model.predict(X_test_s))

    mae     = mean_absolute_error(y_test, preds)
    rmse    = np.sqrt(mean_squared_error(y_test, preds))
    r2      = r2_score(y_test, preds)
    acc_15  = float(np.mean(np.abs(preds - y_test) <= ETA_ACCURACY_THRESHOLD * y_test)) * 100
    mape    = float(np.mean(np.abs((preds - y_test) / np.clip(y_test, 1, None)))) * 100

    log.info("─" * 50)
    log.info("  %s", label)
    log.info("  MAE    = %.2f min", mae)
    log.info("  RMSE   = %.2f min", rmse)
    log.info("  R²     = %.4f", r2)
    log.info("  MAPE   = %.1f %%", mape)
    log.info("  Acc±15%%= %.2f %%", acc_15)
    log.info("─" * 50)

    return dict(
        model=label, mae=mae, rmse=rmse, r2=r2,
        mape=mape, acc_15=acc_15, preds=preds,
    )


def compute_graph_advantage(base: dict, graph: dict) -> dict:
    delta_mae   = graph["mae"]   - base["mae"]
    delta_rmse  = graph["rmse"]  - base["rmse"]
    delta_r2    = graph["r2"]    - base["r2"]
    delta_acc   = graph["acc_15"]- base["acc_15"]
    delta_mape  = graph["mape"]  - base["mape"]

    log.info("═" * 50)
    log.info("  GRAPH ADVANTAGE (Graph − Baseline)")
    log.info("  ΔMAE   = %+.2f min  (%s)", delta_mae,
             "✓ BETTER" if delta_mae < 0 else "✗ WORSE")
    log.info("  ΔRMSE  = %+.2f min  (%s)", delta_rmse,
             "✓ BETTER" if delta_rmse < 0 else "✗ WORSE")
    log.info("  ΔR²    = %+.4f      (%s)", delta_r2,
             "✓ BETTER" if delta_r2 > 0 else "✗ WORSE")
    log.info("  ΔMAPE   = %+.1f %%   (%s)", delta_mape,
             "✓ BETTER" if delta_mape < 0 else "✗ WORSE")
    log.info("  ΔAcc±15%% = %+.2f pp  (%s)", delta_acc,
             "✓ BETTER" if delta_acc > 0 else "✗ WORSE")
    log.info("═" * 50)

    return dict(
        delta_mae=delta_mae, delta_rmse=delta_rmse,
        delta_r2=delta_r2,   delta_acc_15=delta_acc,
        delta_mape=delta_mape,
    )


def export_results(
    test_trips: pd.DataFrame,
    base_res: dict,
    graph_res: dict,
    delta: dict,
    cfg: ETAConfig,
) -> tuple[Path, Path]:

    out_df = test_trips[["trip_uuid", "source_center", "destination_center",
                          "route_type", "segment_osrm_time", TARGET]].copy()
    out_df["baseline_pred_min"] = np.round(base_res["preds"],  2)
    out_df["graph_pred_min"]    = np.round(graph_res["preds"], 2)
    out_df["baseline_err_pct"]  = np.abs(base_res["preds"]  - out_df[TARGET]) / out_df[TARGET] * 100
    out_df["graph_err_pct"]     = np.abs(graph_res["preds"] - out_df[TARGET]) / out_df[TARGET] * 100
    out_df["graph_wins"]        = out_df["graph_err_pct"] < out_df["baseline_err_pct"]

    trip_path = cfg.output_dir / "eta_model_results.csv"
    out_df.to_csv(trip_path, index=False)
    log.info("Per-trip predictions → %s", trip_path)

    bench = pd.DataFrame([
        {"metric": "MAE (min)",      "baseline": round(base_res["mae"],  2),
         "graph": round(graph_res["mae"],  2), "delta": round(delta["delta_mae"],   2)},
        {"metric": "RMSE (min)",     "baseline": round(base_res["rmse"], 2),
         "graph": round(graph_res["rmse"], 2), "delta": round(delta["delta_rmse"],  2)},
        {"metric": "R²",             "baseline": round(base_res["r2"],   4),
         "graph": round(graph_res["r2"],   4), "delta": round(delta["delta_r2"],    4)},
        {"metric": "MAPE (%)",       "baseline": round(base_res["mape"], 2),
         "graph": round(graph_res["mape"], 2), "delta": round(delta["delta_mape"],  2)},
        {"metric": "Acc±15% (%)",    "baseline": round(base_res["acc_15"], 2),
         "graph": round(graph_res["acc_15"], 2), "delta": round(delta["delta_acc_15"], 2)},
    ])
    bench_path = cfg.output_dir / "eta_benchmark_report.csv"
    bench.to_csv(bench_path, index=False)
    log.info("Benchmark report → %s", bench_path)

    return trip_path, bench_path


DARK_BG     = "#0f1117"
CARD_BG     = "#1a1d27"
ACCENT_BLUE = "#4f8ef7"
ACCENT_RED  = "#f75f4f"
ACCENT_GOLD = "#f7c948"
ACCENT_GRN  = "#05c46b"
TEXT_MAIN   = "#e8eaf0"
TEXT_MUTED  = "#7a7f94"
GRID_COLOR  = "#2a2d3a"


def _base_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT_MAIN, "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED, "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "text.color": TEXT_MAIN, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10,
    })


def plot_benchmark(
    base_res: dict,
    graph_res: dict,
    test_trips: pd.DataFrame,
    delta: dict,
    cfg: ETAConfig,
) -> Path:
    _base_style()

    actuals  = test_trips[TARGET].values
    base_p   = base_res["preds"]
    graph_p  = graph_res["preds"]

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                   left=0.07, right=0.97, top=0.91, bottom=0.07)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    ax = axes[0]
    lim = np.percentile(actuals, 95)
    mask = actuals <= lim
    ax.scatter(actuals[mask], base_p[mask],  alpha=0.3, s=6,
               color=ACCENT_RED,  label=f"Baseline MAE={base_res['mae']:.1f}")
    ax.scatter(actuals[mask], graph_p[mask], alpha=0.3, s=6,
               color=ACCENT_BLUE, label=f"Graph    MAE={graph_res['mae']:.1f}")
    ax.plot([0, lim], [0, lim], "--", color=TEXT_MUTED, lw=1.2, label="Perfect")
    ax.set_title("Actual vs Predicted ETA (95th pct range)", fontweight="bold")
    ax.set_xlabel("Actual ETA (min)"); ax.set_ylabel("Predicted ETA (min)")
    ax.legend(fontsize=7.5, framealpha=0.2, edgecolor=GRID_COLOR)
    ax.grid(True, linestyle="--")

    ax = axes[1]
    base_err  = np.abs(base_p  - actuals) / np.clip(actuals, 1, None) * 100
    graph_err = np.abs(graph_p - actuals) / np.clip(actuals, 1, None) * 100
    bins = np.linspace(0, 100, 60)
    ax.hist(base_err[base_err <= 100],  bins=bins, alpha=0.55,
            color=ACCENT_RED,  label="Baseline", density=True)
    ax.hist(graph_err[graph_err <= 100], bins=bins, alpha=0.55,
            color=ACCENT_BLUE, label="Graph",    density=True)
    ax.axvline(15, color=ACCENT_GOLD, lw=1.5, linestyle="--",
               label="15% SLA threshold")
    ax.set_title("Absolute % Error Distribution", fontweight="bold")
    ax.set_xlabel("Abs % Error"); ax.set_ylabel("Density")
    ax.legend(fontsize=7.5, framealpha=0.2, edgecolor=GRID_COLOR)
    ax.grid(True, linestyle="--")

    ax = axes[2]
    metrics  = ["MAE\n(min)", "RMSE\n(min)", "MAPE\n(%)"]
    base_v   = [base_res["mae"],  base_res["rmse"],  base_res["mape"]]
    graph_v  = [graph_res["mae"], graph_res["rmse"], graph_res["mape"]]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, base_v,  width=w, label="Baseline (XGBoost)",
                color=ACCENT_RED,  alpha=0.85, linewidth=0)
    b2 = ax.bar(x + w/2, graph_v, width=w, label="Graph (T-GCN + LightGBM)",
                color=ACCENT_BLUE, alpha=0.85, linewidth=0)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7, color=TEXT_MAIN)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Error Metrics — Lower is Better", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.2, edgecolor=GRID_COLOR)
    ax.grid(True, axis="y", linestyle="--")

    ax = axes[3]
    ax.axis("off")
    lines = [
        ("GRAPH ADVANTAGE REPORT", None, TEXT_MAIN,  13),
        ("", None, None, 9),
        (f"ΔMAE    = {delta['delta_mae']:+.2f} min",
         delta["delta_mae"] < 0,   None, 10),
        (f"ΔRMSE   = {delta['delta_rmse']:+.2f} min",
         delta["delta_rmse"] < 0,  None, 10),
        (f"ΔR²     = {delta['delta_r2']:+.4f}",
         delta["delta_r2"] > 0,    None, 10),
        (f"ΔMAPE   = {delta['delta_mape']:+.1f} %",
         delta["delta_mape"] < 0,  None, 10),
        (f"ΔAcc±15% = {delta['delta_acc_15']:+.2f} pp",
         delta["delta_acc_15"] > 0, None, 10),
        ("", None, None, 9),
        (f"Baseline Acc±15%:  {base_res['acc_15']:.1f} %", None, TEXT_MUTED, 9),
        (f"Graph    Acc±15%:  {graph_res['acc_15']:.1f} %", None, TEXT_MUTED, 9),
    ]
    y_pos = 0.95
    for text, is_better, color_override, fs in lines:
        if not text:
            y_pos -= 0.05
            continue
        if color_override:
            color = color_override
        elif is_better is True:
            color = ACCENT_GRN
        elif is_better is False:
            color = ACCENT_RED
        else:
            color = TEXT_MUTED
        ax.text(0.08, y_pos, text, transform=ax.transAxes,
                fontsize=fs, color=color, fontweight="bold" if fs >= 10 else "normal",
                va="top")
        y_pos -= 0.09 if fs >= 10 else 0.07

    fig.suptitle(
        "T-GCN Graph-Enhanced ETA Prediction — Benchmark Report\n"
        "(Zhao et al. 2019 architecture · XGBoost baseline vs LightGBM + GCN + GraphSAGE)",
        fontsize=12, fontweight="bold", color=TEXT_MAIN, y=0.98,
    )

    path = cfg.output_dir / "eta_benchmark_charts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info("Benchmark chart → %s", path)
    return path


def run(cfg: ETAConfig | None = None) -> dict:
    if cfg is None:
        cfg = ETAConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  T-GCN ETA Pipeline — START")
    log.info("  Zhao et al. (2019) §3.3 architecture")
    log.info("═" * 60)
    t0 = time.perf_counter()

    trip_df, node_df, edge_df = load_data(cfg)

    node_map, X_nodes, A_hat, A_raw = build_graph_objects(node_df, edge_df)

    log.info("GCN propagation (Eq. 2, %d layers, hidden=%d, out=%d) …",
             cfg.gcn_layers, cfg.gcn_hidden, cfg.gcn_out)
    gcn_emb  = gcn_propagate(X_nodes, A_hat, cfg.gcn_hidden, cfg.gcn_out,
                              cfg.gcn_layers, cfg.random_seed)

    log.info("GraphSAGE mean aggregation (%d hops) …", cfg.sage_hops)
    sage_emb = graphsage_aggregate(X_nodes, A_raw, cfg.sage_hops)

    trip_df  = gru_temporal_features(trip_df)

    emb_df   = attach_graph_embeddings(trip_df, node_map, gcn_emb, sage_emb)

    (X_base_tr, X_base_te,
     X_graph_tr, X_graph_te,
     y_tr, y_te, test_trips) = split(trip_df, emb_df, cfg)

    log.info("Train: %d  Test: %d", len(y_tr), len(y_te))
    log.info("Baseline feature dim: %d  |  Graph feature dim: %d",
             X_base_tr.shape[1], X_graph_tr.shape[1])

    baseline_model = train_baseline(X_base_tr,  y_tr, cfg)
    graph_model    = train_graph_model(X_graph_tr, y_tr, cfg)

    base_res  = evaluate(baseline_model, X_base_te,  y_te, "BASELINE (XGBoost)")
    graph_res = evaluate(graph_model,    X_graph_te, y_te, "GRAPH MODEL (T-GCN + LightGBM)")

    delta = compute_graph_advantage(base_res, graph_res)

    trip_out, bench_out = export_results(
        test_trips, base_res, graph_res, delta, cfg
    )
    chart_out = plot_benchmark(base_res, graph_res, test_trips, delta, cfg)

    elapsed = time.perf_counter() - t0
    log.info("═" * 60)
    log.info("  Pipeline COMPLETE in %.1fs", elapsed)
    log.info("═" * 60)

    return {
        "baseline": base_res, "graph": graph_res, "delta": delta,
        "outputs": {"trips": trip_out, "bench": bench_out, "chart": chart_out},
    }


if __name__ == "__main__":
    run()
