from __future__ import annotations

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

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tgcn_eta")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# T-GCN §3.3.1 — GCN hyperparameters
GCN_HIDDEN_DIM: Final[int]   = 64     # §4.3: hidden units; optimal range [32, 128]
GCN_OUTPUT_DIM: Final[int]   = 32
GCN_LAYERS: Final[int]       = 2      # 2-layer GCN (Eq. 2)
SAGE_HOPS: Final[int]        = 2      # GraphSAGE 2-hop aggregation

# GRU-style temporal memory (corridor rolling window)
GRU_SPAN: Final[int]         = 5      # EWM span ≈ GRU hidden horizon

# ETA accuracy threshold (within X% of actual)
ETA_ACCURACY_THRESHOLD: Final[float] = 0.15

# Target column
TARGET: Final[str] = "segment_actual_time"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

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

    # Feature groups
    baseline_features: list[str] = field(default_factory=lambda: [
        # Core OSRM predictions
        "segment_osrm_time",
        "segment_osrm_distance",
        "actual_distance_to_destination",
        # Trip-level context
        "factor",
        "lead_time_min",
        # Temporal signals
        "hour_of_day",
        "is_weekend",
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
        "month_sin", "month_cos",
        # Route type
        "route_type_id",
    ])

    graph_extra_features: list[str] = field(default_factory=lambda: [
        # Corridor-level delay history (OD-level aggregate from Task 1)
        "od_mean_delay",
        "od_median_delay",
        "od_p90_delay",
        "od_std_delay",
        "od_reliability",
        "od_flow_weight",
        # Node-level hub features from Task 2
        "src_node_src_outbound_trips",
        "src_node_src_mean_delay",
        "src_node_src_reliability",
        "dst_node_dst_inbound_trips",
        "dst_node_dst_mean_delay",
        # Temporal lag features (GRU-style memory)
        "rolling_delay_3t",
        "rolling_delay_7t",
        "ewm_delay",
        "corridor_trip_rank",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: ETAConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading datasets …")
    trip_df = pd.read_csv(cfg.trip_path, low_memory=False)
    node_df = pd.read_csv(cfg.node_path)
    edge_df = pd.read_csv(cfg.edge_path)

    # Fill lag-feature NaNs with corridor medians (first-trip cold-start)
    lag_cols = ["rolling_delay_3t", "rolling_delay_7t", "rolling_delay_14t", "ewm_delay"]
    for col in lag_cols:
        if col in trip_df.columns:
            trip_df[col] = trip_df[col].fillna(trip_df["od_median_delay"])

    log.info(
        "Loaded  trips=%d  hubs=%d  corridors=%d",
        len(trip_df), len(node_df), len(edge_df),
    )
    return trip_df, node_df, edge_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GRAPH CONSTRUCTION  (Zhao et al. §3.1 Definition 1)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_objects(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> tuple[dict[str, int], np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """
    Build the node feature matrix X and the normalised adjacency Â.

    Graph definition (Zhao et al. §3.1):
      G = (V, E)
      V = hub centres (nodes)
      E = OD corridors weighted by median_delay_ratio (edges)
      A = adjacency matrix ∈ R^{N×N}
      Â = D̃^{-½} Ã D̃^{-½}   (symmetric normalisation, Eq. 2)

    Node features X ∈ R^{N×P}:
      The 5 normalised centrality metrics from Task 2 + breach stats
      (mirrors the node attribute features of Zhao et al. §3.1 Def. 2)
    """
    log.info("Building graph objects (N=%d nodes, E=%d edges) …", len(node_df), len(edge_df))

    # Integer node-ID mapping
    node_ids  = node_df["node_id"].tolist()
    node_map  = {nid: i for i, nid in enumerate(node_ids)}
    N         = len(node_ids)

    # ── Node feature matrix X (Def. 2) ───────────────────────────────────
    node_feat_cols = [
        "in_degree_norm", "out_degree_norm", "betweenness_norm",
        "closeness_norm", "clustering_norm",
        "sla_breach_rate", "importance_score",
    ]
    # Normalise to [0, 1]
    X_raw = node_df[node_feat_cols].values.astype(np.float32)
    col_min, col_max = X_raw.min(0), X_raw.max(0)
    col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    X_nodes = (X_raw - col_min) / col_range   # shape [N, P]

    # ── Adjacency matrix A (Def. 1) ──────────────────────────────────────
    rows, cols, weights = [], [], []
    for _, row in edge_df.iterrows():
        s = node_map.get(row["source_center"])
        d = node_map.get(row["destination_center"])
        if s is None or d is None:
            continue
        w = float(row["median_delay_ratio"])
        # Undirected (add both directions) – mirrors Zhao et al. symmetric A
        rows += [s, d]; cols += [d, s]; weights += [w, w]

    A = sp.csr_matrix(
        (weights, (rows, cols)), shape=(N, N), dtype=np.float32
    )

    # ── Symmetric normalisation: Â = D̃^{-½} Ã D̃^{-½}  (Eq. 2) ──────────
    A_tilde  = A + sp.eye(N, dtype=np.float32)   # Ã = A + I (self-loops)
    deg      = np.asarray(A_tilde.sum(1)).flatten()
    D_inv_sq = sp.diags(np.where(deg > 0, deg ** -0.5, 0.0))
    A_hat    = D_inv_sq @ A_tilde @ D_inv_sq      # symmetric normalised

    log.info("Graph: X_nodes %s | A_hat nnz=%d", X_nodes.shape, A_hat.nnz)
    return node_map, X_nodes, A_hat, A


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SPECTRAL GCN PROPAGATION  (Zhao et al. §3.3.1 / Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────

def gcn_propagate(
    X: np.ndarray,
    A_hat: sp.csr_matrix,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    seed: int = 42,
) -> np.ndarray:
    """
    2-layer Spectral GCN (Eq. 2):
        H^{(0)} = X
        H^{(1)} = ReLU(Â · H^{(0)} · W_0)
        H^{(2)} = ReLU(Â · H^{(1)} · W_1)

    W_0 ∈ R^{P×hidden}, W_1 ∈ R^{hidden×out} are randomly initialised
    with Xavier uniform (used here as fixed random projection — a valid
    approximation for unsupervised graph feature extraction when labels
    are not available at embedding time).

    In a full training regime, these weights would be jointly trained
    with the prediction head via backprop. Here we use them as a
    graph-topology-aware linear map that encodes neighbourhood structure
    into the embedding — identical to the spatial dependence step of
    Zhao et al. §3.3.1.
    """
    rng     = np.random.default_rng(seed)
    P       = X.shape[1]
    dims    = [P] + [hidden_dim] * (n_layers - 1) + [out_dim]

    H = X.copy()
    for i in range(n_layers):
        fan_in, fan_out = dims[i], dims[i + 1]
        # Xavier uniform initialisation
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        W     = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32)
        # Â · H · W  then ReLU
        H = np.maximum(0, (A_hat @ H) @ W)   # sparse × dense → dense

    return H   # [N, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GRAPHSAGE MEAN AGGREGATION  (multi-hop neighbourhood enrichment)
# ─────────────────────────────────────────────────────────────────────────────

def graphsage_aggregate(
    X: np.ndarray,
    A: sp.csr_matrix,
    n_hops: int,
) -> np.ndarray:
    """
    GraphSAGE mean aggregation (Hamilton et al. 2017):
        h_v^{(k)} = MEAN([h_v^{(k-1)}] ∪ {h_u^{(k-1)} : u ∈ N(v)})

    This produces richer neighbourhood embeddings than a single GCN hop
    by explicitly averaging multi-hop neighbours. Used as a complementary
    spatial representation alongside the spectral GCN embedding.
    """
    # Row-normalise A to get mean-aggregation weights
    deg     = np.asarray(A.sum(1)).flatten()
    D_inv   = sp.diags(np.where(deg > 0, 1.0 / deg, 0.0))
    A_norm  = D_inv @ A   # row-stochastic: each row sums to 1

    H = X.copy()
    embs = [H]
    for _ in range(n_hops):
        H_neigh = A_norm @ H              # mean of neighbours
        H       = (H + H_neigh) / 2.0    # MEAN([self] ∪ neighbours)
        embs.append(H)

    return np.concatenate(embs, axis=1)   # [N, P*(n_hops+1)]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GRU-STYLE TEMPORAL STATE  (Zhao et al. §3.3.2 / Eq. 3-6)
# ─────────────────────────────────────────────────────────────────────────────

def gru_temporal_features(trip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate the GRU temporal state (Eq. 3-6) for each corridor using
    an exponentially weighted memory of past delay observations.

        h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t         (Eq. 6)

    Here u_t (update gate) ≈ EWM decay parameter (span=5),
    and c_t (cell candidate) ≈ current trip delay_ratio.
    This produces a per-corridor temporal hidden state that captures
    recent delay trends — equivalent to the GRU's temporal memory.

    Returns the input df with two new columns:
        gru_h_delay  : GRU-approximate temporal hidden state (delay memory)
        gru_h_speed  : GRU-approximate temporal hidden state (speed memory)
    """
    log.info("Computing GRU-style temporal corridor states …")

    df = trip_df.sort_values(["source_center", "destination_center", "od_start_time"]).copy()
    corridor = ["source_center", "destination_center"]

    # h_t for delay (update gate ≈ EWM span=5, matching GRU's hidden horizon)
    df["gru_h_delay"] = (
        df.groupby(corridor, observed=True)["delay_ratio"]
        .transform(lambda x: x.shift(1).ewm(span=GRU_SPAN, min_periods=1).mean())
        .fillna(df["od_median_delay"])
    )

    # h_t for speed deviation (separate GRU state for speed channel)
    df["gru_h_speed"] = (
        df.groupby(corridor, observed=True)["speed_deviation"]
        .transform(lambda x: x.shift(1).ewm(span=GRU_SPAN, min_periods=1).mean())
        .fillna(0.0)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ATTACH GRAPH EMBEDDINGS TO TRIPS
# ─────────────────────────────────────────────────────────────────────────────

def attach_graph_embeddings(
    trip_df: pd.DataFrame,
    node_map: dict[str, int],
    gcn_emb: np.ndarray,
    sage_emb: np.ndarray,
) -> pd.DataFrame:
    """
    For each trip, look up the GCN and GraphSAGE embeddings of its
    source and destination nodes, then concatenate them as trip-level
    graph features:

        φ(trip) = [gcn(src) ∥ gcn(dst) ∥ sage(src) ∥ sage(dst)]

    Matches the TripETAPredictor forward pass in the original code:
        combined = cat([src_emb, dst_emb, trip_feats])
    """
    log.info("Attaching graph embeddings to %d trips …", len(trip_df))

    # GCN embeddings
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

    # Name columns
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split(
    trip_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    cfg: ETAConfig,
) -> tuple:
    """
    Honour the pre-assigned train/test split from the preprocessing pipeline.
    Preserves temporal ordering integrity — no random shuffling of the time axis.
    """
    train_mask = trip_df["data"] == "training"
    test_mask  = ~train_mask

    def _safe_get(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        missing = set(cols) - set(present)
        if missing:
            log.warning("  Feature columns missing: %s", missing)
        return df[present]

    # Baseline features (no graph)
    X_base      = _safe_get(trip_df, cfg.baseline_features).values.astype(np.float32)
    # Full graph-enriched features
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ETAConfig,
) -> XGBRegressor:
    """
    XGBoost baseline — trip-level features only, no graph context.
    Hyperparameters from Zhao et al. §4.3 spirit (150 estimators, lr 0.1).
    """
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
    model._scaler = scaler     # attach scaler for prediction
    return model


def train_graph_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ETAConfig,
) -> LGBMRegressor:
    """
    LightGBM graph model — GCN + GraphSAGE embeddings + corridor lag features.

    LightGBM chosen for its superior handling of high-dimensional sparse
    feature matrices (the GCN embedding columns are dense but many).
    Loss: MAE (l1) — matches T-GCN's L1 loss function (Eq. 7).
    """
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
        objective        = "regression_l1",   # MAE loss — Eq. 7
        reg_lambda       = 1.0,
        reg_alpha        = 0.1,
        random_state     = cfg.random_seed,
        n_jobs           = cfg.n_jobs,
        verbose          = -1,
    )
    model.fit(X_train_s, y_train)
    model._scaler = scaler
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — EVALUATION  (Zhao et al. §4.2 metrics)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
) -> dict:
    """
    Compute the full metric suite from Zhao et al. §4.2:
      MAE   (Eq. 9) — primary optimisation target
      RMSE  (Eq. 8)
      R²    (Eq. 11)
      Accuracy — % of trips within ETA_ACCURACY_THRESHOLD of actual
    """
    X_test_s = model._scaler.transform(X_test)
    preds    = np.maximum(0, model.predict(X_test_s))   # ETA ≥ 0

    mae     = mean_absolute_error(y_test, preds)
    rmse    = np.sqrt(mean_squared_error(y_test, preds))
    r2      = r2_score(y_test, preds)
    # Zhao et al. Eq. 10 — fraction within threshold
    acc_15  = float(np.mean(np.abs(preds - y_test) <= ETA_ACCURACY_THRESHOLD * y_test)) * 100
    # MAPE
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
    """
    Explicitly measure and log the 'graph advantage' as required.
    Δmetric = graph - baseline (negative = improvement for MAE/RMSE/MAPE)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    test_trips: pd.DataFrame,
    base_res: dict,
    graph_res: dict,
    delta: dict,
    cfg: ETAConfig,
) -> tuple[Path, Path]:

    # Per-trip prediction CSV
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

    # Scalar benchmark table
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Panel A: Actual vs Predicted scatter ─────────────────────────────
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

    # ── Panel B: Error distribution (KDE via histogram) ──────────────────
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

    # ── Panel C: Metric bar comparison ───────────────────────────────────
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

    # ── Panel D: Graph advantage summary card ────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: ETAConfig | None = None) -> dict:
    if cfg is None:
        cfg = ETAConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  T-GCN ETA Pipeline — START")
    log.info("  Zhao et al. (2019) §3.3 architecture")
    log.info("═" * 60)
    t0 = time.perf_counter()

    # ── Load ────────────────────────────────────────────────────────────
    trip_df, node_df, edge_df = load_data(cfg)

    # ── Spatial: Build graph + compute GCN & GraphSAGE embeddings ───────
    node_map, X_nodes, A_hat, A_raw = build_graph_objects(node_df, edge_df)

    log.info("GCN propagation (Eq. 2, %d layers, hidden=%d, out=%d) …",
             cfg.gcn_layers, cfg.gcn_hidden, cfg.gcn_out)
    gcn_emb  = gcn_propagate(X_nodes, A_hat, cfg.gcn_hidden, cfg.gcn_out,
                              cfg.gcn_layers, cfg.random_seed)

    log.info("GraphSAGE mean aggregation (%d hops) …", cfg.sage_hops)
    sage_emb = graphsage_aggregate(X_nodes, A_raw, cfg.sage_hops)

    # ── Temporal: GRU-style corridor state (Eq. 3-6) ────────────────────
    trip_df  = gru_temporal_features(trip_df)

    # ── Attach graph embeddings to trip records ──────────────────────────
    emb_df   = attach_graph_embeddings(trip_df, node_map, gcn_emb, sage_emb)

    # ── Split ────────────────────────────────────────────────────────────
    (X_base_tr, X_base_te,
     X_graph_tr, X_graph_te,
     y_tr, y_te, test_trips) = split(trip_df, emb_df, cfg)

    log.info("Train: %d  Test: %d", len(y_tr), len(y_te))
    log.info("Baseline feature dim: %d  |  Graph feature dim: %d",
             X_base_tr.shape[1], X_graph_tr.shape[1])

    # ── Train ────────────────────────────────────────────────────────────
    baseline_model = train_baseline(X_base_tr,  y_tr, cfg)
    graph_model    = train_graph_model(X_graph_tr, y_tr, cfg)

    # ── Evaluate ─────────────────────────────────────────────────────────
    base_res  = evaluate(baseline_model, X_base_te,  y_te, "BASELINE (XGBoost)")
    graph_res = evaluate(graph_model,    X_graph_te, y_te, "GRAPH MODEL (T-GCN + LightGBM)")

    # ── Measure graph advantage ──────────────────────────────────────────
    delta = compute_graph_advantage(base_res, graph_res)

    # ── Export ───────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # In a notebook environment, __name__ is '__main__', but we don't want
    # argparse to try and parse kernel arguments. Instead, we'll directly call run().
    # If you intend to run this as a standalone script from the command line
    # with arguments, you would re-enable the argparse logic.
    run()
