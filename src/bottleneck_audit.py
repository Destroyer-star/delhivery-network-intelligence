import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bottleneck_audit")


SLA_BREACH_RATIO: Final[float] = 1.20

CHRONIC_DELAY_RATIO: Final[float] = 1.20

CRITICAL_HUB_SCORE_PERCENTILE: Final[float] = 0.85

NORM_SCALE: Final[float] = 10.0

TOP_N_LABELS: Final[int] = 5
TOP_N_CHART: Final[int] = 20


@dataclass
class AuditConfig:
    input_path:   Path = Path("/mnt/user-data/outputs/delivery_processed.csv")
    output_dir:   Path = Path("/mnt/user-data/outputs")
    sla_threshold: float = SLA_BREACH_RATIO
    chronic_threshold: float = CHRONIC_DELAY_RATIO
    top_n_chart:  int   = TOP_N_CHART
    top_n_labels: int   = TOP_N_LABELS
    dpi:          int   = 180


REQUIRED_COLS = {
    "trip_uuid", "source_center", "destination_center",
    "delay_ratio", "od_trip_count", "source_name", "destination_name",
}

def load(cfg: AuditConfig) -> pd.DataFrame:
    log.info("Loading processed dataset from %s", cfg.input_path)
    df = pd.read_csv(cfg.input_path, low_memory=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info("Loaded %d rows × %d cols", *df.shape)
    return df


def build_corridor_metrics(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    log.info("Building corridor-level metrics …")

    corridor = (
        df.groupby(["source_center", "destination_center"], observed=True)
        .agg(
            total_trips         =("trip_uuid",    "count"),
            median_delay_ratio  =("delay_ratio",  "median"),
            mean_delay_ratio    =("delay_ratio",  "mean"),
            p90_delay_ratio     =("delay_ratio",  lambda x: x.quantile(0.90)),
            std_delay_ratio     =("delay_ratio",  "std"),
            sla_breaches        =("delay_ratio",  lambda x: (x > cfg.sla_threshold).sum()),
            od_mean_dist_km     =("od_mean_dist_km",  "first") if "od_mean_dist_km" in df.columns
                                  else ("segment_osrm_distance", "mean"),
            od_reliability      =("od_reliability",   "first") if "od_reliability" in df.columns
                                  else ("delay_ratio", lambda x: (x <= cfg.sla_threshold).mean()),
        )
        .reset_index()
    )

    corridor["breach_rate"]      = corridor["sla_breaches"] / corridor["total_trips"]
    corridor["is_chronic"]       = corridor["median_delay_ratio"] > cfg.chronic_threshold

    total_breaches               = corridor["sla_breaches"].sum()
    corridor["breach_share_pct"] = 100.0 * corridor["sla_breaches"] / max(total_breaches, 1)

    log.info(
        "Corridors: %d total | %d chronic (median delay > %.0f%% of OSRM)",
        len(corridor),
        corridor["is_chronic"].sum(),
        (cfg.chronic_threshold - 1) * 100,
    )
    return corridor


def build_graph(corridor_df: pd.DataFrame) -> nx.DiGraph:
    log.info("Constructing directed graph …")
    G = nx.DiGraph()

    for _, row in corridor_df.iterrows():
        G.add_edge(
            row["source_center"],
            row["destination_center"],
            weight          = row["median_delay_ratio"],
            total_trips     = row["total_trips"],
            breach_rate     = row["breach_rate"],
            is_chronic      = row["is_chronic"],
            sla_breaches    = row["sla_breaches"],
            breach_share_pct= row["breach_share_pct"],
        )

    log.info(
        "Graph: %d nodes | %d edges | density=%.5f | weakly-connected components=%d",
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.density(G),
        nx.number_weakly_connected_components(G),
    )
    return G


def compute_centrality(G: nx.DiGraph) -> dict[str, dict]:
    log.info("Computing centrality metrics (this may take ~30s for large graphs) …")
    t0 = time.perf_counter()

    metrics = {
        "in_degree"    : dict(G.in_degree()),
        "out_degree"   : dict(G.out_degree()),
        "betweenness"  : nx.betweenness_centrality(G, weight="weight", normalized=True),
        "closeness"    : nx.closeness_centrality(G),
        "clustering"   : nx.clustering(G.to_undirected()),
    }

    log.info("Centrality computed in %.1fs", time.perf_counter() - t0)
    return metrics


def _min_max_normalise(series: pd.Series, scale: float = NORM_SCALE) -> pd.Series:
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(0.0, index=series.index)
    return scale * (series - series.min()) / rng


def build_node_registry(
    G: nx.DiGraph,
    metrics: dict[str, dict],
    df: pd.DataFrame,
    cfg: AuditConfig,
) -> pd.DataFrame:
    log.info("Building hub node registry …")

    rows = []
    for node in G.nodes():
        src_match = df.loc[df["source_center"] == node, "source_name"]
        dst_match = df.loc[df["destination_center"] == node, "destination_name"]
        hub_name  = src_match.iat[0] if not src_match.empty else (
                    dst_match.iat[0] if not dst_match.empty else node)

        hub_trips  = df[(df["source_center"] == node) | (df["destination_center"] == node)]
        sla_breaches = int((hub_trips["delay_ratio"] > cfg.sla_threshold).sum())
        total_hub_trips = len(hub_trips)

        rows.append({
            "node_id"            : node,
            "hub_name"           : hub_name,
            "in_degree"          : metrics["in_degree"][node],
            "out_degree"         : metrics["out_degree"][node],
            "betweenness"        : metrics["betweenness"][node],
            "closeness"          : metrics["closeness"][node],
            "clustering"         : metrics["clustering"][node],
            "sla_breach_count"   : sla_breaches,
            "total_trips"        : total_hub_trips,
            "sla_breach_rate"    : sla_breaches / max(total_hub_trips, 1),
        })

    node_df = pd.DataFrame(rows)

    for metric in ["in_degree", "out_degree", "betweenness", "closeness", "clustering"]:
        node_df[f"{metric}_norm"] = _min_max_normalise(node_df[metric])

    node_df["importance_score"] = (
        node_df["in_degree_norm"]
        + node_df["out_degree_norm"]
        + node_df["betweenness_norm"]
        + node_df["closeness_norm"]
        + node_df["clustering_norm"]
    )

    threshold = node_df["importance_score"].quantile(CRITICAL_HUB_SCORE_PERCENTILE)
    node_df["is_critical"] = node_df["importance_score"] >= threshold

    total_breaches = node_df["sla_breach_count"].sum()
    node_df["breach_share_pct"] = 100.0 * node_df["sla_breach_count"] / max(total_breaches, 1)

    node_df = node_df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    log.info(
        "Hub registry: %d hubs | %d critical (top %.0f%%)",
        len(node_df), node_df["is_critical"].sum(), (1 - cfg.sla_threshold) * 100
    )
    log.info(
        "Top hub: %s  importance=%.2f  sla_breaches=%d",
        node_df.iloc[0]["hub_name"],
        node_df.iloc[0]["importance_score"],
        node_df.iloc[0]["sla_breach_count"],
    )

    return node_df


def compute_metric_correlation(node_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["in_degree", "out_degree", "betweenness", "closeness", "clustering"]
    corr = node_df[cols].corr().round(4)
    log.info("Centrality correlation matrix:\n%s", corr.to_string())
    return corr


DARK_BG     = "#0f1117"
CARD_BG     = "#1a1d27"
ACCENT_BLUE = "#4f8ef7"
ACCENT_RED  = "#f75f4f"
ACCENT_GOLD = "#f7c948"
TEXT_MAIN   = "#e8eaf0"
TEXT_MUTED  = "#7a7f94"
GRID_COLOR  = "#2a2d3a"


def _apply_base_style() -> None:
    plt.rcParams.update({
        "figure.facecolor"  : DARK_BG,
        "axes.facecolor"    : CARD_BG,
        "axes.edgecolor"    : GRID_COLOR,
        "axes.labelcolor"   : TEXT_MUTED,
        "axes.titlecolor"   : TEXT_MAIN,
        "xtick.color"       : TEXT_MUTED,
        "ytick.color"       : TEXT_MUTED,
        "grid.color"        : GRID_COLOR,
        "grid.linewidth"    : 0.6,
        "grid.alpha"        : 0.8,
        "text.color"        : TEXT_MAIN,
        "font.family"       : "DejaVu Sans",
        "font.size"         : 9,
        "axes.titlesize"    : 10,
        "axes.labelsize"    : 8.5,
    })


def plot_bottleneck_analysis(
    node_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    cfg: AuditConfig,
) -> Path:
    _apply_base_style()

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                   left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    top_hubs = node_df.nlargest(cfg.top_n_chart, "sla_breach_count").iloc[::-1]
    bar_colors = [ACCENT_RED if c else ACCENT_BLUE for c in top_hubs["is_critical"]]
    bars = ax_a.barh(top_hubs["hub_name"].str.split("_").str[0],
                     top_hubs["sla_breach_count"],
                     color=bar_colors, height=0.65, linewidth=0)
    ax_a.set_title("Top Hubs by SLA Breach Count", pad=10, fontweight="bold")
    ax_a.set_xlabel("SLA Breaches (delay_ratio > 1.2×OSRM)")
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_a.grid(True, axis="x", linestyle="--")
    ax_a.tick_params(axis="y", labelsize=7)
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ACCENT_RED,
               markersize=8, label="Critical Hub"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ACCENT_BLUE,
               markersize=8, label="Standard Hub"),
    ]
    ax_a.legend(handles=legend_elems, loc="lower right", fontsize=7,
                framealpha=0.15, edgecolor=GRID_COLOR)

    sc = ax_b.scatter(
        node_df["betweenness"],
        node_df["sla_breach_count"],
        c=node_df["importance_score"],
        cmap=LinearSegmentedColormap.from_list("rb", [ACCENT_BLUE, ACCENT_GOLD, ACCENT_RED]),
        s=np.clip(node_df["total_trips"] / node_df["total_trips"].max() * 120, 10, 120),
        alpha=0.75,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax_b, pad=0.02)
    cbar.set_label("Importance Score", color=TEXT_MUTED, fontsize=7.5)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    med_bet = node_df["betweenness"].median()
    med_sla = node_df["sla_breach_count"].median()
    ax_b.axvline(med_bet, color=GRID_COLOR, linewidth=1.2, linestyle="--")
    ax_b.axhline(med_sla, color=GRID_COLOR, linewidth=1.2, linestyle="--")

    x_max, y_max = node_df["betweenness"].max(), node_df["sla_breach_count"].max()
    ax_b.text(x_max * 0.98, y_max * 0.97, "HIGH RISK\nCHOKEPOINT",
              ha="right", va="top", color=ACCENT_RED, fontsize=6.5, fontweight="bold", alpha=0.7)
    ax_b.text(x_max * 0.02, y_max * 0.97, "HIGH BREACH\nLOW CENTRALITY",
              ha="left", va="top", color=ACCENT_GOLD, fontsize=6.5, fontweight="bold", alpha=0.7)

    top_annotate = node_df.nlargest(cfg.top_n_labels, "sla_breach_count")
    for _, row in top_annotate.iterrows():
        ax_b.annotate(
            row["hub_name"].split("_")[0],
            (row["betweenness"], row["sla_breach_count"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=6.5, fontweight="bold", color=TEXT_MAIN,
            arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, lw=0.6),
        )
    ax_b.set_title("Betweenness Centrality vs. SLA Breaches", pad=10, fontweight="bold")
    ax_b.set_xlabel("Betweenness Centrality (delay-weighted)")
    ax_b.set_ylabel("SLA Breach Count")
    ax_b.grid(True, linestyle="--")

    chronic = (
        corridor_df[corridor_df["is_chronic"]]
        .nlargest(cfg.top_n_chart, "sla_breaches")
        .iloc[::-1]
    )
    chronic["label"] = (
        chronic["source_center"].str[:6] + "→" + chronic["destination_center"].str[:6]
    )
    c_colors = [ACCENT_RED if r > 0.5 else ACCENT_GOLD for r in chronic["breach_rate"]]
    ax_c.barh(chronic["label"], chronic["sla_breaches"],
              color=c_colors, height=0.65, linewidth=0)
    ax_c.set_title("Top Chronic Corridors by SLA Breach Count", pad=10, fontweight="bold")
    ax_c.set_xlabel("SLA Breaches on Corridor")
    ax_c.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_c.grid(True, axis="x", linestyle="--")
    ax_c.tick_params(axis="y", labelsize=7)

    mask  = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap  = LinearSegmentedColormap.from_list("corr", ["#1a3a6b", CARD_BG, "#6b1a1a"])
    sns.heatmap(
        corr_matrix,
        ax=ax_d,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor=DARK_BG,
        annot_kws={"size": 7.5, "color": TEXT_MAIN},
        cbar_kws={"shrink": 0.75, "pad": 0.02},
    )
    ax_d.set_title("Centrality Metric Correlation Matrix\n(cf. Liu et al. 2023, Table 3)",
                   pad=10, fontweight="bold")
    ax_d.tick_params(axis="x", labelsize=8, rotation=20)
    ax_d.tick_params(axis="y", labelsize=8, rotation=0)
    ax_d.collections[0].colorbar.ax.yaxis.set_tick_params(
        color=TEXT_MUTED, labelcolor=TEXT_MUTED, labelsize=7)

    fig.suptitle(
        "Logistics Network — Bottleneck & Corridor Audit",
        fontsize=15, fontweight="bold", color=TEXT_MAIN, y=0.97,
    )

    n_critical = node_df["is_critical"].sum()
    n_chronic  = corridor_df["is_chronic"].sum()
    fig.text(
        0.5, 0.002,
        f"Critical hubs (top 15% importance score): {n_critical}  |  "
        f"Chronic corridors (median delay >20% OSRM): {n_chronic}  |  "
        f"SLA breach threshold: delay_ratio > {cfg.sla_threshold:.2f}×OSRM",
        ha="center", va="bottom", fontsize=7, color=TEXT_MUTED,
    )

    path = cfg.output_dir / "hub_bottleneck_analysis.png"
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info("Saved 4-panel chart → %s", path)
    return path


def plot_importance_ranking(node_df: pd.DataFrame, cfg: AuditConfig) -> Path:
    _apply_base_style()

    top = node_df.nlargest(cfg.top_n_chart, "importance_score").iloc[::-1]
    labels = top["hub_name"].str.split("_").str[0]

    metrics_norm = ["in_degree_norm", "out_degree_norm",
                    "betweenness_norm", "closeness_norm", "clustering_norm"]
    metric_labels = ["In-Degree", "Out-Degree", "Betweenness", "Closeness", "Clustering"]
    colors = [ACCENT_BLUE, "#4fd6f7", ACCENT_GOLD, "#9b5de5", "#05c46b"]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    left = np.zeros(len(top))
    for col, label, color in zip(metrics_norm, metric_labels, colors):
        vals = top[col].values
        ax.barh(labels, vals, left=left, label=label, color=color,
                height=0.65, linewidth=0, alpha=0.92)
        left += vals

    crit_thresh = node_df[node_df["is_critical"]]["importance_score"].min()
    ax.axvline(crit_thresh, color=ACCENT_RED, linewidth=1.5,
               linestyle="--", label=f"Critical threshold ({crit_thresh:.1f})")

    ax.set_title("Hub Importance Score — Metric Breakdown\n(Liu et al. §5.2: normalised centrality sum)",
                 pad=14, fontweight="bold", color=TEXT_MAIN, fontsize=11)
    ax.set_xlabel("Importance Score (0–50)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.15,
              edgecolor=GRID_COLOR, ncol=2)
    ax.grid(True, axis="x", linestyle="--")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(0, 52)

    fig.tight_layout()
    path = cfg.output_dir / "hub_importance_scores.png"
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log.info("Saved importance ranking chart → %s", path)
    return path


def export_tables(
    node_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    cfg: AuditConfig,
) -> tuple[Path, Path]:

    hub_path = cfg.output_dir / "hub_audit_metrics.csv"
    cor_path = cfg.output_dir / "corridor_audit_metrics.csv"

    hub_export_cols = [
        "node_id", "hub_name",
        "in_degree", "out_degree", "betweenness", "closeness", "clustering",
        "in_degree_norm", "out_degree_norm", "betweenness_norm",
        "closeness_norm", "clustering_norm",
        "importance_score", "is_critical",
        "sla_breach_count", "sla_breach_rate", "breach_share_pct", "total_trips",
    ]
    node_df[hub_export_cols].to_csv(hub_path, index=False)
    log.info("Saved hub audit metrics → %s  (%d hubs)", hub_path, len(node_df))

    corridor_df.sort_values(
        ["is_chronic", "sla_breaches"], ascending=[False, False]
    ).to_csv(cor_path, index=False)
    log.info("Saved corridor audit metrics → %s  (%d corridors)", cor_path, len(corridor_df))

    return hub_path, cor_path


def print_summary(node_df: pd.DataFrame, corridor_df: pd.DataFrame, G: nx.DiGraph) -> None:
    top5_hubs = node_df.nlargest(5, "importance_score")
    top5_cors = corridor_df.nlargest(5, "sla_breaches")

    log.info("=" * 60)
    log.info("  AUDIT SUMMARY")
    log.info("=" * 60)
    log.info("Network:  %d hubs | %d corridors | density=%.5f",
             G.number_of_nodes(), G.number_of_edges(), nx.density(G))
    log.info("Critical hubs (top 15%% importance): %d", node_df["is_critical"].sum())
    log.info("Chronic corridors (median >20%% OSRM): %d", corridor_df["is_chronic"].sum())
    log.info("Total SLA breaches in dataset: %d", node_df["sla_breach_count"].sum())
    log.info("-" * 60)
    log.info("TOP 5 CHOKEPOINT HUBS:")
    for i, row in top5_hubs.iterrows():
        log.info(
            "  #%d  %-30s  importance=%.2f  betweenness=%.5f  breaches=%d",
            i + 1, row["hub_name"].split("(")[0].strip(),
            row["importance_score"], row["betweenness"], row["sla_breach_count"],
        )
    log.info("-" * 60)
    log.info("TOP 5 CHRONIC CORRIDORS:")
    for i, row in top5_cors.reset_index(drop=True).iterrows():
        log.info(
            "  #%d  %-12s → %-12s  median_delay=%.2fx  breaches=%d  breach_rate=%.1f%%",
            i + 1, row["source_center"][:12], row["destination_center"][:12],
            row["median_delay_ratio"], row["sla_breaches"],
            row["breach_rate"] * 100,
        )
    log.info("=" * 60)


def run(cfg: AuditConfig | None = None) -> dict:
    if cfg is None:
        cfg = AuditConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  Bottleneck & Corridor Audit Pipeline — START")
    log.info("═" * 60)
    t_start = time.perf_counter()

    df           = load(cfg)
    corridor_df  = build_corridor_metrics(df, cfg)
    G            = build_graph(corridor_df)
    metrics      = compute_centrality(G)
    node_df      = build_node_registry(G, metrics, df, cfg)
    corr_matrix  = compute_metric_correlation(node_df)

    print_summary(node_df, corridor_df, G)

    hub_path, cor_path = export_tables(node_df, corridor_df, cfg)
    chart_path   = plot_bottleneck_analysis(node_df, corridor_df, corr_matrix, cfg)
    rank_path    = plot_importance_ranking(node_df, cfg)

    log.info("═" * 60)
    log.info("  Audit COMPLETE in %.1fs", time.perf_counter() - t_start)
    log.info("═" * 60)

    return {
        "node_df"      : node_df,
        "corridor_df"  : corridor_df,
        "graph"        : G,
        "hub_metrics"  : hub_path,
        "corridor_metrics": cor_path,
        "chart"        : chart_path,
        "rank_chart"   : rank_path,
    }


if __name__ == "__main__":
    run()
