"""
plot_patristic_thresholds.py

Example plotting code for choosing the Step 2 patristic thresholds (d*, eps).
These are exploratory helpers, not part of the pipeline proper — adapt the
threshold windows and output paths to your own data.

All functions take `patristic_long`: a long-format DataFrame with columns
Query, Reference, Patristic, cluster (as produced by patristic_distances.py).
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _long_to_matrix(sub, q="Query", r="Reference", val="Patristic"):
    names = sorted(set(sub[q]) | set(sub[r]))
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    D = np.zeros((k, k))
    for a, b, d in zip(sub[q], sub[r], sub[val]):
        i, j = idx[a], idx[b]
        D[i, j] = D[j, i] = d
    return names, D


def plot_pooled(patristic_long, save_path=None):
    """Pooled patristic distances + per-cluster diameters (overview)."""
    diam = patristic_long.groupby("cluster")["Patristic"].max().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].hist(patristic_long["Patristic"], bins=80)
    ax[0].set(xlabel="patristic distance (substitutions)", ylabel="pairs",
              title="pooled within-cluster patristic distances")
    ax[1].hist(diam.values, bins=min(24, len(diam)))
    ax[1].set(xlabel="per-cluster diameter", ylabel="clusters",
              title="per-cluster patristic diameters")
    ax[2].bar(diam.index.astype(str), diam.values)
    ax[2].set(xlabel="cluster", ylabel="diameter", title="diameter per cluster (sorted)")
    ax[2].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_per_cluster_distributions(patristic_long, bins=40, ncols=4, save_path=None):
    """One patristic-distance histogram per cluster, in a grid."""
    clusters = sorted(patristic_long["cluster"].unique())
    n = len(clusters)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, cl in zip(axes, clusters):
        vals = patristic_long.loc[patristic_long["cluster"] == cl, "Patristic"].values
        ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="none")
        ax.set_title(f"{cl}\nn={len(vals)}  median={pd.Series(vals).median():.4f}  "
                     f"max={vals.max():.4f}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("patristic distance", fontsize=8)
        ax.set_ylabel("pairs", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_threshold_sweeps(patristic_long, ncols=6, n_steps=40, method="average",
                          save_path=None):
    """Per-cluster 'number of groups vs. threshold' sweep, each scaled to that
    cluster's own max distance."""
    clusters = sorted(patristic_long["cluster"].unique())
    n = len(clusters)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, cl in zip(axes, clusters):
        _, D = _long_to_matrix(patristic_long[patristic_long["cluster"] == cl])
        tmax = float(D.max()) * 1.02
        ts = np.linspace(tmax / n_steps, tmax, n_steps)
        Z = linkage(squareform(D, checks=False), method=method)
        ks = [len(set(fcluster(Z, t=t, criterion="distance"))) for t in ts]
        ax.plot(ts, ks, marker="o", ms=2.5, lw=1, color="#4C72B0")
        ax.set_title(str(cl), fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("threshold", fontsize=8)
        ax.set_ylabel("groups", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_threshold_sweeps_zoom(patristic_long, tmin=0.004, tmax=0.006, step=1e-4,
                               chosen=0.0055, ncols=6, method="average", save_path=None):
    """Same sweep over a FIXED window (same x-axis for every cluster), with the
    chosen cutoff marked. tmin/tmax/chosen are dataset-specific — set them from
    plot_threshold_sweeps() output for your data."""
    clusters = sorted(patristic_long["cluster"].unique())
    n = len(clusters)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, cl in zip(axes, clusters):
        _, D = _long_to_matrix(patristic_long[patristic_long["cluster"] == cl])
        ts = np.arange(tmin, tmax + step / 2, step)
        Z = linkage(squareform(D, checks=False), method=method)
        ks = np.array([len(set(fcluster(Z, t=t, criterion="distance"))) for t in ts])
        ax.plot(ts, ks, marker="o", ms=3, lw=1, color="#4C72B0")
        if chosen is not None and tmin <= chosen <= tmax:
            k_chosen = int(ks[np.argmin(np.abs(ts - chosen))])
            ax.axvline(chosen, color="#C44E52", lw=1, ls="--", alpha=0.8)
            ax.set_title(f"{cl}  (k={k_chosen} @ {chosen})", fontsize=9)
        else:
            ax.set_title(str(cl), fontsize=9)
        ax.set_xlim(tmin, tmax)
        ax.set_xticks([tmin, tmax])
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("threshold", fontsize=8)
        ax.set_ylabel("groups", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
