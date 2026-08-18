#!/usr/bin/env python3
"""
select_baseline_reps.py

STEP 1 of representative selection.

Dereplicate each cluster down to 1-2 baseline representatives using a
pairwise core-distance matrix, and flag the clusters that 2 reps can't
adequately cover so they can be handed to STEP 2 (SNP-based sub-sampling).

The selection is a greedy 2-center cover with a radius test:
  * pick the cluster's medoid (most central genome) as rep1;
  * if every member is within d* of rep1, one rep is enough;
  * otherwise add the genome farthest from rep1 as rep2;
  * if 2 reps still leave someone more than d* away, the cluster has >2-fold
    internal structure -> flag it for step 2.

Using the medoid's *coverage radius* (not the cluster diameter) is deliberate:
a clonal star-burst cluster can have a large diameter (its tips are far from
each other) yet be perfectly covered by one central rep. Diameter would
over-split those; coverage radius does not.

------------------------------------------------------------------------------
Inputs
------
dist_long : DataFrame, long-format pairwise distances.
    One row per genome pair with two genome-name columns and a core-distance
    column (default columns: query, reference, core). For large panels this
    table can be huge (~5e7 rows for 10k genomes); the code immediately drops
    all cross-cluster pairs, so only the within-cluster remainder is held per
    cluster. If the raw file is too big to load at once, read it in chunks and
    keep only rows whose two genomes share a cluster before concatenating.
clusters : DataFrame, one row per genome.
    Needs a genome-name column and a cluster column (defaults: genome,
    cluster). May carry quality columns (e.g. contig_count, n50) used to nudge
    the chosen rep toward the cleanest assembly among near-central candidates.

Outputs
-------
reps_df    : chosen representatives  -> columns [cluster, genome, role]
summary_df : per-cluster diagnostics -> [cluster, size, diameter,
             coverage_radius, n_reps, flagged_for_snp, missing_pairs]

Command-line usage
------------------
Run the two-step workflow from the shell:

  # 1. inspect the within-cluster distance scale to choose d*
  python select_baseline_reps.py report \
      --dist distances.csv --clusters clusters.csv

  # 2. select representatives with the chosen threshold
  python select_baseline_reps.py select \
      --dist distances.csv --clusters clusters.csv --d-star 0.001 \
      --reps-out reps.csv --summary-out summary.csv

See `python select_baseline_reps.py --help` for all options, including the
column-name flags used to match your own input files.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_submatrix(genomes, sub_df, q_col, r_col, core_col):
    """Assemble a symmetric core-distance matrix for one cluster's genomes."""
    idx = {g: i for i, g in enumerate(genomes)}
    k = len(genomes)
    D = np.full((k, k), np.nan)
    np.fill_diagonal(D, 0.0)
    for q, r, c in zip(sub_df[q_col], sub_df[r_col], sub_df[core_col]):
        i, j = idx[q], idx[r]
        D[i, j] = D[j, i] = c
    n_missing = int(np.isnan(D).sum() // 2)           # undirected missing pairs
    if n_missing:                                     # don't let a missing pair
        D[np.isnan(D)] = np.nanmax(D)                 # fake a tight cluster
    return D, n_missing


def _pick_quality_near(anchor, D, quality_rank_local, window):
    """Among genomes within `window` core-distance of local index `anchor`,
    return the local index of the best-quality one (lowest rank). Ties broken
    by proximity to the anchor. Returns `anchor` itself if no quality given."""
    if quality_rank_local is None:
        return anchor
    near = np.where(D[anchor] <= window)[0]           # includes anchor
    if near.size == 0:
        return anchor
    return min(near, key=lambda i: (quality_rank_local[i], D[anchor, i]))


def _within_cluster_rows(dist_long, clusters, genome_col, cluster_col,
                         q_col, r_col, core_col):
    """Keep only distance rows whose two genomes share a cluster. Raises a
    clear error if nothing matches (the usual cause is a genome-name mismatch
    between the distance table and the cluster table) and warns if only some
    genomes are covered."""
    g2c = dict(zip(clusters[genome_col], clusters[cluster_col]))
    qc = dist_long[q_col].map(g2c).values
    rc = dist_long[r_col].map(g2c).values
    mask = (qc == rc) & ~pd.isna(qc)

    if not mask.any():
        dist_names = set(dist_long[q_col]) | set(dist_long[r_col])
        comp_names = set(clusters[genome_col])
        overlap = len(dist_names & comp_names)
        raise ValueError(
            "No within-cluster distance pairs found. The genome names in the "
            f"distance table don't match '{genome_col}' in the cluster table.\n"
            f"  name overlap : {overlap} of {len(comp_names)} cluster genomes\n"
            f"  distance-table examples : {list(dist_names)[:3]}\n"
            f"  cluster-table examples  : {list(comp_names)[:3]}\n"
            "Reconcile the identifiers (some pipelines sanitize names, e.g. "
            "'.'/'-' -> '_'), or pass the matching column via --genome-col / "
            "--q-col / --r-col.")

    # soft warning if a lot of cluster genomes never appear in the table
    matched = (set(dist_long[q_col]) | set(dist_long[r_col])) & set(clusters[genome_col])
    frac = len(matched) / max(len(set(clusters[genome_col])), 1)
    if frac < 0.9:
        warnings.warn(
            f"Only {frac:.0%} of cluster genomes appear in the distance table; "
            "clusters with missing pairs are handled cautiously (missing pairs "
            "filled with the cluster's max distance).")

    within = dist_long.loc[mask, [q_col, r_col, core_col]].copy()
    within["_cluster"] = qc[mask]
    return within


# ---------------------------------------------------------------------------
# choosing d*: report the within-cluster distance scale
# ---------------------------------------------------------------------------
def within_cluster_distance_report(
    dist_long, clusters,
    genome_col="genome", cluster_col="cluster",
    q_col="query", r_col="reference", core_col="core",
    percentiles=(50, 75, 90, 95, 99),
):
    """Summarize the pooled within-cluster core-distance distribution and the
    per-cluster diameters, to help pick d*. Returns (pooled_distances,
    per_cluster_diameters)."""
    wc = _within_cluster_rows(dist_long, clusters, genome_col, cluster_col,
                              q_col, r_col, core_col)
    within = wc[core_col].values
    cl = wc["_cluster"].values

    diam = pd.Series(within).groupby(cl).max()
    print("Pooled within-cluster core distances:")
    for p in percentiles:
        print(f"  p{p:<2} = {np.percentile(within, p):.6g}")
    print(f"  max = {within.max():.6g}")
    print("\nPer-cluster diameters (max within-cluster distance):")
    for p in percentiles:
        print(f"  p{p:<2} = {np.percentile(diam.values, p):.6g}")
    print(f"  max = {diam.values.max():.6g}")
    print(f"\nClusters with diameter > each candidate d* "
          f"(= clusters that would get a 2nd rep / be examined):")
    for p in percentiles:
        d = np.percentile(diam.values, p)
        print(f"  d*={d:.6g}  ->  {(diam.values > d).sum()} clusters")
    return within, diam


# ---------------------------------------------------------------------------
# main: step-1 selection
# ---------------------------------------------------------------------------
def select_baseline_reps(
    dist_long, clusters,
    genome_col="genome", cluster_col="cluster",
    q_col="query", r_col="reference", core_col="core",
    d_star=None,
    quality_cols=None,            # e.g. [("contig_count", True), ("n50", False)]
    quality_window_frac=0.5,
):
    if d_star is None:
        raise ValueError(
            "Set d_star (core-distance threshold). Run "
            "within_cluster_distance_report() first to choose it.")
    if d_star <= 0:
        raise ValueError(f"d_star must be positive, got {d_star}.")

    # ---- 0. global quality ranking (rank 0 = best assembly) -----------------
    if quality_cols:
        for entry in quality_cols:
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                raise ValueError(
                    "quality_cols must be a list of (column, ascending) pairs, "
                    f"e.g. [('contig_count', True), ('n50', False)]; got {entry!r}.")
        missing = [c for c, _ in quality_cols if c not in clusters.columns]
        if missing:
            raise ValueError(
                f"quality_cols reference column(s) not in the cluster table: "
                f"{missing}. Available columns: {list(clusters.columns)}.")
        ranked = clusters.sort_values(
            [c for c, _ in quality_cols],
            ascending=[asc for _, asc in quality_cols])
        quality_rank = {g: i for i, g in enumerate(ranked[genome_col])}
    else:
        quality_rank = None

    # ---- 1. keep only within-cluster distance rows --------------------------
    within = _within_cluster_rows(dist_long, clusters, genome_col, cluster_col,
                                  q_col, r_col, core_col)

    window = quality_window_frac * d_star
    rep_rows, summary_rows = [], []

    # ---- 2. process each cluster --------------------------------------------
    for cluster, members in clusters.groupby(cluster_col):
        genomes = list(members[genome_col])
        k = len(genomes)

        if k == 1:                                    # singleton -> itself
            rep_rows.append({"cluster": cluster, "genome": genomes[0], "role": "solo"})
            summary_rows.append({"cluster": cluster, "size": 1, "diameter": 0.0,
                                 "coverage_radius": 0.0, "n_reps": 1,
                                 "flagged_for_snp": False, "missing_pairs": 0})
            continue

        sub_df = within[within["_cluster"] == cluster]
        D, n_missing = _build_submatrix(genomes, sub_df, q_col, r_col, core_col)
        local_rank = [quality_rank[g] for g in genomes] if quality_rank else None
        diameter = float(D.max())

        # rep1 = medoid (most central), nudged toward quality nearby
        medoid = int(np.argmin(D.sum(axis=1)))
        rep1 = _pick_quality_near(medoid, D, local_rank, window)
        cov1 = float(D[rep1].max())                   # coverage radius of 1 rep

        if cov1 <= d_star:                            # one rep covers everyone
            rep_rows.append({"cluster": cluster, "genome": genomes[rep1], "role": "medoid"})
            summary_rows.append({"cluster": cluster, "size": k, "diameter": diameter,
                                 "coverage_radius": cov1, "n_reps": 1,
                                 "flagged_for_snp": False, "missing_pairs": n_missing})
        else:                                         # add the farthest genome
            far = int(np.argmax(D[rep1]))
            rep2 = _pick_quality_near(far, D, local_rank, window)
            cov2 = float(np.minimum(D[rep1], D[rep2]).max())
            covered = cov2 <= d_star
            rep_rows.append({"cluster": cluster, "genome": genomes[rep1], "role": "medoid"})
            if rep2 != rep1:
                rep_rows.append({"cluster": cluster, "genome": genomes[rep2], "role": "farthest"})
            summary_rows.append({"cluster": cluster, "size": k, "diameter": diameter,
                                 "coverage_radius": cov2,
                                 "n_reps": 1 if rep2 == rep1 else 2,
                                 "flagged_for_snp": not covered, "missing_pairs": n_missing})

    reps_df = pd.DataFrame(rep_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("size", ascending=False).reset_index(drop=True)
    return reps_df, summary_df


# ---------------------------------------------------------------------------
# command-line interface
# ---------------------------------------------------------------------------
def _read_table(path):
    """Read a CSV/TSV table, inferring the separator from the extension."""
    sep = "\t" if str(path).lower().endswith((".tsv", ".tab", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def _add_common_args(p):
    p.add_argument("--dist", required=True,
                   help="Long-format pairwise distance table (CSV/TSV).")
    p.add_argument("--clusters", required=True,
                   help="Per-genome cluster table (CSV/TSV).")
    p.add_argument("--genome-col", default="genome",
                   help="Genome-name column in the cluster table (default: genome).")
    p.add_argument("--cluster-col", default="cluster",
                   help="Cluster column in the cluster table (default: cluster).")
    p.add_argument("--q-col", default="query",
                   help="First genome-name column in the distance table (default: query).")
    p.add_argument("--r-col", default="reference",
                   help="Second genome-name column in the distance table (default: reference).")
    p.add_argument("--core-col", default="core",
                   help="Core-distance column in the distance table (default: core).")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="select_baseline_reps.py",
        description="STEP 1 representative selection: dereplicate each cluster "
                    "to 1-2 baseline representatives and flag clusters needing "
                    "SNP-based sub-sampling.")
    sub = parser.add_subparsers(dest="command", required=True)

    # report subcommand
    p_report = sub.add_parser(
        "report",
        help="Summarize the within-cluster distance scale to help choose d*.")
    _add_common_args(p_report)
    p_report.add_argument(
        "--percentiles", default="50,75,90,95,99",
        help="Comma-separated percentiles to report (default: 50,75,90,95,99).")

    # select subcommand
    p_select = sub.add_parser(
        "select", help="Select 1-2 representatives per cluster.")
    _add_common_args(p_select)
    p_select.add_argument(
        "--d-star", type=float, required=True,
        help="Core-distance coverage threshold. Use `report` to choose it.")
    p_select.add_argument(
        "--quality-cols", default=None,
        help="Optional quality tie-breakers as comma-separated col:order pairs, "
             "where order is 'asc' or 'desc' (best first). "
             "Example: 'contig_count:asc,n50:desc'.")
    p_select.add_argument(
        "--quality-window-frac", type=float, default=0.5,
        help="Fraction of d* within which a better-quality genome may replace "
             "the medoid/farthest pick (default: 0.5).")
    p_select.add_argument(
        "--reps-out", default="reps.csv",
        help="Output path for chosen representatives (default: reps.csv).")
    p_select.add_argument(
        "--summary-out", default="summary.csv",
        help="Output path for per-cluster diagnostics (default: summary.csv).")
    return parser


def _parse_quality_cols(spec):
    """Parse 'contig_count:asc,n50:desc' -> [('contig_count', True), ('n50', False)]."""
    if not spec:
        return None
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Bad --quality-cols entry {item!r}; expected 'column:asc' or "
                "'column:desc'.")
        col, order = item.rsplit(":", 1)
        order = order.lower()
        if order not in ("asc", "desc"):
            raise ValueError(
                f"Bad sort order {order!r} for column {col!r}; use 'asc' or 'desc'.")
        out.append((col.strip(), order == "asc"))
    return out or None


def main(argv=None):
    args = _build_parser().parse_args(argv)

    dist_long = _read_table(args.dist)
    clusters = _read_table(args.clusters)

    if args.command == "report":
        percentiles = tuple(int(p) for p in args.percentiles.split(","))
        within_cluster_distance_report(
            dist_long, clusters,
            genome_col=args.genome_col, cluster_col=args.cluster_col,
            q_col=args.q_col, r_col=args.r_col, core_col=args.core_col,
            percentiles=percentiles)
        return 0

    if args.command == "select":
        quality_cols = _parse_quality_cols(args.quality_cols)
        reps_df, summary_df = select_baseline_reps(
            dist_long, clusters,
            genome_col=args.genome_col, cluster_col=args.cluster_col,
            q_col=args.q_col, r_col=args.r_col, core_col=args.core_col,
            d_star=args.d_star,
            quality_cols=quality_cols,
            quality_window_frac=args.quality_window_frac)

        reps_df.to_csv(args.reps_out, index=False)
        summary_df.to_csv(args.summary_out, index=False)

        n_flagged = int(summary_df["flagged_for_snp"].sum())
        print(f"Wrote {len(reps_df)} representatives across "
              f"{summary_df['cluster'].nunique()} clusters -> {args.reps_out}")
        print(f"Wrote per-cluster diagnostics -> {args.summary_out}")
        print(f"{n_flagged} cluster(s) flagged for STEP 2 (SNP sub-sampling).")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
