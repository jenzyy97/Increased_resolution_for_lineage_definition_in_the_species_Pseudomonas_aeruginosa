#!/usr/bin/env python3
"""
select_representatives.py

STEP 2e (part 2): pick representative genomes per cluster from per-cluster
patristic distances (long format: Query, Reference, Patristic, cluster),
produced by patristic_distances.py.

These are patristic distances from per-cluster RAxML-NG (GTR+G) trees; they are
NOT recombination-corrected.

Strategy
--------
For each cluster, independently:
  1. Build the patristic distance matrix from the long-format rows.
  2. Collapse near-duplicate genomes (pairwise distance <= eps) into groups,
     keeping ONE delegate per group, chosen by assembly quality (fewer contigs,
     then higher N50; with scaffold_* fallback if contig_* are missing).
  3. Greedy max-min coverage on the delegates: seed with the medoid, then
     repeatedly add the delegate farthest from its nearest chosen rep, until
     every delegate is within d_star of some rep. Returns >= 1 rep; a tight
     cluster fully covered by its medoid returns exactly 1.

Choosing d_star and eps
-----------------------
Both thresholds are on the patristic (substitutions/site) scale and are
dataset-specific. Choose them from the per-cluster distance distributions and
threshold sweeps (see the project README), not from the defaults below, which
are only the values used in the original analysis.

Name joining
------------
Tip labels from patristic_distances.py are the exact Newick labels (trees are
read with preserve_underscores=True), so they join directly to the genome name
column in the quality table with no space/underscore translation. The default
join column is 'panaroo_name'; override via genome_col if yours differs.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd


# ---- default tunables (dataset-specific; override per dataset) -------------
D_STAR = 0.0055     # coverage radius (patristic, subs/site)
EPS = 0.0005        # near-duplicate collapse radius (~an order below D_STAR)
# ---------------------------------------------------------------------------


def get_quality(name, comp_index, warn_missing=True):
    """Return a (contig_count, -N50) sort key for a genome; lower sorts better
    (fewer contigs, then higher N50). Uses contig_* if present, else falls back
    to scaffold_*. A name absent from the quality table sorts last."""
    row = comp_index.get(name)
    if row is None:
        if warn_missing:
            warnings.warn(
                f"genome '{name}' not found in the quality table; it will be "
                "treated as lowest quality when choosing a delegate. This "
                "usually means tip labels and the quality-table key column "
                "don't match.")
        return (np.inf, 0.0)
    count = row.get("contig_count")
    n50 = row.get("contig_N50")
    if pd.isna(count):
        count = row.get("scaffold_count")
    if pd.isna(n50):
        n50 = row.get("scaffold_N50")
    count = np.inf if pd.isna(count) else float(count)
    n50 = 0.0 if pd.isna(n50) else float(n50)
    return (count, -n50)


def long_to_matrix(sub):
    """Rebuild a square patristic matrix from one cluster's long-format rows."""
    names = sorted(set(sub["Query"]) | set(sub["Reference"]))
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    D = np.zeros((k, k))
    for a, b, d in zip(sub["Query"], sub["Reference"], sub["Patristic"]):
        i, j = idx[a], idx[b]
        D[i, j] = D[j, i] = d
    return names, D


def collapse_near_duplicates(names, D, eps, quality_fn):
    """Single-linkage grouping at threshold eps; one delegate per group chosen
    by quality_fn. Returns (delegate_names, delegate_indices)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    k = len(names)
    if k == 1:
        return names[:], [0]
    Z = linkage(squareform(D, checks=False), method="single")
    labels = fcluster(Z, t=max(eps, 1e-12), criterion="distance")
    delegates, deleg_idx = [], []
    for lab in sorted(set(labels)):
        members = [i for i in range(k) if labels[i] == lab]
        best = min(members, key=lambda i: quality_fn(names[i]))
        delegates.append(names[best])
        deleg_idx.append(best)
    return delegates, deleg_idx


def max_min_cover(sub_names, subD, d_star):
    """Greedy farthest-point coverage. Seed = medoid (min total distance); add
    the farthest-from-nearest-rep delegate until all are within d_star."""
    m = len(sub_names)
    if m == 1:
        return [sub_names[0]]
    medoid = int(np.argmin(subD.sum(axis=1)))
    reps = [medoid]
    while True:
        nearest = subD[:, reps].min(axis=1)
        far = int(np.argmax(nearest))
        if nearest[far] <= d_star:
            break
        reps.append(far)
    return [sub_names[i] for i in reps]


def select_for_cluster(sub, comp_index, d_star=D_STAR, eps=EPS,
                       warn_missing=True):
    names, D = long_to_matrix(sub)
    qf = lambda name: get_quality(name, comp_index, warn_missing=warn_missing)
    deleg_names, deleg_idx = collapse_near_duplicates(names, D, eps, qf)
    subD = D[np.ix_(deleg_idx, deleg_idx)]
    reps = max_min_cover(deleg_names, subD, d_star)
    return reps, len(names), len(deleg_names)


def select_all(long_df, comp, genome_col="panaroo_name",
               d_star=D_STAR, eps=EPS, verbose=True):
    """Run selection over every cluster.

    `comp` must have the genome-name column `genome_col` plus quality columns
    (contig_count / contig_N50, with scaffold_* as fallback). Returns a tidy
    DataFrame: cluster, representative (tip label), <genome_col>."""
    comp_index = comp.set_index(genome_col).to_dict("index")
    rows, summary = [], []
    for cl in sorted(long_df["cluster"].unique()):
        sub = long_df[long_df["cluster"] == cl]
        reps, n_tips, n_deleg = select_for_cluster(sub, comp_index, d_star, eps)
        for r in reps:
            rows.append({"cluster": cl, "representative": r, genome_col: r})
        summary.append((cl, n_tips, n_deleg, len(reps)))
    out = pd.DataFrame(rows)
    if verbose:
        print(f"{'cluster':<30}{'tips':>6}{'after_dedup':>13}{'reps':>6}")
        for cl, nt, nd, nr in summary:
            print(f"{str(cl):<30}{nt:>6}{nd:>13}{nr:>6}")
        print(f"\nTotal representatives across {len(summary)} clusters: {len(out)}")
    return out


def _build_parser():
    p = argparse.ArgumentParser(
        prog="select_representatives.py",
        description="Pick per-cluster representatives from patristic distances "
                    "by near-duplicate collapse + greedy max-min coverage.")
    p.add_argument("--dist", required=True,
                   help="Long-format patristic table (CSV): "
                        "Query, Reference, Patristic, cluster.")
    p.add_argument("--quality", required=True,
                   help="Genome quality table (CSV) with the genome-name column "
                        "and contig_count/contig_N50 (+ scaffold_* fallback).")
    p.add_argument("--genome-col", default="panaroo_name",
                   help="Genome-name column in the quality table, matching the "
                        "tree tip labels (default: panaroo_name).")
    p.add_argument("--d-star", type=float, default=D_STAR,
                   help=f"Coverage radius on the patristic scale "
                        f"(default: {D_STAR}). Dataset-specific.")
    p.add_argument("--eps", type=float, default=EPS,
                   help=f"Near-duplicate collapse radius (default: {EPS}). "
                        "Dataset-specific.")
    p.add_argument("--out", default="selected_representatives.csv",
                   help="Output CSV path (default: selected_representatives.csv).")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    long_df = pd.read_csv(args.dist)
    comp = pd.read_csv(args.quality)
    reps = select_all(long_df, comp, genome_col=args.genome_col,
                      d_star=args.d_star, eps=args.eps)
    reps.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
