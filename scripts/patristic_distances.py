#!/usr/bin/env python3
"""
patristic_distances.py

STEP 2e (part 1): compute patristic distance matrices from per-cluster
maximum-likelihood trees, for use as the distance input to representative
selection (select_representatives.py).

Patristic distance between two tips is the sum of branch lengths along the path
connecting them through the tree. These distances come from the per-cluster
RAxML-NG trees built in Step 2d (a GTR+G model over the concatenated core-gene
matrix). They are therefore standard ML branch-length distances and are NOT
recombination-corrected; if recombination correction is required, a tool such
as Gubbins would need to be run upstream and its corrected tree used here
instead.

Usage
-----
    import patristic_distances as pdist

    # one cluster
    names, D = pdist.tree_to_patristic_matrix("cluster17.raxml.bestTree")

    # all clusters at once -> long-format table (Query, Reference, Patristic,
    # cluster), the same shape as the Step 1 distance table, so the same
    # pooled-distribution / per-cluster d* exploration code works here too.
    long_df = pdist.batch_patristic_long(
        tree_paths={"17": "cluster17.raxml.bestTree",
                    "42": "cluster42.raxml.bestTree"}
    )

Note on tip names
-----------------
Trees are read with preserve_underscores=True so tip labels are returned
exactly as written in the Newick file (underscores kept, not converted to
spaces). This lets the labels join directly to a genome/quality table without
any space<->underscore translation.
"""

import os

import dendropy
import numpy as np
import pandas as pd


def tree_to_patristic_matrix(tree_path, schema="newick"):
    """
    Load a tree and return (tip_names, D) where D is a symmetric numpy
    patristic-distance matrix, with D[i, j] aligned to tip_names[i]/[j].

    Uses dendropy's phylogenetic distance matrix, which computes all pairwise
    patristic distances in one pass over the tree.
    """
    tree = dendropy.Tree.get(path=tree_path, schema=schema,
                             preserve_underscores=True)
    pdm = tree.phylogenetic_distance_matrix()
    taxa = list(tree.taxon_namespace)
    names = [t.label for t in taxa]
    k = len(names)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = pdm.patristic_distance(taxa[i], taxa[j])
            D[i, j] = D[j, i] = d
    return names, D


def patristic_long_from_tree(tree_path, cluster_label=None, schema="newick"):
    """One cluster's patristic distances as a long DataFrame:
    columns Query, Reference, Patristic[, cluster]."""
    names, D = tree_to_patristic_matrix(tree_path, schema=schema)
    k = len(names)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            rows.append((names[i], names[j], D[i, j]))
    df = pd.DataFrame(rows, columns=["Query", "Reference", "Patristic"])
    if cluster_label is not None:
        df["cluster"] = cluster_label
    return df


def batch_patristic_long(tree_paths, schema="newick", verbose=True):
    """
    tree_paths : dict {cluster_label: path_to_tree}

    Returns one concatenated long DataFrame across all clusters:
    columns Query, Reference, Patristic, cluster. Mirrors the shape of the
    Step 1 distance table so the same pooled-distribution / d* exploration
    code works unchanged.
    """
    frames = []
    for cluster, path in tree_paths.items():
        try:
            frame = patristic_long_from_tree(path, cluster_label=cluster,
                                             schema=schema)
            frames.append(frame)
            if verbose:
                print(f"  {cluster}: {os.path.basename(path)} -> "
                      f"{frame.shape[0]} pairs")
        except Exception as e:
            print(f"  [fail] {cluster} ({path}): {e}")
    if not frames:
        raise RuntimeError("No trees were successfully parsed.")
    return pd.concat(frames, ignore_index=True)


def report_patristic_distribution(long_df, percentiles=(50, 75, 90, 95, 99)):
    """Summarize the pooled patristic distance distribution and per-cluster
    diameters, to help choose the Step 2e coverage threshold d*."""
    pooled = long_df["Patristic"].values
    diam = long_df.groupby("cluster")["Patristic"].max()

    print("Pooled patristic distances (all clusters, all pairs):")
    for p in percentiles:
        print(f"  p{p:<2} = {np.percentile(pooled, p):.6g}")
    print(f"  max = {pooled.max():.6g}")

    print("\nPer-cluster patristic diameters:")
    for p in percentiles:
        print(f"  p{p:<2} = {np.percentile(diam.values, p):.6g}")
    print(f"  max = {diam.values.max():.6g}")
    return pooled, diam
