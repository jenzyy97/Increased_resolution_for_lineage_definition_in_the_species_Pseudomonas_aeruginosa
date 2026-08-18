#!/usr/bin/env python3
"""
build_within_cluster_dists.py

STEP 0 (optional): build a within-cluster pairwise core/accessory distance
table for input to select_baseline_reps.

Use this when PopPUNK assigned samples against a reference database without
recomputing a full all-vs-all distance matrix (e.g. an assign run without
`--update-db`), so the pairwise distances needed for representative selection
were never calculated. Only *within-cluster* pairs are ever needed downstream,
so this script computes them one cluster at a time with pp-sketchlib's
`query dist --subset`, instead of running a full N x N all-vs-all over the whole
panel. For a large panel that difference is substantial (the full matrix scales
with the square of the sample count; the within-cluster pairs do not).

For each multi-member cluster it:
  1. writes a subset file of the cluster's member names,
  2. runs `sketchlib query dist <db> --subset members -o <prefix>` (all-vs-all
     within the subset; a single db argument means self-comparison),
  3. extracts the distances to a long TSV with poppunk_extract_distances.py
     (the same extractor PopPUNK uses, so the columns match a normal
     PopPUNK distance table),
  4. tags the rows with the cluster and accumulates them.

Singletons are skipped — select_baseline_reps assigns them their lone member.

Returns one combined dist_long (Query, Reference, Core, Accessory, + cluster
column) ready to pass straight into select_baseline_reps /
within_cluster_distance_report.

Preconditions
-------------
* `sketch_db` is the prefix (no file extension) of the sketch database holding
  the query genomes. This is either the query sketches from the assign run, or
  a fresh `sketchlib sketch` built with the SAME -k / -s parameters as the
  reference database, so that distances are comparable.
* The sample names in `clusters[genome_col]` must match the sample names in the
  sketch database.
* `out_dir` is used as a scratch directory: intermediate subset files and
  sketchlib outputs are written there and, unless `keep_intermediate=True`,
  deleted afterward. Point it at a fresh/empty directory — the cleanup step
  removes files matching the per-cluster output patterns.
* The `sketchlib` and `poppunk_extract_distances.py` commands must be available
  on PATH (or passed explicitly via sketchlib_cmd / extract_cmd).

Notes
-----
* `adj_random` is off by default: it corrects for k-mer matches expected by
  chance, which is negligible at the very small within-cluster distances and
  requires random matches to have been added to the database. Turn it on only
  if you want exact PopPUNK-convention distances and the database has had
  `sketchlib add random` run on it.
"""

import glob
import os
import shutil
import subprocess

import pandas as pd


def build_within_cluster_dists(
    clusters,
    sketch_db,
    genome_col="popPUNK_query_id",
    cluster_col="popPUNK_cluster",
    out_dir="cluster_dists",
    cpus=24,
    sketchlib_cmd="sketchlib",
    extract_cmd="poppunk_extract_distances.py",
    adj_random=False,
    keep_intermediate=False,
    verbose=True,
):
    # fail early on a missing external tool, rather than once per cluster
    for tool in (sketchlib_cmd, extract_cmd):
        if shutil.which(tool) is None and not os.path.exists(tool):
            raise FileNotFoundError(
                f"Required command '{tool}' was not found on PATH. Install it "
                "or pass its path via sketchlib_cmd / extract_cmd.")

    os.makedirs(out_dir, exist_ok=True)

    members_by_cluster = clusters.groupby(cluster_col)[genome_col].apply(list)
    multi = members_by_cluster[members_by_cluster.map(len) >= 2]
    if verbose:
        print(f"{len(multi)} multi-member clusters to process "
              f"({(members_by_cluster.map(len) == 1).sum()} singletons skipped)")

    frames, failed = [], []
    for i, (cluster, members) in enumerate(multi.items(), 1):
        tag = str(cluster).replace("/", "_").replace(" ", "_")
        subset_path = os.path.join(out_dir, f"{tag}.members.txt")
        prefix = os.path.join(out_dir, tag)
        tsv = prefix + ".tsv"

        with open(subset_path, "w") as fh:
            fh.write("\n".join(map(str, members)) + "\n")

        try:
            # 1) within-cluster all-vs-all distances (single db = self-compare)
            cmd = [sketchlib_cmd, "query", "dist", sketch_db,
                   "--subset", subset_path, "-o", prefix, "--cpus", str(cpus)]
            if adj_random:
                cmd.append("--adj-random")
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL if not verbose else None)

            # 2) extract to a long TSV (same format as a PopPUNK distance table)
            subprocess.run([extract_cmd, "--distances", prefix, "--output", tsv],
                           check=True,
                           stdout=subprocess.DEVNULL if not verbose else None)

            df = pd.read_csv(tsv, sep="\t")
            df[cluster_col] = cluster
            frames.append(df)
        except subprocess.CalledProcessError as e:
            failed.append((cluster, str(e)))
            if verbose:
                print(f"  [fail] cluster {cluster}: {e}")
        finally:
            if not keep_intermediate:
                for f in ([subset_path, tsv] + glob.glob(prefix + ".npy")
                          + glob.glob(prefix + ".pkl")
                          + glob.glob(prefix + ".dists.*")):
                    if os.path.exists(f):
                        os.remove(f)

        if verbose and i % 25 == 0:
            print(f"  ...{i}/{len(multi)} clusters done")

    if not frames:
        raise RuntimeError("No distances were produced — check sketch_db path, "
                           "the sketchlib/extract commands, and that member "
                           "names match the sketch database.")

    dist_long = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"\nassembled {len(dist_long)} within-cluster distance rows "
              f"from {len(frames)} clusters; {len(failed)} failed")
    return dist_long, failed
