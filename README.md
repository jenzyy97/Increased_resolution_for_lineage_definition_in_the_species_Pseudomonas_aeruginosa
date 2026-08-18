## poppunk-representative-selection

A pipeline for reducing PopPUNK clusters to 1-2 baseline
representative genomes using pairwise core distances, and flagging clusters whose internal structure needs
finer sub-sampling which are passed to Maximum Likelihood (ML) Tree inference.

This is intended to be used on pairwise core distances calculated from PopPUNK.

## Workflow

**Step 0 (optional) — build within-cluster distances.**
If PopPUNK was run by assigning samples against a reference database without
recomputing the full distance matrix (an assign run without `--update-db`), not
all pairwise distances are calculated, and the within-cluster distances needed
downstream will be missing. In that case, build a long-format pairwise core
distance table first with `build_within_cluster_dists.py`, which uses
pp-sketchlib to compute distances. Because only *within-cluster* pairs are ever
needed, it runs `sketchlib query dist --subset` once per cluster rather than a
full all-vs-all over the whole panel. This requires that the sketch database's
sample names match your cluster table, and that the sketches were built with the
same `-k`/`-s` parameters as the reference database so the distances are
comparable. If PopPUNK already produced a full distance table, skip this step.

**Step 1 — select baseline representatives.**
`select_baseline_representatives.py` dereplicates each cluster to 1–2 representatives and
flags the clusters two representatives can't cover within a distance threshold
`d*`. (See the Step 1 section below for details.)

**Step 2 — sub-sampling via ML tree (inferred using core gene SNPs).**
Clusters flagged in Step 1 are passed to the sub-sampling step.

### Note on identifier consistency
The same genome identifier has to travel through the whole pipeline, 
because the final step joins tree tip labels back to your genome table by exact string match.

Throughout this pipeline the per-genome table is called {cluster_df} and carries:
1. {popPUNK_query_id}: the genome name (used as genome_col)
2. {popPUNK_cluster}: the cluster assignment (used as cluster_col)
3. {contig_count}, {contig_N50}: assembly-quality columns used as tie-breakers
(with scaffold_count / scaffold_N50 as option fallbacks)

For the pipeline to connect correctly, {popPUNK_query_id} must match:
1. The sample names in the pp-sketchlib database in STEP 0;
2. The sequence headers in the per-gene FASTA files that flow through
MAFFT -> ClipKIT -> PhyKIT -> RAxML-NG 
and become the tips labels in each cluster's ML tree in STEP 2.



## Example driver

clusters: one row per genome, carrying (at minimum) two pieces of information:
   - a genome name that matches the sketch database sample names
   - a cluster assignment

By default these are read from columns named 'popPUNK_query_id' and
'popPUNK_cluster'. The column names are configurable — if yours differ, pass
them via the genome_col / cluster_col arguments rather than renaming your data.
Optional quality columns (e.g. contig_count, n50) may be included for
tie-breaking during representative selection.


### Step 0: Compute within-cluster all-vs-all distances (only if needed)

```python
import os
import build_within_cluster_dists as bw
import select_baseline_representatives as sbr

dist_long, failed = bw.build_within_cluster_dists(
    cluster_df,
    sketch_db="path/to/poppunk_db/poppunk_clusters",  # sketch db prefix, no extension
    genome_col="popPUNK_query_id",
    cpus=2,
)

# OPTIONAL: CACHE THE DISTANCE TABLE SO STEP 0 DOESN'T HAVE TO BE RERUN.
# Reload later with: dist_long = pd.read_parquet("path/to/within_cluster_dists.parquet")
dist_long.to_parquet("path/to/within_cluster_dists.parquet")
```

### Step 1: Choose d* from the distance report, then select representatives

#### Step 1a: Choosing d*

d* is the core-distance coverage threshold: a representative "covers" every
cluster member within d* of it. The report summarizes the distance scale;
the histograms below make it easier to pick a value by eye.

```python
pooled, diam = sbr.within_cluster_distance_report(
    	dist_long, 
	cluster_df, 
	genome_col="popPUNK_query_id"
	)

# pooled = every within-cluster pairwise core distance
# diam   = per-cluster diameter (max within-cluster distance), indexed by cluster

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(pooled, bins=100)
ax[0].set(xlabel="within-cluster core distance (all pairs)", ylabel="pairs",
          title="pooled pairwise distances")
ax[1].hist(diam.values, bins=100)
ax[1].set(xlabel="per-cluster diameter", ylabel="clusters",
          title="per-cluster diameters")
plt.tight_layout()
plt.savefig("path/to/poppunk_cluster_distance_distribution.svg",
            format="svg", bbox_inches="tight")
plt.show()
```

#### Step 1b: Select representatives based on chosen d*

```python
### CHANGE ###
d_star = 0.0046   # read off the distributions above
### -------------------#
reps_df, summary_df = sbr.select_baseline_reps(
    dist_long, cluster_df, genome_col="popPUNK_query_id",
    d_star=d_star,
    quality_cols=[("contig_count", True), ("contig_N50", False)],
    # contig_count ascending (fewer contigs = better);
    # contig_N50 descending (larger N50 = better)
)

# Save summary_df and reps_df.
summary_df.to_csv(f"PATH/TO/cluster_representatives_summary_d={d_star}.csv", index = False)
reps_df.to_csv(f"PATH/TO/cluster_representatives_d={d_star}.csv", index = False)
```


### Step 2: Phylogeny-based sub-sampling of flagged clusters

Clusters that Step 1 flagged (`flagged_for_snp = True` in `summary.csv`) have
more internal structure than two representatives can cover. For these, we build
per-cluster phylogenies and pick representatives by patristic distance.

Stages 1–5 use standard phylogenetics tools, run as follows. Software versions:
MAFFT `<v7.526>`, ClipKIT `<v2.12.0>`, PhyKIT `<v2.1.2>`, RAxML-NG? `<v2.0.2>`.


For each flagged cluster, build ML tree based on core gene alignments, 
then sub-sample based on a similar pairwise distance strategy.
Isolate flagged clusters.

```python
flagged = summary_df.loc[summary_df['flagged_for_snp'] == True].copy()
```

#### Step 2a: Build core gene alignments

*See pan-genome filtering for extracting core genes and writing fasta files.*
Each cluster should be handled separately for all steps.
Each core gene per cluster should also be handled separately.

MAFFT v7.526 Example
mafft --auto --thread 1 core_gene1.fasta > core_gene1.aln
>> output: core_gene1.aln

#### Step 2b: Clean alignments

ClipKIT v2.12.0 Example
clipkit -m smart-gap core_gene1.aln -o core_gene1_clipkit_cleaned.aln
>> output: core_gene1_clipkit_cleaned.aln

#### Step 2c: Create concatenation matrix

Build alignment_list.txt: one clipkit-cleaned alignment path per line, 
for one cluster (all core genes).

PhyKIT v2.1.2 Example
phykit create_concatenation_matrix -a alignment_list.txt -p phykit_concatenation_matrix

>> outputs: phykit_concatenation_matrix.fa (matrix),
            phykit_concatenation_matrix.partition,
  	        phykit_concatenation_matrix.occupancy.

#### Step 2d: ML tree inference

Infer one ML tree per cluster from its concatenated matrix.
ML search only (no bootstrapping): only branch lengths are used downstream,
for patristic-distance rep selection. 
A single GTR+G model is applied across the whole matrix.
Fixed seed for reproducibility.

RAxML-NG v2.0.2 Example
raxml-ng --search --msa phykit_concatenation_matrix.fa --model GTR+G --prefix <cluster_name> --threads 4 --seed 42

>> key output: cluster_name.raxml.bestTree  (used in Step 2e)


#### Step 2e: Patristic distances and select representatives from tree

Patristic distances are calculated per cluster from the cluster_name.raxml.bestTree files.
*These are standard ML branch-length (patristic) distances and are not recombination-corrected.
A new d* is selected based on per-cluster dynamics.
Clusters are collapsed based on d* and patristic matrix.


##### Step 2e.1: Calculating patristic distances

Point these at your RAxML-NG output:
  tree_dir     : directory containing one subdirectory per cluster
  cluster_dirs : the per-cluster subdirectory names (e.g. os.listdir(tree_dir))
Each cluster's subdirectory is expected to hold its <prefix>.raxml.bestTree.

```python
import glob, os
import patristic_distances as pdist

tree_dir = "PATH/TO/raxml_output"
cluster_dirs = os.listdir(tree_dir)

tree_paths = {}
for d in cluster_dirs:                         # one directory per cluster
    hits = glob.glob(os.path.join(tree_dir, d, "*.raxml.bestTree"))
    if hits:
        tree_paths[d] = hits[0]
    else:
        print(f"[warn] no .raxml.bestTree found in {d}")

patristic_long = pdist.batch_patristic_long(tree_paths)   # Query, Reference, Patristic, cluster
```

##### Step 2e.2: Choosing d*

The plotting helpers below live in `examples/plot_patristic_thresholds.py`
(import them rather than redefining). All take `patristic_long` — the
long-format table (`Query, Reference, Patristic, cluster`) from Step 2e.1 — and
each returns a matplotlib figure you can save.

```python
from examples.plot_patristic_thresholds import (
    plot_pooled,
    plot_per_cluster_distributions,
    plot_threshold_sweeps,
    plot_threshold_sweeps_zoom,
)

# 1. Overview: pooled patristic distances + per-cluster diameters.
fig = plot_pooled(patristic_long)
fig.savefig("PATH/TO/MLtree_clusters_patristic_distance_distribution.svg",
            format="svg", bbox_inches="tight")

# 2. Per-cluster distributions, one panel each, to spot outlier clusters/tips.
fig = plot_per_cluster_distributions(patristic_long, ncols=4)
fig.savefig("PATH/TO/per_cluster_patristic.png", dpi=150, bbox_inches="tight")

# 3. Full-range sweep: number of average-linkage (UPGMA) groups vs. threshold,
#    each panel scaled to its own cluster's max distance. Flat stretches mark
#    stable partitions — use this to locate a candidate cutoff region.
fig = plot_threshold_sweeps(patristic_long)
fig.savefig("PATH/TO/all_clusters_sweep.svg", format="svg", bbox_inches="tight")
```

Read a candidate `d*` off the full-range sweep (a value sitting on a flat,
stable stretch shared by most clusters), then confirm it with a zoomed sweep
over a fixed window.

```python
###---------SET d*---------------#
d_star = 0.0055   # chosen coverage threshold (patristic, subs/site): a rep "covers"
                  #   every member within d_star of it; members farther than this from
                  #   any chosen rep trigger an additional rep. Dataset-specific —
                  #   re-pick from the sweeps for your own data.
tmin   = 0.004    # left edge of the zoom window; set just below d_star so the cutoff
                  #   sits inside the plotted range. Must satisfy tmin <= d_star.
tmax   = 0.006    # right edge of the zoom window; set just above d_star. tmin–tmax
                  #   should bracket d_star and span where the curves flatten.
                  #   Must satisfy d_star <= tmax (or the cutoff line is not drawn).
step   = 1e-4     # threshold increment across the window. Aim for ~20 points:
                  #   here (tmax - tmin) / step = 20.
#--------------------------------#

# 4. Zoomed sweep over the fixed window, with the chosen cutoff marked. Each
#    panel's title reports how many groups (k) that cluster splits into at d_star.
fig = plot_threshold_sweeps_zoom(
    patristic_long, tmin=tmin, tmax=tmax, step=step, chosen=d_star,
)
fig.savefig("PATH/TO/all_clusters_sweep_zoom.svg", format="svg", bbox_inches="tight")
```

The `d_star` set here is the value carried into representative selection in the
next step — the figures and the actual rep-picking must use the same number.

##### Step 2e.3: Select representatives based on chosen d* from Step 2e.2

per cluster: 
1. build the patristic matrix; 
2. collapse near-duplicate genomes (distance ≤ eps) into groups, keeping one delegate each, 
chosen by assembly quality (fewer contigs, then higher N50, with scaffold_* fallback); 
3. greedy max-min coverage on the delegates — seed with the medoid, 
then repeatedly add the delegate farthest from its nearest chosen rep until all are within d*.

```python
#--------- SET -------------------------------------------- #

d_star=0.0055   # dataset-specific; from the sweeps above
eps=0.0005      # near-duplicate collapse radius

#-----------------------------------------#


from select_patristic_representatives import select_all

reps = select_all(
    patristic_long, cluster_df,
    genome_col="popPUNK_query_id",
    d_star = d_star,
    eps = eps
)
```
