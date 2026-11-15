# Neuronal effector expression prediction in worm and mouse

Grouped data and code for *Low alignment of neuronal effector expression predictions between worm and mouse at 20-nucleus resolution*, v0.5.9, by Freeman Hui.

A source model uses 136 transcription-factor families to predict variation in 115 effector families among groups of 20 nuclei. The four studies contribute 1,118 target groups from 11 preparations. On these measured groups, the six unadapted cross-species correlations ranged from −0.042 to +0.035.

## Refit the models

Run from the repository root in a Python 3.11 environment.

```sh
python -m pip install -r requirements-refit.txt
python refit.py
```

This fits the TF model and a model using 3,131 background families at the same effective degrees of freedom. Each source selects its ridge setting in five source-only folds. The program scores each of the 12 source–target comparisons using both source scaling and target-X scaling. It writes 48 results to `reconstructed/refitted.tsv` and compares correlation, amplitude and centred R² with their saved values.

To refit the separately tuned backgrounds and all 128 count-matched panels as well, run the following command. It writes 3,168 rows across 132 model configurations.

```sh
python refit.py --models TF matched_background Nfull Nfull_df --all-panels --output reconstructed/all-refits.tsv
```

## What the matrices contain

Each expression file in `refit/groups/` contains a group-by-TF matrix (`TF`), a group-by-background matrix (`Nfull`), measured effector expression (`y`), weights, study and preparation labels, and a group identifier. The identifier is the study, preparation and a sequential group number. `axes.json` gives the ordered family identifiers. The nine `*-matching.npz` files contain the five-fold assignments and feature indices for the matched background panels. In within-study comparisons each source half predicts the other half.

| Study | Preparations | Groups in each evaluation half | Groups in full source for transfer |
| --- | ---: | ---: | ---: |
| StAnge | 3 | 384 | 768 |
| Gao | 3 | 13 | Target only |
| Iida | 1 | 71 | 142 |
| Russ | 4 | 91 | 184 |

Russ has 182 target groups and 184 full-source groups because incomplete groups are discarded separately in each half. The [preparation records](manifests/path-a/v059-referee-sources-2026-11-15/preparation-conditions.tsv) link the original samples and the St. Ange, Gao, Iida and Russ publications. Arc scBaseCount supplied the StAnge measurements under CC0. Please credit the source studies when reusing their data.

The manuscript's supplementary PDF provides the per-nucleus expression and cell-to-group membership required to construct these groups and rerun the shuffles, log-ratio refits and group-size checks. Section S5 of the supplement describes those commands. The group construction and diagnostic code is also provided in `refit/`. Figure 2 can be redrawn from the grouped score summaries with Matplotlib and `plot_transfer.py`.

The reference run used Python 3.11.17 and one OpenBLAS 0.3.34 thread. `SHA256SUMS.txt` lists the files in this version. Citation metadata is in `CITATION.cff`.
