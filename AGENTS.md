# nmf-tools — notes for coding agents

Weighted NMF of large sample-by-peak matrices (DHS index, binary accessibility), plus
reordering, labeling and plotting of the resulting components. Built on scikit-learn's
multiplicative-update NMF; a Nextflow pipeline fits it from an AnnData zarr.
User docs are in `README.md` (Python API, pipeline config, outputs).

Downstream: regulome Figure 3 notebooks (https://github.com/vierstralab/regulome) and
genome-tools' `plot_components.dhs` (imports `component_barplot` lazily), so keep public
names and signatures stable.

## Where things are

| Need | Look in |
|---|---|
| Fit / project (`NMFModel`), weights I/O (`read_weights`, `get_mock_weights`) | `nmf_tools/nmf/nmf.py` (re-exported by `nmf_tools.nmf`) |
| The solver (`WeightedNMF`, subclass of sklearn `NMF`) | `nmf_tools/nmf/weighted_NMF.py` |
| Order components / records, `MatrixReordering` | `nmf_tools/matrix_reordering/` (`order_matrix` in `__init__.py`) |
| Hierarchical clustering of records | `nmf_tools/matrix_reordering/hcluster.py` |
| Peak labels by dominant components | `nmf_tools/component_labels/labels_at_tr.py` (`compute_labels`, `compute_labels_absolute`, `create_label_matrix`) |
| Record (DHS) indices per cluster label, optionally subsampled (`get_indices_by_cluster`) | `nmf_tools/component_labels/indices.py` |
| Stacked barplots (`component_barplot`, `component_barplot_at_scale`, `component_barplot_with_dendrogram`, `plot_top_contributing_samples`) | `nmf_tools/plotting/matrices_barplots.py` |
| Downsampled heatmaps | `nmf_tools/plotting/heatmap_plots.py` |
| Rows/grids of panels with broadcast arguments | `nmf_tools/plotting/layout/__init__.py` (`plot_row_panels`, `plot_grid_panels`, `BroadcastIter`) |
| Component names/colors (`component_data`) | `nmf_tools/utils/component_data.py` |
| Sample weights (inverse category size x SPOT, capping) | `nmf_tools/utils/sample_weights.py` |
| Aggregating AnnData by sample category, information content | `nmf_tools/information_content.py` |
| Pipeline | `nmf_tools/nmf/nextflow/` (`nmf_from_anndata.nf`, `top_samples.nf`, scripts in `bin/`) |

`nmf_tools/main.py` is a stub (no console entry point). The `topSamples` workflow in
`nmf_from_anndata.nf` is marked defunct.

## Matrix orientation

W and H share the component axis everywhere except in `NMFModel` output:

| Where | Layout | W | H |
|---|---|---|---|
| Pipeline files `{prefix}.W.npy` / `{prefix}.H.npy`, AnnData `obsm['nmf_W']` / `varm['nmf_H']` | records x components | samples x components | peaks x components |
| Reordering, labels, plotting functions | components x records | components x samples | components x peaks |
| `NMFModel` (sklearn convention, `X ≈ W @ H`, `X` samples x peaks) | mixed | samples x components | components x peaks |

- Everything in `matrix_reordering`, `component_labels` and `plotting` takes
  `(n_components, n_records)`: transpose both matrices from saved files or AnnData
  (`visualize_nmf.py` loads both files with `.T`); from `NMFModel` output pass `W.T` and `H` as is.
- `project_samples` returns samples x components; `project_peaks` returns components x peaks
  (same orientation as `W` and `H` from `fit_transform`).
- In the regulome AnnData the fit is stored as `obsm['nmf_W']` and `varm['nmf_H']`;
  `component_data_from_anndata` reads `var[['nmf_index', 'nmf_color', 'nmf_name',
  'nmf_short_name']]` and `uns['nmf_components_plot_order']`.

## NMFModel

- Defaults: `solver='mu'`, `beta_loss='frobenius'`, `init='nndsvda'`, `max_iter=1000`,
  `tol=1e-4`, `random_state=0`, `verbose=True`. `extra_params` overrides any of them (printed).
- `X` is converted to float64 CSR. Weights must be 1D numpy arrays (`.shape` is checked).
- Normalize weights to sum to their length before passing them (`read_weights` does;
  `_, w = utils.sample_weights.normalize_weights(w)` for weights built in Python; it returns
  `(cap, weights)`). `NMFModel` uses them as given, and in `scaled_X` mode their scale changes
  the fit. The pipeline renormalizes them after masking samples/peaks.
- `fit_mode='weighted'`: weights enter the multiplicative updates (frobenius only).
  `fit_mode='scaled_X'`: `X` is scaled by `sqrt(weights)` on both axes, and `W`, `H` are
  divided back afterwards.
- `print` is used for progress; "Error at init" is printed even with `verbose=False`.
- `get_transform_params_WH_to_ref` / `apply_transform_WH` align a new fit to a reference
  (component order and scale); rows are components there.

`WeightedNMF` imports private sklearn API (`sklearn.decomposition._nmf`, `_validate_params`,
`_check_params`; `validate_data` on sklearn >= 1.6, `_validate_data` before). Checked with
sklearn 1.6.1 and 1.7.2; a sklearn upgrade is the first suspect if fitting breaks.

## Reordering and plotting

- `order_components(W, by=...)` returns a `MatrixReordering` with `row_order` (1D, or 2D for a
  different component order per record, e.g. `by='primary'`); `order_records(W, by=...)` returns
  one with `col_order`. `by` can be a `MatrixReordering`, which is how one order is reused
  across panels (`component_barplot(..., order_records_by=rec_order, order_components_by=comp_order)`).
- `order_matrix` returns `tops, bottoms, components_order, records_order`.
  `normalize_for_plotting=True` (default) scales each record to sum to 1 in the plot;
  `order_records_kwargs=dict(normalize=True)` only affects how records are ordered. Other
  keyword arguments of `component_barplot` go to `order_matrix`, so a misspelled one raises
  `TypeError`.
- `component_data` is a DataFrame with `index` (row in W/H), `color`, `name`, `short_name`.
  Bar colors are taken after sorting by `index`; its row order sets panel order in
  `plot_top_contributing_samples`. Make one with `get_mock_component_data(W)` or
  `component_data_from_ref` (match a new fit to a reference's names/colors).
- Most plot functions are wrapped with `in_vierstra_style` (packaged `vierstragroup_matplotlibrc`).

## Pipeline

- Scripts in `bin/` import `args_parser` as a sibling module, not from the package: run them as
  `python3 bin/run_NMF.py ...` (what Nextflow does via `$moduleDir/bin`).
- They need `genome_tools` (`read_zarr_backed`), `pyyaml`, `anndata`, `zarr`, `dask`, which are not
  in `pyproject.toml`, and `top_samples_track` calls `wiggletools` and `bedGraphToBigWig`;
  `environment.yml` provides all of them.
- Inputs the scripts assume: `adata.layers['binary']`; `obs['normalized_density_bw']`
  (`find_top_samples.py`); `obs['sample_label']`, else `core_ontology_term` + `SPOT3_score`
  (`visualize_nmf.py`); `var['dist_tss']` is optional.
- `params.config` defaults (`conda`, `chrom_sizes`) and the `Altius` profile are cluster-specific.

## Gotchas

- Peaks with no signal in the masked samples are dropped (`non_zero_peaks_mask.txt`), so `H` has
  fewer peaks than the AnnData; index with that mask.
- With a sample mask and `project_masked_samples: false`, `W` has rows only for the samples in the
  mask (`samples_mask.txt`).
- The `visualize_nmf` process has `errorStrategy 'ignore'`, so a failure only shows as missing PDFs.

## Working in this repo

- Match the surrounding style: numpy-style docstrings in `matrix_reordering`/`component_labels`,
  plain functions over classes, no type-annotation overhaul.
- No test suite. Quick check: fit `NMFModel(4)` on a random binary matrix in both fit modes,
  project samples and peaks, and run `component_barplot(H, get_mock_component_data(H))`.
