# nmf-tools

nmf-tools is a package that contains wrapper code (on top of sklearn NMF) to decompose large sample-by-peak chromatin accessibility matrices (e.g. DHS/ATAC-seq index matrices) into interpretable components using a **weighted non-negative matrix factorization (NMF)**. Weighting lets each sample and each peak contribute to the fit according to a user-supplied importance (e.g. downweighting redundant samples) without altering the underlying matrix. The package provides the factorization itself, component-matching/reordering utilities, labeling of peaks by dominant component, and visualization, wired together into an end-to-end Nextflow pipeline.

The core algorithm is:
- **Weighted NMF** — extends scikit-learn's multiplicative-update NMF (`solver='mu'`, `beta_loss='frobenius'`, NNDSVDA init) with independent per-sample (`W_weights`) and per-peak (`H_weights`) weights, in one of two modes:
  - `weighted` — weights enter the multiplicative-update loss directly.
  - `scaled_X` — the input matrix is pre-scaled by `sqrt(weights)` on both axes before fitting, and `W`/`H` are rescaled back afterward.
- **Projection** — new samples or new peaks can be projected onto an already-fit set of components (`H` or `W` held fixed), e.g. to score samples that were masked out of the original fit.
- **Component matching/reordering** — aligns components across independent NMF runs (e.g. a new run vs. a reference) so component indices, signs, and scales are comparable.
- **Labeling & visualization** — assigns each peak a label listing the smallest set of components explaining a target fraction of its signal, and produces stacked-barplot/heatmap visualizations of sample and component composition.

## Table of contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Python API](#python-api)
- [Nextflow pipeline](#nextflow-pipeline)
  - [Config file](#config-file)
  - [Running the pipeline](#running-the-pipeline)
  - [Outputs](#outputs)
- [Authors](#authors)

## Repository layout

- [`nmf_tools/nmf/`](nmf_tools/nmf) — core algorithm: [`nmf.py`](nmf_tools/nmf/nmf.py) (`NMFModel`, the public API) and [`weighted_NMF.py`](nmf_tools/nmf/weighted_NMF.py) (`WeightedNMF`, the sklearn-derived solver).
- [`nmf_tools/nmf/nextflow/`](nmf_tools/nmf/nextflow) — the Nextflow pipeline (`nmf_from_anndata.nf`, `top_samples.nf`) and its executable scripts ([`bin/`](nmf_tools/nmf/nextflow/bin)): `run_NMF.py`, `visualize_nmf.py`, `find_top_samples.py`, `args_parser.py`.
- [`nmf_tools/matrix_reordering/`](nmf_tools/matrix_reordering) — `order_matrix()` / `MatrixReordering`: reorders and matches components and records (e.g. for comparing runs or aligning to a reference decomposition).
- [`nmf_tools/component_labels/`](nmf_tools/component_labels) — `compute_labels()`: assigns each peak a component-membership label from `H`.
- [`nmf_tools/plotting/`](nmf_tools/plotting) — stacked-barplot and heatmap visualizations of `W`/`H`.
- [`nmf_tools/utils/`](nmf_tools/utils) — component color/naming metadata ([`component_data.py`](nmf_tools/utils/component_data.py)) and sample-weight helpers ([`sample_weights.py`](nmf_tools/utils/sample_weights.py)).
- [`nmf_tools/information_content.py`](nmf_tools/information_content.py) — information-content calculations for DHS annotation.

## Installation

### Prerequisites
`conda` (recommended: `mamba`), `python >= 3.8`

### Create the conda environment
The Nextflow pipeline's environment ([`nmf_tools/nmf/nextflow/environment.yml`](nmf_tools/nmf/nextflow/environment.yml)) also installs nmf-tools and genome-tools, and is the easiest way to get a working setup:
```bash
mamba env create -n nmf -f nmf_tools/nmf/nextflow/environment.yml
conda activate nmf
```

### Install nmf-tools directly
```bash
git clone https://github.com/vierstralab/nmf-tools.git
cd nmf-tools
pip install -e .
```

## Python API

The main entry point is `NMFModel` ([`nmf_tools/nmf/nmf.py`](nmf_tools/nmf/nmf.py)), a thin wrapper around `WeightedNMF` that handles weighting and coerces input to `scipy.sparse.csr_matrix`:

```python
from nmf_tools.nmf import NMFModel

model = NMFModel(n_components=30, fit_mode='weighted')

# X: samples x peaks. W_weights/H_weights: 1D arrays matching X's axes; default uniform.
W, H = model.fit_transform(X, W_weights=sample_weights, H_weights=peak_weights)
# X ≈ W @ H  (W: samples x components, H: components x peaks)

W_new = model.project_samples(X_new, H, W_weights=new_sample_weights, H_weights=peak_weights)  # H fixed
H_new = model.project_peaks(X, W, H_weights=new_peak_weights, W_weights=sample_weights)  # W fixed
```

`extra_params` overrides any underlying sklearn `NMF` parameter (`max_iter`, `tol`, `alpha_W`, `l1_ratio`, ...). Weights are normalized internally to sum to the number of samples/peaks, so only their relative scale matters.

## Nextflow pipeline

The pipeline fits NMF on an `AnnData`/zarr matrix, visualizes the result, and extracts per-component top-contributing samples (with aggregated BigWig tracks).

### Config file

`run_NMF.py`/`visualize_nmf.py`/`find_top_samples.py` all read the same YAML config, keyed by `prefix` (one entry per NMF run):

```yaml
my_run:
  anndata: /path/to/matrix.zarr         # required; AnnData with a 'binary' layer (samples x peaks)
  n_components: 28                      
  sample_mask_eval: "qc_pass == True"   # pandas .eval() over adata.obs; default: no mask
  dhs_mask_eval: null                   # pandas .eval() over adata.var; default: no mask
  samples_weights: /path/to/weights.npy # .npy or .tsv with id/weight columns; default: uniform
  peaks_weights: null                   # default: uniform
  mode: scaled_X                        # 'weighted' or 'scaled_X'
  extra_params: {}                      # overrides passed to NMFModel
  project_masked_samples: false         # project masked-out samples onto the fit
```

### Running the pipeline

```bash
nextflow run nmf_tools/nmf/nextflow/nmf_from_anndata.nf \
  -profile Altius -resume \
  --conda /path/to/conda/env \
  --nmf_config config.yaml \
  --nmf_params prefixes.txt \
  --outdir ./output \
  --top_count 10 \
  --chrom_sizes GRCh38_no_alts.chrom_sizes
```
`prefixes.txt` is a one-column CSV listing the `prefix` keys from `config.yaml` to run. To (re-)generate only the visualizations from existing `W`/`H` output, use the `visualize` entry point instead:
```bash
nextflow run nmf_tools/nmf/nextflow/nmf_from_anndata.nf -entry visualize -profile Altius --nmf_config config.yaml --nmf_params prefixes.txt
```

### Outputs

Per `prefix`, under `{outdir}/nmf/{prefix}/`:
- `{prefix}.W.npy`, `{prefix}.H.npy` — the fit (`W`: samples x components, `H`: **peaks** x components — transposed relative to the `X = W @ H` convention used internally).
- `{prefix}.non_zero_peaks_mask.txt`, `{prefix}.samples_mask.txt` — which peaks/samples were used in the fit.
- `*.pdf` — composition barplots and heatmaps (e.g. `Barplot_all_DHSs.pdf`, `Hierarchical_barplot_all_samples.pdf`, `Top20_all_samples_barplot.pdf`, plus `Distance_to_tss.pdf` if `dist_tss` is in the peak metadata).
- `{prefix}.top_samples.tsv` and `top_samples/{prefix}.{component}.top_samples.bw` — top-N samples per component (by `W` loading) and their averaged density track.
- `{prefix}.component_tracks_meta.tsv` — maps each component to its aggregated BigWig path.

## Authors

Developed by (alphabetically by last name):

- Sergey Abramov
- Alexandr Boytsov

Altius Institute for Biomedical Sciences
