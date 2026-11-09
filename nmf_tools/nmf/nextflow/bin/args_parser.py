import numpy as np
import argparse
import yaml

import dataclasses
import pandas as pd
import scipy.sparse as sp

from genome_tools.data.anndata import read_zarr_backed
from nmf_tools.nmf import read_weights, get_mock_weights


@dataclasses.dataclass
class NMFInputData:
    matrix: sp.csr_matrix
    n_components: int
    samples_mask: np.ndarray
    peaks_mask: np.ndarray
    samples_weights: np.ndarray
    peaks_weights: np.ndarray
    samples_metadata: pd.DataFrame
    dhs_metadata: pd.DataFrame
    mode: str
    extra_params: dict
    project_masked_samples: bool

DEFAULTS = {
    "sample_mask_eval": None,
    "dhs_mask_eval": None,
    "samples_weights": None,
    "peaks_weights": None,
    "mode": "weighted",
    "extra_params": {},
    "n_components": 30,
    "project_masked_samples": False
}

# -------------------
# Entry point
# -------------------
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NMF input from AnnData + YAML config")
    parser.add_argument("prefix", help="Sample prefix defined in config YAML")
    parser.add_argument("config", help="YAML config file")
    return parser


def parse_nmf_args(prefix: str, config_file: str) -> NMFInputData:
    """Load NMF input data from YAML config."""
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    if prefix not in cfg:
        raise ValueError(f"Prefix {prefix} not found in {config_file}")
    job_cfg = {**DEFAULTS, **cfg[prefix]}  # merge with defaults

    if "anndata" not in job_cfg:
        raise ValueError(f"Config for {prefix} must contain 'anndata'")

    return _parse_from_anndata(job_cfg)


def _parse_from_anndata(cfg) -> NMFInputData:
    print("Reading AnnData")
    adata = read_zarr_backed(cfg["anndata"])
    matrix = adata.layers["binary"]  # samples x peaks

  
    if cfg["sample_mask_eval"] is not None:
        samples_mask = mask_from_metadata(adata.obs, cfg["sample_mask_eval"])
    else:
        samples_mask = np.ones(adata.n_obs, dtype=bool)

    if cfg["dhs_mask_eval"] is not None:
        peaks_mask = mask_from_metadata(adata.var, cfg["dhs_mask_eval"])
    else:
        peaks_mask = np.ones(adata.n_vars, dtype=bool)

    if cfg["samples_weights"] is not None:
        samples_weights = read_weights(cfg["samples_weights"], names=adata.obs.index).values
    else:
        samples_weights = get_mock_weights(matrix, which='W')

    if cfg["peaks_weights"] is not None:
        peaks_weights = read_weights(cfg["peaks_weights"], names=adata.var.index).values
    else:
        peaks_weights = get_mock_weights(matrix, which='H')

    assert cfg["mode"] in ["weighted", "scaled_X"]
    return NMFInputData(
        matrix=matrix,
        samples_mask=samples_mask,
        peaks_mask=peaks_mask,
        samples_weights=samples_weights,
        peaks_weights=peaks_weights,
        samples_metadata=adata.obs,
        dhs_metadata=adata.var,
        mode=cfg["mode"],
        n_components=cfg["n_components"],
        extra_params=cfg["extra_params"],
        project_masked_samples=cfg["project_masked_samples"],
    )

def mask_from_metadata(metadata: pd.DataFrame, eval: str):
    if eval is not None:
        return metadata.eval(eval).values
    return np.ones(len(metadata), dtype=bool)
