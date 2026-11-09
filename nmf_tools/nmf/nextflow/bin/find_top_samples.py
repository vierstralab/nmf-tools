import pandas as pd
import numpy as np
import os

from args_parser import setup_parser, parse_nmf_args


def main(W: np.ndarray, densitity_files: pd.Series, topX: int, suffix: str):
    assert W.shape[1] == densitity_files.shape[0]
    top_samples = []
    major_component = np.argmax(W, axis=0)
    for component in range(W.shape[0]):
        component_is_major_indices = np.where(major_component == component)[0]
        sorted_indices = component_is_major_indices[
            np.argsort(W[component, component_is_major_indices])[::-1]
        ][:topX]
        for sample in sorted_indices[:topX]:
            path = densitity_files.iloc[sample]
            sample_id = densitity_files.index[sample]
            try: 
                os.symlink(
                    path,
                    f'{component}.{sample_id}.component_{suffix}.bw'
                )
            except TypeError:
                print(path)
                raise
            top_samples.append([sample_id, component])
    return pd.DataFrame.from_records(top_samples, columns=['sample_id', 'component'])


if __name__ == "__main__":
    parser = setup_parser()

    print('Adding options to parser')
    parser.add_argument('W', help='W matrix of perform NMF decomposition')
    parser.add_argument('--top', type=int, default=10, help='Number of top samples to select')

    args = parser.parse_args()
    nmf_data = parse_nmf_args(args.prefix, args.config)
    W = np.load(args.W).T

    density_tracks = nmf_data.samples_metadata['normalized_density_bw']
    if not nmf_data.project_masked_samples:
        # W has rows only for the samples used in the fit
        density_tracks = density_tracks[nmf_data.samples_mask]
    top_samples = main(W, density_tracks, topX=args.top, suffix=args.prefix)
    top_samples.to_csv(f"{args.prefix}.top_samples.tsv", index=False, sep="\t")