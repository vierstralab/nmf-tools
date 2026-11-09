import numpy as np

from nmf_tools.nmf import NMFModel
from nmf_tools.utils.sample_weights import normalize_weights

from args_parser import NMFInputData, setup_parser, parse_nmf_args


# Internal function to run NMF decomposition as part of pipeline
def main(nmf_input_data: NMFInputData):
    # Data preparation
    nmf_model = NMFModel(
        n_components=nmf_input_data.n_components,
        extra_params=nmf_input_data.extra_params,
        fit_mode=nmf_input_data.mode
    )

    if nmf_input_data.samples_mask.sum() < nmf_input_data.samples_mask.shape[0]:
        X = nmf_input_data.matrix[nmf_input_data.samples_mask, :]
    else:
        X = nmf_input_data.matrix
    
    peaks_mask = nmf_input_data.peaks_mask & (X.sum(axis=0) > 0).A1
    X = X[:, peaks_mask]

    # weights are normalized over all samples/peaks; renormalize the masked ones
    _, W_weights = normalize_weights(nmf_input_data.samples_weights[nmf_input_data.samples_mask])
    _, H_weights = normalize_weights(nmf_input_data.peaks_weights[peaks_mask])
    W, H = nmf_model.fit_transform(
        X,
        W_weights=W_weights,
        H_weights=H_weights
    )
    
    # Optional, projects masked samples
    if nmf_input_data.project_masked_samples:
        W = nmf_model.project_samples(
            nmf_input_data.matrix[:, peaks_mask],
            H,
            W_weights=nmf_input_data.samples_weights,
            H_weights=H_weights
        )

    return W, H, peaks_mask


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()

    nmf_data = parse_nmf_args(args.prefix, args.config)

    W_np, H_np, peaks_mask = main(nmf_data)

    print('Saving results')
    np.save(f'{args.prefix}.W.npy', W_np) # samples x components
    np.save(f'{args.prefix}.H.npy', H_np.T) # peaks x components
    np.savetxt(f'{args.prefix}.non_zero_peaks_mask.txt', peaks_mask, fmt="%d")
    np.savetxt(f'{args.prefix}.samples_mask.txt', nmf_data.samples_mask, fmt="%d")