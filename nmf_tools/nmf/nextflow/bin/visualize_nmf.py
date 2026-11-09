import numpy as np
import matplotlib.pyplot as plt

from args_parser import NMFInputData, setup_parser, parse_nmf_args

from nmf_tools.utils.component_data import get_component_data
from nmf_tools.plotting.matrices_barplots import component_barplot, component_barplot_at_scale, plot_top_contributing_samples 


def plot_dist_tss(H, dist_tss, component_data, ax=None):
    assert H.shape[1] == len(dist_tss)
    max_component = np.argmax(H, axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    for i, row in component_data.iterrows():
        data = np.abs(dist_tss[max_component == row['index']])
        ax.plot(np.sort(data), np.linspace(0, 1, len(data)), color=row['color'])
    ax.set_xlim(-50, 5000)
    ax.set_xlabel('Distance to TSS')
    ax.set_ylabel('Cumulative proportion of DHSs')
    return ax


def main(
        nmf_data: NMFInputData,
        W: np.ndarray,
        H: np.ndarray,
        non_zero_peaks_mask: np.ndarray,
        vis_path: str,
        component_data=None
    ):
    if component_data is None:
        component_data = get_component_data(W)

    print('Order samples by component contribution')
    project_masked_samples = nmf_data.project_masked_samples

    metadata = nmf_data.samples_metadata
    samples_mask = nmf_data.samples_mask

    binary_matrix = nmf_data.matrix[:, non_zero_peaks_mask] # samples x DHSs
    dhs_meta = nmf_data.dhs_metadata[non_zero_peaks_mask]

    if not project_masked_samples:
        metadata = metadata[samples_mask]


    ######## Plot DHSs #########
    if 'dist_tss' in dhs_meta.columns:
        ax = plot_dist_tss(H, dhs_meta['dist_tss'].values, component_data)
        plt.savefig(f'{vis_path}.Distance_to_tss.pdf', bbox_inches='tight', transparent=True)
        plt.close(plt.gcf())

    print('All DHSs')
    rng = np.random.default_rng(42)
    downsampled_indices = rng.choice(H.shape[1], size=min(10_000, H.shape[1]), replace=False)
    fig, ax = plt.subplots(figsize=(20, 2))
    ax, _, _ = component_barplot(
        H[:, downsampled_indices],
        component_data,
        order_records_by='primary',
        normalize_for_plotting=True,
        ax=ax,
        plotting_kwargs=dict(rasterized=True)
    )
    plt.savefig(f'{vis_path}.Barplot_all_DHSs.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    print('All DHSs not normalized')
    fig, ax = plt.subplots(figsize=(20, 2))
    ax, _, _ = component_barplot(
        H[:, downsampled_indices],
        component_data,
        normalize_for_plotting=False,
        order_records_by='primary',
        ax=ax,
        plotting_kwargs=dict(rasterized=True)
    )
    plt.savefig(f'{vis_path}.Barplot_all_DHSs.not_norm.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    #Only reproduced DHSs
    print('>=4 peaks supporting a DHS')
    reproduced_peaks = binary_matrix[:, downsampled_indices].sum(axis=0).A1 >= 4
    fig, ax = plt.subplots(figsize=(20, 2))
    ax, _, _ = component_barplot(
        H[:, downsampled_indices][:, reproduced_peaks],
        component_data,
        normalize_for_plotting=True,
        order_records_by='primary',
        ax=ax,
        plotting_kwargs=dict(rasterized=True)
    )
    plt.savefig(f'{vis_path}.Barplot_DHS_supported_by_4+samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


    ######### Plot samples #########
    print('All samples')
    fig, ax = plt.subplots(figsize=(20, 2))
    ax, _, _ = component_barplot(
        W,
        component_data,
        order_records_by='primary',
        ax=ax,
        plotting_kwargs=dict(rasterized=True)
    )
    plt.savefig(f'{vis_path}.Barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)
    if project_masked_samples:
        print('Reference samples set')
        fig, ax = plt.subplots(figsize=(20, 2))
        ax, _, _ = component_barplot(
            W[:, samples_mask],
            component_data,
            order_records_by='primary',
            ax=ax,
            plotting_kwargs=dict(rasterized=True)
        )
        plt.savefig(f'{vis_path}.Barplot_reference_train_samples.pdf', transparent=True, bbox_inches='tight')
        plt.close(fig)
    

    annotations = metadata["sample_label"].values
    print('Detailed barplot all samples')
    if project_masked_samples:
        label_colors = np.where(samples_mask, 'k', 'r')
    else:
        label_colors = None
    fig, _, _ = component_barplot_at_scale(
        W,
        component_data,
        order_records_by='primary',
        records_labels=annotations,
        label_colors=label_colors
    )

    plt.savefig(f'{vis_path}.Detailed_barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    print('Hierarchical barplot all samples')
    fig, _, _ = component_barplot_at_scale(
        W,
        component_data,
        records_labels=annotations,
        label_colors=label_colors,
        order_records_by='hierarchical',
        order_components_by='cluster',
        normalize_for_plotting=False
    )
    plt.savefig(f'{vis_path}.Hierarchical_barplot_all_samples.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

    print('Hierarchical barplot reference samples')
    if project_masked_samples:
        fig, _, _ = component_barplot_at_scale(
            W[:, samples_mask],
            component_data,
            records_labels=annotations[samples_mask],
            order_records_by='hierarchical',
            order_components_by='cluster',
            normalize_for_plotting=False
        )
        plt.savefig(f'{vis_path}.Hierarchical_barplot_reference_samples.pdf', transparent=True, bbox_inches='tight')
        plt.close(fig)

    print('Top 20 samples per component')
  
    axes = plot_top_contributing_samples(
        W,
        annotations,
        component_data=component_data,
        top_count=20,
    )
    plt.savefig(f'{vis_path}.Top20_all_samples_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close(plt.gcf())

    axes = plot_top_contributing_samples(
        W,
        annotations,
        component_data=component_data,
        top_count=20,
        common_scale=True
    )
    plt.savefig(f'{vis_path}.Top20_all_samples_barplot.common_scale.pdf', bbox_inches='tight', transparent=True)
    plt.close(plt.gcf())

    if project_masked_samples:
        axes = plot_top_contributing_samples(
            W[:, samples_mask],
            annotations[samples_mask],
            component_data=component_data,
            top_count=20,
        )
        plt.savefig(f'{vis_path}.Top20_reference_samples_barplot.pdf', bbox_inches='tight', transparent=True)
        plt.close(plt.gcf())

        axes = plot_top_contributing_samples(
            W[:, samples_mask],
            annotations[samples_mask],
            component_data=component_data,
            top_count=20,
            common_scale=True
        )
        plt.savefig(f'{vis_path}.Top20_reference_samples_barplot.common_scale.pdf', bbox_inches='tight', transparent=True)
        plt.close(plt.gcf())


if __name__ == '__main__':
    print('Visualizing NMF results')
    parser = setup_parser()

    print('Adding options to parser')
    parser.add_argument('W', help='W matrix of perform NMF decomposition')
    parser.add_argument('H', help='H matrix of perform NMF decomposition')
    parser.add_argument('non_zero_peaks_mask', help='Non-zero peaks mask')
    parser.add_argument('--outpath', help='Path to save visualizations', default='./')
    args = parser.parse_args()

    nmf_data = parse_nmf_args(args.prefix, args.config)
    
    if 'sample_label' not in nmf_data.samples_metadata.columns:
        if 'SPOT3_score' not in nmf_data.samples_metadata.columns or 'core_ontology_term' not in nmf_data.samples_metadata.columns:
            raise ValueError('sample_label column is missing in samples metadata')
        print('Assuming sample_label is core_ontology_term + SPOT3_score')
        nmf_data.samples_metadata.loc[:, 'sample_label'] = (
            nmf_data.samples_metadata['core_ontology_term'].astype(str) + " " + 
            nmf_data.samples_metadata['SPOT3_score'].astype(str).str[:3]
        )

    W = np.load(args.W).T # NMF components x samples
    H = np.load(args.H).T # NMF components x peaks
    non_zero_peaks_mask = np.loadtxt(args.non_zero_peaks_mask, dtype=bool)
    outprefix = f"{args.outpath}/{args.prefix}"
    main(nmf_data, W, H, non_zero_peaks_mask, outprefix)

