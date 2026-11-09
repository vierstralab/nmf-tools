import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import scipy.cluster.hierarchy as sch

from nmf_tools import in_vierstra_style
from nmf_tools.matrix_reordering import order_matrix
from nmf_tools.matrix_reordering.hcluster import hierarchical_clustering


@in_vierstra_style
def plot_stacked_barplot(bottoms, tops, colors, xvals=None, ax=None, orient='horizontal', **kwargs):
    assert bottoms.shape == tops.shape
    assert bottoms.shape[0] == len(colors)

    fb_tops = np.repeat(tops, 2, axis=1)
    fb_bottoms = np.repeat(bottoms, 2, axis=1)
    if xvals is None:
        xvals = np.arange(bottoms.shape[1] + 1)
    xvals = np.concatenate([[xvals[0]], np.repeat(xvals[1:-1], 2), [xvals[-1]]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(len(xvals)/200, 2) if orient == 'horizontal' else (2, len(xvals)/200))
    for btms, tps, color in zip(fb_bottoms, fb_tops, colors):
        if orient == 'horizontal':
            ax.fill_between(xvals, btms, tps, lw=0, color=color, **kwargs)
            ax.set_xlim(0, xvals[-1])
            ax.set_ylim(0, tops.max())
        elif orient == 'vertical':
            ax.fill_betweenx(xvals, btms, tps, lw=0, color=color, **kwargs)
            ax.set_ylim(0, xvals[-1])
            ax.set_xlim(0, tops.max())

    return ax


@in_vierstra_style
def component_barplot(matrix, component_data, box_lw=0.15, ax=None, plotting_kwargs=None, **order_matrix_kwargs):
    tops, bottoms, components_order, records_order = order_matrix(
        matrix,
        **order_matrix_kwargs
    )

    if plotting_kwargs is None:
        plotting_kwargs = {}

    ax = plot_stacked_barplot(bottoms, tops, component_data.sort_values('index')['color'],
                              ax=ax, orient='horizontal', **plotting_kwargs)

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(box_lw)

    return ax, components_order, records_order


@in_vierstra_style
def component_barplot_at_scale(
    matrix, component_data, records_labels,
    label_colors=None, figsize=None, bars_per_panel=100, **kwargs
):
    assert len(records_labels) == matrix.shape[1]
    
    tops, bottoms, components_order, records_order = order_matrix(matrix, **kwargs)
    order = records_order.col_order

    colors = component_data.sort_values('index')['color']
    
    n_chunks = np.ceil(matrix.shape[1] / bars_per_panel).astype(int)
    
        
    if figsize is None:
        figsize = (20, 4 * n_chunks)
    fig, axes = plt.subplots(n_chunks, 1, figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=1.5)

    if n_chunks == 1:
        axes = [axes]
    
    maxv = np.max(tops)
    for k in tqdm(np.arange(n_chunks)):
        ax = axes[k]
        sl = slice(bars_per_panel * k, bars_per_panel * (k + 1), 1)
        num_elements = order[sl].shape[0]

        plot_stacked_barplot(
            bottoms[:, sl], tops[:, sl], colors,
            ax=ax, orient='horizontal',
        )
        ax.set_xticks(np.arange(num_elements) + 0.5)
        ax.set_xticklabels(
            np.asarray(records_labels)[order[sl]],
            rotation=90
        )
        ax.set_xlim(0, bars_per_panel)
        if label_colors is not None:
            assert len(label_colors) == matrix.shape[1]
            for xtick, col in zip(ax.get_xticklabels(), label_colors[sl]):
                xtick.set_color(col)
    
    for ax in axes:
        ax.set_ylim(0, maxv*1.05)

    return fig, components_order, records_order


@in_vierstra_style
def component_barplot_hcl(matrix, component_data, ax=None, separator_lw=0.5,
                          cluster_threshold=0.7, criterion='distance',
                          linkage_matrix=None, normalize_for_plotting=False,
                          records_labels=None,
                          **kwargs):
    if linkage_matrix is None:
        linkage_matrix = hierarchical_clustering(matrix, **kwargs)
    
    tops, bottoms, components_order, records_order = order_matrix(
        matrix,
        order_components_by='cluster',
        order_records_by='hierarchical',
        order_records_kwargs=dict(linkage_matrix=linkage_matrix),
        order_components_kwargs=dict(
            linkage_matrix=linkage_matrix,
            cluster_threshold=cluster_threshold,
            criterion=criterion
        ),
        normalize_for_plotting=normalize_for_plotting,
    )

    clusters = sch.fcluster(linkage_matrix, t=cluster_threshold, criterion=criterion)


    ax = plot_stacked_barplot(bottoms, tops, component_data.sort_values('index')['color'],
                         ax=ax, orient='horizontal')
    
    if records_labels is not None:
        records_labels = records_order(records_labels)
        ax.set_xticks(np.arange(len(records_labels)) + 0.5)
        ax.set_xticklabels(records_labels, rotation=90)
    else:
        ax.set_xticks([])
    ax.set_xlim(0, matrix.shape[1])
    
    for i in np.where(np.diff(records_order(clusters)) != 0)[0]:
        ax.axvline(i+1, color='k', lw=separator_lw)

    return ax, components_order, records_order


@in_vierstra_style
def component_barplot_with_dendrogram(
    matrix, component_data, records_labels=None,
    linkage_matrix=None,
    cluster_threshold=0.7,
    criterion='distance',
    normalize_for_plotting=False,
    figsize=None,
    **kwargs
):
    if records_labels is not None:
        assert len(records_labels) == matrix.shape[1]

    if linkage_matrix is None:
        linkage_matrix = hierarchical_clustering(matrix, **kwargs)

    if figsize is None:
        figsize=(20, 4)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    gs = gridspec.GridSpec(2, 1, hspace=0)

    ax1 = fig.add_subplot(gs[0])
    sch.dendrogram(linkage_matrix, color_threshold=cluster_threshold, ax=ax1)
    ax1.axhline(cluster_threshold, color='r')
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gs[1])
    ax2, components_order, records_order = component_barplot_hcl(
        matrix, component_data,
        ax=ax2,
        linkage_matrix=linkage_matrix,
        cluster_threshold=cluster_threshold,
        criterion=criterion,
        normalize_for_plotting=normalize_for_plotting,
        records_labels=records_labels,
    )

    return ax1, ax2, components_order, records_order


@in_vierstra_style
def plot_component_top_barplot(data, labels, color, ax=None, top_count=15):
    n_samples = data.shape[0]
    top_count_actual = min(top_count, n_samples)

    sorted_indices = np.argsort(data)[-top_count_actual:]
    sorted_data = data[sorted_indices]
    sorted_names = np.asarray(labels)[sorted_indices]

    ax.barh(np.arange(top_count_actual), sorted_data, color=color)
    ax.set_yticks(np.arange(top_count_actual))
    ax.tick_params(length=3, pad=1)
    ax.set_ylim(top_count_actual - top_count - 0.5, top_count_actual - 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
    ax.set_yticklabels(sorted_names, ha='right', va='center', fontsize=4, color='k')
    if len(sorted_data) > 0:
        ax.set_xlim(0, 1.1 * sorted_data.max())
    return ax


def plot_top_contributing_samples(W, annotations, component_data, ncols=5, top_count=10, common_scale=False, figsize=None, wspace=1, hspace=0.3, component_is_major=False):
    n_components = W.shape[0]

    ncols = int(np.ceil(ncols))
    nrows = int(np.ceil(n_components / ncols))
    
    if figsize is None:
        figsize = ncols * 3.2, nrows * 2.7

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    gs = gridspec.GridSpec(nrows, ncols, wspace=wspace, hspace=hspace)

    axes = []
    xlims = []
    for i, (_, row) in enumerate(component_data.iterrows()):
        ax = fig.add_subplot(gs[i])
        W_slice = W[row['index'], :]
        if component_is_major:
            major_comp_mask = np.argmax(W, axis=0) == row['index']
            W_slice = W_slice[major_comp_mask]
            annots = annotations[major_comp_mask]
        else:
            annots = annotations

        ax = plot_component_top_barplot(
            W_slice,
            annots,
            color=row['color'],
            ax=ax,
            top_count=top_count
        )
        ax.set_title(f'{row["name"]}', fontsize=4, pad=2)
        xlims.append(ax.get_xlim())
        axes.append(ax)
    if common_scale:
        for ax in axes:
            ax.set_xlim(0, max(xlims, key=lambda x: x[1])[1])
    return axes
