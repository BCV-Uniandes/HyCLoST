import sys
from pathlib import Path
import anndata as ad
import os
import matplotlib.pyplot as plt
import matplotlib
import squidpy as sq
import numpy as np

meru_path = Path(__file__).parent.parent
sys.path.append(str(meru_path)) 
from meru.evaluation.metrics import get_metrics

def log_pred_image(adata, pred_layer, gt_layer, n_genes: int = 2, guiding_metric: str = "mse", save_path="", split=""):
    
    ### 1. Get the selected genes
    assert not (pred_layer is None), 'predictions layer not present in the adata'

    # Get partition names of current dataset
    partitions = list(adata.obs.split.unique())
    # Get dict of adatas for corresponding split
    split = adata.obs.split.unique()[0] if split == "" else split
    adatas_dict = {split: adata[adata.obs.split == split, :]}

    # Compute and add detailed metrics for each split

    # Get detailed metrics from partition
    detailed_metrics = get_metrics(
        gt_mat = adata.to_df(layer=gt_layer).values,
        pred_mat = adata.to_df(layer=pred_layer).values,
        mask = adata.to_df(layer='mask').values,
        detailed=True
    )

    # Add detalied metrics to global adata
    adata.var[f'pcc_{split}'] = detailed_metrics['detailed_PCC-Gene']
    adata.var[f'r2_{split}'] = detailed_metrics['detailed_R2-Gene']
    adata.var[f'mse_{split}'] = detailed_metrics['detailed_mse_gene']
    adata.var[f'mae_{split}'] = detailed_metrics['detailed_mae_gene']
    adata.var[f'avg_err_{split}'] = detailed_metrics['detailed_error_gene']
    
    # Get ordering split
    order_split = 'test' if 'test' in partitions else 'val'
    # Get selected genes based on the best and worse guiding_metric values
    n_top = adata.var.nlargest(n_genes, columns=f'{guiding_metric}_{order_split}').index.to_list()
    n_botom = adata.var.nsmallest(n_genes, columns=f'{guiding_metric}_{order_split}').index.to_list()
    selected_genes = {f"top_{guiding_metric}": n_top, f"bottom_{guiding_metric}": n_botom}

    ### 2. Get the first slide in the input adata.
    slide_adata = {p: list(adatas_dict[p].obs.slide_id.unique())[0] for p in partitions}
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide_adata[split]].copy()

    for set_name, genes_list in selected_genes.items():
        log_genes_set(genes_list, slide_adata, gt_layer, pred_layer, set_name, split=split, save_path=save_path)

def log_genes_set(genes, slide_adata, gt_layer, pred_layer, set_name, split = 'test', mask_layer='mask', guiding_metric: str = "mse", save_path=""):

    # Get the slide
    slide = list(slide_adata.obs.slide_id.unique())[0]
    slide_adata.uns['spatial'] = {slide: slide_adata.uns['spatial'][slide]}
    # Define order of rows in dict
    order_dict = {}
    for i, gene in enumerate(genes):
        order_dict[gene] = i

    # Declare figure TODO: modify number of columns if needed (ncols = mask + # of pred methods + gt + linear plot)
    num_cols = 4
    fig, ax = plt.subplots(nrows=len(genes), ncols=num_cols, layout='constrained')
    fig.set_size_inches(18, 4 * len(genes))

    # Iterate over the genes
    for g in genes:         
        # Get current row
        row = order_dict[g]
        # Get min and max of the selected top genes in the slide
        gene_min_pred = slide_adata[:, g].layers[pred_layer].min() 
        gene_max_pred = slide_adata[:, g].layers[pred_layer].max()
        
        gene_min_gt = slide_adata[:, g].layers[gt_layer].min() 
        gene_max_gt = slide_adata[:, g].layers[gt_layer].max() 
        
        gene_min = min([gene_min_pred, gene_min_gt])
        gene_max = max([gene_max_pred, gene_max_gt])
        
        # Set PCC spackle
        pcc_preds = str(round(slide_adata.var[f"pcc_{split}"][g], 3))
        # Set MSE spackle
        mse_preds = str(round(slide_adata.var[f"mse_{split}"][g], 3))

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
        
        plot_mask = slide_adata.layers[mask_layer]
        plot_mask = plot_mask.astype(bool) 
        # valores reales en donde la máscara es True y NaN en donde la mascara el False
        gt_masked = np.where(plot_mask, slide_adata.layers[gt_layer], np.nan)
        slide_adata.layers["gt_masked"] = gt_masked
        
        # Plot layers
        sq.pl.spatial_scatter(slide_adata, color=[g], layer="gt_masked", fig=fig, ax=ax[row,0], cmap='jet', norm=norm, colorbar=True, title="", na_color="black")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=gt_layer, fig=fig, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
        sq.pl.spatial_scatter(slide_adata, color=[g], layer=pred_layer, fig=fig, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title="", na_color="black")
        
        # Set titles
        metric_value = pcc_preds if guiding_metric=="pcc" else mse_preds
        ax[row, 2].set_title(f'{guiding_metric.upper()} = {metric_value}', fontsize='xx-large')
        
        # Set y labels
        ax[row,0].set_ylabel(f'{g}:\n{slide}\n', fontsize='x-large') # Add gene name to the corresponding row
        ax[row,0].set_xticks([])
        ax[row,0].set_yticks([])
        ax[row,1].set_ylabel('')
        ax[row,2].set_ylabel('')
        
        # Set x labels 
        ax[row,0].set_xlabel('')
        ax[row,1].set_xlabel('')
        ax[row,2].set_xlabel('')
        
        # Define gene adata
        gene_adata = slide_adata[:,g].copy()

        # Define models prediction and ground truth (only masked spots)
        ceros_gt = [gene_adata.layers[gt_layer][gene_adata.layers[mask_layer]==True]][0]
        ceros_pred = [gene_adata.layers[pred_layer][gene_adata.layers[mask_layer]==True]][0]

        # Plot gen predictions and ground truth
        color = ["#1a8899", "#a5e1e3"]
        ax[row,3].plot(ceros_gt, ceros_gt, color="black", linestyle="-", label="Ground Truth")
        ax[row,3].plot(ceros_gt, ceros_pred, color=color[1], marker="o",  markersize=3, linestyle="None", label="Preds")
        ax[row,3].set_xlabel("Ground Truth", fontsize="large")
        ax[row,3].set_ylabel("Prediction", fontsize="large")
        ax[row,3].legend(loc="upper left", fontsize="large")
    
    
    # Format figure
    for i, axis in enumerate(ax.flatten()):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if ((i+1)%num_cols) != 0: 
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
    
    # Set PCC
    pcc_metric = str(round(slide_adata.var[f"pcc_{split}"][genes[0]], 3))
    # Set MSE
    mse_metric = str(round(slide_adata.var[f"mse_{split}"][genes[0]], 3))

    metric_value = pcc_metric if guiding_metric=="pcc" else mse_metric

    # Set titles
    ax[0, 0].set_title(f"Masked Map", fontsize='xx-large')
    ax[0, 1].set_title('Ground Truth', fontsize='xx-large')
    ax[0, 2].set_title(f'Preds\n{guiding_metric.upper()} = {metric_value}', fontsize='xx-large')

    # FIXME: change file extension to PDF
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'preds_{guiding_metric}_{set_name}.png'), bbox_inches='tight')
    # TODO: Log image to wandb if necessary
