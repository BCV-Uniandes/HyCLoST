# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
import pandas as pd
import torchvision.transforms as T
from loguru import logger
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import anndata as ad
import umap
import matplotlib.pyplot as plt
import os

from meru import lorentz as L
from meru.evaluation.metrics import get_metrics
from meru.models import MERU, CLIPBaseline
from meru.utils.utils import seed_everything

seed_everything()

class GeneExpressionPrediction:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        num_workers: int = 4,
        image_size: int = 224,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            num_workers: Number of CPU works to parallelize data loading for
                extracting features.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        self._num_workers = num_workers

        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(self, train_loader, test_loader, args, model: MERU | CLIPBaseline, top_k: int = 50) -> dict[str, float]:
        model = model.eval()

        # Get the features of all gene expression vectors as well as the original gene expression values
        encode_genes = args.encode_genes
        _, train_text_feats, train_labels, _, = _encode_dataset(train_loader, model, project=True, encode_genes=encode_genes)
        test_image_feats, _, test_labels, test_masks, = _encode_dataset(test_loader, model, project=True, encode_genes=encode_genes)

        # Evaluate in small batches of 256 instances.
        all_predictions, all_labels, all_masks = [], [], []
        
        for _feats, _labels, _masks in zip(test_image_feats.split(256), test_labels.split(256), test_masks.split(256)):
            if _feats.dim() == 2:
                predictions = self.get_preds(model, _feats, train_text_feats, train_labels, top_k)
            else: # eight symmetry test is active
                _feats = _feats.permute(1,0,2)
                sample_predictions = []
                for i in range(len(_feats)):
                    img_sym = _feats[i]
                    pred = self.get_preds(model, img_sym, train_text_feats, train_labels, top_k)
                    sample_predictions.append(pred)

                predictions = torch.stack(sample_predictions).mean(0)
            
            all_predictions.append(predictions)
            all_labels.append(_labels)
            all_masks.append(_masks)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        metrics = get_metrics(all_labels, all_predictions, all_masks)
        return metrics, (all_predictions, all_labels)

    def get_preds(self, model, _feats, train_text_feats, train_labels, top_k):
        
        # Compute pairwise similarity depending on model type:
        if isinstance(model, MERU):
            scores = L.pairwise_inner(_feats, train_text_feats, model.curv.exp())
        else:
            scores = _feats @ train_text_feats.T

        _, topk_idx = torch.topk(scores.cpu(), k=top_k, dim=1)
        exp_vectors = train_labels[topk_idx]
        predictions = exp_vectors.mean(dim=1)
        
        return predictions

def _encode_dataset(
    data_loader: DataLoader,
    model: MERU | CLIPBaseline,
    project: bool,
    encode_genes: bool
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_text_feats, all_labels, all_masks = [], [], [], []

    for batch in tqdm(data_loader, desc=f"Extracting image feats"):
        if encode_genes:
            images, labels, masks, encoded_genes = batch['patches'], batch["exp_matrix"], batch['mask'], batch['encoded_genes']
        else:
            images, labels, masks, encoded_genes = batch['patches'], batch["exp_matrix"], batch['mask'], batch['exp_matrix']
            
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)

        with torch.inference_mode():
            if images.dim() == 5: # true when images are processed raw (not pre-encoded with UNI)
                all_syms = []
                for i in range(len(images)):
                    all_syms.append(model.encode_image(images[i].to(model.device), project))
                image_feats = torch.stack(all_syms).permute(1,0,2)
            else:
                image_feats = model.encode_image(images.to(model.device), project)
            
            text_feats = model.encode_text(list(encoded_genes), project)
        
        all_image_feats.append(image_feats)
        all_text_feats.append(text_feats)
        all_labels.append(labels)
        all_masks.append(masks)

    return torch.cat(all_image_feats, dim=0), torch.cat(all_text_feats, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_masks, dim=0)

def visualize_embeddings_eucledian(
        model,
        image_feats,
        text_feats
):

    image_feats_euc = L.log_map0(image_feats, model.curv.exp().cpu())
    text_feats_euc  = L.log_map0(text_feats, model.curv.exp().cpu())

    # Concatenar para UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    all_feats = torch.cat([image_feats_euc.cpu(), text_feats_euc.cpu()], dim=0)
    embedding_2d = reducer.fit_transform(all_feats.cpu().numpy())

    # Separar índices
    B = image_feats.shape[0]
    plt.figure(figsize=(8,8))
    plt.scatter(embedding_2d[:B,0], embedding_2d[:B,1], label="Images", alpha=0.7)
    plt.scatter(embedding_2d[B:,0], embedding_2d[B:,1], label="Text", alpha=0.7)
    plt.legend()
    plt.savefig("image_text_embeddings.png")
    
def visualize_embedding_poincare(image_feats: torch.Tensor,
                              text_feats: torch.Tensor,
                              curv: torch.Tensor,
                              save_path: str = "poincare_embeddings_scheduler.png"):
    """
    Visualize MERU hyperbolic embeddings in the Poincaré disk.

    Args:
        image_feats: Tensor of shape (B, D), Lorentz hyperboloid image embeddings.
        text_feats: Tensor of shape (B, D), Lorentz hyperboloid text embeddings.
        curv: Scalar tensor, hyperbolic curvature (c > 0).
        save_path: Optional path to save the figure. If None, just show.
    """
    # Ensure CPU and float
    image_feats = image_feats.detach().cpu().float()
    text_feats  = text_feats.detach().cpu().float()
    c = curv.detach().cpu().float()
    sqrt_c = torch.sqrt(c)

    # Convert to Poincaré disk
    def hyperboloid_to_poincare(x_hyper):
        x0 = x_hyper[:, 0:1]
        xs = x_hyper[:, 1:]
        return xs / (x0 + 1.0)

    image_poincare = hyperboloid_to_poincare(image_feats).numpy()
    text_poincare  = hyperboloid_to_poincare(text_feats).numpy()

    # Hyperbolic distance to origin
    def hyperbolic_distance_from_origin(x_hyper):
        x0 = x_hyper[:, 0]
        arg = torch.clamp(sqrt_c * x0, min=1.0 + 1e-5)
        return (torch.acosh(arg) / sqrt_c).numpy()

    image_dist = hyperbolic_distance_from_origin(image_feats)
    text_dist  = hyperbolic_distance_from_origin(text_feats)

    B = image_feats.shape[0]

    # Plot
    plt.figure(figsize=(7, 7))
    
    # Images
    plt.scatter(image_poincare[:, 0], image_poincare[:, 1],
                c=image_dist, cmap="viridis", label="Images",
                s=50, alpha=0.8, edgecolor='k')

    # Text
    plt.scatter(text_poincare[:, 0], text_poincare[:, 1],
                c=text_dist, cmap="plasma", label="Text",
                s=50, alpha=0.8, edgecolor='k')

    # Boundary of Poincaré disk
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1.5)
    plt.gca().add_artist(circle)
    plt.gca().set_aspect('equal', adjustable='box')

    # Colorbar and labels
    plt.colorbar(label="Hyperbolic distance to origin")
    plt.legend()
    plt.title("MERU embeddings in Poincaré disk")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)

