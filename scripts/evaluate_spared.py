# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a trained model using implementations from `meru.evaluation` module.
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import anndata as ad
import os

meru_path = Path(__file__).parent.parent
sys.path.append(str(meru_path))

from meru.config import LazyConfig, LazyFactory, SpaREDData
from meru.utils.checkpointing import CheckpointManager
import meru.utils.distributed as dist
from meru.utils.utils import seed_everything
from meru.utils.visualize_predictions import log_pred_image

seed_everything()

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--config", help="Path to an evaluation config file (.py)")
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--prediction_layer", default="c_StepB_log1p", help="Prediction layer.")
_AA("--dataset", default="villacampa_lung_organoid", help="SpaRED datasets.")
_AA("--top_k", type=int, default=50, help="Top k to use during evaluation.")
_AA("--num_genes", type=int, default=128, help="TNum genes in dataset.")
_AA("--encode_genes", action="store_true", help="Activa geneformer encoding (por defecto False).")

def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    _C = LazyConfig.load(_A.config)
    # Modify flexible elements in config
    _C_TRAIN['model']['textual']['context_length'] = _A.num_genes
    del _A.num_genes

    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C.evaluator)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

    # Get dataset
    collection_processed = ad.read_h5ad(os.path.join(f'/media/lambda003/SSD5/pcardenasg/ST/MICCAI25/SpaRED_data/original/{_A.dataset}.h5ad'))
    # FIXME: reordenando los genes al orden original de SpaRED
    collection_processed = collection_processed[:, np.argsort(collection_processed.var['original_index'].astype(int))].copy()
    collection_processed.var.index = collection_processed.var['gene_ids']
    # Train dataloader
    train_adata = collection_processed[collection_processed.obs["split"] == "train"]
    print(f"Training split is of shape {train_adata.shape}")
    train_data = SpaREDData(adata=train_adata, args=_A, split_name="train")
    train_loader = DataLoader(train_data, 
                        batch_size=_C_TRAIN.train.total_batch_size // dist.get_world_size(), 
                        shuffle=False, 
                        pin_memory=True, 
                        drop_last=False, 
                        num_workers=_C_TRAIN.train.num_workers)
    
    # Test dataloader
    test_split = "test" if "test" in collection_processed.obs["split"].unique() else "val"
    test_adata = collection_processed[collection_processed.obs["split"] == test_split]
    print(f"Test split is of shape {test_adata.shape}")
    test_data = SpaREDData(adata=test_adata, args=_A, split_name=test_split)
    test_loader = DataLoader(test_data, 
                        batch_size=_C_TRAIN.train.total_batch_size // dist.get_world_size(), 
                        shuffle=False, 
                        pin_memory=True, 
                        drop_last=False, 
                        num_workers=_C_TRAIN.train.num_workers)
    
    metrics, results = evaluator(train_loader, test_loader, _A, model, top_k=_A.top_k)
    predictions, _ = results
    test_adata.layers["meru_preds"] = predictions.numpy()
    log_pred_image(test_adata, pred_layer='meru_preds', gt_layer=_A.prediction_layer, save_path=f"visualization_results/{_A.dataset}")

    # Log results for copy-pasting to spreadsheet, including checkpoint path.
    print(metrics)

if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)

#CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_spared.py --config configs/gene_expression_prediction.py --checkpoint-path /home/dvegaa/meru/outputs_uni/villacampa_lung_organoid/checkpoint_final.pth --train-config configs/train_meru_spared.py --dataset villacampa_lung_organoid