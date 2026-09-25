# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core module for lazily instantiating objects from arbitrary configs. Many design
choices in this module are heavily influenced by Detectron2.
"""
from __future__ import annotations

import builtins
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn, optim
from torch.distributed.algorithms.ddp_comm_hooks import default as ddph
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import squidpy as sq

from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
import meru.utils.distributed as dist
from meru.utils.utils import seed_everything
from UNI.spared.configs import EightSymmetry

__all__ = ["callable_to_str", "LazyCall", "LazyConfig", "LazyFactory"]

REPO_ROOT = Path(__file__).resolve().parent.parent


_CFG_PACKAGE_NAME = "meru._cfg_loader"
"""
Shared module namespace to import all config objects.
"""

torch.set_num_threads(1)
seed_everything()

@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist."
            )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            spec = importlib.machinery.ModuleSpec(
                _CFG_PACKAGE_NAME + "." + os.path.basename(cur_file),
                None,
                origin=cur_file,
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = DictConfig(module.__dict__[name], flags={"allow_objects": True})
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


def callable_to_str(some_callable: Callable) -> str:
    # Return module and name of a callable (function or class) for OmegaConf.
    return f"{some_callable.__module__}.{some_callable.__qualname__}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed, but
    returns a dict that describes the call. Only supports keyword arguments.
    """

    def __init__(self, target: Callable):
        if not callable(target):
            raise TypeError(f"LazyCall target must be a callable! Got {target}")
        self.T = target

    def target_str(self):
        return

    def __call__(self, **kwargs):
        # Pop `_target_` if it already exists in kwargs. This happens when the
        # callable target is changed while keeping everything else same.
        _ = kwargs.pop("_target_", None)

        # Put current target first; it reads better in printed/saved output.
        kwargs = {"_target_": callable_to_str(self.T), **kwargs}
        return DictConfig(content=kwargs, flags={"allow_objects": True})


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load(filename: str | Path) -> DictConfig:
        """
        Load a config file (either Python or YAML).

        Args:
            filename: absolute path or relative path w.r.t. current directory.
        """
        filename = str(filename).replace("/./", "/")
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")

        if filename.endswith(".py"):
            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _CFG_PACKAGE_NAME + "." + os.path.basename(filename),
                }
                with open(filename) as f:
                    content = f.read()
                # Compile first with filename to make filename appear in stacktrace
                exec(compile(content, filename, "exec"), module_namespace)

            # Collect final objects in config:
            ret = OmegaConf.create(flags={"allow_objects": True})

            for name, value in module_namespace.items():
                # Ignore "private" variables (starting with underscores).
                if name.startswith("_"):
                    continue

                if isinstance(value, (DictConfig, dict)):
                    value = DictConfig(value, flags={"allow_objects": True})
                    ret[name] = value

                if isinstance(value, (ListConfig, list)):
                    value = ListConfig(value, flags={"allow_objects": True})
                    ret[name] = value
        else:
            with open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        return ret

    @staticmethod
    def save(cfg: DictConfig, filename: str) -> None:
        """
        Save a config object as YAML file. (same as :meth:`OmegaConf.save`).
        """
        OmegaConf.save(cfg, filename, resolve=False)

    @staticmethod
    def apply_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
        """
        Return a new config by applying overrides (provided as dotlist). See
        https://hydra.cc/docs/advanced/override_grammar/basic/ for dotlist syntax.
        """
        return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


class LazyFactory:
    """
    Provides a clean interface to easily construct essential objects from input
    lazy configs (omegaconf): dataloader, model, optimizer, and LR scheduler.
    """

    @staticmethod
    def build_dataloader(cfg: DictConfig):
        # Instantiate dataset and wrap in dataloader.
        return DataLoader(
            instantiate(cfg.dataset),
            num_workers=cfg.train.num_workers,
            batch_size=cfg.train.total_batch_size // dist.get_world_size(),
            drop_last=True,
            pin_memory=True,
        )

    @staticmethod
    def build_model(cfg: DictConfig, device: torch.device | None = None):
        # Get the current device as set for current distributed process.
        # Check `launch` function in `meruu.utils.distributed` module.
        device = device or torch.cuda.current_device()
        model = instantiate(cfg.model).to(device)

        # Wrap model in DDP if using more than one GPUs.
        if dist.get_world_size() > 1:
            model = DistributedDataParallel(model, [device], **cfg.train.ddp)

            # Optionally add FP16 compression hook with AMP.
            if cfg.train.amp and cfg.train.ddp_fp16_compression:
                model.register_comm_hook(state=None, hook=ddph.fp16_compress_hook)

        return model

    @staticmethod
    def build_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
        # Iterate named parameters of the model. Use internal `module` for DDP.
        if isinstance(model, DistributedDataParallel):
            model = model.module

        # Add model as an input to `set_weight_decay_per_param`.
        cfg.optim.optimizer.params.model = model
        return instantiate(cfg.optim.optimizer)

    @staticmethod
    def build_lr_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
        return instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)


class SpaREDData(torch.utils.data.Dataset):
    def __init__(self, adata, args, split_name = "train"):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            adata (ad.AnnData): An anndata object with the data of the entire dataset.
            args (argparse): parser with the values necessary for data processing.
            split_name (str): name of the data split being processed. Useful for identifying which data split the model is being tested on.
        """

        self.args = args
        self.adata = adata
        self.pred_layer = args.prediction_layer
        self.split_name = split_name
        self.active_embedding = args.active_embedding

        if self.args.num_neighs > 0:
            # Get adjacency matrix.
            self.adj_mat = None
            self.neigh_dim = 0
            self.get_adjacency(self.args.num_neighs)

        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])
        # Retreive the mask from the adata, where "False" corresponds to the values that contain the median as expression value.
        self.median_imp_mask = torch.tensor(self.adata.layers['mask'])
        # Get image patches
        tissue_tiles = adata.obsm["patches_scale_1.0"]
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        tissue_tiles = torch.from_numpy(tissue_tiles)
        tissue_tiles = tissue_tiles.permute(0,3,1,2)
        self.embeddings = tissue_tiles/255

        # Get unique identifier
        self.IDs = adata.obsm["spatial"]

        # Load CSV containing gene encoded data
        if self.args.gene_encoder == "geneformer":
            self.gene_encoded = pd.read_csv(f"{REPO_ROOT}/embeddings/embs_geneformer/{self.args.dataset}/{self.split_name}/encoded_genes.csv", index_col=0)
            self.gene_encoded = torch.tensor(self.gene_encoded.values, dtype=torch.float32)

        elif self.args.gene_encoder == "autoencoder":
            self.gene_encoded = torch.load(f"{REPO_ROOT}/embeddings/results/{self.args.prediction_layer}/{self.args.dataset}/gene_embeddings_{self.split_name}.pt", weights_only=False)
            self.gene_encoded = torch.tensor(self.gene_encoded, dtype=torch.float32)

        elif self.args.gene_encoder == "language":
            self.gene_encoded = torch.load(f"{REPO_ROOT}/embeddings/lang_embeds/{self.args.prediction_layer}/{self.args.dataset}/lang_gene_embeddings_{self.split_name}.pt", weights_only=False)
            self.gene_encoded = torch.tensor(self.gene_encoded, dtype=torch.float32)

        elif self.args.gene_encoder == "nicheformer":
            self.gene_encoded = np.load(f"{REPO_ROOT}/embeddings/spared_embeddings/{self.args.prediction_layer}/{self.args.dataset}/{self.split_name}.npy")
            self.gene_encoded = torch.tensor(self.gene_encoded, dtype=torch.float32)

        if not self.args.raw_patches:
            # Get precomputed image embeddings
            if not self.active_embedding:
                # Retrieve image embeddings that do not consider image transformetions
                print("Loading pre-computed image embeddings...")
                self.embeddings = []
                for idx in range(len(self.adata)):
                    if self.args.eight_sym_test and split_name != "train":
                        self.embeddings.append(np.load(f"{REPO_ROOT}/embeddings/aug_features/{self.args.dataset}/{self.split_name}/{float(self.IDs[idx][0])}_{float(self.IDs[idx][1])}.npy"))
                    else:
                        self.embeddings.append(np.load(f"{REPO_ROOT}/embeddings/features/{self.args.dataset}/{self.split_name}/{float(self.IDs[idx][0])}_{float(self.IDs[idx][1])}.npy"))
                self.embeddings = torch.tensor(self.embeddings)

            else:
                embs_folder = "tuned_features" if self.args.uni_finetuned else "aug_features"
                # Retrieve image embeddings created with image transformations for data augmentation
                self.embeddings = []
                for idx in range(len(self.adata)):
                    self.embeddings.append(np.load(f"{REPO_ROOT}/embeddings/{embs_folder}/{self.args.dataset}/{self.split_name}/{float(self.IDs[idx][0])}_{float(self.IDs[idx][1])}.npy"))
                self.embeddings = torch.tensor(self.embeddings)

        else:
            # Define image transformations - IMPORTANT: Normalization is done during the model's forward pass
            if split_name == "train":     
                self.transforms = Compose(
                    [RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomApply([RandomRotation((90, 90))], p=0.5)]
                    )
            else: 
                self.transforms = Compose([EightSymmetry()])
        
    def get_adjacency(self, k = 6):
        """
        This function calls for squidpy's spatial_neighbors function to obtain the k closest neighbors of each spot in self.adata.
        """
        self.adj_mat = torch.zeros((self.adata.shape[0], self.adata.shape[0]))
        init_pos = 0
        # Get num_neighs nearest neighbors for each spot in each slide
        for i in range(len(self.adata.obs.slide_id.unique())):
            slide = self.adata.obs.slide_id.unique()[i]
            slide_adata = self.adata[self.adata.obs.slide_id == slide].copy()
            sq.gr.spatial_neighbors(slide_adata, coord_type='generic', n_neighs=k)
            
            # Save slide adjacency in general adjacency matrix
            slide_adj = torch.tensor(slide_adata.obsp['spatial_connectivities'].todense())
            self.adj_mat[init_pos:init_pos+slide_adj.shape[0], init_pos:init_pos+slide_adj.shape[0]] = slide_adj
            init_pos += slide_adj.shape[0]
        
        if not (self.adata.obsp['spatial_connectivities'].sum(0) == k).all():
            self.neigh_dim = 1

    def build_data(self, idx):
        # Get gene emebddings
        if self.args.encode_genes:
            encoded_gene = self.gene_encoded[idx]
        # Get expression data
        exp_matrix = self.expression_mtx[idx].type('torch.FloatTensor') # Original dtype was 'torch.float64'
        # Get mask
        median_imp_mask = self.median_imp_mask[idx]
        # Get image embeddings if needed
        embeddings = self.embeddings[idx]

        if self.active_embedding:
            if self.split_name == "train":
                # Randomly select one of the multiple image variations obtained through data augmentation
                embeddings = embeddings[torch.randint(0, len(embeddings), (1,))].squeeze(0)
            else:
                if not self.args.eight_sym_test:
                    # Select only the first sample, which corresponds to the original symmetry
                    embeddings = embeddings[0]

        if self.args.raw_patches:
            embeddings = self.transforms(embeddings)

        # Include neighborhood information if necessary
        embeddings_neighs = None
        if self.args.num_neighs > 0:
            neighs_ids = self.adj_mat[idx,:] if self.neigh_dim else self.adj_mat[:,idx]

            exp_neighs = self.expression_mtx[neighs_ids==1,:].type('torch.FloatTensor')
            neighs_mask = self.median_imp_mask[neighs_ids==1,:]
            embeddings_neighs = self.embeddings[neighs_ids==1,:]
            if self.active_embedding:
                if self.split_name == "train":
                    # Randomly select one of the multiple image variations obtained through data augmentation
                    embeddings_neighs = embeddings_neighs[:, torch.randint(0, len(embeddings_neighs), (1,)), :].squeeze(1)
                else:
                    if not self.args.eight_sym_test:
                        # Select only the first sample, which corresponds to the original symmetry
                        embeddings_neighs = embeddings_neighs[:,0,:]

            if self.args.encode_genes:
                enc_genes_neighs = self.gene_encoded[neighs_ids==1,:]    

        if self.args.encode_genes:
            data = (exp_matrix, median_imp_mask, encoded_gene, embeddings, embeddings_neighs)
        else:
            data = (exp_matrix, median_imp_mask, embeddings, embeddings_neighs)
        return data
        
    def __getitem__(self, idx):
        item = {'split_name': self.split_name}
        # Get expression and median imputation mask of neighborhoods
        if self.args.encode_genes:
            item['exp_matrix'], item['mask'], item["encoded_genes"], item['patches'], embeddings_neighs = self.build_data(idx)
        else:
            item['exp_matrix'], item['mask'], item['patches'], embeddings_neighs = self.build_data(idx)

        if self.args.num_neighs > 0:
            item["patches_neighs"] = embeddings_neighs

        return item

    def __len__(self):
        return len(self.adata)