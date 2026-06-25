# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a MERU or CLIP model based on parameters specified by a config file.
"""
import argparse
import time
import random
from pathlib import Path

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path
import anndata as ad
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from datetime import datetime
from hydra.utils import instantiate

meru_path = Path(__file__).parent.parent
sys.path.append(str(meru_path))

import meru.utils.distributed as dist
from meru.config import LazyConfig, LazyFactory, SpaREDData
from meru.utils.checkpointing import CheckpointManager
from meru.utils.timer import Timer
from meru.utils.utils import seed_everything

torch.set_num_threads(1)

seed_everything()

# fmt: off
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--output-dir", default="./output",
    help="Path to a directory to save checkpoints and job logs.",
)
parser.add_argument(
    "--resume", action="store_true",
    help="Whether to resume training from `--output-dir`. This script will find "
    "the last saved checkpoint and resume training. It is user's responsibility "
    "to provide matching config file in `--config`.",
)
parser.add_argument(
    "--validation-period", type=int, default=100, help="Checkpoint log validation metrics."
)
parser.add_argument(
    "--log-period", type=int, default=100,
    help="Log to stdout/tensorboard periodically (only main process).",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="Integer in [0, num_machines) to specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training, it defaults to "
    "localhost for single-machine training.",
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides (key-value pairs)."
)
parser.add_argument(
    "--dataset", default="villacampa_lung_organoid",
    help="Name of the dataset.",
)
parser.add_argument(
    "--prediction_layer", default="c_StepB_log1p",
    help="Prediction layer.",
)
parser.add_argument(
    "--top_k", type=int, default=50,
    help="Top k to use during evaluation.",
)
parser.add_argument(
    "--entail_weight", type=float, default=0.2,
    help="Weight of entailment loss on overall model loss.",
)
parser.add_argument(
    "--neigh_weight", type=float, default=0.3,
    help="Weight of neighbor loss on overall model loss.",
)
parser.add_argument(
    "--num_genes", type=int, default=128,
    help="Amount of genes in dataset.",
)
parser.add_argument(
    "--num_neighs", type=int, default=0,
    help="Amount of neighbors to retrieve for context.",
)
parser.add_argument(
    "--encode_genes", action="store_true",
    help="Activa geneformer encoding (por defecto False)."
)
parser.add_argument(
    "--gene_encoder", default="geneformer",
    help="Defines method for gene embedding."
)
parser.add_argument(
    "--ckpts_path", type=str, default="",
    help="Path to checkpoints when only performing test.",
)
parser.add_argument(
    "--only_test", action="store_true",
    help="Whether to only perform test and skip training.",
)
parser.add_argument(
    "--entail_mode", type=str, default="text",
    help="Determines which data mode defines hierarchy for the entailment loss.",
)
parser.add_argument(
    "--active_embedding", action="store_true",
    help="Gets UNI image embeddings on the loop instead of loading pre-saved embeddings."
)
parser.add_argument(
    "--eight_sym_test", action="store_true",
    help="If True, for each sample the model will output a prediction on its eight symmetries as in STNet."
)
parser.add_argument(
    "--raw_patches", action="store_true",
    help="If True, don't use precomputed image embeddings but rather keep 2D patches as model input."
)
parser.add_argument(
    "--uni_finetuned", action="store_true",
    help="If True, use precomputed image embeddings from finetuned uni."
)
parser.add_argument(
    "--multi_k_valid", action="store_true",
    help="If True, validate with multiple top_k values."
)
# fmt: on

def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    # Create a config object and perform common setup.
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg_dict = OmegaConf.to_container(_C, resolve=True)

    args_dict = vars(_A)   # or _A.__dict__

    # Modify flexible elements in config
    cfg_dict['model']['entail_weight'] = _A.entail_weight
    _C['model']['entail_weight'] = _A.entail_weight
    del _A.entail_weight
    cfg_dict['model']['textual']['context_length'] = _A.num_genes
    _C['model']['textual']['context_length'] = _A.num_genes
    del _A.num_genes
    cfg_dict['model']['entail_mode'] = _A.entail_mode
    _C['model']['entail_mode'] = _A.entail_mode
    del _A.entail_mode
    cfg_dict['model']['neigh_weight'] = _A.neigh_weight
    _C['model']['neigh_weight'] = _A.neigh_weight
    del _A.neigh_weight

    # Merge both configs into one dict
    full_config = {
        "lazy_config": cfg_dict,
        "args": args_dict,
        "rank": RANK,
        "world_size": WORLD_SIZE,
    }

    # Start wandb configs
    if dist.is_main_process():
        wandb.init(
            project='MERU',
            entity = 'spared_v2',
            config=full_config,
            name=exp_name
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
    
    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark

    # Create output directory and save config in it.
    if not _A.only_test:
        output_dir = os.path.join(_A.output_dir, _A.dataset, exp_name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        LazyConfig.save(_C, output_dir / "config.yaml")

        # Create a logger for each process which writes to a separate log-file.
        logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")

        # Print process info, config and args.
        logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
        logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
        logger.info(OmegaConf.to_yaml(_C))

        logger.info("Command line args:")
        for arg in vars(_A):
            logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    else:
        output_dir = os.path.join(os.path.dirname(_A.ckpts_path), str(_A.top_k))
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )

    # Prepare data and dataloaders

    #dataloader = LazyFactory.build_dataloader(_C)
    collection_processed = ad.read_h5ad(os.path.join(f'/home/pcardenasg/ST/MICCAI25/SpaRED_data/original/{_A.dataset}.h5ad'))
    # FIXME: reordenando los genes al orden original de SpaRED
    collection_processed = collection_processed[:, np.argsort(collection_processed.var['original_index'].astype(int))].copy()
    collection_processed.var.index = collection_processed.var['gene_ids']
    # Load training data
    train_adata = collection_processed[collection_processed.obs["split"] == "train"]
    train_data = SpaREDData(adata=train_adata, args=_A, split_name="train")
    # Define generator
    g = torch.Generator()
    g.manual_seed(_C.train.seed + RANK)
    train_dataloader = DataLoader(train_data, 
                            batch_size=_C.train.total_batch_size // dist.get_world_size(), 
                            shuffle=_C.train.shuffle, 
                            pin_memory=True, 
                            drop_last=False, 
                            num_workers=_C.train.num_workers,
                            generator=g)
    
    # Load validation data
    valid_adata = collection_processed[collection_processed.obs["split"] == "val"]
    valid_data = SpaREDData(adata=valid_adata, args=_A, split_name="val")
    valid_dataloader = DataLoader(valid_data, 
                            batch_size=_C.train.total_batch_size // dist.get_world_size(), 
                            shuffle=False, 
                            pin_memory=True, 
                            drop_last=False, 
                            num_workers=_C.train.num_workers)

    #tokenizer = Tokenizer()

    model = LazyFactory.build_model(_C, device)
    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = amp.GradScaler(enabled=_C.train.amp)
    
    # Initiate and evaluator
    evaluator = instantiate(_C.train.evaluator)

    checkpoint_manager = CheckpointManager(
        output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    start_iteration = checkpoint_manager.resume() if _A.resume else 0

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(train_dataloader)
    timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)

    # Create tensorboard writer, only in main process.
    if dist.is_main_process():
        tboard = SummaryWriter(log_dir=output_dir)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    if not _A.only_test:
        epoch = 0
        dataloader_iter = iter(train_dataloader)

        # barra de progreso (solo en el main process)
        progress = tqdm(
            range(start_iteration + 1, _C.train.num_iterations + 1),
            disable=not dist.is_main_process(),
            dynamic_ncols=True,
            desc="Training"
        )
        
        best_mse_top_ks = {f"k_{_A.top_k}": np.inf}
        if _A.multi_k_valid:
            best_mse_top_ks = {"k_1": np.inf, "k_20": np.inf, "k_50": np.inf, "k_80": np.inf}

        for k in best_mse_top_ks:
            os.makedirs(os.path.join(output_dir, k))

        for iteration in progress:
            # --- fetch batch con reinicio automático del iterador ---
            t0 = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                # avanza la época del sampler distribuido (si aplica)
                if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                    train_dataloader.sampler.set_epoch(epoch)
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            data_time = time.perf_counter() - t0

            timer.tic()
            optimizer.zero_grad()
            with amp.autocast(enabled=_C.train.amp):
                # Get image and text (tokens) from batch and pass through model.

                #tokens = list(batch["exp_matrix"])
                if _A.encode_genes:
                    tokens = list(batch["encoded_genes"])
                else:
                    tokens = list(batch["exp_matrix"])

                patches_neighs = batch.get("patches_neighs", None)
                if patches_neighs is not None:
                    patches_neighs = patches_neighs.to(device)

                output_dict = model(batch["patches"].to(device), tokens, patches_neighs)
                loss = output_dict["loss"]
                # Get entailment and contrastive loss separetly
                entailment_loss = output_dict["logging"]["entailment_loss"]
                contrastive_loss = output_dict["logging"]["contrastive_loss"]
                neighbor_loss = output_dict["logging"]["neighbor_loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()
                timer.toc()

                # Log statistics to terminal and tensorboard.
                if iteration % _A.log_period == 0:
                    timer_stats = (
                        f"Iter {timer.iteration} | Time (sec): {data_time:.3f} data, "
                        f"{timer.deltas[-1]:.3f} model | ETA: {timer.eta_hhmm}"
                    )

                    log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
                    for key, value in output_dict["logging"].items():
                        log_str += f" [{key} {value:.3f}]"

                    logger.info(log_str)

                    if dist.is_main_process():
                        if scheduler is not None:
                            tboard.add_scalar("lr", scheduler.get_last_lr()[0], iteration)
                        else:
                            tboard.add_scalar("lr", _C.optim.optimizer.lr, iteration)
                        tboard.add_scalar("amp_scale", scaler.get_scale(), iteration)
                        for name, _loss in output_dict["logging"].items():
                            tboard.add_scalar(f"train/{name}", _loss, iteration)

                # ---- actualizar tqdm ----
                if dist.is_main_process():
                    # muestra algunos stats útiles en la barra
                    if scheduler is not None:
                        lr = float(scheduler.get_last_lr()[0])
                    else:
                        lr = _C.optim.optimizer.lr        
                    logs = {"loss": float(loss.item()),
                            "entailment_loss": float(entailment_loss.item()),
                            "contrastive_loss": float(contrastive_loss.item()),
                            "neighbor_loss": float(neighbor_loss.item()),
                            "lr": lr}
                    progress.set_postfix(logs)
                    wandb.log(logs, step=iteration)

            # Validate and log metrics
            if iteration % _A.validation_period == 0 and dist.is_main_process():
                for top_k, best_mse in best_mse_top_ks.items():
                    k = int(top_k.split("_")[1])
                    new_best = val_and_save(
                        train_dataloader,
                        valid_dataloader,
                        model,
                        evaluator,
                        k,
                        iteration,
                        checkpoint_manager,
                        best_mse
                    )
                    best_mse_top_ks[top_k] = new_best

        for top_k, best_mse in best_mse_top_ks.items():
            wandb.log({f"Best_Val_MSE_K{top_k}": best_mse})
        
    ckpts = _A.ckpts_path if _A.only_test else os.path.join(output_dir, f"k_{_A.top_k}", "checkpoint_best.pth") 
    # Evaluate model
    CheckpointManager(model=model).load(ckpts)    
    # Test dataloader
    test_split = "test" if "test" in collection_processed.obs["split"].unique() else "val"
    test_adata = collection_processed[collection_processed.obs["split"] == test_split]
    print(f"Test split is of shape {test_adata.shape}")

    test_data = SpaREDData(adata=test_adata, args=_A, split_name=test_split)
    test_dataloader = DataLoader(test_data, 
                        batch_size=_C.train.total_batch_size // dist.get_world_size(), 
                        shuffle=False, 
                        pin_memory=True, 
                        drop_last=False, 
                        num_workers=_C.train.num_workers)
    
    test_metrics, (test_preds, test_labels) = evaluator(train_dataloader, test_dataloader, _A, model, top_k=_A.top_k)
    test_logs = {f"Test_MSE_K{ _A.top_k}": test_metrics["MSE"],
                 f"Test_PCC_K{_A.top_k}": test_metrics["PCC-Gene"]}
    
    print("test MSE: ", test_metrics['MSE'])
    print("test PCC-Gene: ", test_metrics['PCC-Gene'])    

    wandb.log(test_logs)
    preds_dir = os.path.join(output_dir, "preds", "eight_sym_test") if _A.eight_sym_test else os.path.join(output_dir, "preds")

    os.makedirs(preds_dir, exist_ok=True)
    torch.save(
        {
            _A.prediction_layer: test_labels.detach().cpu().numpy(),
            "prediction": test_preds.detach().cpu().numpy(),
        },
        os.path.join(preds_dir, "test.pt")
    )

def val_and_save(train_dataloader, valid_dataloader, model, evaluator, top_k, iteration, checkpoint_manager, best_mse):

    metrics, results = evaluator(train_dataloader, train_dataloader, _A, model, top_k=top_k)
    logs = {f"Train_MSE_K{top_k}": metrics["MSE"]}
    wandb.log(logs, step=iteration)

    metrics, results = evaluator(train_dataloader, valid_dataloader, _A, model, top_k=top_k)
    logs = {f"Val_MSE_K{top_k}": metrics["MSE"]}
    wandb.log(logs, step=iteration)

    if metrics["MSE"] < best_mse:
        best_mse = metrics["MSE"]
        checkpoint_manager.step(iteration, f"k_{top_k}")
        print(f"Saving best model with MSE {best_mse}")

    return best_mse

if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A,),
        )
        # "tcp://127.0.0.1:5648" (default)
        # "tcp://127.0.0.1:5649" 
        # "tcp://127.0.0.1:5650"
        # "tcp://127.0.0.1:5651"
        # "tcp://127.0.0.1:5652"
        # "tcp://127.0.0.1:5653"
        # "tcp://127.0.0.1:5654"
        # "tcp://127.0.0.1:5655"
        # "tcp://127.0.0.1:5656"
        # "tcp://127.0.0.1:5657"

#CUDA_VISIBLE_DEVICES=5 python scripts/train_spared.py --config configs/train_meru_spared.py --num-gpus 1 --output-dir ./outputs_0.5 --dist-url tcp://127.0.0.1:5648 --dataset
#CUDA_VISIBLE_DEVICES=7 python scripts/train_spared.py --config configs/train_meru_spared.py --num-gpus 1 --output-dir ./outputs_geneformer_uni --entail_weight 0.5 --num_genes 768 --dist-url tcp://127.0.0.1:5644 --dataset