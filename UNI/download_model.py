import os
import torch
from torchvision import transforms
import timm
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
import anndata as ad
from torch.utils.data import DataLoader
from spared.configs import SpaREDData
import numpy as np
import argparse
from uni.downstream.utils import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--prediction_layer", default="c_StepB_log1p",
    help="Prediction layer.")
parser.add_argument(
    "--batch_size", default=128,
    help="Batch size in dataloader.")
parser.add_argument(
    "--shuffle", default=True,
    help="Wheather to shuffle in dataloader")
parser.add_argument(
    "--num_workers", default=0,
    help="Number of workers in dataloader.")
parser.add_argument(
    "--dataset", default="villacampa_lung_organoid",
    help="Dataset to use")


def main(args):
    print("-----------------------------------------------------------")
    print(f"   Running UNI model for {args.dataset}")
    print("-----------------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_dir = "models/ckpts/uni2-h/"

    timm_kwargs = {'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24, #24
                'init_values': 1e-5, 
                'embed_dim': 1536, #1536 
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
                }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    #model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.eval()
    model.to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    transform
    model, transform = get_encoder(enc_name='uni2-h', device=device, assets_dir="/home/dvegaa/MERU_SpaRED/UNI/models/ckpts/encoder/")

    # get path to example data
    collection_processed = ad.read_h5ad(os.path.join(f'/home/pcardenasg/ST/MICCAI25/SpaRED_data/original/{args.dataset}.h5ad'))
    # FIXME: reordenando los genes al orden original de SpaRED
    collection_processed = collection_processed[:, np.argsort(collection_processed.var['original_index'].astype(int))].copy()
    collection_processed.var.index = collection_processed.var['gene_ids']
    # Load training data

    splits = collection_processed.obs["split"].unique().tolist()
    for split in splits:
        print(f"Processing {split} split")
        adata = collection_processed[collection_processed.obs["split"] == split]
        data = SpaREDData(adata=adata, prediction_layer=args.prediction_layer, split_name=split)
        dataloader = DataLoader(data, 
                                batch_size=args.batch_size, 
                                shuffle=args.shuffle, 
                                pin_memory=True, 
                                drop_last=False, 
                                num_workers=args.num_workers)


        # extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
        features = extract_patch_features_from_dataloader(model, dataloader, split=split)

        # convert these to torch
        feats = torch.Tensor(features['embeddings'])
        labels = torch.Tensor(features['labels'])

        dir = f"aug_features/{args.dataset}/{split}"
        save_features(feats, labels, dir=dir)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)