import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader, split):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch in tqdm(dataloader):
        if isinstance(batch['patches'], list):
            batch['patches'] = torch.stack(batch['patches']).permute(1,0,2,3,4)
        batch = {k: v.to(device) for k, v in batch.items()}
        for i in range(len(batch['patches'])):
            aug_imgs = batch['patches'][i]
            with torch.inference_mode():
                embeddings = model(aug_imgs).detach().cpu()
                assert not torch.isnan(embeddings).any()

            all_labels.append(batch["labels"][i].detach().cpu())
            all_embeddings.append(embeddings)

    asset_dict = {
        "labels": np.vstack(all_labels).astype(np.float32),
        "embeddings": np.stack(all_embeddings).astype(np.float32)
    }

    return asset_dict
