import torch
import numpy as np
import torchvision
from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Normalize

class SpaREDData(torch.utils.data.Dataset):
    def __init__(self, adata, prediction_layer, split_name = "train", pre_masked = False):
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            adata (ad.AnnData): An anndata object with the data of the entire dataset.
            args (argparse): parser with the values necessary for data processing.
            split_name (str): name of the data split being processed. Useful for identifying which data split the model is being tested on.
            pre_masked (str, optional): specifies if the data incoming has already been masked for testing purposes. 
                    * If True, __getitem__() will return the random mask that was used to mask the original expression 
                    values instead of the median imputation mask, as well as the gt expressions and the masked data.
        """

        self.adata = adata
        self.pred_layer = prediction_layer
        self.split_name = split_name
        self.pre_masked = pre_masked
        # Get image patches
        tissue_tiles = adata.obsm["patches_scale_1.0"]
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        tissue_tiles = torch.from_numpy(tissue_tiles)
        tissue_tiles = tissue_tiles.permute(0,3,1,2)
        self.embeddings = tissue_tiles/255

        # Get unique identifier
        self.IDs = adata.obsm["spatial"]
        
        # Define image transformations   
        if split_name == "train":     
            self.transforms = Compose(
                [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomApply([RandomRotation((90, 90))], p=0.5)]
                )
        else: 
            self.transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), EightSymmetry()])
        
    def build_data(self, idx):
        # Get image embeddings if needed
        embedding = self.embeddings[idx]
        if self.split_name == "train":
            aug_patches = []
            for _ in range(8):
                aug_img = self.transforms(embedding)
                aug_patches.append(aug_img)
            embedding = torch.stack(aug_patches)
        else:
            embedding = self.transforms(embedding)

        id = self.IDs[idx]
        data = (embedding, id)
        return data
        
    def __getitem__(self, idx):
        
        item = {}
        # Get expression and median imputation mask of neighborhoods
        item["patches"], item["labels"] = self.build_data(idx)

        return item

    def __len__(self):
        return len(self.adata)
    
class EightSymmetry(object):
    """Returns a tuple of the eight symmetries resulting from rotation and reflection.
    
    This behaves similarly to TenCrop.
    This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.
    Example:
     transform = Compose([
         EightSymmetry(), # this is a tuple of PIL Images
         Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
     ])
    """
    # This class function was taken fron the original ST-Net repository at:
    # https://github.com/bryanhe/ST-Net/blob/43022c1cb7de1540d5a74ea2338a12c82491c5ad/stnet/transforms/eight_symmetry.py#L3
    def __call__(self, img):
        identity = lambda x: x
        ans = []
        for i in [identity, torchvision.transforms.RandomHorizontalFlip(1)]:
            for j in [identity, torchvision.transforms.RandomVerticalFlip(1)]:
                for k in [identity, torchvision.transforms.RandomRotation((90, 90))]:
                    ans.append(i(j(k(img))))
        return tuple(ans)

    def __repr__(self):
        return self.__class__.__name__ + "()"