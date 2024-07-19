import numpy as np
import torch
import torch.nn as nn
from fastai.vision.all import *
import gc
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
from easyfsl.modules import resnet12
from easyfsl.utils import plot_images, sliding_average
from statistics import mean
import copy
from easyfsl.utils import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fsl_dataset(Dataset):
    ''' Make dataset with im_pths as list and targets as tensor'''
    def __init__(self, im_pths, targets, num_augmentations, transform = None, target_transform = None):
        self.im_pths = im_pths
        self.targets = targets
        self.num_augmentations = num_augmentations
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.im_pths)*self.num_augmentations
        
    def __getitem__(self, idx):
        actual_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations
        
        im_pth = str(self.im_pths[actual_idx])
        image = Image.fromarray(cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB))
        label = self.targets[actual_idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    # Defining additional method called `get_labels` for few-shot-learning with easyfsl
    def get_labels(self):
        return self.targets.tolist()
    
    
    
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


