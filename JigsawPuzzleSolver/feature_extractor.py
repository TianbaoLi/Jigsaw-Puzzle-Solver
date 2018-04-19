from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import resnet_fmap
from jigsaw_image_loader import JigsawImageLoader

data_dir = 'ILSVRC2012_img_val/train'
#image_dataset = datasets.ImageFolder(data_dir, data_transform)
image_dataset = JigsawImageLoader(data_dir, slice=3)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

dataset_size = len(image_dataset)
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in ['train', 'val']}
#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_dataset.classes



def back_transform(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)

    return inp


def imshow(inp, title=None):
    plt.imshow(back_transform(inp))
    plt.axis('off')
    if title is not None:
        plt.title(title)

def plot_kernels(tensor, num_cols=8):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def plot_jigsaw(tensor):
    num_tiles = tensor.shape[0]
    num_cols = int(np.sqrt(num_tiles))
    num_rows = num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(back_transform(tensor[i]))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def gen_feature_map(model):
    model.train(False)
    for data in dataloader:
        # get the inputs
        origins, jigsaws, orders, tiles = data
        batch_size = jigsaws.shape[0]
        for i in range(batch_size):
            origin = origins[i]
            jigsaw = jigsaws[i]
            order_index = orders[i]
            tile = tiles[i]
            imshow(origin)
            plot_jigsaw(jigsaw)
            #plot_jigsaw(tile)
            slice_length = tile[0].shape[1]

            # store the edge feature map of all tiles
            # dim1: index of tiles
            # dim2: direction: up, down, left, right
            # dim3: tile length * 64 (ReLU1 output maps) * 2 (use 2 rows near the edge)
            fmp = torch.zeros(9, 4, slice_length * 64 * 2)

            for i in range(jigsaw.shape[0]):
                tile = jigsaw[i]
                index = order_index[i]
                input = tile[None, :, :]
                out = torchvision.utils.make_grid(input)
                imshow(out)

                # wrap them in Variable
                input = Variable(input.cuda())
                # get ReLU outputs as the feature maps
                (layer1_out, output) = model(input)
                # only use the first block as the low level info
                first_map = layer1_out.data.cpu().numpy()[0]
                plot_kernels(first_map, int(np.sqrt(first_map.shape[0])))


model_resnet34 = resnet_fmap.resnet34(pretrained=True)
model_resnet34 = model_resnet34.cuda()
gen_feature_map(model_resnet34)
