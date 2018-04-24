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
import argparse
import shutil
import resnet_fmap
from jigsaw_image_loader import JigsawImageLoader


data_dir = 'ILSVRC2012_img_val/train'

parser = argparse.ArgumentParser(description='Extract ReLU1 features from pre-trained RestNet-34')
parser.add_argument('--slice', default=3, type=int, help='slice per edge')
parser.add_argument('--amount', default=1000, type=int, help='amount tile pairs for each class (neighbor/ non-neighbor')
parser.add_argument('--batch', default=16, type=int, help='batch size')


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


def gen_feature_map(data, model, output_dir, output_index, slice_per_edge=3): # data_amount for both true and false
    data_amount = (slice_per_edge * (slice_per_edge - 1) * 2)
    neighbor = torch.zeros(data_amount, 2 * 64 * 56 * 2 + 224 * 2 * 3 * 2 + 1)
    neighbor_count = 0
    non_neighbor = torch.zeros(data_amount, 2 * 64 * 56 * 2 + 224 * 2 * 3 * 2 + 1)
    non_neighbor_count = 0

    # get the inputs
    origins, jigsaws, orders, tiles = data
    batch_size = jigsaws.shape[0]
    for b in range(batch_size):
        origin = origins[b]
        jigsaw = jigsaws[b]
        order_index = orders[b]
        tile = tiles[b]
        #imshow(origin)
        #plot_jigsaw(jigsaw)

        # store the edge feature map of all tiles
        # dim1: index of tiles
        # dim2: direction: up, down, left, right
        # dim3:
        #   feature map: 64 (ReLU1 output maps) * 56 (ReLU output edge length) * 2 (use 2 rows near the edge)
        #   color: 224 (tile edge length) * 2 (use 2 rows near the edge) * 3 (RGB)
        feature_map = torch.zeros(slice_per_edge ** 2, 4, 64 * 56 * 2 + 224 * 2 * 3)

        for i in range(jigsaw.shape[0]):
            tile = jigsaw[i]
            index = order_index[i]
            input = tile[None, :, :]
            out = torchvision.utils.make_grid(input)
            #imshow(out)

            # wrap them in Variable
            input = Variable(input.cuda())
            # get ReLU outputs as the feature maps
            (layer1_out, output) = model(input)
            # only use the first block as the low level info
            first_map = layer1_out.data[0]
            # generate the edge feature map of each tile of up(0), down(1), left(2), right(3)
            # fill each filter
            for j in range(64):
                feature_map[int(index)][0][j * 56 * 2: j * 56 * 2 + 56 * 2] = torch.cat((first_map[j][0], first_map[j][1]), 0).view(1, -1)
                feature_map[int(index)][1][j * 56 * 2: j * 56 * 2 + 56 * 2] = torch.cat((first_map[j][-2], first_map[j][-1]), 0).view(1, -1)
                feature_map[int(index)][2][j * 56 * 2: j * 56 * 2 + 56 * 2] = torch.cat((first_map[j][:, 0], first_map[j][:, 1]), 0).view(1, -1)
                feature_map[int(index)][3][j * 56 * 2: j * 56 * 2 + 56 * 2] = torch.cat((first_map[j][:, -2], first_map[j][:, -1]), 0).view(1, -1)
            # fill RGB info
            feature_map[int(index)][0][64 * 56 * 2: 64 * 56 * 2 + 224 * 2 * 3] = torch.cat((tile[:, 0], tile[:, 1]), 0).view(1, -1)
            feature_map[int(index)][1][64 * 56 * 2: 64 * 56 * 2 + 224 * 2 * 3] = torch.cat((tile[:, -2], tile[:, -1]), 0).view(1, -1)
            feature_map[int(index)][2][64 * 56 * 2: 64 * 56 * 2 + 224 * 2 * 3] = torch.cat((tile[:, :, 0], tile[:, :, 1]), 0).view(1, -1)
            feature_map[int(index)][3][64 * 56 * 2: 64 * 56 * 2 + 224 * 2 * 3] = torch.cat((tile[:, :, -2], tile[:, :, -1]), 0).view(1, -1)

            first_map = first_map.cpu().numpy()
            #plot_kernels(first_map, int(np.sqrt(first_map.shape[0])))

        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[0]):
                if i >= j:
                    continue
                if neighbor_count >= data_amount and non_neighbor_count >= data_amount:
                    break
                if j - i == 1 and neighbor_count < data_amount:
                    neighbor[neighbor_count][0: 64 * 56 * 2 + 224 * 2 * 3] = feature_map[i][3]
                    neighbor[neighbor_count][64 * 56 * 2 + 224 * 2 * 3: 64 * 56 * 2 + 224 * 2 * 3 + 64 * 56 * 2 + 224 * 2 * 3] = feature_map[j][2]
                    neighbor[neighbor_count][-1] = 1
                    neighbor_count += 1
                elif j - i == slice_per_edge and neighbor_count < data_amount:
                    neighbor[neighbor_count][0: 64 * 56 * 2 + 224 * 2 * 3] = feature_map[i][1]
                    neighbor[neighbor_count][64 * 56 * 2 + 224 * 2 * 3: 64 * 56 * 2 + 224 * 2 * 3 + 64 * 56 * 2 + 224 * 2 * 3] = feature_map[j][0]
                    neighbor[neighbor_count][-1] = 1
                    neighbor_count += 1
                elif non_neighbor_count < data_amount:
                    non_neighbor[non_neighbor_count][0: 64 * 56 * 2 + 224 * 2 * 3] = feature_map[i][3]
                    non_neighbor[non_neighbor_count][64 * 56 * 2 + 224 * 2 * 3: 64 * 56 * 2 + 224 * 2 * 3 + 64 * 56 * 2 + 224 * 2 * 3] = feature_map[j][2]
                    non_neighbor[non_neighbor_count][-1] = 0
                    non_neighbor_count += 1

                    if non_neighbor_count >= data_amount:
                        break

                    non_neighbor[non_neighbor_count][0: 64 * 56 * 2 + 224 * 2 * 3] = feature_map[i][1]
                    non_neighbor[non_neighbor_count][64 * 56 * 2 + 224 * 2 * 3: 64 * 56 * 2 + 224 * 2 * 3 + 64 * 56 * 2 + 224 * 2 * 3] = feature_map[j][0]
                    non_neighbor[non_neighbor_count][-1] = 0
                    non_neighbor_count += 1

    feature_data = torch.cat((neighbor, non_neighbor), 0)
    torch.save(feature_data, output_dir + '/' + str(output_index))


def main():
    args = parser.parse_args()
    SLICE_PER_EDGE = args.slice
    DATA_AMOUNT = args.amount
    BATCH_SIZE = args.batch
    WORKER_THREAD = 4

    image_dataset = JigsawImageLoader(data_dir, slice=SLICE_PER_EDGE)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_THREAD)

    model_resnet34 = resnet_fmap.resnet34(pretrained=True)
    model_resnet34 = model_resnet34.cuda()
    model_resnet34.train(False)

    try:
        shutil.rmtree('EdgeFeatures_' + str(SLICE_PER_EDGE) + '_' + str(DATA_AMOUNT))
    except:
        pass

    output_dir = 'EdgeFeatures_' + str(SLICE_PER_EDGE) + '_' + str(DATA_AMOUNT)
    os.mkdir(output_dir)
    output_index = 0
    output_up_range = DATA_AMOUNT // (SLICE_PER_EDGE * (SLICE_PER_EDGE - 1) * 2 * 2) + 1

    for data in dataloader:
        if output_index >= output_up_range:
            break
        gen_feature_map(data, model_resnet34, output_dir, output_index, slice_per_edge=SLICE_PER_EDGE)
        output_index += 1

    #for filename in os.listdir(output_dir):
    #    feature_data = torch.load(output_dir + '/' + filename)
    #    print(filename, feature_data.shape)


if __name__ == '__main__':
    main()