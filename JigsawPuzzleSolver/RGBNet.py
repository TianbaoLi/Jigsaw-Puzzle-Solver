import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import time
import copy
import argparse
from FeatureDataset import FeatureDataset, FeatureDatasetGenerator


parser = argparse.ArgumentParser(description='Train Jigsaw Puzzle Solver')
parser.add_argument('data', type=str, help='Path to feature file')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs for training')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for SGD optimizer')


class RGBNet(nn.Module):

    def __init__(self):
        super(RGBNet, self).__init__()
        # fcs
        self.fc1 = nn.Linear(2 * 224 * 2 * 3, 336)
        self.fc2 = nn.Linear(336, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # Set model to training mode
            else:
                model.train(False) # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            item_amount = 0

            # Iterate over data.
            for tile in dataloaders[phase]:
                tile_inputs, tile_labels = tile
                item_amount = tile_inputs.shape[1]
                for i in range(item_amount):
                    # get the inputs
                    inputs = tile_inputs[:, i, :]
                    labels = tile_labels[:, i]

                    slice_1 = inputs[:, 0: 64 * 56 * 2 + 224 * 2 * 3]
                    slice_2 = inputs[:, 64 * 56 * 2 + 224 * 2 * 3: 64 * 56 * 2 + 224 * 2 * 3 + 64 * 56 * 2 + 224 * 2 * 3]

                    slice_1 = slice_1[:, - 224 * 2 * 3:]
                    slice_2 = slice_2[:, - 224 * 2 * 3:]
                    input = feature_data = torch.cat((slice_1, slice_2), 0)

                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.long().cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (dataset_sizes[phase] * item_amount)
            epoch_acc = int(running_corrects) / (dataset_sizes[phase] * item_amount)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def LoadData(path, ratio=0.2):
    feature_data = torch.load(path)
    N = int(feature_data.shape[0])
    feature_data = feature_data[torch.randperm(N)]
    train = feature_data[: int(N * (1 - ratio))]
    val = feature_data[int(N * (1 - ratio)):]
    x_train = train[:, : -1]
    y_train = train[:, -1].long()
    #y_train = torch.zeros(x_train.shape[0], 2, dtype=torch.int64)
    #for i in range(y.shape[0]):
    #    y_train[i][y[i]] = 1

    x_val = val[:, : -1]
    y_val = val[:, -1].long()
    #y_val = torch.zeros(x_val.shape[0], 2, dtype=torch.int64)
    #for i in range(y.shape[0]):
    #    y_val[i][y[i]] = 1

    return x_train, y_train, x_val, y_val



def main():
    args = parser.parse_args()
    DATA = args.data

    #(x_train, y_train, x_val, y_val) = LoadData('EdgeFeatures_3')
    #datasets = {'train': TensorDataset(x_train, y_train),
    #            'val': TensorDataset(x_val, y_val)}

    generator = FeatureDatasetGenerator(DATA)
    train_dataset, val_dataset = generator.generate()
    datasets = {'train': train_dataset, 'val': val_dataset}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    net = RGBNet()
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.3)

    model_conv = train_model(net, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, args.epochs)


if __name__ == '__main__':
    main()
