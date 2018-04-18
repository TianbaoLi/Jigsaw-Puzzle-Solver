import numpy as np
import torchvision.transforms as transforms
import torch
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS

import matplotlib.pyplot as plt
from PIL import Image


class JigsawImageLoader(ImageFolder):
    def __init__(self, root):
        self.data_path = root
        self.permutations = self.__retrive_permutations(1000)
        self.__image_transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
        self.__augment_tile = transforms.Compose([
                    #transforms.RandomCrop(64),
                    transforms.Resize((224,224),Image.BILINEAR),
                    #transforms.Lambda(rgb_jittering),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         #std =[0.229, 0.224, 0.225])
                                         ])
        super(ImageFolder, self).__init__(root, default_loader, IMG_EXTENSIONS)

    def __getitem__(self, index):
        img, target = ImageFolder.__getitem__(self, index)

        img = self.__image_transformer(img)
        origin = img
        img = transforms.ToPILImage()(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            #m, s = tile.view(3,-1).mean(dim=1).numpy(), tile.view(3,-1).std(dim=1).numpy()
            #s[s==0]=1
            #norm = transforms.Normalize(mean=m.tolist(),std=s.tolist())
            #tile = norm(tile)
            tiles[n] = tile
        
        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data,0)
        tiles = torch.stack(tiles)


        return origin, data, self.permutations[order], tiles


    def __retrive_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:,:,ch] += np.random.randint(-2,2)
    im[im>255] = 255
    im[im<0] = 0
    return im.astype('uint8')
