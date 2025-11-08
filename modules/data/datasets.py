import time
from typing import Tuple
from glob import glob
import os
import cv2 as cv
import numpy as np
import torch
from modules.data.utils import PatchExtractor, PatchExtractorTensor, resizeTensor
from modules.models import utils
from modules.models.raft.utils import coords_grid
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class BasicTransform:

    def transform(self, inputs, args):
        return inputs

    def __call__(self, inputs: dict, **kwargs):
        return self.transform(inputs, kwargs)

class TensorTransform(BasicTransform):

    def __init__(self):
        self.toTensor = ToTensor()

    def transform(self, inputs, args):
        res = {}
        for k in inputs:
            res[k] = self.toTensor(inputs[k])
        return res

class AddGridTransform(BasicTransform):

    def transform(self, inputs, args):
        _, h, w = inputs['image1'].shape
        gridX = torch.linspace(0, h - 1, steps = h).view(-1, 1, 1).expand(h, w, 1)
        gridY = torch.linspace(0, w - 1, steps = w).view(1, -1, 1).expand(h, w, 1)
        grid = torch.cat((gridY, gridX), dim = -1)
        inputs['grid'] = grid
        return inputs

class AugHomoTransform(BasicTransform):

    def transform(self, inputs, args):
        if 'homo' in args:
            homo = args['homo']
            offset = np.random.randint(0, 10, size = (4, 2))
            homo += offset
        return inputs

class AugmentTransform(BasicTransform):

    def transform(self, inputs, args):
        image1 = inputs['image1']
        image2 = inputs['image2']

        # Randomly shift brightness
        random_brightness = torch.randn(1).uniform_(0.7, 1.3)
        img1_aug = image1 * random_brightness
        random_brightness = torch.randn(1).uniform_(0.7, 1.3)
        img2_aug = image2 * random_brightness

        # Randomly shift color
        white = torch.ones([image1.size()[1], image1.size()[2]])
        random_colors = torch.randn(3).uniform_(0.7, 1.3)
        color_image = torch.stack([white * random_colors[i] for i in range(3)], dim = 0)
        img1_aug *= color_image

        random_colors = torch.randn(3).uniform_(0.7, 1.3)
        color_image = torch.stack([white * random_colors[i] for i in range(3)], dim = 0)
        img2_aug *= color_image

        # clip
        img1_aug = torch.clamp(img1_aug, 0, 1)
        img2_aug = torch.clamp(img2_aug, 0, 1)

        inputs['image1'] = img1_aug
        inputs['image2'] = img2_aug

        return inputs

class ResizeTransform(BasicTransform):

    def __init__(self, minsize):
        self.minsize = minsize

    def transform(self, inputs, args):
        # _, h, w = inputs['image1'].shape
        _, _, _, _, h, w, h2, w2 = inputs['homoParam']
        if self.minsize >= max(h, w):
            return inputs
        inputs['origin1'] = inputs['image1']
        inputs['origin2'] = inputs['image2']
        if 'mask1' in inputs:
            inputs['orimask1'] = inputs['mask1']
            inputs['orimask2'] = inputs['mask2']
        factor = (min(self.minsize, h) / h, min(self.minsize, w) / w)
        # dsize = (min(self.minsize, h), min(self.minsize, w))
        inputs['image1'] = F.interpolate(inputs['image1'][None], scale_factor = factor, mode = 'bilinear')[0]
        inputs['image2'] = F.interpolate(inputs['image2'][None], scale_factor = factor, mode = 'bilinear')[0]
        if 'mask1' in inputs:
            inputs['mask1'] = F.interpolate(inputs['mask1'][None], scale_factor = factor, mode = 'bilinear')[0]
            inputs['mask2'] = F.interpolate(inputs['mask2'][None], scale_factor = factor, mode = 'bilinear')[0]
        # _, hh, ww = inputs['image1'].shape
        if 'homo' in args:
            homo = args['homo']
            homo[..., 0] = homo[..., 0] * factor[1]
            homo[..., 1] = homo[..., 1] * factor[0]
        if 'c1' in inputs:
            c1 = inputs['c1']
            c1[0] = c1[0] * factor[1]
            c1[1] = c1[1] * factor[0]
            c2 = inputs['c2']
            c2[0] = c2[0] * factor[1]
            c2[1] = c2[1] * factor[0]
        # if 'grid' in inputs:
        #     inputs['grid'] = F.interpolate(inputs['grid'].permute(2, 0, 1)[None],
        #                                    scale_factor = factor, mode = 'bilinear')[0].permute(1, 2, 0)
        return inputs

class PatchTransform(BasicTransform):

    def __init__(self, patchSize: int, padding: int):
        super().__init__()
        self.extractor = PatchExtractorTensor(patchSize - 2 * padding)
        self.pad = nn.ZeroPad2d(padding)
        self.validKeys = {'image1', 'image2', 'mask1', 'mask2', 'weight', 'grid'}

    def transform(self, inputs, args):
        keys = inputs.keys()
        images = [inputs[k] for k in keys if k in self.validKeys]
        images = self.extractor(*images)
        images = iter(images)
        res = {}
        for k in keys:
            if k in self.validKeys:
                res[k] = next(images)
            else:
                res[k] = inputs[k]
        return res

class PatchGridTransform(BasicTransform):

    def __init__(self, patchSize: int, padding: int, factors = None):
        self.patchSize = patchSize - 2 * padding
        self.pad = nn.ZeroPad2d(padding)
        self.validKeys = {'image1', 'image2', 'mask1', 'mask2', 'weight'}
        if factors is None:
            factors = [1., 1.25, 1.5, 1.75, 2.]
        self.factors = factors

    def transform(self, inputs, args):
        crop = self.patchSize
        resize = np.random.choice([int(crop * i) for i in self.factors])
        res = {}
        idw, idh = -1, -1
        flip = np.random.rand() < 0.5
        for k in self.validKeys:
            if k not in inputs:
                continue
            image = inputs[k]
            image = resizeTensor(image, minSize = resize, points = [inputs['c1'], inputs['c2']] if idw == -1 else None)
            _, h, w = image.shape
            if idw == -1:
                idw, idh = np.random.randint(w - crop + 1), np.random.randint(h - crop + 1)
            image = TF.crop(image, idh, idw, crop, crop)
            if flip:
                image = TF.hflip(image)
            res[k] = image

        gridX = torch.linspace(0, h - 1, steps = h).view(1, -1, 1).expand(1, h, w)
        gridY = torch.linspace(0, w - 1, steps = w).view(1, 1, -1).expand(1, h, w)
        grid = torch.cat((gridY, gridX), dim = 0)
        grid = TF.crop(grid, idh, idw, crop, crop)
        if flip:
            grid = TF.hflip(grid)
        res['grid'] = grid.permute(1, 2, 0)
        res['h'] = torch.tensor(h)
        res['w'] = torch.tensor(w)

        for k in inputs:
            if k not in res:
                res[k] = inputs[k]

        return res

class HomoTransform(BasicTransform):

    def __init__(self, dual = False, asGrid = False, addHomo = False, givenH = None):
        super().__init__()
        self.asGrid = asGrid
        self.dual = dual
        self.addHomo = addHomo
        self.givenH = givenH

    def transform(self, inputs, args):
        homo = args['homo']
        image1 = inputs['image1']
        image2 = inputs['image2']
        homo = torch.tensor(homo)
        _, h, w = image1.shape
        _, h2, w2 = image1.shape

        if self.givenH is not None:
            _, h2, w2 = image2.shape
            H1, H2 = self.givenH
            H1, H2, outh, outw = utils.centralizeH(h, w, H1, H2, h2, w2)
        elif self.dual:
            # _, h2, w2 = image2.shape
            H1, H2 = utils.computeAndDecomposeH(h, w, homo)
            H1, H2, outh, outw = utils.centralizeH(h, w, H1, H2, h2, w2)
        else:
            H1, H2, outh, outw = utils.computePairedImageH(h2, w2, homo)
        outh, outw = outh.round().int().item(), outw.round().int().item()

        res = dict()
        res['homoParam'] = (H1, H2, outh, outw, h, w, h2, w2)

        # print(outh, outw)
        image1, grid1 = utils.warpHomo(image1, H1, h, w, outh, outw, returnGrid = True)
        image2, grid2 = utils.warpHomo(image2, H2, h2, w2, outh, outw, returnGrid = True)

        if self.asGrid:
            res['grid1'] = grid1
            res['grid2'] = grid2
            res['raw1'] = inputs['image1']
            res['raw2'] = inputs['image2']

        if 'mask' in args:
            mask = args['mask']
            if isinstance(mask, tuple):
                mask1 = mask[0]
                mask2 = mask[1]
            else:
                mask1 = mask2 = mask
            mask1 = utils.warpHomo(mask1, H1, h, w, outh, outw)
            mask2 = utils.warpHomo(mask2, H2, h2, w2, outh, outw)
            res['mask1'] = mask1
            res['mask2'] = mask2

        res['image1'] = image1
        res['image2'] = image2
        if self.addHomo:
            res['homo'] = homo[0]
        for key in inputs:
            if key not in res:
                res[key] = inputs[key]
        return res

class WeightTransform(BasicTransform):

    def __init__(self, center = True, weight = False):
        self.center = center
        self.weight = weight

    def transform(self, inputs, args):
        H1, H2, outh, outw, h, w, h2, w2 = inputs['homoParam']

        c1 = torch.tensor([[[w / 2, h / 2]]])
        c2 = torch.tensor([[[w2 / 2, h2 / 2]]])
        c1 = utils.warpPointsHomo(c1, H1)
        c2 = utils.warpPointsHomo(c2, H2)

        if self.center:
            inputs['c1'] = c1[0, 0]
            inputs['c2'] = c2[0, 0]

        if self.weight:
            weight = utils.getDualFlowWeight(outh, outw, c1[0], c2[0]) # (1, h, w)
            inputs['weight'] = weight

        return inputs

class DropTransform(BasicTransform):

    def __init__(self, keys):
        self.keys = keys

    def transform(self, inputs, args):
        for k in self.keys:
            inputs.pop(k)
        return inputs


class ImagePairDataset(Dataset):

    def __init__(self, dataRoot: str, dualDir: Tuple[str, str] = ('input1', 'input2'), transforms = None):
        """
        Read image pairs from given data root.
        :param dataRoot: data root
        :param dualDir: 2 sub dirs for image pairs under the data root
        :param transforms: preprocess
        """
        super().__init__()
        self.path1 = sorted(glob(os.path.join(dataRoot, dualDir[0], '*')))
        self.path2 = sorted(glob(os.path.join(dataRoot, dualDir[1], '*')))
        assert(len(self.path1) == len(self.path2))
        if transforms is None:
            transforms = [TensorTransform()]
        self.transforms = transforms

    def __getitem__(self, index):
        image1 = cv.imread(self.path1[index]).copy()
        image2 = cv.imread(self.path2[index]).copy()
        res = dict(
            image1 = image1,
            image2 = image2
        )
        for trans in self.transforms:
            res = trans(res)
        return res

    def __len__(self):
        return len(self.path1)

class ImagePairPatchDataset(ImagePairDataset):

    def __init__(self, dataRoot: str, patchSize: int, padding: int = 0,
                 dualDir: Tuple[str, str] = ('input1', 'input2'), transforms = None):
        """
        Read image pairs from given data root, and extract a corresponding pair of patches.
        :param dataRoot: data root
        :param dualDir: 2 sub dirs for image pairs under the data root
        :param transforms: preprocess
        """
        if transforms is None:
            transforms = [TensorTransform(), PatchTransform(patchSize, padding)]
        super().__init__(dataRoot, dualDir, transforms)

class ImagePairWithHomoPatchDataset(ImagePairPatchDataset):

    def __init__(self, dataRoot: str, patchSize: int, padding: int = 0,
                 dualDir: Tuple[str, str] = ('input1', 'input2'), homoDir: str = 'homo', transforms = None,
                 mask = False, dual = False):
        self.path3 = sorted(glob(os.path.join(dataRoot, homoDir, '*')))
        self.path4 = sorted(glob(os.path.join(dataRoot, 'mask1', '*'))) if os.path.exists(
            os.path.join(dataRoot, 'mask1')) else None
        self.path5 = sorted(glob(os.path.join(dataRoot, 'mask2', '*'))) if os.path.exists(
            os.path.join(dataRoot, 'mask2')) else None
        if transforms is None:
            transforms = [TensorTransform(), HomoTransform(dual = dual), PatchTransform(patchSize, padding)]
        super().__init__(dataRoot, patchSize, padding, dualDir, transforms)
        assert len(self.path3) == len(self.path1)
        self.mask = mask

    def __getitem__(self, index):
        image1 = cv.imread(self.path1[index])
        image2 = cv.imread(self.path2[index])
        homo = np.load(self.path3[index]).reshape((-1, 4, 2)).astype('float32')
        res = dict(
            image1 = image1,
            image2 = image2
        )
        if self.mask:
            if self.path4 is not None:
                mask1 = cv.imread(self.path4[index])
                mask2 = cv.imread(self.path5[index])
                mask1 = ToTensor()(mask1)
                mask2 = ToTensor()(mask2)
                mask = (mask1, mask2)
            else:
                h, w, _ = image1.shape
                mask = torch.ones((1, h, w))

            for trans in self.transforms:
                res = trans(res, homo = homo, mask = mask)
        else:
            for trans in self.transforms:
                res = trans(res, homo = homo)
        return res


class ImagePairWithHomoAdaptivePatchDataset(ImagePairWithHomoPatchDataset):

    def __init__(self, dataRoot: str, patchSize: int, padding: int = 0,
                 dualDir: Tuple[str, str] = ('input1', 'input2'), homoDir: str = 'homo', transforms = None,
                 mask = False, dual = False, factors = None, aug = True):
        if transforms is None:
            transforms = [TensorTransform(), HomoTransform(dual = dual),
                          WeightTransform(center = True), PatchGridTransform(patchSize, padding, factors),
                          DropTransform(['homoParam'])]
            if aug:
                transforms.insert(1, AugHomoTransform())
        super().__init__(dataRoot, patchSize, padding, dualDir, homoDir, transforms, mask)


class ImagePairWithHomoDataset(ImagePairWithHomoPatchDataset):

    def __init__(self, dataRoot: str,
                 dualDir: Tuple[str, str] = ('input1', 'input2'), homoDir: str = 'homo', transforms = None,
                 mask = False, dual = False):
        if transforms is None:
            transforms = [TensorTransform(), HomoTransform(dual = dual)]
        super().__init__(dataRoot, 0, 0, dualDir, homoDir, transforms, mask)


class ImagePairWithHomoAdaptiveDataset(ImagePairWithHomoDataset):

    def __init__(self, dataRoot: str,
                 dualDir: Tuple[str, str] = ('input1', 'input2'), homoDir: str = 'homo', transforms = None,
                 mask = False, dual = False, resize = 0, asGrid = False, givenH = None):
        if transforms is None:
            transforms = [TensorTransform(), HomoTransform(dual = dual, asGrid = asGrid, addHomo = True, givenH = givenH),
                          WeightTransform(center = True, weight = False), #AddGridTransform(),
                          DropTransform(['homoParam'])]
            if resize > 0:
                transforms.insert(2, ResizeTransform(resize))
        super().__init__(dataRoot, dualDir, homoDir, transforms, mask)
