from typing import Sequence
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2 as cv
import numpy as np
from PIL import Image

def resizeImg(image, minSize = 256, divisor = 0):
    """
    :param image
    :param minSize: reshape the image so that the smaller side equals minSize
    :param divisor: if not 0, pad the image so that the shape of it can be divided by divisor
    """
    h, w, _ = image.shape
    factor = max(minSize / h, minSize / w)

    image = cv.resize(image, dsize = None, fx = factor, fy = factor)
    if divisor != 0:
        h, w, _ = image.shape
        dh = ((h - 1) // 8 + 1) * 8 - h
        dw = ((w - 1) // 8 + 1) * 8 - w
        image = np.pad(image, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)))

    return image

class PatchExtractor:

    def __init__(self, patchSize: int, factors: Sequence[float] = None):
        if factors is None:
            factors = [1., 1.5, 2.]
        self.patchSize = patchSize
        self.factors = factors

    def transform(self, images):
        """
        Randomly crop a patch with size crop × crop
        """
        crop = self.patchSize
        resize = np.random.choice([int(crop * i) for i in self.factors])
        res = []
        idw, idh = -1, -1
        flip = np.random.rand() < 0.5
        for image in images:
            image = resizeImg(image, minSize = resize)
            image = Image.fromarray(image)
            if idw == -1:
                w, h = image.size
                idw, idh = np.random.randint(w - crop + 1), np.random.randint(h - crop + 1)
            image = image.crop((idw, idh, idw + crop, idh + crop))
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            res.append(np.asarray(image))
        return res

    def __call__(self, *images):
        return self.transform(images)

def resizeTensor(image, minSize = 256, divisor = 0, points = None):
    """
    :param image
    :param minSize: reshape the image so that the smaller side equals minSize
    :param divisor: if not 0, pad the image so that the shape of it can be divided by divisor
    :param points: inplace
    """
    _, h, w = image.shape
    factor = max(minSize / h, minSize / w)
    th, tw = round(factor * h), round(factor * w)

    image = F.interpolate(image[None, ...], size = (th, tw), mode = 'bilinear')[0]
    if points is not None:
        for p in points:
            p *= factor
    if divisor != 0:
        _, h, w = image.shape
        dh = ((h - 1) // 8 + 1) * 8 - h
        dw = ((w - 1) // 8 + 1) * 8 - w
        image = F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))

    return image

class PatchExtractorTensor:

    def __init__(self, patchSize: int, factors: Sequence[float] = None):
        if factors is None:
            factors = [1., 1.25, 1.5, 1.75, 2.]
        self.patchSize = patchSize
        self.factors = factors

    def transform(self, images):
        """
        Randomly crop a patch with size crop × crop
        """
        crop = self.patchSize
        resize = np.random.choice([int(crop * i) for i in self.factors])
        res = []
        idw, idh = -1, -1
        flip = np.random.rand() < 0.5
        for image in images:
            image = resizeTensor(image, minSize = resize)
            _, h, w = image.shape
            if idw == -1:
                idw, idh = np.random.randint(w - crop + 1), np.random.randint(h - crop + 1)
            image = TF.crop(image, idh, idw, crop, crop)
            if flip:
                image = TF.hflip(image)
            res.append(image)
        return res

    def __call__(self, *images):
        return self.transform(images)

class PatchExtractorHomo:

    def __init__(self, patchSize: int, factors: Sequence[float] = None):
        if factors is None:
            factors = [1., 1.25, 1.5, 1.75, 2.]
        self.patchSize = patchSize
        self.factors = factors

    def __call__(self, H1, H2, outh, outw):
        # H (b, 3, 3); outsize (b, )
        b = H1.shape[0]
        crop = self.patchSize
        resize = np.random.choice([int(crop * i) for i in self.factors], b)
        resize = torch.from_numpy(resize).to(H1.device)
        factorh = outh / resize
        factorw = outw / resize
        scaleFactor = torch.minimum(factorh, factorw)
        scaleH = torch.zeros((b, 3, 3)).to(H1.device)
        scaleH[:, 2, 2] = 1
        scaleH[:, 0, 0] = scaleFactor
        scaleH[:, 1, 1] = scaleFactor

        outh /= scaleFactor
        outw /= scaleFactor
        outh = outh.round() - crop
        outw = outw.round() - crop
        choice = torch.rand(b).to(outh.device)
        croph = outh * choice
        cropw = outw * choice
        croph = croph.floor()
        cropw = cropw.floor()
        transH = torch.eye(3).unsqueeze(0).tile((b, 1, 1)).to(H1.device)
        transH[:, 0, 2] = cropw
        transH[:, 1, 2] = croph

        H1 = H1 @ scaleH @ transH
        H2 = H2 @ scaleH @ transH

        return H1, H2

