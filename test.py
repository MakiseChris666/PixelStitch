import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from modules.data.datasets import ImagePairWithHomoAdaptiveDataset, DataLoader
from modules.models.raft.raft import RAFTStitch
from modules.models.utils import getGrid
from modules.flow_viz import flow_to_image
from modules.models import utils
import cv2 as cv

def scale(img, minsize = 352):
    _, _, h, w = img.shape
    factor = max(minsize / h, minsize / w)
    img = F.interpolate(img, scale_factor = factor, mode = 'bilinear')
    return img

def resize8(image, returnParam = False):
    _, _, h, w = image.shape
    dh = ((h - 1) // 8 + 1) * 8 - h
    dw = ((w - 1) // 8 + 1) * 8 - w
    image = F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
    if returnParam:
        return image, dh, dw
    return image

def resizeBack(image, dh, dw):
    _, _, h, w = image.shape
    image = image[..., dh // 2:h - (dh - dh // 2), dw // 2:w - (dw - dw // 2)]
    return image

@torch.no_grad()
def test(args):

    model = RAFTStitch(args)
    model.load_state_dict(torch.load(args.save_path), strict = False)
    model = model.to(args.device)
    model.eval()

    dataset = ImagePairWithHomoAdaptiveDataset(args.data_root, mask = True, dual = True, asGrid = False,
                                               resize = args.resize)
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.workers,
                        pin_memory = True)
    loaderBar = tqdm(loader, total = len(loader))

    for (i, data) in enumerate(loaderBar):
        image1 = data['image1'].to(args.device)
        image2 = data['image2'].to(args.device)
        mask1 = data['mask1'].to(args.device)
        mask2 = data['mask2'].to(args.device)

        image1, dh, dw = resize8(image1, returnParam = True)
        image2 = resize8(image2)
        imagePair = torch.cat([image1, image2], dim = 0) # (2, c, h, w)

        flows, match = model(imagePair, args.iters)
        flow = flows[-1]
        flow = resizeBack(flow, dh, dw)

        flow = flow.permute(0, 2, 3, 1)
        weight1, weight2 = utils.getAdaptiveWeight(flow[[0]], flow[[1]],
                                                   data['c1'].to(args.device), data['c2'].to(args.device))

        _, h, w, _ = flow.shape
        flow[..., 0] /= w / 2
        flow[..., 1] /= h / 2

        flow[[0]] *= weight1[[0]]
        flow[[1]] *= weight2[[0]]

        grid = getGrid(h, w).to(args.device)
        flowGrid = flow + grid

        ## you may visualize the optical flow here
        # flowDraw = torch.empty_like(flow)
        # flowDraw[...] = flow
        # flowDraw[..., 0] *= w / 2
        # flowDraw[..., 1] *= h / 2
        # flowImage1 = flow_to_image(flowDraw[0].cpu().numpy())
        # flowImage2 = flow_to_image(flowDraw[1].cpu().numpy())

        if 'origin1' in data:
            image1 = data['origin1'].to(args.device)
            image2 = data['origin2'].to(args.device)
            mask1 = data['orimask1'].to(args.device)
            mask2 = data['orimask2'].to(args.device)
            _, _, h, w = image1.shape
            flowGrid = F.interpolate(flowGrid.permute(0, 3, 1, 2), size = (h, w), mode = 'bilinear').permute(0, 2, 3, 1)

        if 'origin1' not in data:
            image1 = resizeBack(image1, dh, dw)
            image2 = resizeBack(image2, dh, dw)

        nonmask = torch.all(image1 < 0.03, dim = 1, keepdim = True).float()
        image2 = F.grid_sample(image2, flowGrid[[1]])
        image1 = F.grid_sample(image1, flowGrid[[0]])

        # - this code block is some naive post-processings to prevent the warped images from blurred edges
        # which is caused by the fusion of edge pixels and invalid (black) pixels outside the image
        # - should possibly be solved by modifying the 'grid_sample' function
        # nonmask = F.grid_sample(nonmask, flowGrid[[0]])
        # nonmask[nonmask > 0.0] = 1
        # nonmask[nonmask < 1] = 0
        # nonmask_ = nonmask.clone()
        # nonmask[:, :, :-1, :] += nonmask_[:, :, 1:, :]
        # nonmask[:, :, :, :-1] += nonmask_[:, :, :, 1:]
        # nonmask[:, :, 1:, :] += nonmask_[:, :, :-1, :]
        # nonmask[:, :, :, 1:] += nonmask_[:, :, :, :-1]
        # nonmask = torch.clamp(nonmask, 0, 1)
        # mask2 = F.grid_sample(mask2, flowGrid[[1]])
        # mask1 = F.grid_sample(mask1, flowGrid[[0]])
        # # mask1[mask1 < 0.9] = 0
        # # mask1[mask1 > 0] = 1
        # # mask2[mask2 < 0.9] = 0
        # # mask2[mask2 > 0] = 1
        # image1 = image1 / (mask1 + 1e-7)
        # image2 = image2 / (mask2 + 1e-7)
        # # mask1[torch.mean(mask1, dim = 1, keepdim = True) < 0.05] = 0
        # # mask2[torch.mean(mask1, dim = 1, keepdim = True) < 0.05] = 0
        # mask1 *= torch.mean(mask1, dim = 1, keepdim = True) >= 0.05
        # mask2 *= torch.mean(mask2, dim = 1, keepdim = True) >= 0.05
        # mask1[mask1 > 0] = 1
        # mask2[mask2 > 0] = 1
        # image1 = image1 * mask1 * (1 - nonmask)
        # image2 = image2 * mask2
        # # image2 = W.grid_sample(raw2, grid2)
        # # image1 = W.grid_sample(raw1, grid1)
        # # mask2 = W.grid_sample(mask, grid2)
        # # mask1 = W.grid_sample(mask, grid1)
        # mask = mask1 * mask2

        image1 = image1[0].cpu().numpy().transpose(1, 2, 0)
        image2 = image2[0].cpu().numpy().transpose(1, 2, 0)
        mask1 = mask1[0].cpu().numpy().transpose(1, 2, 0)
        mask2 = mask2[0].cpu().numpy().transpose(1, 2, 0)

        imageSum = image1 + image2
        blend = image1 * image1 / (imageSum + 1e-6) + image2 * image2 / (imageSum + 1e-6)
        blend = (blend * 255).astype('uint8')

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        blendPath = os.path.join(args.out_dir, 'blend')
        warpPath = os.path.join(args.out_dir, 'warp2')
        image1Path = os.path.join(args.out_dir, 'warp1')
        mask1Path = os.path.join(args.out_dir, 'mask1')
        mask2Path = os.path.join(args.out_dir, 'mask2')
        if not os.path.exists(blendPath):
            os.mkdir(blendPath)
        if not os.path.exists(warpPath):
            os.mkdir(warpPath)
        if not os.path.exists(image1Path):
            os.mkdir(image1Path)
        if not os.path.exists(mask1Path):
            os.mkdir(mask1Path)
        if not os.path.exists(mask2Path):
            os.mkdir(mask2Path)

        cv.imwrite(os.path.join(blendPath, '%06d.jpg' % i), blend)
        cv.imwrite(os.path.join(warpPath, '%06d.jpg' % i), (image2 * 255).round().astype('uint8'))
        cv.imwrite(os.path.join(image1Path, '%06d.jpg' % i), (image1 * 255).round().astype('uint8'))
        cv.imwrite(os.path.join(mask1Path, '%06d.jpg' % i), (mask1 * 255).round().astype('uint8'))
        cv.imwrite(os.path.join(mask2Path, '%06d.jpg' % i), (mask2 * 255).round().astype('uint8'))

        torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--data-root', type = str, default = '/path/to/your/dataset')
    parser.add_argument('--out-dir', type = str, default = '/path/to/save/outputs')
    parser.add_argument('--save-path', type = str, default = './checkpoints/ckpt.pth')
    parser.add_argument('--workers', type = int, default = 1)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--resize', type = int, default = 0)
    parser.add_argument('--ftdown', type = int, default = 8)

    parser.add_argument('--iters', type = int, default = 4)
    parser.add_argument('--corr-kernel-size', type = int, default = 13)

    args = parser.parse_args()
    args.radius = args.corr_kernel_size // 2
    test(args)