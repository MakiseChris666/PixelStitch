from .utils import *
import torch
from .torch_DLT import tensor_DLT
from modules.models.utils import torch_homo
import numpy as np

def solveH(h, w, dstMotion):
    src = torch.Tensor([[0, 0], [w, 0], [0, h], [w, h]]).unsqueeze(0).expand(*dstMotion.shape).to(dstMotion.device)
    dst = src + dstMotion
    H = tensor_DLT(src, dst)
    return H

def warpHomo(img, H, img_h, img_w, out_height, out_width, returnGrid = False):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        if not type(out_height) == int:
            out_height = out_height.round().int().item()
            out_width = out_width.round().int().item()
        f = True
    else:
        f = False

    M_tensor = torch_homo.getSizeTensor(out_height, out_width).to(img.device)
    N_tensor = torch_homo.getSizeTensor(img_h, img_w).to(img.device)
    N_tensor_inv = torch.inverse(N_tensor)

    H_mat = torch.matmul(torch.matmul(N_tensor_inv, H), M_tensor)

    if type(out_height) == int or out_height.dtype == torch.int:
        output_H = torch_homo.transformer(img, H_mat, (out_height, out_width), returnGrid)
        if f:
            output_H = output_H[0] if not returnGrid else (a[0] for a in output_H)
    else:
        output_H = []
        for i in range(img.shape[0]):
            output_H.append(torch_homo.transformer(img[[i]], H_mat[[i]], (out_height[i], out_width[i]), returnGrid))
    return output_H

def warpPointsHomo(pts, H):
    H = torch.inverse(H)
    ones = torch.ones((pts.shape[0], pts.shape[1], 1), device = pts.device)
    pts = torch.cat([pts, ones], dim = -1)
    pts = H @ pts.permute(0, 2, 1)
    pts[:, :2, :] /= pts[:, [2], :]
    return pts.permute(0, 2, 1)[..., :2]

def computePairedImageH(h, w, dstMotion):
    H = solveH(h, w, dstMotion)

    corner = torch.Tensor([[0, 0], [w, 0], [0, h], [w, h]]).to(H.device).unsqueeze(0).expand(*dstMotion.shape)
    hcorner = warpPointsHomo(corner, H)
    pts = torch.cat([corner, hcorner], dim = 1) # (b, 8, 2)
    width_max, _ = torch.max(pts[..., 0], dim = 1)
    width_min, _ = torch.min(pts[..., 0], dim = 1)
    height_max, _ = torch.max(pts[..., 1], dim = 1)
    height_min, _ = torch.min(pts[..., 1], dim = 1)

    out_width = width_max - width_min
    out_height = height_max - height_min

    H1 = torch.tensor([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]]).unsqueeze(0).tile(dstMotion.shape[0], 1, 1).to(H.device)
    H1[:, 0, 2] = width_min
    H1[:, 1, 2] = height_min
    H = H @ H1

    return H1, H, out_height, out_width

def centralizeH(h, w, H1, H2 = None, h2 = None, w2 = None):
    corner = torch.Tensor([[0, 0], [w, 0], [0, h], [w, h]]).to(H1.device).unsqueeze(0).tile(H1.shape[0], 1, 1)
    hcorner = warpPointsHomo(corner, H1)
    if H2 is not None:
        if h2 is not None and w2 is not None:
            corner2 = torch.Tensor([[0, 0], [w2, 0], [0, h2], [w2, h2]]).to(H1.device).unsqueeze(0).tile(H1.shape[0], 1, 1)
        else:
            corner2 = corner
        hcorner2 = warpPointsHomo(corner2, H2)
        hcorner = torch.cat([hcorner, hcorner2], dim = 1)
    pts = torch.cat([corner, hcorner], dim = 1)  # (b, 12, 2)
    width_max, _ = torch.max(pts[..., 0], dim = 1)
    width_min, _ = torch.min(pts[..., 0], dim = 1)
    height_max, _ = torch.max(pts[..., 1], dim = 1)
    height_min, _ = torch.min(pts[..., 1], dim = 1)

    out_width = width_max - width_min
    out_height = height_max - height_min

    transH = torch.tensor([[1., 0., width_min],
                       [0., 1., height_min],
                       [0., 0., 1.]]).unsqueeze(0).tile(H1.shape[0], 1, 1).to(H1.device)
    H1 = H1 @ transH
    if H2 is not None:
        H2 = H2 @ transH
        return H1, H2, out_height, out_width

    return transH, H1, out_height, out_width

def computePairedImageH2(h, w, dstMotion):
    H = solveH(h, w, dstMotion)

    corner = torch.Tensor([[0, 0], [w, 0], [0, h], [w, h]]).to(H.device).unsqueeze(0)
    hcorner1 = warpPointsHomo(corner, H[[0]])
    hcorner2 = warpPointsHomo(corner, H[[1]])
    pts = torch.cat([corner, hcorner1, hcorner2], dim = 1) # (b, 8, 2)
    width_max, _ = torch.max(pts[..., 0], dim = 1)
    width_min, _ = torch.min(pts[..., 0], dim = 1)
    height_max, _ = torch.max(pts[..., 1], dim = 1)
    height_min, _ = torch.min(pts[..., 1], dim = 1)

    out_width = width_max - width_min
    out_height = height_max - height_min

    H1 = torch.tensor([[1., 0., width_min],
                       [0., 1., height_min],
                       [0., 0., 1.]]).unsqueeze(0).tile(dstMotion.shape[0], 1, 1).to(H.device)
    H = H @ H1

    return H1, H, out_height, out_width

def splitHomo(h, w, dstMotion, split):
    src1 = torch.Tensor([[0, 0], [w, 0], [0, h], [w, h]]).unsqueeze(0).expand(*dstMotion.shape).to(dstMotion.device)
    src2 = src1 + dstMotion
    dst = src1 + dstMotion * split
    H2 = torch_DLT.tensor_DLT(src1, dst)
    H1 = torch_DLT.tensor_DLT(src2, dst)

    return H1, H2

def computeAndDecomposeH(h, w, dstMotion):
    # print(dstMotion)
    H = solveH(h, w, dstMotion)
    # print(H)
    return decomposeH(H)

def decomposeH(H):
    L, P = torch.linalg.eig(H)
    L = torch.diag_embed(L ** 0.5)
    # print(L, P)
    H2 = P @ L @ torch.inverse(P)
    H1 = torch.inverse(H2)
    # print(H1.shape, H2.shape)
    H1 = H1.real
    H2 = H2.real
    return H1, H2

def scaleH(H, factor):
    H[:, [0, 1], 2] /= factor
    H[:, 2, [0, 1]] *= factor
    return H