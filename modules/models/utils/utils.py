import torch
import torch.nn.functional as F
import numpy as np

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def getGrid(h, w):
    gridY = torch.linspace(-1, 1, steps = h).view(1, -1, 1, 1).expand(1, h, w, 1)
    gridX = torch.linspace(-1, 1, steps = w).view(1, 1, -1, 1).expand(1, h, w, 1)
    grid = torch.cat((gridX, gridY), dim = 3)
    return grid

def getMesh(h, w, gridh, gridw, batchSize = 1):
    ww = torch.matmul(torch.ones([gridh + 1, 1]), torch.unsqueeze(torch.linspace(0., w - 1, gridw + 1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, h - 1, gridh + 1), 1), torch.ones([1, gridw + 1]))
    ori_pt = torch.cat((hh.unsqueeze(2), ww.unsqueeze(2)), 2)  # (grid_h+1)*(grid_w+1)*2
    return ori_pt.round().long().unsqueeze(0).tile(batchSize, 1, 1, 1)

def gridSampleOnMesh(mesh, flowGrid):
    meshX = mesh[..., 0]
    meshY = mesh[..., 1]
    res = []
    for x, y, f in zip(meshX, meshY, flowGrid):
        res.append(f[x, y])
    res = torch.stack(res, dim = 0)
    return res

def getDualFlowWeight(h, w, c1, c2):
    # c1, c2: (b, 2)
    # 呃呃这里的c1c2是用homo算的，所以格式是wh
    refvec = F.normalize(c1 - c2, dim = 1) # 向量c21，2那边为0，1那边为1
    gridX = torch.linspace(0, h - 1, steps = h).view(1, -1, 1, 1).expand(1, h, w, 1)
    gridY = torch.linspace(0, w - 1, steps = w).view(1, 1, -1, 1).expand(1, h, w, 1)
    grid = torch.cat((gridY, gridX), dim = 3).to(c1.device)
    grid = grid - c2[:, None, None, :]
    proj = torch.sum(grid * refvec[:, None, None, :], dim = -1)
    mx, mn = proj.max(), proj.min()
    weight = (proj - mn) / (mx - mn)
    return weight

def getAdaptiveWeight(flow1, flow2, c1, c2, h = None, w = None, grid = None):
    """
    flow in pixel level
    """
    if h is None or w is None:
        b, h, w, _ = flow1.shape
    b = flow1.shape[0]
    refvec = F.normalize(c1 - c2, dim = 1)  # 向量c21，2那边为0，1那边为1

    if grid is None:
        gridX = torch.linspace(0, h - 1, steps = h).view(1, -1, 1, 1).expand(1, h, w, 1)
        gridY = torch.linspace(0, w - 1, steps = w).view(1, 1, -1, 1).expand(1, h, w, 1)
        grid = torch.cat((gridY, gridX), dim = 3).to(c1.device)

    '''
    ((grid + w * flow) - c2) \cdot refvec = proj
    (proj - mn) / len = w or (mx - proj) / len = w
    proj = w * len + mn or mx - w * len = ((grid + w * flow) - c2) \cdot refvec
    (gridX + w * flowX - c2X) * refX + (gridY + w * flowY - c2Y) * refY = w * len + mn
    w * (flowX * refX + flowY * refY) + (gridX - c2X) * refX + (gridY - c2Y) * refY = w * len + mn
    w = (mn - (gridX - c2X) * refX - (gridY - c2Y) * refY) / (flowX * refX + flowY * refY - len)
    '''

    corner = torch.zeros((b, 4, 2)).to(c1.device)
    corner[:, 1, 0] = w - 1
    corner[:, 3, 0] = w - 1
    corner[:, 2, 1] = h - 1
    corner[:, 3, 1] = h - 1
    mvals = torch.sum(corner * refvec[:, None, :], dim = -1)
    mx, _ = torch.max(mvals, dim = 1)
    mn, _ = torch.min(mvals, dim = 1)
    mx = mx[:, None, None]
    mn = mn[:, None, None]
    len = mx - mn

    weight2 = ((mn - torch.sum(grid * refvec[:, None, None, :], dim = -1)) /
               (torch.sum(flow2 * refvec[:, None, None, :], dim = -1) - len)).unsqueeze(-1)
    weight1 = ((mx - torch.sum(grid * refvec[:, None, None, :], dim = -1)) /
               (torch.sum(flow1 * refvec[:, None, None, :], dim = -1) + len)).unsqueeze(-1)
    torch.clamp_(weight1, min = 0, max = 1)
    torch.clamp_(weight2, min = 0, max = 1)

    return weight1, weight2

def convertFlow(flowGrid):
    """
    Convert a grid from `align_corners = False` form to `align_corners = True` form
    """
    b, h, w, _ = flowGrid.shape
    coords = coords_grid(b, h, w, flowGrid.device)
    coords = F.grid_sample(coords, flowGrid)
    coords[..., 0] = coords[..., 0] / (w - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (h - 1) * 2 - 1
    return coords.permute(0, 2, 3, 1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    flow = torch.zeros((1, 10, 10, 2))
    flow[0, 1, 2] = torch.Tensor([8, 9])
    weight1, weight2 = getAdaptiveWeight(flow, flow, torch.Tensor([[2, 2]]), torch.Tensor([[8, 8]]))
    weight2 = weight2[0].numpy()
    plt.imshow(weight2, cmap = 'gray')
    plt.show()