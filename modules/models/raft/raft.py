import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.models.raft.update import BasicUpdateBlock, ImageMatch
from modules.models.raft.extractor import BasicEncoder, SmallEncoder
from modules.models.raft.corr import CorrBlock, AlternateCorrBlock
from modules.models.raft.utils import upsample_flow, initialize_flow, upflow8


class RAFTStitch(nn.Module):
    def __init__(self, args):
        super(RAFTStitch, self).__init__()
        self.args = args

        # self.hidden_dim = hdim = 96
        # self.context_dim = cdim = 64
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        if 'num_levels' in args:
            args.corr_levels = args.num_levels
        else:
            args.corr_levels = 1
        if 'radius' in args:
            args.corr_radius = args.radius
        else:
            args.corr_radius = 3

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        # self.fnet = SmallEncoder(output_dim = 256, norm_fn = 'instance', dropout = args.dropout)
        # self.cnet = SmallEncoder(output_dim = hdim + cdim, norm_fn = 'batch', dropout = args.dropout)
        # self.update_block = SmallUpdateBlock(self.args, hidden_dim = hdim)

        self.fnet = BasicEncoder(output_dim = 256, norm_fn = 'instance', dropout = args.dropout)
        self.cnet = BasicEncoder(input_dim = 3, output_dim = hdim + cdim, norm_fn = 'batch', dropout = args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim = hdim)

        self.image_match = ImageMatch()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, imagePair, iters = 4, weight2 = None, flow_init = None, upsample = True, dual = True, fmap = None, corr_fn = None):
        """ Estimate optical flow between a pair of frames """

        imagePair = imagePair.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        b = imagePair.shape[0]
        # run the feature network
        if fmap is None:
            fmap = self.fnet(imagePair)
        if dual:
            fmap1 = fmap
            fmap2 = torch.roll(fmap1, b // 2, dims = 0)
        else:
            fmap1 = fmap[b // 2:]
            fmap2 = fmap[:b // 2]

        if corr_fn is None:
            if self.args.alternate_corr:
                corr_fn = AlternateCorrBlock(fmap2, fmap1, num_levels = self.args.corr_levels, radius = self.args.corr_radius)
            else:
                corr_fn = CorrBlock(fmap2, fmap1, num_levels = self.args.corr_levels, radius = self.args.corr_radius)

        # run the context network
        if dual:
            if weight2 is None:
                cnet = self.cnet(imagePair)
            else:
                cnet = self.cnet(torch.cat([imagePair, weight2], dim = 1))
        else:
            cnet = self.cnet(imagePair[:b // 2])
        net, inp = torch.split(cnet, [hdim, cdim], dim = 1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = initialize_flow(fmap1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        matchPred = None
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if upsample:
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0, factor = self.args.ftdown)
                else:
                    flow_up = upsample_flow(coords1 - coords0, up_mask)
            else:
                flow_up = coords1 - coords0

            # 返回像素级相对光流
            flow_predictions.append(flow_up)

            if matchPred is None and self.training:
                matchPred = self.image_match(imagePair)

        return flow_predictions, matchPred