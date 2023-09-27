import torch
import torch.nn as nn
import torch.nn.functional as F

from corr import CorrBlock
from extractor import BasicEncoder
from utils.utils import coords_grid, upflow8

from model.encoder_gmflownet import BasicConvEncoder, POLAUpdate, MixAxialPOLAUpdate
from model.update_skflow import SKMotionEncoder6_Deep_nopool_res, PCBlock4_Deep_nopool_res
from model.modules import GCL, GGAM

autocast = torch.cuda.amp.autocast


class GAFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        print(' ---------- Model: GAFlow ---------- ')
        self.args = args
        args.corr_levels = 4
        args.corr_radius = 4
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.args.UpdateBlock = 'SKUpdateBlock6_Deep_nopoolres_AllDecoder'
        self.args.k_conv = [1, 15]
        self.args.PCUpdater_conv = [1, 7] 

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if args.dataset == 'sintel':
            self.fnet = nn.Sequential(
                            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                            MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
                        )
        else:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )
        
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.gcl = GCL(embed_dim=256, depth=1, args=args)
        self.update_sk = SKUpdate(self.args, hidden_dim=hdim) 
        self.zero = nn.Parameter(torch.zeros(12))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        # coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        # coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False, flow_gt=None):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        gcl_fmap1 = self.gcl(fmap1)
        gcl_fmap2 = self.gcl(fmap2)
        corr_fn = CorrBlock(gcl_fmap1, gcl_fmap2, radius=self.args.corr_radius)

        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        coords1_init = coords1

        softCorrMap = None
        if self.args.dataset == 'sintel':
            # Correlation as initialization
            N, fC, fH, fW = fmap1.shape
            corr_gm = torch.einsum('b c n, b c m -> b n m', fmap1.view(N, fC, -1), fmap2.view(N, fC, -1))
            corr_gm = corr_gm / torch.sqrt(torch.tensor(fC).float())
            corrMap = corr_gm

            #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
            softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if self.args.global_flow and not flow_init:
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            # coords_y = coords_index // fW
            coords_y = torch.div(coords_index, fW, rounding_mode='trunc')

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        elif flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_sk(net, inp, corr, flow, itr)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions, softCorrMap


class SKUpdate(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = GGAM(args, 128)

    def forward(self, net, inp, corr, flow, itr=None):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(inp, motion_features, itr)

        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

