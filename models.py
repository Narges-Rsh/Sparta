import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

#from graph import Graph
from st_graph_conv_block import ConvBlock
#Graph Encoder
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
#from reformer_pytorch import ReformerEncDec
#from reformer_pytorch import Reformer



class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img
    
class ConvEncoder2(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder2(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class GraphEncoder_2(nn.Module):
    def __init__(self, in_channels, h_dim=4, graph_args=None, split_seqs=True, eiw=True,
                 dropout=0.0, input_frames=12, conv_oper=None, act=None, headless=False, **kwargs):
        super(GraphEncoder, self).__init__()
        
        # Load graph
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        dec_1st_residual = kwargs.get('dec_1st_residual', None)
        
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        self.headless = headless
         # build networks
        num_node = self.graph.num_node
        self.fig_per_seq = 2
        if split_seqs:
            self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.dropout = dropout
        self.act = get_act(act)  # Activation function for conv block. Defaults to ReLU

        self.in_channels = in_channels
        self.h_dim = h_dim

        arch_dict = {'enc_ch_fac': [4, 4, 4, 6, 6, 6, 8, 8, 4],
                     'enc_stride': [1, 1, 2, 1, 1, 3, 1, 1, 1],
                     'dec_ch_fac': [4, 8, 8, 6, 6, 6],
                     'dec_stride': [1, 3, 1, 1, 2, 1]}

        self.enc_ch_fac = arch_dict['enc_ch_fac']
        self.enc_stride = arch_dict['enc_stride']
        self.dec_ch_fac = arch_dict['dec_ch_fac']
        self.dec_stride = arch_dict['dec_stride']

        self.out_bn  = kwargs.get('out_bn', False)
        self.out_act = kwargs.get('out_act', False)
        self.out_res = kwargs.get('out_res', False)
        self.gen_ae(self.enc_ch_fac,
                    self.enc_stride,
                    self.dec_ch_fac,
                    self.dec_stride,
                    dec_1st_residual=dec_1st_residual)
        # self.lastact = nn.Sigmoid()
        downsample_factor = np.multiply.reduce(np.array(self.enc_stride))
        self.hidden_dim = (input_frames / downsample_factor) * num_node * h_dim * self.fig_per_seq
        self.hidden_dim *= self.enc_ch_fac[-1]

        # Edge weighting
        if eiw and (not conv_oper.startswith('sagc')):
            self.ei_enc = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc])
            # self.ei_dec = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_dec])
        else:
            self.ei_enc = [1] * len(self.st_gcn_enc)
            # self.ei_dec = [1] * len(self.st_gcn_dec)
    
    def forward(self, x):
        if self.fig_per_seq == 1:
            if len(x.size()) == 4:
                x = x.unsqueeze(4)
        # Return to (N*M, c, t, v) structure
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x.float())
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_enc, self.ei_enc):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.contiguous()
        x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x_ref = x
        x_size = x.size()
        x = x.contiguous()
        x = x.view(N, -1)
        x = torch.sigmoid(x)
        return x, x_size, x_ref
    
    def gen_ae(self, enc_ch_fac, enc_stride, dec_ch_fac=None, dec_stride=None, symmetric=True, dec_1st_residual=True):
        if dec_ch_fac is not None or dec_stride is not None:
            symmetric = False
        # if symmetric:
        #     dec_ch_fac = enc_ch_fac[::-1]
        #     dec_stride = enc_stride[::-1]
        self.build_enc(enc_ch_fac, enc_stride)
        # self.build_dec(dec_ch_fac, dec_stride, dec_1st_residual=dec_1st_residual)

    def build_enc(self, enc_ch_fac, enc_stride):
        """
        Generate and encoder according to a series of dimension factors and strides
        """
        if len(enc_ch_fac) != len(enc_stride):
            raise Exception("Architecture error")

        enc_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper, 'act': self.act,
                       'headless': self.headless} for _ in enc_ch_fac]
        enc_kwargs[0] = {'residual': False, **enc_kwargs[0]}
        enc_kwargs[-1] = {'out_act': False, **enc_kwargs[-1]}  # No Relu for final encoder layer
        st_gcn_enc = [ConvBlock(self.in_channels, enc_ch_fac[0] * self.h_dim, self.kernel_size, enc_stride[0],
                                **enc_kwargs[0])]
        for i in range(1, len(enc_ch_fac)):
            st_gcn_enc.append(
                ConvBlock(enc_ch_fac[i - 1] * self.h_dim, enc_ch_fac[i] * self.h_dim, self.kernel_size, enc_stride[i],
                          **enc_kwargs[i]))
        self.st_gcn_enc = nn.ModuleList(st_gcn_enc)
        
        
class GraphDecoder(nn.Module):
    def __init__(self, in_channels, eiw=True, h_dim =8, conv_oper=None, graph_args=None, split_seqs=True,
                 dropout=0.0, input_frames=12, act=None, headless=False, **kwargs):
        super(GraphDecoder, self).__init__()
        # self.x_size = x_size
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        dec_1st_residual = kwargs.get('dec_1st_residual', None)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        self.headless = headless
        self.h_dim = h_dim
        self.in_channels = in_channels
    
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        
        num_node = self.graph.num_node
        self.fig_per_seq = 2
        if split_seqs:
            self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = get_act(act)  # Activation function for conv block. Defaults to ReLU

        arch_dict = {'enc_ch_fac': [4, 4, 4, 6, 6, 6, 8, 8, 4],
                     'enc_stride': [1, 1, 2, 1, 1, 3, 1, 1, 1],
                     'dec_ch_fac': [4, 8, 8, 6, 6, 6],
                     'dec_stride': [1, 3, 1, 1, 2, 1]}

        self.enc_ch_fac = arch_dict['enc_ch_fac']
        self.enc_stride = arch_dict['enc_stride']
        self.dec_ch_fac = arch_dict['dec_ch_fac']
        self.dec_stride = arch_dict['dec_stride']

        self.out_bn  = kwargs.get('out_bn', False)
        self.out_act = kwargs.get('out_act', False)
        self.out_res = kwargs.get('out_res', False)
        self.gen_ae(self.enc_ch_fac,
                    self.enc_stride,
                    self.dec_ch_fac,
                    self.dec_stride,
                    dec_1st_residual=dec_1st_residual)

        
        # Edge weighting
        if eiw and (not conv_oper.startswith('sagc')):
            # self.ei_enc = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc]) # self.st_gcn_enc self.A
            self.ei_dec = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_dec])
        else:
            # self.ei_enc = [1] * len(self.st_gcn_enc)
            self.ei_dec = [1] * len(self.st_gcn_dec)
    
    def forward(self, z, x_size):
        # Decoding layers
        # x = z.view(self.x_size)
        # N, C, T, V, M = self.x_size
        N, C, T, V, M = x_size
        x = z.view(N, int(C/2), T, V, M)
        # x = z.view(x_size)
        # N, C, T, V, M = x_size
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, int(C/2), T, V)
        for ind, (layer_, importance) in enumerate(zip(self.st_gcn_dec, self.ei_dec)):
            if type(layer_) == ConvBlock:
                x, _ = layer_(x, self.A * importance)  # A graph convolution
            else:
                x = layer_(x)  # An upsampling layer

        x, _ = self.dec_final_gcn(x, self.A * self.ei_dec[-1])  # Final layer has no upsampling
        if self.fig_per_seq == 1:
            return x

        NM, c, t, v = x.size()
        x = x.view(N, M, c, t, v)
        mu_x = x.permute(0, 2, 3, 4, 1).contiguous()
        return mu_x
    
    def gen_ae(self, enc_ch_fac, enc_stride, dec_ch_fac=None, dec_stride=None, symmetric=True, dec_1st_residual=True):
        if dec_ch_fac is not None or dec_stride is not None:
            symmetric = False
        if symmetric:
            dec_ch_fac = enc_ch_fac[::-1]
            dec_stride = enc_stride[::-1]
        # self.build_enc(enc_ch_fac, enc_stride)
        self.build_dec(dec_ch_fac, dec_stride, dec_1st_residual=dec_1st_residual)
        
    def build_dec(self, dec_ch_fac, dec_stride, dec_1st_residual=True):
        if len(dec_ch_fac) != len(dec_stride):
            raise Exception("Architecture error")
        dec_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper,
                       'act': self.act, 'headless': self.headless, } for _ in dec_ch_fac]
        dec_kwargs[1] = {'residual': dec_1st_residual, **dec_kwargs[1]}
        dec_kwargs += [{'residual': self.out_res, 'out_act': self.out_act, 'out_bn': self.out_bn, **dec_kwargs[0]}]
        st_gcn_dec = []
        for i in range(1, len(dec_ch_fac)):
            if dec_stride[i] != 1:
                st_gcn_dec.append(nn.Upsample(scale_factor=(dec_stride[i], 1), mode='bilinear'))

            st_gcn_dec.append(ConvBlock(dec_ch_fac[i - 1] * self.h_dim, dec_ch_fac[i] * self.h_dim, self.kernel_size, 1))

        # Add output layer back to in_channels w/o relu, bn or residuals
        if dec_kwargs[-1]['conv_oper'].startswith(('sagc')):
            dec_kwargs[-1]['conv_oper'] = 'gcn'
        self.dec_final_gcn = ConvBlock(dec_ch_fac[i] * self.h_dim, self.in_channels, self.kernel_size, 1,
                                       **(dec_kwargs[-1]))
        self.st_gcn_dec = nn.ModuleList(st_gcn_dec)
#######################################################

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


# def anneal_kl(args, vae, iteration):
#     if args.dataset == 'shapes':
#         warmup_iter = 7000
#     elif args.dataset == 'faces':
#         warmup_iter = 2500

#     if args.lambda_anneal:
#         vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
#     else:
#         vae.lamb = 0
#     if args.beta_anneal:
#         vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
#     else:
#         vae.beta = args.beta



def get_act(act_type):
    if act_type is None:
        return nn.ReLU(inplace=True)
    if act_type.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type.lower() == 'mish':
        return Mish()


class Mish(nn.Module):
    """
    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))
  


#Positional encoding is added to the input to the Transformer to give the model 
#information about the position of each token in the sequence.
# This is crucial because the Transformer architecture itself does not inherently 
# model sequence order. 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device='cuda:0'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding = self.encoding.to(device)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.encoding[:, :x.size(1)].detach()
    
# class TransformerEncoder(nn.Module):
#     def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
#         super(TransformerEncoder, self).__init__()
#         self.positional_encoding = PositionalEncoding(d_model, max_len, device)
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=dropout) 
#             for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         x = self.positional_encoding(x)
#         for layer in self.layers:
#             x = layer(x)
#         return x
# class TransformerDecoder(nn.Module):
#     def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
#         super(TransformerDecoder, self).__init__()
#         self.positional_encoding = PositionalEncoding(d_model, max_len, device)
#         self.layers = nn.ModuleList([
#             nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout=dropout) 
#             for _ in range(num_layers)
#         ])

#     def forward(self, tgt, memory):
#         tgt = self.positional_encoding(tgt)
#         for layer in self.layers:
#             tgt = layer(tgt, memory)
#         return tgt
    
# class TransformerPredictor(nn.Module):
#     def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
#         super(TransformerPredictor, self).__init__()
#         self.encoder = TransformerEncoder(d_model, num_heads, d_feedforward, num_layers, max_len, device, dropout=dropout)
#         self.decoder = TransformerDecoder(d_model, num_heads, d_feedforward, num_layers, max_len, device, dropout=dropout)

#     def forward(self, src, tgt):
#         memory = self.encoder.forward(src)
#         output = self.decoder.forward(tgt, memory)
#         return output
    
#     def mse (self, x):
#         batch_size, seq_lenx2, d = x.shape
#         pred = self.forward(x, x)
#         return torch.mean((x-pred)**2, dim=(1, 2))

#D_model: The dimensionality of the input and output feature vectors
#  (often referred to as the embedding size).
#num_heads: The number of attention heads in the multi-head attention mechanism.

#batch_first=True means the input tensor's shape will be
#(batch_size, sequence_length, features) instead of the default 
# (sequence_length, batch_size, features).




class TransformerPredictor(nn.Module):
    #i added input dimension to this
    def __init__(self,d_model, num_heads, d_feedforward, num_layers, max_seq_len=1000, device='cuda:0', dropout=0.1):
        #d_feedforward=64, d_model=36= embedding dimension, num-heads+2, num_layers=2, dropout=0.1
        super(TransformerPredictor, self).__init__()

        

        self.encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True) 

        self.decoder_layers = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True) 

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, device)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_layers)
        self.device = device

    def forward(self, src, tgt):
        
        #print(f"[DEBUG] SRC after projection: {src.shape}")
        #print(f"[DEBUG] Entered forward with tgt shape: {tgt.shape}")
        src = self.positional_encoding(src)  #tokenization:t: ([512,24, 36]) (batch, sequence length, embedding dimension)
        #embeddding dimension: the size of the feature vector for each token
        #24:sequnce lenght: the number of tokens in each sequence
        memory = self.encoder.forward(src)
        
        tgt = self.positional_encoding(tgt)
        output = self.decoder.forward(tgt, memory)  #([512, 24, 36])
        return output
    
    def mse (self, x):       #why??
        batch_size, seq_lenx2, d = x.shape
        pred = self.forward(x, x)



class ReformerPredictor(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, bucket_size=6, device='cuda:0'):
        super(ReformerPredictor, self).__init__()
        self.device = device

        # Project continuous input to d_model
        # self.input_proj = nn.Linear(input_dim, d_model)
        # self.output_proj = nn.Linear(d_model, input_dim)

        # Encoder
        self.encoder = Reformer(
            dim=d_model,
            depth=num_layers,
            heads=num_heads,
            lsh_dropout=0.1,
            causal=False,  # No need for causal in encoder
            bucket_size=bucket_size,
            n_hashes=4
        ).to(device)

        # Decoder
        self.decoder = Reformer(
            dim=d_model,
            depth=num_layers,
            heads=num_heads,
            lsh_dropout=0.1,
            causal=True,  # Causal in decoder for autoregression
            bucket_size=bucket_size,
            n_hashes=4
        ).to(device)

    def forward(self, src, tgt):
        """
        Args:
            src: (batch, seq_len, input_dim) - continuous input
            tgt: (batch, seq_len, input_dim) - continuous target
        """
        # Project to model dimension
        # src_proj = self.input_proj(src)
        # tgt_proj = self.input_proj(tgt)

        # Encoder

        #print(f"[DEBUG] Entered forward with src shape: {src.shape}")
        #print(f"[DEBUG] Entered forward with tgt shape: {tgt.shape}")
        memory = self.encoder(src)

        # (Optional) Combine memory and tgt_proj (if needed)
        if memory.shape[1] == tgt.shape[1]:
            decoder_input = tgt + memory  # Sum for simplicity
        else:
            # Interpolate to match size
            memory_resized = nn.functional.interpolate(
                memory.permute(0, 2, 1), size=tgt.shape[1], mode='linear'
            ).permute(0, 2, 1)
            decoder_input = tgt + memory_resized

        # Decoder
        output = self.decoder(decoder_input)

        
        print(f"[DEBUG] Entered forward with output shape: {output.shape}")
        return output

    # def mse(self, x):
    #     """
    #     Args:
    #         x: (batch, 2 * seq_len, input_dim)
    #     """
    #     batch_size, seq_lenx2, d = x.shape
    #     print(f"[DEBUG] Input x shape: {x.shape}") 
    #     seq_len = seq_lenx2 // 2

    #     src = x[:, :seq_len, :]  # First half
    #     tgt = x[:, seq_len:, :]  # Second half

    #     pred = self.forward(src, tgt)  # Reconstruction
    #     return nn.functional.mse_loss(pred, tgt)

    def mse(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - sequence to reconstruct
        Returns:
            MSE loss between model output and input x
        """
        print(f"[DEBUG] Input x shape: {x.shape}")

        # Forward pass: using x both as source and target
        pred = self.forward(x, x)  # (batch, seq_len, input_dim)

        # Debug check if gradients are tracked
        print(f"[DEBUG] pred requires_grad: {pred.requires_grad}")

        # Mean Squared Error between prediction and original x
        loss = nn.functional.mse_loss(pred, x)
        print(f"[DEBUG] Loss value: {loss.item()}")

        return loss

    

##Graph Encoder##
# Define your Graph Encoder using a GCN (Graph Convolutional Network) as an example
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 64)
        self.conv2 = pyg_nn.GCNConv(64, out_channels)
    
    def forward(self, graph_data):
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# Function to create the graph representation of keypoints
def create_graph_from_keypoints(data, num_kp, device):
    batch_size, channels, seg_len, num_kp = data.shape  
    node_features = data.permute(0, 2, 3, 1)  # [batch_size, seg_len, num_kp , channels]    
    #(512,24,18,2) 
   

    edge_index = create_edge_index(num_kp).to(device) 
     # Create edges based on your choice, e.g., Euclidean distance  [2,306] : fully connected graph for num_kp nodes
     #a fully connected graph with directed edges
    graph_list = []

    for i in range(seg_len):
        keypoints = node_features[:, i, :, :]  #shape of key points: [batch_size, num_kp, channels]  [512,18,2]
        ## Get keypoint positions for the i-th segment (x and y coordinates) and Extract keypoints for the i-th frame
        #For i = 0, keypoints becomes [32, 18, 2], representing the keypoint data for the first frame across all 32 samples.
        ## You can define how to create edges based on your data
        ## Construct the edge index (graph connectivity, for example, by Euclidean distance)
        # Here you need to define a function to create edges, for simplicity assuming fully connected graph
        #This function would create the edge indices that connect the keypoints based on your choice of connectivity (e.g., fully connected or based on physical relationships between body parts).
        keypoints = keypoints.reshape(-1, channels).to(device)  # Flatten into [batch_size * num_kp, channels]   ([9216, 2])
        ## Create PyG Data object
        graph_data = Data(x=keypoints, edge_index=edge_index)  # Create the graph data object for PyTorch Geometric
        graph_list.append(graph_data)
    
    return graph_list

# Example function to create edges (can use Euclidean distance or predefined body part connectivity)
def create_edge_index(num_kp):
    # Define edges here (this example assumes a fully connected graph)
    edge_index = []
    for i in range(num_kp):
        for j in range(i+1, num_kp):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index





                                       

        
