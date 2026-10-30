import json
import os
import torch
import torch.optim as optim
import shutil
import time
from tqdm import tqdm
from CTR.graph.ucla import Graph

import numpy as np 
import torch.optim as optim
from functools import partial
from datetime import datetime
from utils.eval2 import score_dataset
from utils.schedulers.delayed_sched import *
from torch.utils.tensorboard import SummaryWriter
from utils.schedulers.cosine_annealing_with_warmup import *
import torch.nn as nn
# import ot 
import torch.nn.functional as F
import csv
from models_graph.gcae.gcae import GCAE, Encoder_T 
import torch
import torch.nn as nn
from CTR.model.ctrgcn import Model, Encoder_CTR
from torch.utils.data import DataLoader
#from feeders.feeder_ucla import Feeder
import yaml
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def init_model_params(args, dataset):
    return {
        'pose_shape': dataset["test"][0][0].shape if args.model_confidence else dataset["test"][0][0][:2].shape,
        'hidden_dim': args.model_latent_dim,
        'actnorm_scale': 1.0,
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'device': args.device,
        'model_dist': 'normal'
    }


def dump_args(args, ckpt_dir):
    path = os.path.join(ckpt_dir, "args.json")
    data = vars(args)
    with open(path, 'w') as fp:
        json.dump(data, fp)


def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def get_fn_suffix(args):
    fn_suffix = args.dataset + args.conv_oper
    return fn_suffix


class CostumLoss:
    # Constructor method (initialize instance attributes)
    def __init__(self, loss_type='mse',  a=1, b=1, c=1, d=1):
        self.loss_type = loss_type
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    # Instance method (access and modify instance attributes)
    def calculate(self, batch1, batch2):
        if self.loss_type == 'mse':
            return self.mse_loss(batch1, batch2)
        # elif self.loss_type == 'emd':
        #     return self.emd_loss(batch1, batch2)
        elif self.loss_type == 'ssim':
            return self.ssim_loss(batch1, batch2)
        elif self.loss_type == 'spec':
            return self.spectral_distance_loss(batch1, batch2)
        elif self.loss_type == 'cos':
            return self.cosine_similarity_loss(batch1, batch2)
        elif self.loss_type == 'combined':
            mse = self.mse_loss(batch1, batch2)
            cos = self.cosine_similarity_loss(batch1, batch2)
            # emd = self.emd_loss(batch1, batch2)
            spec = self.spectral_distance_loss(batch1, batch2)
            mse_n = (mse-torch.min(mse))/(torch.max(mse)-torch.min(mse))
            cos_n = (cos-torch.min(cos))/(torch.max(cos)-torch.min(cos)) 
            # emd_n = (emd-torch.min(emd))/(torch.max(emd)-torch.min(emd))
            spec_n = (spec-torch.min(spec))/(torch.max(spec)-torch.min(spec))              
            return self.a*mse_n + self.b*cos_n + self.d*spec_n
        else:
            print("Unrecognized loss!")


    def mse_loss(self, x, x_recon):
        return torch.mean((x-x_recon)**2, dim=(1, 2))

    def calc_adjacency(self, x):
        batch_size, seq_lenx2, d = x.shape
        x = x.view(batch_size, seq_lenx2, 36, 2)
        
        # Compute pairwise L2 distances for all pairs of points
        # Use torch.cdist for efficient distance computation
        pairwise_distances_x = torch.cdist(x, x, p=2)
        
        # Create a mask for upper triangular elements
        mask = torch.triu(torch.ones(36, 36), diagonal=1).bool()
        
        # Apply the mask to the pairwise distances
        adj = pairwise_distances_x[:, :, mask].view(batch_size, seq_lenx2, -1)
        return adj
        
        
    def spectral_distance_loss (self, x, x_recon):
        adj_x = self.calc_adjacency(x)
        adj_x_recon = self.calc_adjacency(x_recon)
        return torch.mean((adj_x-adj_x_recon)**2, dim=(1, 2))

    def ssim_loss(self, batch1, batch2, window_size=11, size_average=False):
        """
        Calculate SSIM loss between two batches of images.

        Args:
        - batch1: First batch of images (batch_size, channels, height, width)
        - batch2: Second batch of images (batch_size, channels, height, width)
        - window_size: Size of the SSIM window (default: 11)
        - size_average: If True, compute the average SSIM over the batch (default: True)

        Returns:
        - loss: SSIM loss for each image in the batch (batch_size,)
        """
        

        batch_size, num_channels, d = batch1.size()
        batch1 = batch1.reshape(batch_size, num_channels, 36, 2)
        batch2 = batch2.reshape(batch_size, num_channels, 36, 2)
        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)

        # Create a 1D Gaussian kernel
        # window = torch.Tensor(nn.functional.gaussian(window_size, window_size / 6.4)).to(batch1.device)
        window = self.gaussian_kernel(window_size, window_size / 6.4).to(batch1.device)
        window = window / window.sum()

        loss = torch.Tensor([]).to(batch1.device)

        for i in range(batch_size):
            for j in range(num_channels):
                img1 = batch1[i, j, :, :].unsqueeze(0)
                img2 = batch2[i, j, :, :].unsqueeze(0)

                mu1 = nn.functional.conv2d(img1, window.view(1, 1, window_size, 1).expand(num_channels, -1, -1, -1), padding=(window_size//2, 0))
                mu2 = nn.functional.conv2d(img2, window.view(1, 1, window_size, 1).expand(num_channels, -1, -1, -1), padding=(window_size//2, 0))
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2

                sigma1_sq = nn.functional.conv2d(img1 * img1, window.view(1, 1, window_size, 1).expand(num_channels, -1, -1, -1), padding=(window_size//2, 0)) - mu1_sq
                sigma2_sq = nn.functional.conv2d(img2 * img2, window.view(1, 1, window_size, 1).expand(num_channels, -1, -1, -1), padding=(window_size//2, 0)) - mu2_sq
                sigma12 = nn.functional.conv2d(img1 * img2, window.view(1, 1, window_size, 1).expand(num_channels, -1, -1, -1), padding=(window_size//2, 0)) - mu1_mu2

                ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
                ssim_map = ssim_map.mean(dim=(-1, -2))

                # Calculate the mean SSIM for the channel
                loss = torch.cat((loss, ssim_map.mean().unsqueeze(0)), 0)

        one_tensor = torch.ones(loss.shape, device=loss.device)
        # Calculate cosine similarity

        if size_average:
            return loss.mean()
        else:
            return one_tensor-loss

    def gaussian_kernel(slef,size, sigma):
        x = torch.arange(size) - size // 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
        return kernel / kernel.sum()


    # def emd_loss(self, batch1, batch2):
    #     """
    #     Calculate Earth Mover's Distance (EMD) between two batches of tensors.

    #     Args:
    #     - batch1: First batch of tensors (batch_size, dim, num_samples)
    #     - batch2: Second batch of tensors (batch_size, dim, num_samples)

    #     Returns:
    #     - emd: EMD loss for each data point in the batch (batch_size,)
    #     """

    #     batch_size, dim, num_samples = batch1.size()
        
    #     emd = torch.zeros(batch_size, device=batch1.device)

    #     for i in range(batch_size):
    #         sample1 = batch1[i].permute(1, 0).reshape(dim, -1)
    #         sample2 = batch2[i].permute(1, 0).reshape(dim, -1)

    #         # Calculate EMD
    #         M = ot.dist(sample1, sample2)
    #         emd_value = ot.emd2(torch.tensor(np.array([]), device=M.device), torch.tensor(np.array([]), device=M.device), M)
    #         emd[i] = torch.tensor(emd_value, device=batch1.device, requires_grad=True)

    #     return emd


    # Gradients can be accessed as batch1.grad and batch2.grad


    def cosine_similarity_loss(self, batch1, batch2):
        """
        Calculate cosine similarity between datapoints in two batches.

        Args:
        - batch1: Tensor of shape (batch_size, feature_dim) for the first batch.
        - batch2: Tensor of shape (batch_size, feature_dim) for the second batch.

        Returns:
        - similarities: Tensor of shape (batch_size,) containing cosine similarities for each pair of datapoints.
        """
        batch_size, seq_len, d = batch1.size()
        batch1 = batch1.reshape(batch_size, -1)
        batch2 = batch2.reshape(batch_size,-1)
        
        # Normalize the input tensors
        batch1 = F.normalize(batch1, p=2, dim=1)
        batch2 = F.normalize(batch2, p=2, dim=1)

        # Calculate the dot product
        dot_product = torch.sum(batch1 * batch2, dim=1)
        one_tensor = torch.ones(dot_product.shape, device=dot_product.device)
        # Calculate cosine similarity
        similarities = dot_product

        return one_tensor-similarities


                                                 
class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  # For checkpoint filename
        self.loss = CostumLoss(args.loss, a=args.a, b=args.b, c=args.c, d=args.d)
        # Loss, Optimizer and Scheduler
        self.expand_ratio = 1
        if args.relative:
            self.expand_ratio = 2
        
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr,)
        
    def adjust_lr(self, epoch, lr=None):
        if self.scheduler is not None:
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
        elif (lr is not None) and (self.args.lr_decay is not None):
            new_lr = lr * (self.args.lr_decay ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            raise ValueError('Missing parameters for LR adjustment')
        return new_lr
    
    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args
        if not os.path.exists(self.args.save_dir):
        # Create the directory
            os.makedirs(self.args.save_dir)
        
        current_time = datetime.now()
        # path_join = os.path.join(self.args.ckpt_dir, filename)
        # path_join = os.path.join(self.args.save_dir, filename + '_' +  current_time.strftime("%Y-%m-%d_%H-%M-%S")+".pth.tar")
        path_join = os.path.join(self.args.save_dir, filename + '_' + str(epoch) + ".pth.tar")
        torch.save(state, path_join)
        if is_best:
            # shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))
            shutil.copy(path_join, os.path.join(self.args.save_dir, 'checkpoint_best.pth.tar'))
    
    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))
    
    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state
    
    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        writer = SummaryWriter(args.model_save_dir)
        best_roc = 0.5
        # train_elbo = []
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'
        if num_epochs is None:  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = args.epochs
        else:
            start_epoch = 0
            
        self.model = self.model.to(args.device)
        if args.token_config == "graph":
            encoder_path = args.encoder_path  # Set your checkpoint path
            encoder = load_trained_encoder(encoder_path, args)  # Load the trained encoder
            encoder.to(args.device)  # Move it to the correct device
            encoder.eval()  # Ensure it's in evaluation mode

        best_loss = float('inf')  # Initialize at the beginning
        best_metrics = {}
        for epoch in range(start_epoch, num_epochs):
            running_loss = 0
            print("Started epoch {}".format(epoch))
            self.model.train()
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:,0:2, :, :].to(torch.float32)
                batch_size, channels, seg_len, num_kp = data.shape

                if args.token_config == "t": 
                    data = data.permute(0, 2, 1, 3)
                    data = data.reshape(batch_size, seg_len, channels*num_kp)


                elif args.token_config == "graph":
                    # encoder_path = args.encoder_path  # Set your checkpoint path
                    # encoder = load_trained_encoder(encoder_path, args)  # Load the trained encoder
                    # encoder.to(args.device)  # Move it to the correct device
                    # encoder.eval()  # Ensure it's in evaluation mode

                    encoded_tokens = encoder(data)
                    print("token shape:", encoded_tokens.shape )
                    batch_size, channels, seg_len, num_kp, _ = encoded_tokens.shape  # Example
                    encoded_tokens = encoded_tokens.view(batch_size, seg_len, -1)  # Reshape to (batch, seq_len, features)
                    #Flattened feature dimension (C × num_kp × _)
                    data = encoded_tokens  # Replace input data with extracted tokens

                elif args.token_config == "CTR":
                    if num_kp < 20:
                        padding_kps = 20 - num_kp  # Number of joints to add (should be 2)
                        padding = torch.zeros((batch_size, channels, seg_len, padding_kps), device=data.device)
                        data = torch.cat((data, padding), dim=-1)  # Add zero joints at the end

                    checkpoint_path =args.CTR_path  # Set your checkpoint path
                    CTR = load_trained_CTR(checkpoint_path, args)  # Load the trained encoder
                    CTR.to(args.device)  # Move it to the correct device
                    CTR.eval()  # Ensure it's in evaluation mode

                    encoded_tokens_CTR = CTR(data)
                    print("token shape:", encoded_tokens_CTR.shape )
                    batch_size, channels, seg_len, num_kp = encoded_tokens_CTR.shape  # Example
                    encoded_tokens_CTR = encoded_tokens_CTR.view(batch_size, seg_len, -1)  # Reshape to (batch, seq_len, features)
                    #Flattened feature dimension (C × num_kp × _)
                    data = encoded_tokens_CTR  # Replace input data with extracted tokens


                elif args.token_config == "pst": 
                    data = data.view(batch_size, seg_len, channels*num_kp)
                elif args.token_config == "kps":
                    if num_kp == 36:
                        absolute = data[:, :, :, 0:18]
                        rel = data[:, :, :, 18:]
                        data = torch.cat([absolute, rel], dim=1)
                        data = data.view(batch_size, -1, 18)
                        data = data.permute(0, 2, 1)
                        print("data shape", data.shape)
                    elif num_kp == 38:
                        absolute = data[:, :, :, 0:19]
                        rel = data[:, :, :, 19:]
                        data = torch.cat([absolute, rel], dim=1)
                        data = data.view(batch_size, -1, 19)
                        data = data.permute(0, 2, 1)
                    else:
                        data = data.view(batch_size, -1, num_kp)
                        data = data.permute(0, 2, 1)
                elif args.token_config == "2ds":
                    if num_kp == 36:
                        absolute = data[:, :, :, 0:18]
                        rel = data[:, :, :, 18:]
                        absolute = absolute.permute(0,2,1,3)
                        absolute = absolute.reshape(batch_size,seg_len,-1)
                        absolute = absolute.permute(0, 2, 1)
                        rel = rel.permute(0,2,1,3)
                        rel = rel.reshape(batch_size,seg_len,-1)
                        rel = rel.permute(0, 2, 1)
                        data = torch.cat([absolute, rel], dim=2)
                    elif num_kp == 38:
                        absolute = data[:, :, :, 0:19]
                        rel = data[:, :, :, 19:]
                        absolute = absolute.permute(0,2,1,3)
                        absolute = absolute.reshape(batch_size,seg_len,-1)
                        absolute = absolute.permute(0, 2, 1)
                        rel = rel.permute(0,2,1,3)
                        rel = rel.reshape(batch_size,seg_len,-1)
                        rel = rel.permute(0, 2, 1)
                        data = torch.cat([absolute, rel], dim=2)
                        
                    else:
                        data = data.permute(0,2,1,3)
                        data = data.reshape(batch_size,seg_len,-1)
                        data = data.permute(0, 2, 1)
                elif args.token_config == "st":
                    if num_kp == 36:
                        absolute = data[:, :, :, 0:18]
                        rel = data[:, :, :, 18:]
                        absolute = absolute.permute(0, 1, 3, 2)
                        absolute = absolute.reshape(batch_size,channels,-1)
                        absolute = absolute.permute(0, 2, 1)
                        rel = rel.permute(0, 1, 3, 2)
                        rel = rel.reshape(batch_size,channels,-1)
                        rel = rel.permute(0, 2, 1)
                        data = torch.cat([absolute, rel], dim=2)
                    elif num_kp == 38:
                        absolute = data[:, :, :, 0:19]
                        rel = data[:, :, :, 19:]
                        absolute = absolute.permute(0, 1, 3, 2)
                        absolute = absolute.reshape(batch_size,channels,-1)
                        absolute = absolute.permute(0, 2, 1)
                        rel = rel.permute(0, 1, 3, 2)
                        rel = rel.reshape(batch_size,channels,-1)
                        rel = rel.permute(0, 2, 1)
                        data = torch.cat([absolute, rel], dim=2)
                        
                    else:
                        data = data.permute(0, 1, 3, 2)
                        data = data.reshape(batch_size,channels,-1)
                        data = data.permute(0, 2, 1)
                    
                self.optimizer.zero_grad()
                # mse_loss = self.model.mse(data)
                # mse_loss.mean().backward()
                recon = self.model.forward(data, data)
                loss = self.loss.calculate(data, recon)
                loss.mean().backward()
                self.optimizer.step()
                # running_loss = mse_loss.sum()
                running_loss = running_loss + loss.sum()
                
            epoch_loss = running_loss/len(self.train_loader.dataset)
            print('[Epoch %03d]  \t training loss: %.10f ' % (
                epoch, epoch_loss))
            writer.add_scalar('Loss', epoch_loss, epoch)
            new_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self.adjust_lr(epoch, new_lr)
            print('lr: {0:.3e}'.format(new_lr))
            # train_elbo.append(epoch_loss)
        
            eval_loss = []
            # dataset_size = len(self.test_loader.dataset)
            self.model.eval()
            with torch.no_grad():
                for i, data_batch in enumerate(tqdm(self.test_loader)):
                    data = data_batch[0].to(args.device, non_blocking=True)
                    data = data[:,0:2, :, :].to(torch.float32)
                    batch_size, channels, seg_len, num_kp = data.shape
                    if args.token_config == "t": 
                        data = data.permute(0, 2, 1, 3)
                        data = data.reshape(batch_size, seg_len, channels*num_kp)


                    elif args.token_config == "graph":
                        encoder_path = args.encoder_path  # Set your checkpoint path
                        encoder = load_trained_encoder(encoder_path, args)  # Load the trained encoder
                        encoder.to(args.device)  # Move it to the correct device
                        encoder.eval()  # Ensure it's in evaluation mode

                        encoded_tokens = encoder(data)
                        print("token shape:", encoded_tokens.shape )
                        batch_size, channels, seg_len, num_kp, _ = encoded_tokens.shape  # Example
                        encoded_tokens = encoded_tokens.view(batch_size, seg_len, -1)  # Reshape to (batch, seq_len, features)
                        #Flattened feature dimension (C × num_kp × _)
                        data = encoded_tokens  # Replace input data with extracted tokens


                    elif args.token_config == "CTR":
                        if num_kp < 20:
                            padding_kps = 20 - num_kp  # Number of joints to add (should be 2)
                            padding = torch.zeros((batch_size, channels, seg_len, padding_kps), device=data.device)
                            data = torch.cat((data, padding), dim=-1)  # Add zero joints at the end
                    
                        checkpoint_path =args.CTR_path  # Set your checkpoint path
                        CTR = load_trained_CTR(checkpoint_path, args)  # Load the trained encoder
                        CTR.to(args.device)  # Move it to the correct device
                        CTR.eval()  # Ensure it's in evaluation mode

                        encoded_tokens_CTR = CTR(data)
                        print("token shape:", encoded_tokens_CTR.shape )
                        batch_size, channels, seg_len, num_kp= encoded_tokens_CTR.shape  # Example
                        encoded_tokens_CTR = encoded_tokens_CTR.view(batch_size, seg_len, -1)  # Reshape to (batch, seq_len, features)
                        #Flattened feature dimension (C × num_kp × _)
                        data = encoded_tokens_CTR  # Replace input data with extracted tokens


                    elif args.token_config == "pst": 
                        data = data.view(batch_size, seg_len, channels*num_kp)
                    elif args.token_config == "kps":
                        if num_kp == 36:
                            absolute = data[:, :, :, 0:18]
                            rel = data[:, :, :, 18:]
                            data = torch.cat([absolute, rel], dim=1)
                            data = data.view(batch_size, -1, 18)
                            data = data.permute(0, 2, 1)
                        elif num_kp == 38:
                            absolute = data[:, :, :, 0:19]
                            rel = data[:, :, :, 19:]
                            data = torch.cat([absolute, rel], dim=1)
                            data = data.view(batch_size, -1, 19)
                            data = data.permute(0, 2, 1)
                        else:
                            data = data.view(batch_size, -1, num_kp)
                            data = data.permute(0, 2, 1)
                    elif args.token_config == "2ds":
                        if num_kp == 36:
                            absolute = data[:, :, :, 0:18]
                            rel = data[:, :, :, 18:]
                            absolute = absolute.permute(0,2,1,3)
                            absolute = absolute.reshape(batch_size,seg_len,-1)
                            absolute = absolute.permute(0, 2, 1)
                            rel = rel.permute(0,2,1,3)
                            rel = rel.reshape(batch_size,seg_len,-1)
                            rel = rel.permute(0, 2, 1)
                            data = torch.cat([absolute, rel], dim=2)
                        elif num_kp == 38:
                            absolute = data[:, :, :, 0:19]
                            rel = data[:, :, :, 19:]
                            absolute = absolute.permute(0,2,1,3)
                            absolute = absolute.reshape(batch_size,seg_len,-1)
                            absolute = absolute.permute(0, 2, 1)
                            rel = rel.permute(0,2,1,3)
                            rel = rel.reshape(batch_size,seg_len,-1)
                            rel = rel.permute(0, 2, 1)
                            data = torch.cat([absolute, rel], dim=2)    
                        else:
                            data = data.permute(0,2,1,3)
                            data = data.reshape(batch_size,seg_len,-1)
                            data = data.permute(0, 2, 1)
                    elif args.token_config == "st":
                        if num_kp == 36:
                            absolute = data[:, :, :, 0:18]
                            rel = data[:, :, :, 18:]
                            absolute = absolute.permute(0, 1, 3, 2)
                            absolute = absolute.reshape(batch_size,channels,-1)
                            absolute = absolute.permute(0, 2, 1)
                            rel = rel.permute(0, 1, 3, 2)
                            rel = rel.reshape(batch_size,channels,-1)
                            rel = rel.permute(0, 2, 1)
                            data = torch.cat([absolute, rel], dim=2)
                        elif num_kp == 38:
                            absolute = data[:, :, :, 0:19]
                            rel = data[:, :, :, 19:]
                            absolute = absolute.permute(0, 1, 3, 2)
                            absolute = absolute.reshape(batch_size,channels,-1)
                            absolute = absolute.permute(0, 2, 1)
                            rel = rel.permute(0, 1, 3, 2)
                            rel = rel.reshape(batch_size,channels,-1)
                            rel = rel.permute(0, 2, 1)
                            data = torch.cat([absolute, rel], dim=2)
                            
                        else:
                            data = data.permute(0, 1, 3, 2)
                            data = data.reshape(batch_size,channels,-1)
                            data = data.permute(0, 2, 1)
                        # data = data.view(batch_size, seg_len, channels*num_kp)
                    recon = self.model.forward(data, data)
                    loss = self.loss.calculate(data, recon)
                    # mse = self.model.mse(data)
                    # eval_loss.extend(mse.cpu().numpy())
                    eval_loss.extend(loss.cpu().numpy())

        #             if epoch_loss < best_loss:  # Save the model if it has the lowest loss
        #                 best_loss = epoch_loss
        #                 best_metrics = {
        #                 "epoch": epoch,
        #                 "auc_roc": auc_roc,
        #                 "auc_pr": auc_pr,
        #                 "eer": eer,
        #                 "eer_th": eer_th,
        #                 "10er": ten_er,
        #                 "10er_th": ten_er_th,
        #                 "model_path": args.model_save_dir,
        #                 "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #                 }


        #                 checkpoint_filename = self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
        #                 print(f"Model saved at epoch {epoch} with loss {best_loss:.6f}")
                
        #         # test_loss /= len(test_loader.dataset)
        #         # print("--> Test set loss {:.7f}".format(test_loss))

        #         writer.close()
                
                    
        #     best_model_file_path = os.path.join(os.path.split(args.model_save_dir)[0], "best_model_log.csv")

        # # Check if file exists
        #     file_exists = os.path.exists(best_model_file_path)

        # # Define correct field names
        #     fieldnames = ["epoch", "auc_ro", "auc_pr", "eer", "eer_th", "10er", "10er_th", "model_path", "timestamp"]

        # # Open file for appending or creating
        #     with open(best_model_file_path, "a" if file_exists else "w", newline="") as file:
        #     #fieldnames = ["epoch", "auc_roc", "auc_pr", "eer", "eer_th", "10er", "10er_th", "model_path", "timestamp"]
        #         csv_writer = csv.DictWriter(file, fieldnames=fieldnames)

        #     # Write header only if file is new
        #         if not file_exists:
        #             csv_writer.writeheader()

        #     # Write metrics as a new row
        #         csv_writer.writerow({
        #             "epoch": epoch, 
        #             "val_loss": best_loss,  # Store the best validation loss
        #             "model_path": args.model_save_dir,
        #             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # })

        #     return checkpoint_filename

            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), self.test_loader.dataset.metadata, args=args)
            writer.add_scalar('AUC ROC', auc_roc, epoch)
            writer.add_scalar('AUC PR', auc_pr, epoch)
            writer.add_scalar('EER', eer, epoch)
            writer.add_scalar('EER TH', eer_th, epoch)
            writer.add_scalar('10ER', fpr_at_target_fnr, epoch)
            writer.add_scalar('10ER TH', threshold_at_target_fnr, epoch)
            
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
            if auc_roc > best_roc:
                best_roc = auc_roc
                
                self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
                print(f"Model saved at epoch {epoch} with AUC ROC: {best_roc:.6f}")
                
        # writer.close()
        
        # if os.path.exists(os.path.split(args.model_save_dir)[0]+"/"+"roc.csv"):
        #     # Open the existing CSV file for appending
        #     with open(os.path.split(args.model_save_dir)[0]+"/"+"roc.csv", "a", newline="") as file:
        #         writer = csv.writer(file)
        #         # Append a new row with the AUC-ROC value and path
        #         writer.writerow([best_roc, args.model_save_dir])
        # else:
        #     # Create a new CSV file "roc.csv" and write the header
        #     with open(os.path.split(args.model_save_dir)[0]+"/"+"roc.csv", "w", newline="") as file:
        #         writer = csv.writer(file)
        #         # Write the header row
        #         writer.writerow(["auc_roc", "model_path"])
        #         # Write the first data row
        #         writer.writerow([best_roc, args.model_save_dir])
        # return checkpoint_filename

        # -------------------- CSV File to Store Best Metrics --------------------
                csv_path = os.path.join(os.path.split(args.model_save_dir)[0], "roc_pr.csv")  # Combined ROC & PR CSV file

        # Check if CSV file exists to decide on writing header
                file_exists = os.path.exists(csv_path)

        # Open the CSV file for appending or creating
                with open(csv_path, "a" if file_exists else "w", newline="") as file:
                    csv_writer = csv.writer(file)

            # Write header if file is new
                    if not file_exists:
                        csv_writer.writerow([
                            "epoch", 
                            "auc_roc", 
                            "auc_pr", 
                            "eer", 
                            "eer_th", 
                            "10er", 
                            "10er_th", 
                            "model_path"
                            ])

                # Append the latest best metrics row
                    csv_writer.writerow([epoch,
                        best_roc,
                        auc_pr,
                        eer,
                        eer_th,
                        fpr_at_target_fnr,
                        threshold_at_target_fnr,
                        args.model_save_dir])

        # -------------------- Finalize --------------------
        

        return checkpoint_filename  # Return the latest checkpoint path if saved


         

                




# Load the trained GCAE encoder
def load_trained_encoder(encoder_path, args):
    """
    Load the trained GCAE model and extract only the encoder part.
    """
    model = GCAE(
        in_channels=2,
        h_dim=args.h_d,     #8,
        input_frames=args.seg_len,  #24,
        dropout=0.1,
        conv_oper="sagc"
    ).to(args.device)

    #checkpoint_encoder = torch.load(encoder_path, map_location=args.device,  weights_only=False)
    #model.load_state_dict(checkpoint_encoder['model_state_dict'])
    #model.eval()  # Set to evaluation mode

    
    checkpoint = torch.load(encoder_path, map_location=args.device,weights_only=False)
    print(f"[DEBUG] Checkpoint keys: {checkpoint.keys()}")

    # Extract only model weights
    weights = checkpoint['state_dict']

    # ✅ Ensure compatibility with DataParallel
    weights = OrderedDict([[k.split('module.')[-1], v.to(args.device)] for k, v in weights.items()])
    model.load_state_dict(weights)
    #model = model.cuda()
    model.eval()

    #model.load_state_dict(checkpoint_encoder)  


    #checkpoint_encoder = torch.load(encoder_path, map_location=args.device)
    #model.load_state_dict(checkpoint_encoder['state_dict'])
    

    encoder = Encoder_T(model)
    return encoder



def load_trained_CTR(CTR_path, args):
    """
    Load the trained GCAE model and extract only the encoder part.
    """
    config_path = "/home/nrashvan/TCVAE_modified/CTR/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Initialize the Model
    graph_args = config["model_args"]["graph_args"]
    model = Model(
        num_class=config["model_args"]["num_class"],
        num_point=config["model_args"]["num_point"],   
        num_person=config["model_args"]["num_person"],
        in_channels=2,  # Adjust if needed
        graph=config["model_args"]["graph"],
        graph_args=graph_args,
    ).to(device)
    weights_path="/home/nrashvan/TCVAE_modified/CTR/runs-59-18762.pt"

    weights = torch.load(CTR_path, map_location=args.device)
    weights = OrderedDict([[k.split('module.')[-1], v.to(device)] for k, v in weights.items()])
    #model.load_state_dict(checkpoint_path_CTR['state_dict'])
    model.load_state_dict(weights)
    model = model.cuda()
    model.eval()  # Set to evaluation mode

    encoder_CTR = Encoder_CTR(model) 
    return encoder_CTR


def init_optimizer(type_str, **kwargs):
    if type_str.lower() == 'adam':
        opt_f = optim.Adam
    else:
        return None

    return partial(opt_f, **kwargs)


def init_scheduler(type_str, lr, epochs, warmup=3):
    sched_f = None
    if type_str.lower() == 'exp_decay':
        sched_f = None
    elif type_str.lower() == 'cosine':
        sched_f = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=epochs)
    elif type_str.lower() == 'cosine_warmup':
        sched_f = partial(CosineAnnealingWarmUpRestarts, T_0=epochs, T_up=warmup)
    elif type_str.lower() == 'cosine_delayed':
        sched_f = partial(DelayedCosineAnnealingLR, delay_epochs=warmup,
                          cosine_annealing_epochs=epochs)
    elif (type_str.lower() == 'tri') and (epochs >= 8):
        sched_f = partial(optim.lr_scheduler.CyclicLR,
                          base_lr=lr, max_lr=lr,
                          step_size_up=epochs//8,
                          mode='triangular2',
                          cycle_momentum=False)
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return sched_f