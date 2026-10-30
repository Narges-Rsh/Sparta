import json
import os
import torch
import torch.optim as optim
import shutil
import time
from tqdm import tqdm
from datetime import datetime

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
from Sparta.models import *

def init_model_params(args, dataset):
    return {      
        'pose_shape': dataset["test"][0][0].shape if args.model_confidence else dataset["test"][0][0][:2].shape,
        'pose_shape': dataset["test"][0][0].shape,
        'hidden_dim': args.model_latent_dim,
        'actnorm_scale': 1.0,
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'device': args.device,
        'model_dist': 'normal'
    }


def dump_args(args, ckpt_dir):     #save model arguments to a json file for reproducibility
    path = os.path.join(ckpt_dir, "args.json")
    data = vars(args)
    with open(path, 'w') as fp:
        json.dump(data, fp)


def calc_reg_loss(model, reg_type='l2', avg=True):  
#Computes regularization loss for a model's parameters to prevent overfitting.
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
    def calculate(self, batch1, batch2): #Custom loss function class for flexibility in loss computations.
        if self.loss_type == 'mse':   #Mean squared error
            return self.mse_loss(batch1, batch2)  
        # elif self.loss_type == 'emd':
        #     return self.emd_loss(batch1, batch2)
        elif self.loss_type == 'ssim':   #Structural similarity loss for images
            return self.ssim_loss(batch1, batch2)
        elif self.loss_type == 'spec':   # Spectral distance loss using adjacency matrices
            return self.spectral_distance_loss(batch1, batch2)
        elif self.loss_type == 'cos':   #Cosine similarity loss
            return self.cosine_similarity_loss(batch1, batch2)
        elif self.loss_type == 'combined':   #Weighted combination of multiple loss types
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

    def gaussian_kernel(slef,size, sigma):   #??
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
        # self.expand_ratio = 1     #I commented this for graph
        # if args.relative:
        #     self.expand_ratio = 2   
        
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

        best_loss = float('inf') 
        for epoch in range(start_epoch, num_epochs):
            running_loss = 0
            #loss_list = []
            #ep_start_time = time.time()
            print("Started epoch {}".format(epoch))
            self.model.train()
            for itern, data_arr in enumerate(tqdm(self.train_loader)): # what are the data_arr[0],[1],[2],[3]
                data = data_arr[0].to(args.device, non_blocking=True)     
                
                data = data[:,0:2, :, :].to(torch.float32)   #([512, 2, 24, 18])
                
                batch_size, channels, seg_len, num_kp = data.shape  #Original: [batch_size, channels, seg_len, num_kp].
                    
                #self.optimizer.zero_grad()    
                
                #for the transformer########
                # recon = self.model.forward(data, data)
                # loss = self.loss.calculate(data, recon)
                # loss.mean().backward()
                # self.optimizer.step()

                self.optimizer.zero_grad()  
                reco_data=self.model(data) #I addeeeeeeed
                reco_loss = self.loss.calculate(data, reco_data)
                
                reg_loss = calc_reg_loss(self.model)
                loss = reco_loss + 1e-3 * 1e-3*  reg_loss          #args.alpha
                loss.mean().backward()  #instead of of loss.backward()
                self.optimizer.step()
                running_loss = running_loss + loss.sum()
                #loss_list.append(loss.item())

            epoch_loss = running_loss/len(self.train_loader.dataset)
            print('[Epoch %03d]  \t training loss: %.10f ' % (
                epoch, epoch_loss))
            writer.add_scalar('Loss', epoch_loss, epoch)
            new_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self.adjust_lr(epoch, new_lr)
            print('lr: {0:.3e}'.format(new_lr))

                
            eval_loss = []
            self.model.eval()
            print("starting the test")
            with torch.no_grad():
                for i, data_batch in enumerate(tqdm(self.test_loader)):
                    data = data_batch[0].to(args.device, non_blocking=True)
                    data = data[:,0:2, :, :].to(torch.float32)
                    batch_size, channels, seg_len, num_kp = data.shape
                    
                reco_data=self.model(data) #I addeeeeeeed
                reco_loss = self.loss.calculate(data, reco_data)
                eval_loss.extend(reco_loss.cpu().numpy())
                #eval_loss.append(reco_loss.mean().item()) 

                if epoch_loss < best_loss:  # Save the model if it has the lowest loss
                    best_loss = epoch_loss
                    self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
                    print(f"Model saved at epoch {epoch} with loss {best_loss:.6f}")
                
                # test_loss /= len(test_loader.dataset)
                # print("--> Test set loss {:.7f}".format(test_loss))

            writer.close()
                
                    
        best_model_file_path = os.path.join(os.path.split(args.model_save_dir)[0], "best_model_log.csv")

        # Check if file exists
        file_exists = os.path.exists(best_model_file_path)

        # Define correct field names
        fieldnames = ["epoch", "val_loss", "model_path", "timestamp"]

        # Open file for appending or creating
        with open(best_model_file_path, "a" if file_exists else "w", newline="") as file:
            #fieldnames = ["epoch", "auc_roc", "auc_pr", "eer", "eer_th", "10er", "10er_th", "model_path", "timestamp"]
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if file is new
            if not file_exists:
                csv_writer.writeheader()

            # Write metrics as a new row
            csv_writer.writerow({
                "epoch": epoch, 
                "val_loss": best_loss,  # Store the best validation loss
                "model_path": args.model_save_dir,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

        return checkpoint_filename



        
        
    
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

