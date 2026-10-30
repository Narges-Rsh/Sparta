import os
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

#from utils.train_utils import calc_reg_loss
#from models.dc_gcae.dc_gcae_training import adjust_lr

class Trainer:
    def __init__(self, args, model, loss, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  # For checkpoint filename
        self.loss = loss

        # Optimizer
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())

        # Scheduler
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.lr, self.args.lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'
        state['args'] = args
        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = os.path.join(self.args.ckpt_dir, filename)
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']})")
        except FileNotFoundError:
            print(f"No checkpoint found at '{filename}'. Skipping...")

    def train(self, log=True, checkpoint_filename=None, args=None, num_epochs=10):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'

        start_epoch = self.args.start_epoch
        best_loss = float('inf')

        self.model.train()
        self.model = self.model.to(args.device)

        for epoch in range(start_epoch, num_epochs):
            loss_list = []
            ep_start_time = time.time()
            print(f"Started epoch {epoch}")

            # **Training Loop**
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:, 0:2, :, :].to(torch.float32)

                self.optimizer.zero_grad()
                reco_data = self.model(data)
                reco_loss = self.loss(data, reco_data)
                reg_loss = calc_reg_loss(self.model)
                loss = reco_loss + 1e-3 * args.alpha * reg_loss

                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())

            epoch_loss = np.mean(loss_list)
            print(f"Epoch {epoch} done, loss: {epoch_loss:.7f}, took: {time.time()-ep_start_time:.3f} sec")

            # Adjust learning rate
            new_lr = self.adjust_lr(epoch)
            print(f'lr: {new_lr:.3e}')

            # **Evaluation Loop**
            eval_loss = self.evaluate(args)

            # Save best model based on validation loss
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)
                print(f"Best model saved at epoch {epoch} with validation loss {best_loss:.6f}")

        return checkpoint_filename, best_loss

    def evaluate(self, args):
        """ Evaluation function using `torch.no_grad()` """
        print("Evaluating...")
        self.model.eval()
        eval_loss = []

        with torch.no_grad():
            for data_batch in tqdm(self.test_loader):
                data = data_batch[0].to(args.device, non_blocking=True)
                data = data[:, 0:2, :, :].to(torch.float32)

                reco_data = self.model(data)
                reco_loss = self.loss(data, reco_data)

                eval_loss.append(reco_loss.item())

        avg_loss = np.mean(eval_loss)
        print(f"--> Validation loss: {avg_loss:.7f}")
        self.model.train()  # Switch back to training mode
        return avg_loss

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state
    




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