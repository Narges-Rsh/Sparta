from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import csv

from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler, CostumLoss
from utils.data_utils import trans_list
from tqdm import tqdm
from utils.eval2 import score_dataset
import yaml
import os
from models import TransformerPredictor
#from models import GraphEncoder
from torch.utils.tensorboard import SummaryWriter

#This function runs the entire workflow, from model training to evaluation
def main ():
    print('sedaye mano darid az Chalotte America!')
    parser = init_parser()
    args = parser.parse_args()
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)
    
    pretrained_model = vars(args).get('model_ckpt_dir', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))
    # model_args = init_model_params(args, dataset)
    expand_ratio = 1
    extra_dim = 0
    if args.relative:
        expand_ratio = 2
    if model_args.token_config == "kps":  #KS-PRP
        input_dim = model_args.seg_len*2     
    elif model_args.token_config == "2ds":   #FS-PRP
        input_dim = model_args.seg_len       
    elif model_args.token_config == "st":    #ST-PRP
        input_dim = 2          
    else:
        if args.traj:
            if args.relative:      
                extra_dim = 4      
            else:
                extra_dim = 2
        input_dim = args.num_kp*2

    
    
    
    
    Transformer_model = TransformerPredictor(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
                                            #(d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
       
    transformer_param = sum(p.numel() for p in Transformer_model.parameters() if p.requires_grad) + sum(p.numel() for p in Transformer_model.decoder.parameters() if p.requires_grad)
    if pretrained_model == None:  
        if not os.path.exists(args.model_save_dir):
        # Create the directory
            os.makedirs(args.model_save_dir)
        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, Transformer_model, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        
        trained_model = trainer.train(checkpoint_filename='trans', args=args)
        
    else:   
        loss_func = CostumLoss(model_args.loss, a=model_args.a, b=model_args.b, c=model_args.c, d=model_args.d)
        checkpoint = torch.load(args.model_ckpt_dir)
        Transformer_model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        Transformer_model.to(args.device)
        eval_loss = []
        dataset_size = len(loader['test'].dataset)
        Transformer_model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(loader['test'])):
                data = data_batch[0].to(args.device, non_blocking=True)
                data = data[:,0:2, :, :].to(torch.float32)
                batch_size, channels, seg_len, num_kp = data.shape
                if args.token_config == "t": 
                    data = data.permute(0, 2, 1, 3)
                    data = data.reshape(batch_size, seg_len, channels*num_kp)
                elif args.token_config == "pst":   #
                    data = data.view(batch_size, seg_len, channels*num_kp)
                elif args.token_config == "kps":  #KS-PRP
                    if num_kp == 36:
                        absolute = data[:, :, :, 0:18]
                        rel = data[:, :, :, 18:]
                        data = torch.cat([absolute, rel], dim=1)
                        data = data.view(batch_size, -1, 18)
                        data = data.permute(0, 2, 1)
                    else:
                        data = data.view(batch_size, -1, 18)
                        data = data.permute(0, 2, 1)
                elif args.token_config == "2ds":    #FS-PRP
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
                        
                    else:
                        data = data.permute(0,2,1,3)
                        data = data.reshape(batch_size,seg_len,-1)
                        data = data.permute(0, 2, 1)
                elif args.token_config == "st":   #??
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
                        
                    else:
                        data = data.permute(0, 1, 3, 2)
                        data = data.reshape(batch_size,channels,-1)
                        data = data.permute(0, 2, 1)
                # data = data.view(batch_size, seg_len, channels*num_kp)
                # mse = Transformer_model.mse(data)
                # eval_loss.extend(mse.cpu().numpy())
                recon = Transformer_model.forward(data, data)
                loss = loss_func.calculate(data, recon)
                eval_loss.extend(loss.cpu().numpy())
        
        auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args)
        print('AUC ROC: {}'.format(auc_roc))
        print('AUC PR: {}'.format(auc_pr))
        print('EER: {}'.format(eer))
        print('EER TH: {}'.format(eer_th))
        print('10ER: {}'.format(fpr_at_target_fnr))
        print('10ER TH: {}'.format(threshold_at_target_fnr))

    
if __name__ == '__main__':    
    main()