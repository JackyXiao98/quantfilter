import torch
import math
from KFOptimizer import KFOptimizer
from train_utils import test, train_zo
from init_utils import base_parse_args, task_init, logger_init
# from fastDP import PrivacyEngine
import argparse
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='LP DPSGD')
    parser = base_parse_args(parser)
    args = parser.parse_args()
    train_dl, test_dl, model, device, sample_size, acc_step, noise = task_init(args)
    log_file = logger_init(args, noise, sample_size//args.mnbs,type=args.log_type)

    if args.algo == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0)
    elif args.algo == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.algo == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        print(args.algo)
        raise RuntimeError("Unknown Algorithm!")
    
    # from torch.optim import lr_scheduler
    if args.scheduler:
        from train_utils import CosineAnnealingWarmupRestarts
        lrscheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, first_cycle_steps= sample_size//args.bs * args.epoch, warmup_steps= (sample_size*args.epoch)//(args.bs*20))
    else:
        lrscheduler = None
        
    if args.kf:
        optimizer = KFOptimizer(model.parameters(), optimizer=optimizer, kappa=args.kappa, gamma=args.gamma)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    for E in range(args.epoch):
        # if args.no_record:
        train_zo(model, train_dl, optimizer, criterion, log_file, device = device, epoch = E, log_frequency = args.log_freq, acc_step = acc_step, lr_scheduler=lrscheduler)
        test(model, test_dl, criterion, log_file, device = device, epoch = E)
        
        
