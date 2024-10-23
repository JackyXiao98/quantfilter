import torch
import math
from KFOptimizer import  KFOptimizer
from train_utils import train_nlp, test_nlp
from init_utils import base_parse_args, logger_init, nlp_task_init
from fastDP import PrivacyEngine
from AdamBC import AdamBC
import argparse
import warnings
import gc

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='LP DPSGD')
    parser = base_parse_args(parser)
    args = parser.parse_args()
    train_dl, test_dl, model, device, sample_size, acc_step, noise, tokenizer = nlp_task_init(args)
    log_file = logger_init(args, noise, sample_size//args.mnbs,type=args.log_type)

    use_manual_noise = not args.clipping and noise>0
    if use_manual_noise:
        noise = noise/args.mnbs
        args.lr = args.lr/acc_step
        print('use manual noise')

    if args.algo == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0)
    elif args.algo == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.algo == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.algo == 'adambc':
        optimizer = AdamBC(model.parameters(), lr=args.lr, dp_batch_size=args.bs, dp_l2_norm_clip=1, dp_noise_multiplier=noise, eps_root=1e-8)
    else:
        print(args.algo)
        raise RuntimeError("Unknown Algorithm!")
    
    start = 0
    
    if args.load_path is not None:
        print("loading optimizer")
        checkpoint = torch.load(args.load_path, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
    if args.scheduler:
        from train_utils import CosineAnnealingWarmupRestarts
        lrscheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, first_cycle_steps= sample_size//args.bs * args.epoch, warmup_steps= (sample_size*args.epoch)//(args.bs*20), last_epoch = start*sample_size//args.bs-1)
    else:
        lrscheduler = None
        
    if args.kf:
        optimizer = KFOptimizer(model.parameters(), optimizer=optimizer, kappa=args.kappa, gamma=args.gamma)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if args.clipping:
        privacy_engine = PrivacyEngine(model, noise_multiplier=noise, numerical_stability_constant=1e-3, grad_accum_steps = acc_step, sample_size= sample_size, batch_size=args.bs, epochs= args.epoch, per_sample_clip=args.clipping, torch_seed_is_fixed=False, clipping_fn=args.clipping_fn, clipping_style=args.clipping_style, max_grad_norm=args.clipping_norm)
        privacy_engine.attach(optimizer)

    for E in range(args.epoch):
        train_nlp(model, train_dl, optimizer, criterion, log_file, device = device, epoch = E, log_frequency = args.log_freq, acc_step = acc_step, lr_scheduler=lrscheduler)
        test_nlp(model, test_dl, criterion, log_file, device = device, epoch = E)
        gc.collect()
        torch.cuda.empty_cache()
        if args.save_freq >0 and E % args.save_freq == 0 and args.save_path is not None:
            if args.kf:
                torch.save({'model':model.state_dict(),'kf_optimizer':optimizer.state_dict(), 'optimizer':optimizer.optimizer.state_dict(), 'epoch':E}, args.save_path)
            else:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':E}, args.save_path)
        
        
