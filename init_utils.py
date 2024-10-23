import torch
import math
from data_utils import generate_Cifar, generate_Mnist, generate_imgnet1k, generate_glue
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
import argparse
import warnings
import timm
import os
from datetime import datetime
try:
    import wandb
    HAS_WANDB = True
except ImportError as e:
    HAS_WANDB = False
from model_utils import LinearModel, CNN5, create_roberta, WideResNet

def base_parse_args(parser):
    # Task arguments
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--tag', default = '', type=str, help='log file tag')
    parser.add_argument('--log_type', default='file',type=str, help='log type (file, wandb)')
    parser.add_argument('--log_path', default = './log', type=str, help='log file path')
    parser.add_argument('--log_freq', default=-1, type=int, help='log frequency during training')
    parser.add_argument('--load_path', default=None, type=str, help='load checkpoint if specified')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint is specified')
    parser.add_argument('--save_freq', default=999, type=int, help='checkpoint saving frequency')

    parser.add_argument('--data', default='cifar100', type=str, help='dataset (cifar10, cifar100)')
    parser.add_argument('--data_path', default='../data', type=str, help='dataset path')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--mnbs', default=32, type=int, help='mini batch size')
    parser.add_argument('--model', default = 'vit_small_patch16_224', type=str, help='trained model')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained weights')

    # Algorithm parameters
    parser.add_argument('--algo', default='sgd', type=str, help='algorithm (sgd/adam)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate list')
    parser.add_argument('--beta', default=0.999, type=float, help='beta for adam')
    parser.add_argument('--epoch', default=3, type=int,help='number of public epochs')
    parser.add_argument('--scheduler',action="store_true" ,help='use 1 cycle lr scheduler')
    parser.add_argument('--kf',action="store_true" ,help='use kalman filter')
    parser.add_argument('--record',action="store_true" ,help='record gamma for KFAdamBC')
    parser.add_argument('--kappa', default=1.5, type=float, help='sigma ratio')
    parser.add_argument('--gamma', default=-1, type=float, help='perturbation stepsize, default: -1 (dynamic according to k_t)')

    # DP parameters
    parser.add_argument('--clipping', action="store_true", help="use gradient clipping")
    parser.add_argument('--noise', default=0, type=float, help='add dp noise, 0: no noise, -1: dp noise by epsilon, >0: manual noise')
    parser.add_argument('--epsilon', default=8, type=float, help='dp privacy, must be larger than 0, used when noise is not specified')
    parser.add_argument('--clipping_norm', default=-1,  type=float, help='clipping style, <=0: automatic, >0: Abadi')
    parser.add_argument('--clipping_style', default='all-layer', type=str, help='clipping style, all-layer, layer-wise, param-wise')
    return parser

def lp_parse_args(parser):
    parser = base_parse_args(parser)
    # LPSGD parameter
    parser.add_argument('--coef_file', default='./coefs/2.csv', type=str, help='coefficients')
    return parser

def galore_parse_args(parser):
    parser = lp_parse_args(parser)
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    return parser
    
def task_init(args):
    device = torch.device('cuda:'+str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
    if args.clipping_norm <=0:
        args.clipping_fn = 'automatic'
        args.clipping_norm = 1
    else:
        args.clipping_fn = 'Abadi'
    model = None
    if args.data == 'cifar10':
        num_classes = 10
        train_dl, test_dl = generate_Cifar(args.mnbs, args.data, args.model, args.data_path)
        sample_size = 50000
    elif args.data == 'cifar100':
        num_classes = 100
        # model = timm.create_model(args.model, pretrained=args.pretrained, num_classes = 100)
        train_dl, test_dl = generate_Cifar(args.mnbs, args.data, args.model, args.data_path)
        sample_size = 50000
    elif args.data == 'imgnet1k':
        num_classes = 1000
        train_dl, test_dl = generate_imgnet1k(args.mnbs, args.data_path)
        temp_train_dl = train_dl()
        sample_size = len(temp_train_dl.dataset)
        del temp_train_dl
    elif args.data == 'mnist':
        model = LinearModel(28*28, 10)
        train_dl, test_dl = generate_Mnist(args.mnbs, args.data_path)
        sample_size = 60000
    if model is None:
        if args.model == 'cnn5':
            model = CNN5(num_classes = num_classes, normalization= True)
        elif args.model == 'wrn_40_4':
            model = WideResNet(depth = 40, num_classes=num_classes, widen_factor=4)
        elif args.model == 'wrn_16_4':
            model = WideResNet(depth = 16, num_classes=num_classes, widen_factor=4)
        elif args.model == 'wrn_28_4':
            model = WideResNet(depth = 28, num_classes=num_classes, widen_factor=4)
        else:
            model = timm.create_model(args.model, pretrained=args.pretrained, num_classes = num_classes)
    model = ModuleValidator.fix(model)
    model.to(device)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict = True)
        # optimizer.load_state_dict(state_dicts['optimizer'])
    
    if args.noise < 0 :
        noise = get_noise_multiplier(target_delta=1.0/(sample_size)**1.1, target_epsilon=args.epsilon, sample_rate=args.bs/sample_size, epochs=args.epoch, accountant='prv')
    else:
        noise = args.noise
    if args.kf:
        c = (1-args.kappa)/(args.gamma*args.kappa)
        norm_factor = math.sqrt(c**2 + (1-c)**2)
        noise = noise/norm_factor
    acc_step = args.bs//args.mnbs
    return train_dl, test_dl, model, device, sample_size, acc_step, noise

def nlp_task_init(args):
    device = torch.device('cuda:'+str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
    if args.clipping_norm <=0:
        args.clipping_fn = 'automatic'
        args.clipping_norm = 1
    else:
        args.clipping_fn = 'Abadi'
    model = None
    train_dl, test_dl, tokenizer, label_to_id, num_labels = generate_glue(args.data, data_path=args.data_path, batch_size=args.mnbs)
    model = create_roberta(label_to_id, num_labels)
    model = ModuleValidator.fix(model)
    model.to(device)
    sample_size = len(train_dl.dataset)
    print(sample_size)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict = True)
        # optimizer.load_state_dict(state_dicts['optimizer'])
    
    if args.noise < 0 :
        noise = get_noise_multiplier(target_delta=1.0/(sample_size)**1.1, target_epsilon=args.epsilon, sample_rate=args.bs/sample_size, epochs=args.epoch)
    else:
        noise = args.noise
    acc_step = args.bs//args.mnbs
    
    return train_dl, test_dl, model, device, sample_size, acc_step, noise, tokenizer

def logger_init(args, noise, steps_per_epoch, type = 'file'):
    if type == 'file' or not HAS_WANDB:
        if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path)
        # if not os.path.isdir(args.log_path+'/G'):
        #     os.makedirs(args.log_path+'/G')
        # datetime object containing current date and time
        log_file_path = '%s/%s'%(args.log_path,args.tag)
        if hasattr(args, 'coef_file'):
            log_file = file_logger(log_file_path, 2, ["acc","loss"], steps_per_epoch, heading = "Data=%s, Model=%s, E=%d, B=%d, lr=%-.6f, sigma=%-.6f, coef=%s"%(args.data, args.model, args.epoch, args.bs, args.lr, noise, args.coef_file))
        else:
            log_file = file_logger(log_file_path, 2, ["acc","loss"], steps_per_epoch, heading = "Data=%s, Model=%s, E=%d, B=%d, lr=%-.6f, sigma=%-.6f"%(args.data, args.model, args.epoch, args.bs, args.lr, noise))
        return log_file
    elif type == 'wandb' and HAS_WANDB:
        log_wanb = wanb_logger(args, noise, steps_per_epoch)
        return log_wanb
    else:
        raise RuntimeError('incorrect logger')
    

class file_logger():
    def __init__(self, path, time_num, item_list, steps_per_epoch, heading = None):
        head = ['time_'+str(i) for i in range(time_num)]
        head_str = ','.join(head)+','+','.join(item_list)
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        self.train_path = path+'_train'+dt_string+'.csv'
        self.test_path = path+'_test'+dt_string+'.csv'
        self.epoch_per_step = 1.0/steps_per_epoch
        self.time_num = time_num
        self.item_length = len(item_list)
        with open(self.train_path,'a') as fp:
            if heading is not None:
                print(heading, file=fp)
            print(head_str, file=fp)
        with open(self.test_path,'a') as fp:
            if heading is not None:
                print(heading, file=fp)
            print(head_str, file=fp)
    
    def update(self, time_list, item_list):
        if len(time_list)!=self.time_num:
            raise RuntimeError('incorrect log time information')
        
        if time_list[1] == -1:
            # test log
            log_info = str(time_list[0])+','+','.join(map(str,item_list))
            with open(self.test_path,'a') as fp:
                print(log_info, file=fp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # train log
            log_info = str(time_list[0]+time_list[1]*self.epoch_per_step)+','+','.join(map(str,item_list))
            with open(self.train_path,'a') as fp:
                print(log_info, file=fp)
        

class wanb_logger():
    def __init__(self, args, noise, steps_per_epoch):
        self.epoch_per_step = 1.0/steps_per_epoch
        tag = args.tag+'_'+args.data+'_'+str(args.epsilon)+'_'+str(args.lr)+'_'+str(args.bs)+'_'+str(args.epoch)
        run_config = dict(vars(args))
        run_config.update({
            "noise": noise,
            "tag": tag,
        })
        wandb.init(
            project='DPLPF',
            entity='entity-name',
            name=tag
        )
        wandb.config.update(run_config, allow_val_change=True)
    def update(self, time_list, item_list):
        if len(time_list)!=2:
            raise RuntimeError('incorrect log time information')
        if time_list[1] == -1:
            # test log
            wandb.log({
                "test_epoch": time_list[0],
                "test_acc": item_list[0],
                "test_loss": item_list[1],
            })
        else:
            # train log
            wandb.log({
                "train_epoch": time_list[0]+time_list[1]*self.epoch_per_step,
                "train_acc": item_list[0],
                "train_loss": item_list[1],
            })