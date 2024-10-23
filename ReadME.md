# Doppler+ (Kalman filter for DP optimizers)

Key requirements:
* deepspeed
* fast-differential-privacy
* pytorch
* opacus
* wandb (optional)
* torchvision
* timm
* transformers
* accelerate
* sentence-transformers
* fire
* ml_swissknife

Provides KFOptimizer class:
```python
kfopt = KFOptimizer(params, optimizer,kappa, gamma) # initalize the optimizer
optimizer.prestep(closure) # loss computation and backward propagation
optimizer.step() # optimizer update
```

## Example script: 

```bash
TAG="Tag_of_task"
EPS=8
python ./run_KFSGD.py \
    --tag ${TAG} --log_type file --log_freq 10 \
    --bs 1000 --mnbs 50 --data cifar100 \
    --algo sgd --lr 0.2 --epoch 120 \
    --clipping --noise -1 --epsion ${EPS} \
    --kf --kappa 0.5 --gamma 0.7
```

## Arguments:
```bash
  --cuda        int     cuda device
  --tag         str     task name
  --log_type    str     log type (file, wandb)
  --log_path    str     log file path (used when log_type=file)
  --log_freq    int     log frequency during training per step
  --load_path   str     load checkpoint if specified
  --save_path   str     save checkpoint is specified
  --save_freq   str     checkpoint saving frequency per epoch
  --data        str     dataset (cifar10, cifar100, mnist, imagenet)
  --data_path   str     dataset path
  --bs          int     logical batch size
  --mnbs        int     physical batch size
  --model       str     trained model (cnn5, vit, wrn, etc., check init_utils for details)
  --pretrained          use pre-trained weights
  --algo        str     algorithm (sgd/adam/adamw)
  --lr          float   learning rate
  --epoch       int     number of epochs
  --scheduler           use cosineanealling with warmup
  --kf                  use kalman filter
  --kappa       float   filter gain (used if --kf)
  --gamma       float   perturbation stepsize (used if --kf)
  --clipping            use gradient clipping
  --noise       float   dp noise level, 0: no noise, -1: dp noise by epsilon, >0: manual noise
  --epsilon     float   dp privacy budget, must be larger than 0, used when noise=-1
  --clipping_norm str   clipping style, <=0: automatic, >0: Abadi
  --clipping_style str  clipping style, all-layer, layer-wise, param-wise
```

## Files:
run_KFSGD.py, run_KFSGD_GLUE.py
> Main script for running the tasks. CV and NLP tasks.

KFOptimizer.py
> contains the DOPPLER+ optimizer, wraps base optimizer as KF optimizer
> Usage: ```optimizer = KFOptimizer(model.parameters(), optimizer=optimizer, kappa=args.kappa, gamma=args.gamma)```

train_utils.py
> contains the training/testing function for one epoch.

init_utils.py
> contains the arg_parser and initialization functions for dataset, models, log files

model_utils.py
> contains self defined models fof CV and NLP tasks

data_utils.py
> contains dataset and data loader, transforms for each dataset

