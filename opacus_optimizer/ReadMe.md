# Usage
This folder is a version compatible with opacus package.

The optimizers take the same set of argumemnts as the optimizer as the coresponding ones in opacus and has two extra arguments:
kappa: float=0.7, and gamma: float=0.5

The corresponding privacy engine is also included, which rewirtes _prepare_optimizer to include KFOptimizers

Example of using the KFOpt optimizers:

```python
from KFprivacy_engine import KF_PrivacyEngine
# ...
# follow the same steps as original opacus training scripts
privacy_engine = KF_PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=max_grad_norm,
        clipping=clipping,
        grad_sample_mode=args.grad_sample_mode,
        kalman=True, # need this argument
        kappa=0.7, # optional
        gamma=0.5 # optional
    )

# ...
# during training:
def closure(): # compute loss and backward, an example adapting the one used in examples/cifar10.py
    output = model(images)
    loss = criterion(output, target)
    loss.backward()
    return output, loss
output, loss = optimizer.step(closure)
optimizer.zero_grad() 
# compute other matrices
# ...
```