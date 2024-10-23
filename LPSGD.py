import torch
from torch.optim.optimizer import Optimizer, required

class LPSGD(Optimizer):
    def __init__(self, params, lr=required, dampening=0,
                 weight_decay=0, a = [1.        , -1.98342955,  0.98549544], b = [0.00016332, 0.00032665, 0.00016332], c = None):
        defaults = dict(lr=lr, dampening=dampening,
                        weight_decay=weight_decay, a = a, b = b, c= c)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LPSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        EPSILON = 1e-5

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            a = group['a']
            b = group['b']
            c = group['c'] 
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * weight_decay)
                d_p = p.grad.data
                param_state = self.state[p]
                if c is not None:
                    if 'exp_avg_sq' not in param_state:
                        param_state['exp_avg_sq'] = torch.zeros_like(d_p).to(d_p)
                        param_state['ct'] = torch.tensor(0).to(d_p)
                    param_state['exp_avg_sq'].mul_(c).addcmul_(d_p, d_p, value=1 - c)
                    param_state['ct'].mul_(c).add_(1 - c)
                if 'bt' not in param_state:
                    param_state['bt'] = torch.zeros(len(b)).to(d_p)
                    param_state['bt'][0] = 1
                else:
                    param_state['bt'] = torch.cat((torch.tensor([1]).to(d_p), param_state['bt'][:-1]))
                norm_factor = torch.inner(torch.tensor(b).to(param_state['bt']), param_state['bt'])

                if len(b) > 1:
                    if not torch.is_tensor(b):
                        b = torch.tensor(b).to(d_p)
                    # d[t] = b[0]g[t] + b[1]g[t-1] + ... + b[n]g[t-n]
                    if 'g_tau' not in param_state:
                        # initialize
                        size = [len(b)-1, d_p.numel()]
                        param_state['g_tau'] = torch.zeros(size, dtype=d_p.dtype).to(d_p)
                        param_state['g_tau'][0] = d_p.reshape(-1).clone()
                        d_p.mul_(b[0])
                    else:
                        # other iterations, update buffer
                        G_temp = d_p.reshape(1,-1).clone()
                        d_p.mul_(b[0])
                        d_p.add_(torch.einsum('i,ij->j', b[1:], param_state['g_tau']).reshape(d_p.size()))
                        param_state['g_tau'] = torch.cat((G_temp, param_state['g_tau'][:-1]))
                        del G_temp
                else:
                    d_p.mul_(b[0])
                if len(a) > 1:
                    if not torch.is_tensor(a):
                        a = torch.tensor(a).to(d_p)
                    # d[t] = a[0]d[t] - a[1]d[t-1] - ... -a[n]d[t-n]
                    if 'm_tau' not in param_state:
                        # initialize
                        size = [len(a)-1, d_p.numel()]
                        param_state['m_tau'] = torch.zeros(size, dtype=d_p.dtype).to(d_p)
                        param_state['at'] = torch.zeros(len(a)-1).to(d_p)
                    else:
                        # other iterations, update buffer
                        d_p.add_(torch.einsum('i,ij->j', a[1:], param_state['m_tau']).reshape(d_p.size()), alpha = -1)
                        norm_factor -= torch.inner(a[1:], param_state['at'])
                    param_state['at'] = torch.cat((norm_factor.reshape(-1), param_state['at'][:-1]))
                    param_state['m_tau'] = torch.cat((d_p.reshape(1,-1).clone(), param_state['m_tau'][:-1]))
                
                if c is not None:
                    denom = param_state['exp_avg_sq'].div(param_state['ct']).sqrt().clamp_min(EPSILON)
                    p.data.addcdiv_(d_p, denom, value = -group['lr']/norm_factor)
                else:
                    p.data.add_(d_p, alpha = -group['lr']/norm_factor)
        return loss