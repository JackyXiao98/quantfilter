import torch
from torch.optim.optimizer import Optimizer, required

class KFSGD(Optimizer):
    def __init__(self, params, lr=required, H=required, weight_decay=0, sigma_q=0, sigma_p=0):
        defaults = dict(lr=lr, H=H,
                        weight_decay=weight_decay, sigma_q=sigma_q, sigma_p=sigma_p)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            H = group['H']
            sigma_p = group['sigma_p']
            sigma_q = group['sigma_q']
            g_list = []
            m_list = []
            d_list = []
            beta_t = None
            norm_fact = None
            # concat as a vector
            for p in group['params']:
                if p.grad is None:
                    continue
                if beta_t is None:
                    if 'beta_t' not in self.state[p]:
                        self.state[p]['beta_t'] = 1
                    if 'norm_fact' not in self.state[p]:
                        self.state[p]['norm_fact'] = 0
                    beta_t = self.state[p]['beta_t'] + sigma_q**2
                    norm_fact = self.state[p]['norm_fact']
                    H = H.to(p)
                g_list.append(p.grad.data.view(-1))
                if 'm_t' not in self.state[p]:
                    self.state[p]['m_t'] = p.grad.clone().reshape(-1).to(p.grad)# torch.zeros_like(p.grad).reshape(-1).to(p.grad) # 
                m_list.append(self.state[p]['m_t'])
                if 'd_t' not in self.state[p]:
                    self.state[p]['d_t'] = torch.zeros_like(p.grad).to(p.grad)
                d_list.append(self.state[p]['d_t'].view(-1))
            g_vector = torch.cat(g_list)
            m_vector = torch.cat(m_list)
            d_vector = torch.cat(d_list)

            # prediction
            m_vector.addmv_(H, d_vector, alpha=-1)
            delta_g = g_vector - m_vector
            # print("delta_g:", torch.norm(delta_g).item())

            k_t = beta_t/(beta_t + sigma_p**2 - sigma_q**2)

            # filter
            m_vector.lerp_(g_vector, k_t)
            norm_fact = 1 #(1-k_t)*norm_fact + k_t
            beta_t = (1-k_t)*beta_t

            offset = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'beta_t' in self.state[p]:
                    self.state[p]['beta_t'] = beta_t
                    self.state[p]['norm_fact'] = norm_fact
                param_num = torch.numel(p)
                self.state[p]['m_t'] = m_vector[offset:offset+param_num]
                offset += param_num

                self.state[p]['d_t'] = p.data.clone().to(p.data)
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * weight_decay)
                p.data.add_(self.state[p]['m_t'], alpha = -group['lr']/norm_fact)
                self.state[p]['d_t'].add_(p.data, alpha=-1)
        return loss

def lambda_clip(param: torch.Tensor, Lambda:torch.Tensor, R: torch.Tensor, C):
    sqrt_L = torch.sqrt(Lambda) * C
    if param.grad is not None:
        size = param.grad.size()
        grad = torch.matmul(R.T, param.grad.reshape(-1,1))
        grad = torch.clamp(grad.reshape(-1), -sqrt_L, sqrt_L)
        param.grad = torch.matmul(R, grad).reshape(size)
    return param

def lambda_clip_2(param: torch.Tensor, Lambda:torch.Tensor, R: torch.Tensor, C):
    sqrt_L = torch.sqrt(Lambda)
    if param.grad is not None:
        size = param.grad.size()
        grad = torch.div(torch.matmul(R.T, param.grad.reshape(-1,1)),sqrt_L)
        norm = torch.linalg.norm(grad)
        C_1 = torch.clamp_max(C/norm, 1)
        grad.mul_(C_1).mul_(sqrt_L)
        # grad = torch.clamp(grad.reshape(-1), -sqrt_L, sqrt_L)
        param.grad = torch.matmul(R, grad).reshape(size)
    return param

def uniform_clip(param: torch.Tensor, C):
    if param.grad is not None:
        param.grad = torch.clamp(param.grad, -C, C)
    return param

def flat_clip(param: torch.Tensor, C):
    if param.grad is not None:
        norm = torch.linalg.norm(param.grad)
        C_1 = torch.clamp_max(C/norm, 1)
        param.grad.mul_(C_1)
    return param

def get_C(dim, Lambda, C):
    dim = torch.tensor(dim)
    C_flat = 1
    C_uniform = 1/torch.sqrt(dim)
    C_lambda = 1/torch.sqrt(torch.sum(Lambda))
    return C_flat*C, C_uniform*C, C_lambda*C

# def apply_noise(param: torch.Tensor, sigma):
#     if param.grad is not None:
#         noise = torch.randn_like(param)*sigma
#         param.grad.add_(noise)
#     return param

def apply_noise(param: torch.Tensor, distribution):
    if param.grad is not None:
        noise = distribution.sample().to(param)
        param.grad.add_(noise)
    return param