import torch
from torch.utils.data import DataLoader
from data_utils import generate_quadratic, synthetic
from optimizer import *
from LPSGD import LPSGD
from KFOptimizer import KFOptimizer
import numpy as np
from tqdm import tqdm

class linearRegression(torch.nn.Module):
    def __init__(self, dim):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(dim, 1,bias=False)
    
    def get_params(self):
        return self.get_parameter('linear.weight')

    def forward(self, x):
        out = self.linear(x)
        return out

#set problem
torch.manual_seed(42)
dim = 100
data_size = 100
bs = 100
criterion = torch.nn.MSELoss(reduction='sum')
sigma = 3e-2
dist = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim)*sigma**2)
device = torch.device('cuda:0')
epoch = 2
test = 1

total_final_loss = 0

for _ in tqdm(range(test)):
    # generate dataset
    x,y,H,w_opt=generate_quadratic(dim=dim, size=data_size)
    Lambda, MatR = torch.linalg.eigh(H*2)
    L = max(Lambda)

    dataset = synthetic(x,y)
    dataloader = DataLoader(dataset, batch_size= bs, shuffle=False)
    model = linearRegression(dim)

    optimizer = torch.optim.SGD(model.parameters(), lr = 1/L)
    optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H=1e-5, sigma_g=sigma)
    # optimizer = LPSGD(model.parameters(), lr=1/L, a=[1,-0.9], b=[0.15, -0.05])

    init_loss = None
    min_loss = None
    for E in range(epoch):
        model.to(device)
        model.train()
        for x,y in dataloader:
            optimizer.prestep()
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y, y_pred)
            if init_loss is None:
                init_loss = loss.item()
                min_loss = loss.item()
            elif loss.item()<min_loss:
                min_loss = loss.item()
            loss.backward()
            # optimizer.hessian_d_product()
            apply_noise(model.get_params(), dist)
            # optimizer.hessian_d_product()
            optimizer.step()
            optimizer.zero_grad()
        # if (E+1)%5 == 0:
    # print(min_loss/init_loss)
    total_final_loss += min_loss/init_loss
print(total_final_loss/test)
# print(torch.cuda.max_memory_allocated())