import torch
from neural import actor
from neural import critic
from torch.utils.tensorboard import SummaryWriter

log_path = 'D:\\'

writer = SummaryWriter(log_path + '_graph_1')

c_n = critic()

writer.add_graph(c_n,torch.randn(1,1291))

writer = SummaryWriter(log_path + '_graph_2')

a_n = actor()

writer.add_graph(a_n,torch.randn(1,955))