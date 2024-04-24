import torch
from g_p_tranformer import actor
from g_p_tranformer import critic
from torch.utils.tensorboard import SummaryWriter

log_path = 'D:\\'

writer = SummaryWriter(log_path + '_graph_1')

c_n = critic(5,5,num_head=2)


writer.add_graph(c_n,torch.randn(1,715))

writer = SummaryWriter(log_path + '_graph_2')

a_n = actor(5,5,num_head=2)

writer.add_graph(a_n,torch.randn(1,379))