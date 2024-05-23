import torch
from g_p_transformer import actor
from g_p_transformer import critic
from torch.utils.tensorboard import SummaryWriter

log_path = 'D:\\'

writer = SummaryWriter(log_path + '_graph_1')

c_n = critic(3, 3, num_head=2)
c_n.eval()


writer.add_graph(c_n, torch.randn(1, 308, 3))

writer = SummaryWriter(log_path + '_graph_2')

a_n = actor(3, 5, num_head=2)
a_n.eval()

writer.add_graph(a_n, torch.randn(1, 172, 3))