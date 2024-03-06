from pickletools import floatnl
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from collections import deque
import random
import math



leaky_relu = nn.LeakyReLU(negative_slope=0.01)	

class net(nn.Module):
	def __init__(self, noise_std:float = 0.1, lr:float = 0.01):
		super(net, self).__init__()
		self.fc1 = nn.Linear(935, 2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.fc3 = nn.Linear(2048, 2048)
		self.fc4 = nn.Linear(2048, 2048)
		self.fc5 = nn.Linear(2048, 2048)
		self.fc6 = nn.Linear(2048, 2048)
		self.fc7 = nn.Linear(2048, 2048)
		self.fc8 = nn.Linear(2048, 2048)
		self.fc9 = nn.Linear(2048, 2048)

		self.fc10 = nn.Linear(2048, 2048)
		self.fc11 = nn.Linear(2048, 2048)
		self.fc12 = nn.Linear(2048, 2048)
		self.fc13 = nn.Linear(2048, 2048)
		self.fc14 = nn.Linear(2048, 2048)
		self.fc15 = nn.Linear(2048, 2048)
		self.fc16 = nn.Linear(2048, 2048)
		self.fc17 = nn.Linear(2048, 2048)
		self.fc18 = nn.Linear(2048, 2048)
		self.fc19 = nn.Linear(2048, 2048)

		self.fc20 = nn.Linear(2048, 2048)
		self.fc21 = nn.Linear(2048, 2048)
		self.fc22 = nn.Linear(2048, 2048)
		self.fc23 = nn.Linear(2048, 2048)
		self.fc24 = nn.Linear(2048, 2048)
		self.fc25 = nn.Linear(2048, 2048)
		self.fc26 = nn.Linear(2048, 2048)
		self.fc27 = nn.Linear(2048, 2048)
		self.fc28 = nn.Linear(2048, 2048)
		self.fc29 = nn.Linear(2048, 2048)
		
		# self.fc30 = nn.Linear(2048, 2048)
		# self.fc31 = nn.Linear(2048, 2048)
		# self.fc32 = nn.Linear(2048, 2048)
		# self.fc33 = nn.Linear(2048, 2048)
		# self.fc34 = nn.Linear(2048, 2048)
		# self.fc35 = nn.Linear(2048, 2048)
		# self.fc36 = nn.Linear(2048, 2048)
		# self.fc37 = nn.Linear(2048, 2048)
		# self.fc38 = nn.Linear(2048, 2048)
		# self.fc39 = nn.Linear(2048, 2048)

		self.fc40 = nn.Linear(2048, 336)
		
		self.ln1 = nn.LayerNorm(2048)
		self.ln2 = nn.LayerNorm(2048)
		self.ln3 = nn.LayerNorm(2048)
		self.ln4 = nn.LayerNorm(2048)
		self.ln5 = nn.LayerNorm(2048)
		self.ln6 = nn.LayerNorm(2048)
		self.ln7 = nn.LayerNorm(2048)
		self.ln8 = nn.LayerNorm(2048)
		self.ln9 = nn.LayerNorm(2048)
		
		self.ln10 = nn.LayerNorm(2048)
		self.ln11 = nn.LayerNorm(2048)
		self.ln12 = nn.LayerNorm(2048)
		self.ln13 = nn.LayerNorm(2048)
		self.ln14 = nn.LayerNorm(2048)
		self.ln15 = nn.LayerNorm(2048)
		self.ln16 = nn.LayerNorm(2048)
		self.ln17 = nn.LayerNorm(2048)
		self.ln18 = nn.LayerNorm(2048)
		self.ln19 = nn.LayerNorm(2048)
		
		self.ln20 = nn.LayerNorm(2048)
		self.ln21 = nn.LayerNorm(2048)
		self.ln22 = nn.LayerNorm(2048)
		self.ln23 = nn.LayerNorm(2048)
		self.ln24 = nn.LayerNorm(2048)
		self.ln25 = nn.LayerNorm(2048)
		self.ln26 = nn.LayerNorm(2048)
		self.ln27 = nn.LayerNorm(2048)
		self.ln28 = nn.LayerNorm(2048)
		self.ln29 = nn.LayerNorm(2048)
		# self.ln19 = nn.LayerNorm(2048)
		# self.ln29 = nn.LayerNorm(2048)		
		# self.ln39 = nn.LayerNorm(2048)


		self.noise_std = noise_std
		self.exp_replay = deque(maxlen = 2048)
		self.state = torch.Tensor()
		self.action = torch.Tensor()
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 0.0000001)

	def forward(self, x:torch.Tensor)->torch.Tensor:
		x = leaky_relu(self.ln1.forward(self.fc1.forward(x)))
		x = leaky_relu(self.ln2.forward(self.fc2.forward(x) + x))
		x = leaky_relu(self.ln3.forward(self.fc3.forward(x) + x))
		x = leaky_relu(self.ln4.forward(self.fc4.forward(x) + x))
		x = leaky_relu(self.ln5.forward(self.fc5.forward(x) + x))
		x = leaky_relu(self.ln6.forward(self.fc6.forward(x) + x))
		x = leaky_relu(self.ln7.forward(self.fc7.forward(x) + x))
		x = leaky_relu(self.ln8.forward(self.fc8.forward(x) + x))
		x = leaky_relu(self.ln9.forward(self.fc9.forward(x) + x))
		
		x = leaky_relu(self.ln10.forward(self.fc10.forward(x) + x))
		x = leaky_relu(self.ln11.forward(self.fc11.forward(x) + x))		
		x = leaky_relu(self.ln12.forward(self.fc12.forward(x) + x))
		x = leaky_relu(self.ln13.forward(self.fc13.forward(x) + x))
		x = leaky_relu(self.ln14.forward(self.fc14.forward(x) + x))
		x = leaky_relu(self.ln15.forward(self.fc15.forward(x) + x))
		x = leaky_relu(self.ln16.forward(self.fc16.forward(x) + x))
		x = leaky_relu(self.ln17.forward(self.fc17.forward(x) + x))
		x = leaky_relu(self.ln18.forward(self.fc18.forward(x) + x))
		x = leaky_relu(self.ln19.forward(self.fc19.forward(x) + x))
		
		x = leaky_relu(self.ln20.forward(self.fc20.forward(x) + x))
		x = leaky_relu(self.ln21.forward(self.fc21.forward(x) + x))		
		x = leaky_relu(self.ln22.forward(self.fc22.forward(x) + x))
		x = leaky_relu(self.ln23.forward(self.fc23.forward(x) + x))
		x = leaky_relu(self.ln24.forward(self.fc24.forward(x) + x))
		x = leaky_relu(self.ln25.forward(self.fc25.forward(x) + x))
		x = leaky_relu(self.ln26.forward(self.fc26.forward(x) + x))
		x = leaky_relu(self.ln27.forward(self.fc27.forward(x) + x))
		x = leaky_relu(self.ln28.forward(self.fc28.forward(x) + x))
		x = leaky_relu(self.ln29.forward(self.fc29.forward(x) + x))
		# x = leaky_relu(self.fc10.forward(x) + x)
		# x = leaky_relu(self.fc11.forward(x) + x)
		# x = leaky_relu(self.fc12.forward(x) + x)
		# x = leaky_relu(self.fc13.forward(x) + x)
		# x = leaky_relu(self.fc14.forward(x) + x)
		# x = leaky_relu(self.fc15.forward(x) + x)
		# x = leaky_relu(self.fc16.forward(x) + x)
		# x = leaky_relu(self.fc17.forward(x) + x)
		# x = leaky_relu(self.fc18.forward(x) + x)
		# x = leaky_relu(self.ln19.forward(self.fc19.forward(x) + x))
		# x = leaky_relu(self.fc20.forward(x) + x)
		# x = leaky_relu(self.fc21.forward(x) + x)
		# x = leaky_relu(self.fc22.forward(x) + x)
		# x = leaky_relu(self.fc23.forward(x) + x)
		# x = leaky_relu(self.fc24.forward(x) + x)
		# x = leaky_relu(self.fc25.forward(x) + x)
		# x = leaky_relu(self.fc26.forward(x) + x)
		# x = leaky_relu(self.fc27.forward(x) + x)
		# x = leaky_relu(self.fc28.forward(x) + x)
		# x = leaky_relu(self.ln29.forward(self.fc29.forward(x) + x))
		# x = leaky_relu(self.fc30.forward(x) + x)
		# x = leaky_relu(self.fc31.forward(x) + x)
		# x = leaky_relu(self.fc32.forward(x) + x)
		# x = leaky_relu(self.fc33.forward(x) + x)
		# x = leaky_relu(self.fc34.forward(x) + x)
		# x = leaky_relu(self.fc35.forward(x) + x)
		# x = leaky_relu(self.fc36.forward(x) + x)
		# x = leaky_relu(self.fc37.forward(x) + x)
		# x = leaky_relu(self.fc38.forward(x) + x)
		# x = leaky_relu(self.ln39.forward(self.fc39.forward(x) + x))
		x = self.fc40.forward(x) / 2048 * 5
		return x

	def explor(self, x:torch.Tensor)->torch.Tensor:
		with torch.no_grad():
			self.state = x.detach().clone()
			action = self.forward(x)
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			self.action = (action + noise).detach().clone()
			#print(2)
			#print(self.action)
		return self.action.detach().clone()

	def learn(self, callback:list):
		state_list = torch.split(self.state, 1, dim = 0)
		action_list = torch.split(self.action, 1, dim = 0)
		if(len(callback) != 0):
			for i in range(len(callback)):
				self.exp_replay.append((state_list[i][0], action_list[i][0], callback[i]))


		if len(self.exp_replay) >= 512:
			batch = random.sample(self.exp_replay, 512)
			state, action, reward = zip(*batch)
				
			state = torch.stack(state)
			action = torch.stack(action)
			reward = torch.FloatTensor(reward).view(-1, 1).cuda()
			# 前向传播
			output = self.forward(state)
			# 损失
			action_probs = torch.distributions.Normal(output, self.noise_std).log_prob(action)
			loss = -torch.mean((action_probs + 0.91893853320467274178032973640562 + math.log(self.noise_std)) * reward)
			# 反向传播
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			return loss.item()
		return 0.0