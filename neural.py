from pickletools import floatnl
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from collections import deque
import random



leaky_relu = nn.LeakyReLU(negative_slope=0.01)	

class net(nn.Module):
	def __init__(self, noise_std:float = 0.1, lr:float = 0.01):
		super(net, self).__init__()
		self.fc1 = nn.Linear(935, 2048)
		self.ln1 = nn.LayerNorm(2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.ln2 = nn.LayerNorm(2048)
		self.fc3 = nn.Linear(2048, 2048)
		self.ln3 = nn.LayerNorm(2048)
		self.fc4 = nn.Linear(2048, 2048)
		self.ln4 = nn.LayerNorm(2048)
		self.fc5 = nn.Linear(2048, 2048)
		self.ln5 = nn.LayerNorm(2048)
		self.fc6 = nn.Linear(2048, 2048)
		self.ln6 = nn.LayerNorm(2048)
		self.fc7 = nn.Linear(2048, 2048)
		self.ln7 = nn.LayerNorm(2048)
		self.fc8 = nn.Linear(2048, 2048)
		self.ln8 = nn.LayerNorm(2048)
		self.fc9 = nn.Linear(2048, 2048)
		self.ln9 = nn.LayerNorm(2048)
		self.fc10 = nn.Linear(2048, 336)
		self.ln10 = nn.LayerNorm(336)
		self.noise_std = noise_std
		self.exp_replay = deque(maxlen = 2048)
		self.human_control = deque(maxlen = 2048)
		self.state = torch.Tensor()
		self.action = torch.Tensor()
		self.opt = optim.AdamW(self.parameters(), lr = lr, weight_decay = 0.00001)

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
		x = torch.tanh(self.ln10.forward(self.fc10.forward(x)) / 2048)
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


		if len(self.exp_replay) >= 256:
			batch = random.sample(self.exp_replay, 256)
			state, action, reward = zip(*batch)
			
			if len(self.human_control) > 0:
				batch = random.sample(self.human_control, min(len(self.human_control), 256))
				h_state, h_action, h_reward = zip(*batch)
				state = state + h_state
				action = action + h_action
				reward = reward + h_reward
				if random.random() < 0.00001:
					self.human_control.popleft()
				
			state = torch.stack(state)
			action = torch.stack(action)
			reward = torch.FloatTensor(reward).view(-1, 1).cuda()
			# 前向传播
			output = self.forward(state)
			# 损失
			action_probs = torch.distributions.Normal(output, self.noise_std).log_prob(action)
			loss = -torch.mean(action_probs * reward)
			# 反向传播
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			return loss.item()
		return 0.0