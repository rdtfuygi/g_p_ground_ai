from pickletools import floatnl
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda


import math



leaky_relu = nn.LeakyReLU(negative_slope=0.01)	


class critic(nn.Module):
	def __init__(self, lr:float = 0.01):
		super(critic, self).__init__()
		self.fc1 = nn.Linear(951, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.fc4 = nn.Linear(1024, 1024)
		self.fc5 = nn.Linear(1024, 1024)
		self.fc6 = nn.Linear(1024, 1024)
		self.fc7 = nn.Linear(1024, 1024)
		self.fc8 = nn.Linear(1024, 1024)
		self.fc9 = nn.Linear(1024, 1024)
		self.fc10 = nn.Linear(1024, 1)
		
		self.ln1 = nn.LayerNorm(1024)
		self.ln2 = nn.LayerNorm(1024)
		self.ln3 = nn.LayerNorm(1024)
		self.ln4 = nn.LayerNorm(1024)
		self.ln5 = nn.LayerNorm(1024)
		self.ln6 = nn.LayerNorm(1024)
		self.ln7 = nn.LayerNorm(1024)
		self.ln8 = nn.LayerNorm(1024)
		self.ln9 = nn.LayerNorm(1024)
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-8, weight_decay = 0.0000001)
		
	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = leaky_relu(self.ln1.forward(self.fc1.forward(x)))
		x = leaky_relu(self.ln2.forward(self.fc2.forward(x)))
		x = leaky_relu(self.ln3.forward(self.fc3.forward(x)))
		x = leaky_relu(self.ln4.forward(self.fc4.forward(x)))
		x = leaky_relu(self.ln5.forward(self.fc5.forward(x)))
		x = leaky_relu(self.ln6.forward(self.fc6.forward(x)))
		x = leaky_relu(self.ln7.forward(self.fc7.forward(x)))
		x = leaky_relu(self.ln8.forward(self.fc8.forward(x)))
		x = leaky_relu(self.ln9.forward(self.fc9.forward(x)))
		x = self.fc10.forward(x)
		return x
	
	def learn(self, s:torch.Tensor, r:torch.Tensor, s_new:torch.Tensor) -> torch.Tensor:
		v = self.forward(s)
		v_new = self.forward(s_new)
		td_e = 0.99 * v_new + r - v
		loss = torch.mean(torch.square(td_e))
		self.opt.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
		self.opt.step()
		return td_e.detach().clone()


class actor(nn.Module):
	def __init__(self, noise_std:float = 0.1, lr:float = 0.01):
		super(actor, self).__init__()
		self.fc1 = nn.Linear(951, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.fc4 = nn.Linear(1024, 1024)
		self.fc5 = nn.Linear(1024, 1024)
		self.fc6 = nn.Linear(1024, 1024)
		self.fc7 = nn.Linear(1024, 1024)
		self.fc8 = nn.Linear(1024, 1024)
		self.fc9 = nn.Linear(1024, 1024)

		self.fc10 = nn.Linear(1024, 1024)
		self.fc11 = nn.Linear(1024, 1024)
		self.fc12 = nn.Linear(1024, 1024)
		self.fc13 = nn.Linear(1024, 1024)
		self.fc14 = nn.Linear(1024, 1024)
		self.fc15 = nn.Linear(1024, 1024)
		self.fc16 = nn.Linear(1024, 1024)
		self.fc17 = nn.Linear(1024, 1024)
		self.fc18 = nn.Linear(1024, 1024)
		self.fc19 = nn.Linear(1024, 1024)
		
		self.fc20 = nn.Linear(1024, 1024)
		self.fc21 = nn.Linear(1024, 1024)
		self.fc22 = nn.Linear(1024, 1024)
		self.fc23 = nn.Linear(1024, 1024)
		self.fc24 = nn.Linear(1024, 1024)
		self.fc25 = nn.Linear(1024, 1024)
		self.fc26 = nn.Linear(1024, 1024)
		self.fc27 = nn.Linear(1024, 1024)
		self.fc28 = nn.Linear(1024, 1024)
		self.fc29 = nn.Linear(1024, 1024)
		
		self.fc30 = nn.Linear(1024, 1024)
		self.fc31 = nn.Linear(1024, 1024)
		self.fc32 = nn.Linear(1024, 1024)
		self.fc33 = nn.Linear(1024, 1024)
		self.fc34 = nn.Linear(1024, 1024)
		self.fc35 = nn.Linear(1024, 1024)
		self.fc36 = nn.Linear(1024, 1024)
		self.fc37 = nn.Linear(1024, 1024)
		self.fc38 = nn.Linear(1024, 1024)
		self.fc39 = nn.Linear(1024, 1024)

		self.fc40 = nn.Linear(1024, 336)
		

		self.ln1 = nn.LayerNorm(1024)
		self.ln2 = nn.LayerNorm(1024)
		self.ln3 = nn.LayerNorm(1024)
		self.ln4 = nn.LayerNorm(1024)
		self.ln5 = nn.LayerNorm(1024)
		self.ln6 = nn.LayerNorm(1024)
		self.ln7 = nn.LayerNorm(1024)
		self.ln8 = nn.LayerNorm(1024)
		self.ln9 = nn.LayerNorm(1024)
		
		self.ln10 = nn.LayerNorm(1024)
		self.ln11 = nn.LayerNorm(1024)
		self.ln12 = nn.LayerNorm(1024)
		self.ln13 = nn.LayerNorm(1024)
		self.ln14 = nn.LayerNorm(1024)
		self.ln15 = nn.LayerNorm(1024)
		self.ln16 = nn.LayerNorm(1024)
		self.ln17 = nn.LayerNorm(1024)
		self.ln18 = nn.LayerNorm(1024)
		self.ln19 = nn.LayerNorm(1024)
		
		self.ln20 = nn.LayerNorm(1024)
		self.ln21 = nn.LayerNorm(1024)
		self.ln22 = nn.LayerNorm(1024)
		self.ln23 = nn.LayerNorm(1024)
		self.ln24 = nn.LayerNorm(1024)
		self.ln25 = nn.LayerNorm(1024)
		self.ln26 = nn.LayerNorm(1024)
		self.ln27 = nn.LayerNorm(1024)
		self.ln28 = nn.LayerNorm(1024)
		self.ln29 = nn.LayerNorm(1024)
		
		self.ln30 = nn.LayerNorm(1024)
		self.ln31 = nn.LayerNorm(1024)
		self.ln32 = nn.LayerNorm(1024)
		self.ln33 = nn.LayerNorm(1024)
		self.ln34 = nn.LayerNorm(1024)
		self.ln35 = nn.LayerNorm(1024)
		self.ln36 = nn.LayerNorm(1024)
		self.ln37 = nn.LayerNorm(1024)
		self.ln38 = nn.LayerNorm(1024)
		self.ln39 = nn.LayerNorm(1024)


		self.noise_std = noise_std
		self.var_range = math.sqrt(noise_std * 12)
		
		self.way=[]

		#self.exp_replay = deque(maxlen = 2048)
		self.state = torch.Tensor()
		self.action = torch.Tensor()
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-8, weight_decay = 0.0000001)
		
		#self.loss_bais = 0.91893853320467274178032973640562 + math.log(self.noise_std)
		#self.reward_bais = 0.0
		#self.reward_l_bais = 0.0

	def forward(self, x:torch.Tensor) -> torch.Tensor:
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
		
		x = leaky_relu(self.ln30.forward(self.fc30.forward(x) + x))
		x = leaky_relu(self.ln31.forward(self.fc31.forward(x) + x))
		x = leaky_relu(self.ln32.forward(self.fc32.forward(x) + x))
		x = leaky_relu(self.ln33.forward(self.fc33.forward(x) + x))
		x = leaky_relu(self.ln34.forward(self.fc34.forward(x) + x))
		x = leaky_relu(self.ln35.forward(self.fc35.forward(x) + x))
		x = leaky_relu(self.ln36.forward(self.fc36.forward(x) + x))
		x = leaky_relu(self.ln37.forward(self.fc37.forward(x) + x))
		x = leaky_relu(self.ln38.forward(self.fc38.forward(x) + x))
		x = leaky_relu(self.ln39.forward(self.fc39.forward(x) + x))
		
		x = self.fc40.forward(x) / 1024 * 5
		return x

	def explor(self, x:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			self.eval()
			self.state = x.detach().clone()
			action = self.forward(x)
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			self.action = (action + noise).detach().clone()

			self.action[:, 40::42] = (torch.rand(action[:, 40::42].size()).cuda() * self.var_range - 0.5 * self.var_range + action[:, 40::42]).round()
			self.action[:, 41::42] = (torch.rand(action[:, 41::42].size()).cuda() * self.var_range - 0.5 * self.var_range + action[:, 41::42]).round()

		return self.action.detach().clone()

	def learn(self, td_e:torch.Tensor,state:torch.Tensor, action:torch.Tensor) -> float:

		# ǰ�򴫲�
		self.train()
		output = self.forward(state)
		# ��ʧ
		action_probs = torch.distributions.Normal(output, self.noise_std).log_prob(action)

		loss = -torch.mean(action_probs * td_e)
		#loss = -torch.mean((action_probs + self.loss_bais) * (reward - self.reward_l_bais))
		# ���򴫲�
		self.opt.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
		self.opt.step()
		return loss.item()