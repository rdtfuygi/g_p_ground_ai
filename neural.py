from enum import auto
from pickletools import floatnl
from tkinter import W
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda

from torch.cuda.amp import autocast, GradScaler


import math


leaky_relu = nn.LeakyReLU(negative_slope=0.01)	

class share_net(nn.Module):
	def __init__(self) -> None:
		super(share_net, self).__init__()
		#l1
		#2点
		self.l1k1fc1 = nn.Linear(4, 4)
		self.l1k1fc2 = nn.Linear(4, 4)
		self.l1k1fc3 = nn.Linear(4, 1)
		#3点
		self.l1k2fc1 = nn.Linear(6, 8)
		self.l1k2fc2 = nn.Linear(8, 8)
		self.l1k2fc3 = nn.Linear(8, 1)
		#2向量
		self.l1k3fc1 = nn.Linear(4, 4)
		self.l1k3fc2 = nn.Linear(4, 4)
		self.l1k3fc3 = nn.Linear(4, 1)
		#3向量
		self.l1k4fc1 = nn.Linear(6, 8)
		self.l1k4fc2 = nn.Linear(8, 8)
		self.l1k4fc3 = nn.Linear(8, 1)
		#2长度
		self.l1k5fc1 = nn.Linear(2, 2)
		self.l1k5fc2 = nn.Linear(2, 2)
		self.l1k5fc3 = nn.Linear(2, 1)
		#3长度
		self.l1k6fc1 = nn.Linear(3, 4)
		self.l1k6fc2 = nn.Linear(4, 4)
		self.l1k6fc3 = nn.Linear(4, 1)
		#2线段
		self.l1k7fc1 = nn.Linear(10, 16)
		self.l1k7fc2 = nn.Linear(16, 16)
		self.l1k7fc3 = nn.Linear(16, 1)
		#3线段
		self.l1k8fc1 = nn.Linear(15, 16)
		self.l1k8fc2 = nn.Linear(16, 16)
		self.l1k8fc3 = nn.Linear(16, 1)
		
		#l2
		#2线段
		self.l2k1fc1 = nn.Linear(18, 32)
		self.l2k1fc2 = nn.Linear(32, 32)
		self.l2k1fc3 = nn.Linear(32, 32)
		self.l2k1fc4 = nn.Linear(32, 1)
		#3线段
		self.l2k2fc1 = nn.Linear(27, 32)
		self.l2k2fc2 = nn.Linear(32, 32)
		self.l2k2fc3 = nn.Linear(32, 32)
		self.l2k2fc4 = nn.Linear(32, 1)
		
		#l3
		self.l3k1fc1 = nn.Linear(306, 512)
		self.l3k1fc2 = nn.Linear(512, 512)
		self.l3k1fc3 = nn.Linear(512, 512)
		self.l3k1fc4 = nn.Linear(512, 512)
		self.l3k1fc5 = nn.Linear(512, 512)
		self.l3k1fc6 = nn.Linear(512, 512)
		self.l3k1fc7 = nn.Linear(512, 512)
		self.l3k1fc8 = nn.Linear(512, 1)
		
		#l4
		self.l4fc1 = nn.Linear(2763, 2048)
		self.l4fc2 = nn.Linear(2048, 2048)
		self.l4fc3 = nn.Linear(2048, 2048)
		self.l4fc4 = nn.Linear(2048, 2048)
		self.l4fc5 = nn.Linear(2048, 2048)
		self.l4fc6 = nn.Linear(2048, 1024)
		self.l4fc7 = nn.Linear(1024, 1024)
		self.l4fc8 = nn.Linear(1024, 1024)
		self.l4fc9 = nn.Linear(1024, 1024)
		
		self.l4fc10 = nn.Linear(1024, 1024)
		
		self.l4bn1 = nn.BatchNorm1d(2763, 1e-5)
		self.l4bn2 = nn.BatchNorm1d(2048, 1e-5)
		self.l4bn3 = nn.BatchNorm1d(2048, 1e-5)
		self.l4bn4 = nn.BatchNorm1d(2048, 1e-5)
		self.l4bn5 = nn.BatchNorm1d(2048, 1e-5)
		self.l4bn6 = nn.BatchNorm1d(2048, 1e-5)
		self.l4bn7 = nn.BatchNorm1d(1024, 1e-5)
		self.l4bn8 = nn.BatchNorm1d(1024, 1e-5)
		self.l4bn9 = nn.BatchNorm1d(1024, 1e-5)
		
		self.l4bn10 = nn.BatchNorm1d(1024, 1e-5)
		
	
	def forward(self, x:torch.Tensor) ->torch.Tensor:
			x = x.view(-1, 106)
		
			x_ = torch.cat((x[:,95:100], x[:,0:100], x[:,0:5]), dim = 1)
		#with autocast(dtype = torch.float16):
			
			l1k1x = torch.cat((x_[:,5:105:5], x_[:,6:105:5], x_[:,10::5], x_[:,11::5]), dim = 1).view(-1, 4)
			l1k1x = leaky_relu(self.l1k1fc1.forward(l1k1x))
			l1k1x = leaky_relu(self.l1k1fc2.forward(l1k1x))
			l1k1x = leaky_relu(self.l1k1fc3.forward(l1k1x))
			
			l1k2x = torch.cat((x_[:,0:100:5], x_[:,1:100:5], x_[:,5:105:5], x_[:,6:105:5], x_[:,10::5], x_[:,11::5]), dim = 1).view(-1, 6)
			l1k2x = leaky_relu(self.l1k2fc1.forward(l1k2x))
			l1k2x = leaky_relu(self.l1k2fc2.forward(l1k2x))
			l1k2x = leaky_relu(self.l1k2fc3.forward(l1k2x))
			
			l1k3x = torch.cat((x_[:,7:105:5], x_[:,8:105:5], x_[:,12::5], x_[:,13::5]), dim = 1).view(-1, 4)
			l1k3x = leaky_relu(self.l1k3fc1.forward(l1k3x))
			l1k3x = leaky_relu(self.l1k3fc2.forward(l1k3x))
			l1k3x = leaky_relu(self.l1k3fc3.forward(l1k3x))
			
			l1k4x = torch.cat((x_[:,2:100:5], x_[:,3:100:5], x_[:,7:105:5], x_[:,8:105:5], x_[:,12::5], x_[:,13::5]), dim = 1).view(-1, 6)
			l1k4x = leaky_relu(self.l1k4fc1.forward(l1k4x))
			l1k4x = leaky_relu(self.l1k4fc2.forward(l1k4x))
			l1k4x = leaky_relu(self.l1k4fc3.forward(l1k4x))
			
			l1k5x = torch.cat((x_[:,9:105:5], x_[:,14::5]), dim = 1).view(-1, 2)
			l1k5x = leaky_relu(self.l1k5fc1.forward(l1k5x))
			l1k5x = leaky_relu(self.l1k5fc2.forward(l1k5x))
			l1k5x = leaky_relu(self.l1k5fc3.forward(l1k5x))
			
			l1k6x = torch.cat((x_[:,4:100:5], x_[:,9:105:5], x_[:,14::5]), dim = 1).view(-1, 3)
			l1k6x = leaky_relu(self.l1k6fc1.forward(l1k6x))
			l1k6x = leaky_relu(self.l1k6fc2.forward(l1k6x))
			l1k6x = leaky_relu(self.l1k6fc3.forward(l1k6x))
			
			l1k7x = torch.cat((x_[:,5:105:5], x_[:,6:105:5], x_[:,7:105:5], x_[:,8:105:5], x_[:,9:105:5], x_[:,10::5], x_[:,11::5], x_[:,12::5], x_[:,13::5], x_[:,14::5]), dim = 1).view(-1, 10)
			l1k7x = leaky_relu(self.l1k7fc1.forward(l1k7x))
			l1k7x = leaky_relu(self.l1k7fc2.forward(l1k7x))
			l1k7x = leaky_relu(self.l1k7fc3.forward(l1k7x))
			
			l1k8x = torch.cat((x_[:,0:100:5], x_[:,1:100:5], x_[:,2:100:5], x_[:,3:100:5], x_[:,4:100:5], x_[:,5:105:5], x_[:,6:105:5], x_[:,7:105:5], x_[:,8:105:5], x_[:,9:105:5], x_[:,10::5], x_[:,11::5], x_[:,12::5], x_[:,13::5], x_[:,14::5]), dim = 1).view(-1, 15)
			l1k8x = leaky_relu(self.l1k8fc1.forward(l1k8x))
			l1k8x = leaky_relu(self.l1k8fc2.forward(l1k8x))
			l1k8x = leaky_relu(self.l1k8fc3.forward(l1k8x))
			
						
			x_ = x[:,0:100].reshape(-1,5)
			x_ = torch.cat((x_, l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x), dim = 1)
			x_ = x_.view(-1, 260)
			x_ = torch.cat((x_[:,247:260], x_, x_[:,0:13]), dim = 1)	

			l2k1x = torch.cat((x_[:,13:273:13], x_[:,14:273:13], x_[:,15:273:13], x_[:,16:273:13], x_[:,17:273:13], x_[:,22:273:13], x_[:,23:273:13], x_[:,24:273:13], x_[:,25:273:13], x_[:,26::13], x_[:,27::13], x_[:,28::13], x_[:,29::13], x_[:,30::13], x_[:,35::13], x_[:,36::13], x_[:,37::13], x_[:,38::13]), dim = 1)
			l2k1x = l2k1x.view(-1, 18)
			l2k1x = leaky_relu(self.l2k1fc1.forward(l2k1x))
			l2k1x = leaky_relu(self.l2k1fc2.forward(l2k1x))
			l2k1x = leaky_relu(self.l2k1fc3.forward(l2k1x))
			l2k1x = leaky_relu(self.l2k1fc4.forward(l2k1x))
			
			l2k2x = torch.cat((x_[:,0:260:13], x_[:,1:260:13], x_[:,2:260:13], x_[:,3:260:13], x_[:,4:260:13], x_[:,9:260:13], x_[:,10:260:13], x_[:,11:260:13], x_[:,12:260:13], x_[:,13:273:13], x_[:,14:273:13], x_[:,15:273:13], x_[:,16:273:13], x_[:,17:273:13], x_[:,22:273:13], x_[:,23:273:13], x_[:,24:273:13], x_[:,25:273:13], x_[:,26::13], x_[:,27::13], x_[:,28::13], x_[:,29::13], x_[:,30::13], x_[:,35::13], x_[:,36::13], x_[:,37::13], x_[:,38::13]), dim = 1).view(-1, 27)
			l2k2x = leaky_relu(self.l2k2fc1.forward(l2k2x))
			l2k2x = leaky_relu(self.l2k2fc2.forward(l2k2x))
			l2k2x = leaky_relu(self.l2k2fc3.forward(l2k2x))
			l2k2x = leaky_relu(self.l2k2fc4.forward(l2k2x))
			
			
			x_ = torch.cat((l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x, l2k1x, l2k2x), dim = 1).view(-1, 200)
			x_ = torch.cat((x,x_), dim = 1)
			
			l3k1x = leaky_relu(self.l3k1fc1.forward(x_))
			l3k1x = leaky_relu(self.l3k1fc2.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc3.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc4.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc5.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc6.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc7.forward(l3k1x)) + l3k1x
			l3k1x = leaky_relu(self.l3k1fc8.forward(l3k1x))


			x_ = torch.cat((x_, l3k1x), dim = 1).view(-1, 2763)
			
			x_ = self.l4fc1.forward(self.l4bn1.forward(x_))
			x_ = self.l4fc2.forward(self.l4bn2.forward(x_)) + x_
			x_ = self.l4fc3.forward(self.l4bn3.forward(x_)) + x_
			x_ = self.l4fc4.forward(self.l4bn4.forward(x_)) + x_
			x_ = self.l4fc5.forward(self.l4bn5.forward(x_)) + x_
			x_ = self.l4fc6.forward(self.l4bn6.forward(x_))
			x_ = self.l4fc7.forward(self.l4bn7.forward(x_)) + x_
			x_ = self.l4fc8.forward(self.l4bn8.forward(x_)) + x_
			x_ = self.l4fc9.forward(self.l4bn9.forward(x_)) + x_
			x_ = self.l4fc10.forward(self.l4bn10.forward(x_)) + x_
			return x_



class critic(nn.Module):
	def __init__(self, share:share_net, lr:float = 0.01):
		super(critic, self).__init__()
		self.share = share
		self.fc21 = nn.Linear(1024, 1024)
		self.fc22 = nn.Linear(1024, 1024)
		self.fc23 = nn.Linear(1024, 1024)
		self.fc24 = nn.Linear(1024, 1024)
		# self.fc25 = nn.Linear(1024, 1024)
		# self.fc26 = nn.Linear(1024, 1024)
		# self.fc27 = nn.Linear(1024, 1024)
		# self.fc28 = nn.Linear(1024, 1024)
		# self.fc29 = nn.Linear(1024, 1024)
		
		self.fc30 = nn.Linear(1024, 1)
		
		self.ln21 = nn.LayerNorm(1024, 1e-5)
		self.ln22 = nn.LayerNorm(1024, 1e-5)
		self.ln23 = nn.LayerNorm(1024, 1e-5)
		self.ln24 = nn.LayerNorm(1024, 1e-5)
		# self.ln25 = nn.LayerNorm(1024, 1e-5)
		# self.ln26 = nn.LayerNorm(1024, 1e-5)
		# self.ln27 = nn.LayerNorm(1024, 1e-5)
		# self.ln28 = nn.LayerNorm(1024, 1e-5)
		# self.ln29 = nn.LayerNorm(1024, 1e-5)
		self.ln30 = nn.LayerNorm(1024, 1e-5)
		
		self.scaler = GradScaler()
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-5)
		

	def forward(self, x:torch.Tensor) -> torch.Tensor:
			x = self.share.forward(x)
		#with autocast(dtype = torch.float16):
			x = leaky_relu(self.fc21.forward(self.ln21.forward(x))) + x
			x = leaky_relu(self.fc22.forward(self.ln22.forward(x))) + x
			x = leaky_relu(self.fc23.forward(self.ln23.forward(x))) + x
			x = leaky_relu(self.fc24.forward(self.ln24.forward(x))) + x
			# x = leaky_relu(self.fc25.forward(self.ln25.forward(x))) + x
			# x = leaky_relu(self.fc26.forward(self.ln26.forward(x))) + x
			# x = leaky_relu(self.fc27.forward(self.ln27.forward(x))) + x
			# x = leaky_relu(self.fc28.forward(self.ln28.forward(x))) + x
			# x = leaky_relu(self.fc29.forward(self.ln29.forward(x))) + x	
			x = self.fc30.forward(self.ln30.forward(x))
			return x
	

	def learn(self, s:torch.Tensor, r:torch.Tensor, s_new:torch.Tensor) -> torch.Tensor:
		
		self.train()
		self.opt.zero_grad()		

		#print(s.size())

		v = self.forward(s)
		v_new = self.forward(s_new)

		#with autocast(dtype = torch.bfloat16):
		td_e = 0.99 * v_new + r - v
		loss = torch.mean(torch.square(td_e))
		
		# self.scaler.scale(loss).backward()
		# self.scaler.unscale_(self.opt)
		# torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
		# self.scaler.step(self.opt)
		# self.scaler.update()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
		self.opt.step()


		return (td_e.detach(), loss.item())


class actor(nn.Module):
	def __init__(self, share:share_net, noise_std:float = 0.1, lr:float = 0.01):
		super(actor, self).__init__()
		
		self.share = share


		self.fc21 = nn.Linear(1024, 1024)
		self.fc22 = nn.Linear(1024, 1024)
		self.fc23 = nn.Linear(1024, 1024)
		self.fc24 = nn.Linear(1024, 1024)
		self.fc25 = nn.Linear(1024, 1024)
		self.fc26 = nn.Linear(1024, 1024)
		self.fc27 = nn.Linear(1024, 1024)
		self.fc28 = nn.Linear(1024, 1024)
		self.fc29 = nn.Linear(1024, 1024)
		
		# self.fc30 = nn.Linear(1024, 1024)
		# self.fc31 = nn.Linear(1024, 1024)
		# self.fc32 = nn.Linear(1024, 1024)
		# self.fc33 = nn.Linear(1024, 1024)
		# self.fc34 = nn.Linear(1024, 1024)
		# self.fc35 = nn.Linear(1024, 1024)
		# self.fc36 = nn.Linear(1024, 1024)
		# self.fc37 = nn.Linear(1024, 1024)
		# self.fc38 = nn.Linear(1024, 1024)
		# self.fc39 = nn.Linear(1024, 1024)

		self.fc40 = nn.Linear(1024, 336)
		


		self.ln21 = nn.LayerNorm(1024, 1e-5)
		self.ln22 = nn.LayerNorm(1024, 1e-5)
		self.ln23 = nn.LayerNorm(1024, 1e-5)
		self.ln24 = nn.LayerNorm(1024, 1e-5)
		self.ln25 = nn.LayerNorm(1024, 1e-5)
		self.ln26 = nn.LayerNorm(1024, 1e-5)
		self.ln27 = nn.LayerNorm(1024, 1e-5)
		self.ln28 = nn.LayerNorm(1024, 1e-5)
		self.ln29 = nn.LayerNorm(1024, 1e-5)
		
		# self.ln30 = nn.LayerNorm(1024, 1e-5)
		# self.ln31 = nn.LayerNorm(1024, 1e-5)
		# self.ln32 = nn.LayerNorm(1024, 1e-5)
		# self.ln33 = nn.LayerNorm(1024, 1e-5)
		# self.ln34 = nn.LayerNorm(1024, 1e-5)
		# self.ln35 = nn.LayerNorm(1024, 1e-5)
		# self.ln36 = nn.LayerNorm(1024, 1e-5)
		# self.ln37 = nn.LayerNorm(1024, 1e-5)
		# self.ln38 = nn.LayerNorm(1024, 1e-5)
		# self.ln39 = nn.LayerNorm(1024, 1e-5)

		self.ln40 = nn.LayerNorm(1024, 1e-5)


		self.noise_std = noise_std
		self.var_range = math.sqrt(noise_std * 12)
		
		#self.way=[]

		#self.exp_replay = deque(maxlen = 2048)
		# self.state = torch.Tensor()
		# self.action = torch.Tensor()
		self.scaler = GradScaler()
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-5)
		
		self.loss_bais = 0.91893853320467274178032973640562 + math.log(self.noise_std)
		#self.reward_bais = 0.0
		#self.reward_l_bais = 0.0
	

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.share.forward(x)
		
		with autocast(dtype = torch.float16):
			x = leaky_relu(self.fc21.forward(self.ln21.forward(x))) + x
			x = leaky_relu(self.fc22.forward(self.ln22.forward(x))) + x
			x = leaky_relu(self.fc23.forward(self.ln23.forward(x))) + x
			x = leaky_relu(self.fc24.forward(self.ln24.forward(x))) + x
			x = leaky_relu(self.fc25.forward(self.ln25.forward(x))) + x
			x = leaky_relu(self.fc26.forward(self.ln26.forward(x))) + x
			x = leaky_relu(self.fc27.forward(self.ln27.forward(x))) + x
			x = leaky_relu(self.fc28.forward(self.ln28.forward(x))) + x
			x = leaky_relu(self.fc29.forward(self.ln29.forward(x))) + x	

			# x = leaky_relu(self.fc30.forward(self.ln30.forward(x))) + x
			# x = leaky_relu(self.fc31.forward(self.ln31.forward(x))) + x
			# x = leaky_relu(self.fc32.forward(self.ln32.forward(x))) + x
			# x = leaky_relu(self.fc33.forward(self.ln33.forward(x))) + x
			# x = leaky_relu(self.fc34.forward(self.ln34.forward(x))) + x

			#x = x.view(-1, 128)	

			# x = leaky_relu(self.fc35.forward(self.ln35.forward(x))) + x
			# x = leaky_relu(self.fc36.forward(self.ln36.forward(x))) + x
			# x = leaky_relu(self.fc37.forward(self.ln37.forward(x))) + x
			# x = leaky_relu(self.fc38.forward(self.ln38.forward(x))) + x
			# x = leaky_relu(self.fc39.forward(self.ln39.forward(x))) + x

			x = self.fc40.forward(self.ln40.forward(x))

			x_ = torch.tanh(x) * 10
			x_[:, 40::42] = x[:, 40::42]
			x_[:, 41::42] = x[:, 41::42]
		return x_
	

	def explor(self, x:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			self.eval()
			#self.state = x.detach().clone()
			action = self.forward(x)
			#with autocast(dtype = torch.float16):
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			
			action_ = (action + noise)

			action_[:, 40::42] = (torch.rand(action[:, 40::42].size()).cuda() * self.var_range - 0.5 * self.var_range + action[:, 40::42]).round()
			action_[:, 41::42] = (torch.rand(action[:, 41::42].size()).cuda() * self.var_range - 0.5 * self.var_range + action[:, 41::42]).round()

		return action_.detach()
	

	def learn(self, td_e:torch.Tensor, state:torch.Tensor, action:torch.Tensor) -> float:

		self.train()
		self.opt.zero_grad()

		output = self.forward(state)		

		#with autocast(dtype = torch.bfloat16):
		action_probs = torch.distributions.Normal(output, self.noise_std).log_prob(action)
			
		with torch.no_grad():
			td_e_ = F.normalize(td_e - torch.mean(td_e), dim = 0)
		loss = -torch.mean(action_probs * td_e_)

		# self.scaler.scale(loss).backward()
		# self.scaler.unscale_(self.opt)
		# torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
		# self.scaler.step(self.opt)
		# self.scaler.update()

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
		self.opt.step()
		return loss.item()