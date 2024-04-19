from pickletools import floatnl
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda

#from torch.cuda.amp import autocast, GradScaler


import math


leaky_relu = nn.LeakyReLU(negative_slope=0.01)

class share_net(nn.Module):
	def __init__(self, lr = 0.01) -> None:
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
		#self.l2_bn = nn.BatchNorm1d(8, eps = 1e-5)
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
		#self.l3_bn = nn.BatchNorm1d(10, eps = 1e-5)

		self.l3k1fc1 = nn.Linear(306, 276)
		self.l3k1fc2 = nn.Linear(276, 215)
		self.l3k1fc3 = nn.Linear(215, 185)
		self.l3k1fc4 = nn.Linear(185, 123)
		self.l3k1fc5 = nn.Linear(123, 93)
		self.l3k1fc6 = nn.Linear(93, 62)
		self.l3k1fc7 = nn.Linear(62, 32)
		self.l3k1fc8 = nn.Linear(32, 1)
		
		#l4
		self.l4resfc = nn.Linear(2764, 1024)
  
		self.l4fc1 = nn.Linear(2764,2416)
		self.l4fc2 = nn.Linear(2416,2068)
		self.l4fc3 = nn.Linear(2068,1720)
		self.l4fc4 = nn.Linear(1720,1372)
		self.l4fc5 = nn.Linear(1372,1024)

		self.l4fc6 = nn.Linear(1024, 1024)
		self.l4fc7 = nn.Linear(1024, 1024)
		self.l4fc8 = nn.Linear(1024, 1024)
		self.l4fc9 = nn.Linear(1024, 1024)
		self.l4fc10 = nn.Linear(1024, 1024)

		self.l4fc11 = nn.Linear(1024, 1024)
		self.l4fc12 = nn.Linear(1024, 1024)
		self.l4fc13 = nn.Linear(1024, 1024)
		self.l4fc14 = nn.Linear(1024, 1024)
		self.l4fc15 = nn.Linear(1024, 1024)
		self.l4fc16 = nn.Linear(1024, 1024)
		self.l4fc17 = nn.Linear(1024, 1024)
		self.l4fc18 = nn.Linear(1024, 1024)
		self.l4fc19 = nn.Linear(1024, 1024)
		self.l4fc20 = nn.Linear(1024, 1024)
		
		self.l4ln1 = nn.LayerNorm(2416, 1e-5)
		self.l4ln2 = nn.LayerNorm(2068, 1e-5)
		self.l4ln3 = nn.LayerNorm(1720, 1e-5)
		self.l4ln4 = nn.LayerNorm(1372, 1e-5)
		self.l4ln5 = nn.LayerNorm(1024, 1e-5)

		self.l4ln6 = nn.LayerNorm(1024, 1e-5)
		self.l4ln7 = nn.LayerNorm(1024, 1e-5)
		self.l4ln8 = nn.LayerNorm(1024, 1e-5)
		self.l4ln9 = nn.LayerNorm(1024, 1e-5)
		self.l4ln10 = nn.LayerNorm(1024, 1e-5)

		self.l4ln11 = nn.LayerNorm(1024, 1e-5)
		self.l4ln12 = nn.LayerNorm(1024, 1e-5)
		self.l4ln13 = nn.LayerNorm(1024, 1e-5)
		self.l4ln14 = nn.LayerNorm(1024, 1e-5)
		self.l4ln15 = nn.LayerNorm(1024, 1e-5)
		self.l4ln16 = nn.LayerNorm(1024, 1e-5)
		self.l4ln17 = nn.LayerNorm(1024, 1e-5)
		self.l4ln18 = nn.LayerNorm(1024, 1e-5)
		self.l4ln19 = nn.LayerNorm(1024, 1e-5)
		self.l4ln20 = nn.LayerNorm(1024, 1e-5)
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-5)
	
	def forward(self, x:torch.Tensor) ->torch.Tensor:
		#with autocast(dtype = torch.float16):
			x = x.view(-1, 955)
			x_s = x[:, -1].clone().view(-1, 1)
			x = x[:, :-1].reshape(-1, 106)
		
			x_ = torch.cat((x[:, 95:100], x[:, 0:100], x[:, 0:5]), dim = 1)
			
			l1k1x = torch.cat((x_[:, 5:105:5], x_[ :, 6:105:5], x_[:, 10::5], x_[:, 11::5]), dim = 1).view(-1, 4)
			l1k1x = torch.tanh(self.l1k1fc1.forward(l1k1x))
			l1k1x = torch.tanh(self.l1k1fc2.forward(l1k1x))
			l1k1x = torch.tanh(self.l1k1fc3.forward(l1k1x))
			
			l1k2x = torch.cat((x_[:, 0:100:5], x_[:, 1:100:5], x_[:, 5:105:5], x_[:, 6:105:5], x_[:, 10::5], x_[:, 11::5]), dim = 1).view(-1, 6)
			l1k2x = torch.tanh(self.l1k2fc1.forward(l1k2x))
			l1k2x = torch.tanh(self.l1k2fc2.forward(l1k2x))
			l1k2x = torch.tanh(self.l1k2fc3.forward(l1k2x))
			
			l1k3x = torch.cat((x_[:, 7:105:5], x_[:, 8:105:5], x_[:, 12::5], x_[:, 13::5]), dim = 1).view(-1, 4)
			l1k3x = torch.tanh(self.l1k3fc1.forward(l1k3x))
			l1k3x = torch.tanh(self.l1k3fc2.forward(l1k3x))
			l1k3x = torch.tanh(self.l1k3fc3.forward(l1k3x))
			
			l1k4x = torch.cat((x_[:, 2:100:5], x_[:, 3:100:5], x_[:, 7:105:5], x_[:, 8:105:5], x_[:, 12::5], x_[:, 13::5]), dim = 1).view(-1, 6)
			l1k4x = torch.tanh(self.l1k4fc1.forward(l1k4x))
			l1k4x = torch.tanh(self.l1k4fc2.forward(l1k4x))
			l1k4x = torch.tanh(self.l1k4fc3.forward(l1k4x))
			
			l1k5x = torch.cat((x_[:, 9:105:5], x_[:, 14::5]), dim = 1).view(-1, 2)
			l1k5x = torch.tanh(self.l1k5fc1.forward(l1k5x))
			l1k5x = torch.tanh(self.l1k5fc2.forward(l1k5x))
			l1k5x = torch.tanh(self.l1k5fc3.forward(l1k5x))
			
			l1k6x = torch.cat((x_[:, 4:100:5], x_[:, 9:105:5], x_[:, 14::5]), dim = 1).view(-1, 3)
			l1k6x = torch.tanh(self.l1k6fc1.forward(l1k6x))
			l1k6x = torch.tanh(self.l1k6fc2.forward(l1k6x))
			l1k6x = torch.tanh(self.l1k6fc3.forward(l1k6x))
			
			l1k7x = torch.cat((x_[:, 5:105:5], x_[:, 6:105:5], x_[:, 7:105:5], x_[:, 8:105:5], x_[:, 9:105:5], x_[:, 10::5], x_[:, 11::5], x_[:, 12::5], x_[:, 13::5], x_[:, 14::5]), dim = 1).view(-1, 10)
			l1k7x = torch.tanh(self.l1k7fc1.forward(l1k7x))
			l1k7x = torch.tanh(self.l1k7fc2.forward(l1k7x))
			l1k7x = torch.tanh(self.l1k7fc3.forward(l1k7x))
			
			l1k8x = torch.cat((x_[:, 0:100:5], x_[:, 1:100:5], x_[:, 2:100:5], x_[:, 3:100:5], x_[:, 4:100:5], x_[:, 5:105:5], x_[:, 6:105:5], x_[:, 7:105:5], x_[:, 8:105:5], x_[:, 9:105:5], x_[:, 10::5], x_[:, 11::5], x_[:, 12::5], x_[:, 13::5], x_[:, 14::5]), dim = 1).view(-1, 15)
			l1k8x = torch.tanh(self.l1k8fc1.forward(l1k8x))
			l1k8x = torch.tanh(self.l1k8fc2.forward(l1k8x))
			l1k8x = torch.tanh(self.l1k8fc3.forward(l1k8x))
			
						
			x_ = x[:, 0:100].reshape(-1, 5)
			x_ = torch.cat((x_, torch.cat((l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x), dim = 1)), dim = 1)
			x_ = x_.view(-1, 260)
			x_ = torch.cat((x_[:, 247:260], x_, x_[:, 0:13]), dim = 1)

			l2k1x = torch.cat((x_[:, 13:273:13], x_[:, 14:273:13], x_[:, 15:273:13], x_[:, 16:273:13], x_[:, 17:273:13], x_[:, 22:273:13], x_[:, 23:273:13], x_[:, 24:273:13], x_[:, 25:273:13], x_[:, 26::13], x_[:, 27::13], x_[:, 28::13], x_[:, 29::13], x_[:, 30::13], x_[:, 35::13], x_[:, 36::13], x_[:, 37::13], x_[:, 38::13]), dim = 1)
			l2k1x = l2k1x.view(-1, 18)
			l2k1x = torch.tanh(self.l2k1fc1.forward(l2k1x))
			l2k1x = torch.tanh(self.l2k1fc2.forward(l2k1x))
			l2k1x = torch.tanh(self.l2k1fc3.forward(l2k1x))
			l2k1x = torch.tanh(self.l2k1fc4.forward(l2k1x))
			
			l2k2x = torch.cat((x_[:, 0:260:13], x_[:, 1:260:13], x_[:, 2:260:13], x_[:, 3:260:13], x_[:, 4:260:13], x_[:, 9:260:13], x_[:, 10:260:13], x_[:, 11:260:13], x_[:, 12:260:13], x_[:, 13:273:13], x_[:, 14:273:13], x_[:, 15:273:13], x_[:, 16:273:13], x_[:, 17:273:13], x_[:, 22:273:13], x_[:, 23:273:13], x_[:, 24:273:13], x_[:, 25:273:13], x_[:, 26::13], x_[:, 27::13], x_[:, 28::13], x_[:, 29::13], x_[:, 30::13], x_[:, 35::13], x_[:, 36::13], x_[:, 37::13], x_[:, 38::13]), dim = 1).view(-1, 27)
			l2k2x = torch.tanh(self.l2k2fc1.forward(l2k2x))
			l2k2x = torch.tanh(self.l2k2fc2.forward(l2k2x))
			l2k2x = torch.tanh(self.l2k2fc3.forward(l2k2x))
			l2k2x = torch.tanh(self.l2k2fc4.forward(l2k2x))
			
			x_ = torch.cat((l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x, l2k1x, l2k2x), dim = 1).view(-1, 200)
			x_ = torch.cat((x, x_), dim = 1)
			
			l3k1x = torch.tanh(self.l3k1fc1.forward(x_))
			l3k1x = torch.tanh(self.l3k1fc2.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc3.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc4.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc5.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc6.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc7.forward(l3k1x))
			l3k1x = torch.tanh(self.l3k1fc8.forward(l3k1x))

			x_ = torch.cat((torch.cat((x_, l3k1x), dim = 1).view(-1, 2763), x_s), dim = 1)

			l4_res_x = self.l4resfc.forward(x_)
			
			x_ = torch.tanh(self.l4ln1.forward(self.l4fc1.forward(x_)))
			x_ = torch.tanh(self.l4ln2.forward(self.l4fc2.forward(x_)))
			x_ = torch.tanh(self.l4ln3.forward(self.l4fc3.forward(x_)))
			x_ = torch.tanh(self.l4ln4.forward(self.l4fc4.forward(x_)))
			x_ = torch.tanh(self.l4ln5.forward(self.l4fc5.forward(x_)) + l4_res_x)

			x_ = torch.tanh(self.l4ln6.forward(self.l4fc6.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln7.forward(self.l4fc7.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln8.forward(self.l4fc8.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln9.forward(self.l4fc9.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln10.forward(self.l4fc10.forward(x_)) + x_)

			x_ = torch.tanh(self.l4ln11.forward(self.l4fc11.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln12.forward(self.l4fc12.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln13.forward(self.l4fc13.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln14.forward(self.l4fc14.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln15.forward(self.l4fc15.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln16.forward(self.l4fc16.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln17.forward(self.l4fc17.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln18.forward(self.l4fc18.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln19.forward(self.l4fc19.forward(x_)) + x_)
			x_ = torch.tanh(self.l4ln20.forward(self.l4fc20.forward(x_)) + x_)
			return x_



class critic(nn.Module):
	def __init__(self, lr:float = 0.01):
		super(critic, self).__init__()

		self.share = share_net()

		self.resfc = nn.Linear(1360, 1)
		
		self.al1k1fc1 = nn.Linear(4, 4)
		self.al1k1fc2 = nn.Linear(4, 4)
		self.al1k1fc3 = nn.Linear(4, 1)

		self.al1k2fc1 = nn.Linear(6, 6)
		self.al1k2fc2 = nn.Linear(6, 6)
		self.al1k2fc3 = nn.Linear(6, 1)
		
		self.al2resfc = nn.Linear(656, 336)

		self.al2fc1 = nn.Linear(656, 592)
		self.al2fc2 = nn.Linear(592, 528)
		self.al2fc3 = nn.Linear(528, 464)
		self.al2fc4 = nn.Linear(464, 400)
		self.al2fc5 = nn.Linear(400, 336)
		
		self.al2ln1 = nn.LayerNorm(592, 1e-5)
		self.al2ln2 = nn.LayerNorm(528, 1e-5)
		self.al2ln3 = nn.LayerNorm(464, 1e-5)
		self.al2ln4 = nn.LayerNorm(400, 1e-5)
		self.al2ln5 = nn.LayerNorm(336, 1e-5)

  
		self.afc1 = nn.Linear(336, 336)
		self.afc2 = nn.Linear(336, 336)
		self.afc3 = nn.Linear(336, 336)


		self.aln1 = nn.LayerNorm(336, 1e-5)
		self.aln2 = nn.LayerNorm(336, 1e-5)
		self.aln3 = nn.LayerNorm(336, 1e-5)



		self.fc21 = nn.Linear(1360,1224)
		self.fc22 = nn.Linear(1224,1088)
		self.fc23 = nn.Linear(1088,952)
		self.fc24 = nn.Linear(952,816)
		self.fc25 = nn.Linear(816,680)
		self.fc26 = nn.Linear(680,544)
		self.fc27 = nn.Linear(544,408)
		self.fc28 = nn.Linear(408,272)
		self.fc29 = nn.Linear(272,136)
		self.fc30 = nn.Linear(136,1)
		
		self.ln21 = nn.LayerNorm(1224, 1e-5)
		self.ln22 = nn.LayerNorm(1088, 1e-5)
		self.ln23 = nn.LayerNorm(952, 1e-5)
		self.ln24 = nn.LayerNorm(816, 1e-5)
		self.ln25 = nn.LayerNorm(680, 1e-5)
		self.ln26 = nn.LayerNorm(544, 1e-5)
		self.ln27 = nn.LayerNorm(408, 1e-5)
		self.ln28 = nn.LayerNorm(272, 1e-5)
		self.ln29 = nn.LayerNorm(136, 1e-5)
		
		#self.scaler = GradScaler()
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-2)
		

	def forward(self, x:torch.Tensor) -> torch.Tensor:
			
			x_0 = self.share.forward(x[:, :-336])

			x_c_ = x[:, -336:].reshape(-1, 42)
			x_c = torch.cat((x_c_[:, -4:-2], x_c_[:, :-2], x_c_[:, :2]), dim = 1)

			x_c1 = torch.cat((x_c[:,2:42:2], x_c[:,3:42:2], x_c[:,4::2], x_c[:,5::2]), dim = 1).view(-1, 4)
			x_c1 = torch.tanh(self.al1k1fc1.forward(x_c1))
			x_c1 = torch.tanh(self.al1k1fc2.forward(x_c1))
			x_c1 = torch.tanh(self.al1k1fc3.forward(x_c1)).view(-1, 20)			

			x_c2 = torch.cat((x_c[:,0:40:2], x_c[:,1:40:2], x_c[:,2:42:2], x_c[:,3:42:2], x_c[:,4::2], x_c[:,5::2]), dim = 1).view(-1, 6)
			x_c2 = torch.tanh(self.al1k2fc1.forward(x_c2))
			x_c2 = torch.tanh(self.al1k2fc2.forward(x_c2))
			x_c2 = torch.tanh(self.al1k2fc3.forward(x_c2)).view(-1, 20)

			x_c_ = torch.cat((x_c_, x_c1, x_c2), dim = 1).view(-1, 656)

			

			x_ = torch.tanh(self.al2ln1.forward(self.al2fc1.forward(x_c_)))
			x_ = torch.tanh(self.al2ln2.forward(self.al2fc2.forward(x_)))
			x_ = torch.tanh(self.al2ln3.forward(self.al2fc3.forward(x_)))
			x_ = torch.tanh(self.al2ln4.forward(self.al2fc4.forward(x_)))
			x_ = torch.tanh(self.al2ln5.forward(self.al2fc5.forward(x_)) + self.al2resfc.forward(x_c_))



			x_ = torch.tanh(self.aln1.forward(self.afc1.forward(x_)) + x_)
			x_ = torch.tanh(self.aln2.forward(self.afc2.forward(x_)) + x_)
			x_ = torch.tanh(self.aln3.forward(self.afc3.forward(x_)) + x_)


			x__ = torch.cat((x_0, x_), dim = 1)
			res_x = self.resfc.forward(x__)

			x__ = leaky_relu(self.ln21.forward(self.fc21.forward(x__)))
			x__ = leaky_relu(self.ln22.forward(self.fc22.forward(x__)))
			x__ = leaky_relu(self.ln23.forward(self.fc23.forward(x__)))
			x__ = leaky_relu(self.ln24.forward(self.fc24.forward(x__)))
			x__ = leaky_relu(self.ln25.forward(self.fc25.forward(x__)))
			x__ = leaky_relu(self.ln26.forward(self.fc26.forward(x__)))
			x__ = leaky_relu(self.ln27.forward(self.fc27.forward(x__)))
			x__ = leaky_relu(self.ln28.forward(self.fc28.forward(x__)))
			x__ = leaky_relu(self.ln29.forward(self.fc29.forward(x__)))

			x__ = self.fc30.forward(x__) + res_x
			return x__
	

	# def learn(self, s:torch.Tensor, a:torch.Tensor, r:torch.Tensor, t:critic, s_new:torch.Tensor, a_new:torch.Tensor):
		
	# 	self.train()
	# 	#self.opt.zero_grad()

	# 	#print(s.size())

	# 	x = torch.cat((s, a), dim = 1)

	# 	v = self.forward(x)
		
	# 	x_new = torch.cat((s_new, a_new), dim = 1)
	# 	v_new = t.forward(x_new)

	# 	#with autocast(dtype = torch.bfloat16):
	# 	td_e = 0.99 * v_new + r - v
	# 	loss = torch.mean(torch.square(td_e))
		
	# 	# self.scaler.scale(loss).backward()
	# 	# self.scaler.unscale_(self.opt)
	# 	# torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
	# 	# self.scaler.step(self.opt)
	# 	# self.scaler.update()

	# 	loss.backward()
	# 	torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
	# 	#self.opt.step()


	# 	return (td_e.detach(), loss.item())


class actor(nn.Module):
	def __init__(self, noise_std:float = 0.1, lr:float = 0.01):
		super(actor, self).__init__()
		
		self.share = share_net()

		self.resfc1 = nn.Linear(1024, 336)
		self.resfc1.weight.data = self.resfc1.weight.data / 32
		self.resfc1.bias.data = self.resfc1.bias.data / 32
		self.resfc2 = nn.Linear(680, 336)
		self.resfc2.weight.data = self.resfc2.weight.data / 32
		self.resfc2.bias.data = self.resfc2.bias.data / 32

		self.fc21 = nn.Linear(1024,989)
		self.fc22 = nn.Linear(989,955)
		self.fc23 = nn.Linear(955,920)
		self.fc24 = nn.Linear(920,886)
		self.fc25 = nn.Linear(886,852)
		self.fc26 = nn.Linear(852,817)
		self.fc27 = nn.Linear(817,783)
		self.fc28 = nn.Linear(783,748)
		self.fc29 = nn.Linear(748,714)
		self.fc30 = nn.Linear(714,680)
		self.fc31 = nn.Linear(680,645)
		self.fc32 = nn.Linear(645,611)
		self.fc33 = nn.Linear(611,576)
		self.fc34 = nn.Linear(576,542)
		self.fc35 = nn.Linear(542,508)
		self.fc36 = nn.Linear(508,473)
		self.fc37 = nn.Linear(473,439)
		self.fc38 = nn.Linear(439,404)
		self.fc39 = nn.Linear(404,370)
		self.fc40 = nn.Linear(370,336)
		self.fc40.weight.data = self.fc40.weight.data / 32
		self.fc40.bias.data = self.fc40.bias.data / 32
		#torch.nn.init.uniform_(self.fc40.weight, -0.001, 0.001)


		self.ln21 = nn.LayerNorm(989, 1e-5)
		self.ln22 = nn.LayerNorm(955, 1e-5)
		self.ln23 = nn.LayerNorm(920, 1e-5)
		self.ln24 = nn.LayerNorm(886, 1e-5)
		self.ln25 = nn.LayerNorm(852, 1e-5)
		self.ln26 = nn.LayerNorm(817, 1e-5)
		self.ln27 = nn.LayerNorm(783, 1e-5)
		self.ln28 = nn.LayerNorm(748, 1e-5)
		self.ln29 = nn.LayerNorm(714, 1e-5)
		self.ln30 = nn.LayerNorm(680, 1e-5)
		self.ln31 = nn.LayerNorm(645, 1e-5)
		self.ln32 = nn.LayerNorm(611, 1e-5)
		self.ln33 = nn.LayerNorm(576, 1e-5)
		self.ln34 = nn.LayerNorm(542, 1e-5)
		self.ln35 = nn.LayerNorm(508, 1e-5)
		self.ln36 = nn.LayerNorm(473, 1e-5)
		self.ln37 = nn.LayerNorm(439, 1e-5)
		self.ln38 = nn.LayerNorm(404, 1e-5)
		self.ln39 = nn.LayerNorm(370, 1e-5)
		#self.ln40 = nn.LayerNorm(336, 1e-5)


		self.noise_std = noise_std
		self.var_range = math.sqrt(noise_std * 12)
		
		#self.way=[]

		#self.exp_replay = deque(maxlen = 2048)
		# self.state = torch.Tensor()
		# self.action = torch.Tensor()
		#self.scaler = GradScaler()
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-2)
		
		self.loss_bais = 0.91893853320467274178032973640562 + math.log(self.noise_std)
		#self.reward_bais = 0.0
		#self.reward_l_bais = 0.0
	

	def forward(self, x:torch.Tensor) -> torch.Tensor:
			x = self.share.forward(x)

		#with autocast(dtype = torch.float16):
			res1_x = self.resfc1.forward(x)

			x = torch.tanh(self.ln21.forward(self.fc21.forward(x)))
			x = torch.tanh(self.ln22.forward(self.fc22.forward(x)))
			x = torch.tanh(self.ln23.forward(self.fc23.forward(x)))
			x = torch.tanh(self.ln24.forward(self.fc24.forward(x)))
			x = torch.tanh(self.ln25.forward(self.fc25.forward(x)))
			x = torch.tanh(self.ln26.forward(self.fc26.forward(x)))
			x = torch.tanh(self.ln27.forward(self.fc27.forward(x)))
			x = torch.tanh(self.ln28.forward(self.fc28.forward(x)))
			x = torch.tanh(self.ln29.forward(self.fc29.forward(x)))
			x = torch.tanh(self.ln30.forward(self.fc30.forward(x)))

			res2_x = self.resfc2.forward(x)

			x = leaky_relu(self.ln31.forward(self.fc31.forward(x)))
			x = leaky_relu(self.ln32.forward(self.fc32.forward(x)))
			x = leaky_relu(self.ln33.forward(self.fc33.forward(x)))
			x = leaky_relu(self.ln34.forward(self.fc34.forward(x)))
			x = leaky_relu(self.ln35.forward(self.fc35.forward(x)))
			x = leaky_relu(self.ln36.forward(self.fc36.forward(x)))
			x = leaky_relu(self.ln37.forward(self.fc37.forward(x)))
			x = leaky_relu(self.ln38.forward(self.fc38.forward(x)))
			x = leaky_relu(self.ln39.forward(self.fc39.forward(x)))

			x = torch.tanh(self.fc40.forward(x) + res1_x + res2_x)

			# x_ = leaky_relu(x) * 2
			# x_[:, 40::42] = x[:, 40::42] * 20 % 20
			# x_[:, 41::42] = x[:, 41::42] * 20 % 20
			return x
	

	def explor(self, x:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			#self.eval()
			#self.state = x.detach().clone()
			action = self.forward(x)
			
			# with autocast(dtype = torch.float16):
				
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			
			action_ = torch.clamp(action + noise, -1, 1)

			# action_[:, 40::42] = (torch.rand(action[:, 40::42].size()).cuda() * 2 * self.var_range - 1 * self.var_range + action[:, 40::42]).round() % 20
			# action_[:, 41::42] = (torch.rand(action[:, 41::42].size()).cuda() * 2 * self.var_range - 1 * self.var_range + action[:, 41::42]).round() % 20

		return action_.detach()
	

	# def learn(self, td_e:torch.Tensor, state:torch.Tensor, action:torch.Tensor) -> float:

	# 	self.train()
	# 	#self.opt.zero_grad()

	# 	output = self.forward(state)

	# 	#with autocast(dtype = torch.bfloat16):
	# 	action_probs = torch.distributions.Normal(output, self.noise_std).log_prob(action)
			
	# 	with torch.no_grad():
	# 		td_e_ = F.normalize(td_e - torch.mean(td_e), dim = 0)
	# 	loss = -torch.mean(action_probs * td_e_)

	# 	# self.scaler.scale(loss).backward()
	# 	# self.scaler.unscale_(self.opt)
	# 	# torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
	# 	# self.scaler.step(self.opt)
	# 	# self.scaler.update()

	# 	loss.backward()
	# 	torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
	# 	#self.opt.step()
	# 	return loss.item()