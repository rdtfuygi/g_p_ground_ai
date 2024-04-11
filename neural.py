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
		self.l2_bn = nn.BatchNorm1d(8, eps = 1e-5)
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
		self.l3_bn = nn.BatchNorm1d(10, eps = 1e-5)

		self.l3k1fc1 = nn.Linear(306, 276)
		self.l3k1fc2 = nn.Linear(276, 215)
		self.l3k1fc3 = nn.Linear(215, 185)
		self.l3k1fc4 = nn.Linear(185, 123)
		self.l3k1fc5 = nn.Linear(123, 93)
		self.l3k1fc6 = nn.Linear(93, 62)
		self.l3k1fc7 = nn.Linear(62, 32)
		self.l3k1fc8 = nn.Linear(32, 1)
		
		#l4
		self.l4resfc = nn.Linear(2764, 2048)
  
		self.l4fc1 = nn.Linear(2764, 2691)
		self.l4fc2 = nn.Linear(2691, 2620)
		self.l4fc3 = nn.Linear(2620, 2548)
		self.l4fc4 = nn.Linear(2548, 2477)
		self.l4fc5 = nn.Linear(2477, 2405)
		self.l4fc6 = nn.Linear(2405, 2334)
		self.l4fc7 = nn.Linear(2334, 2262)
		self.l4fc8 = nn.Linear(2262, 2191)
		self.l4fc9 = nn.Linear(2191, 2119)
		self.l4fc10 = nn.Linear(2119, 2048)
		
		self.l4ln1 = nn.LayerNorm(2691, 1e-5)
		self.l4ln2 = nn.LayerNorm(2620, 1e-5)
		self.l4ln3 = nn.LayerNorm(2548, 1e-5)
		self.l4ln4 = nn.LayerNorm(2477, 1e-5)
		self.l4ln5 = nn.LayerNorm(2405, 1e-5)
		self.l4ln6 = nn.LayerNorm(2334, 1e-5)
		self.l4ln7 = nn.LayerNorm(2262, 1e-5)
		self.l4ln8 = nn.LayerNorm(2191, 1e-5)
		self.l4ln9 = nn.LayerNorm(2119, 1e-5)
		self.l4ln10 = nn.LayerNorm(2048, 1e-5)
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-5)
	
	def forward(self, x:torch.Tensor) ->torch.Tensor:
		#with autocast(dtype = torch.float16):
			x_s = x[:,-1].clone().view(-1, 1)
			x = x[:,:-1].reshape(-1, 106)
		
			x_ = torch.cat((x[:,95:100], x[:,0:100], x[:,0:5]), dim = 1)
			
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
			x_ = torch.cat((x_, self.l2_bn.forward(torch.cat((l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x), dim = 1))), dim = 1)
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
			
			
			x_ = self.l3_bn.forward(torch.cat((l1k1x, l1k3x, l1k5x, l1k7x, l1k2x, l1k4x, l1k6x, l1k8x, l2k1x, l2k2x), dim = 1)).view(-1, 200)
			x_ = torch.cat((x, x_), dim = 1)
			
			l3k1x = leaky_relu(self.l3k1fc1.forward(x_))
			l3k1x = leaky_relu(self.l3k1fc2.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc3.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc4.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc5.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc6.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc7.forward(l3k1x))
			l3k1x = leaky_relu(self.l3k1fc8.forward(l3k1x))

			x_ = torch.cat((torch.cat((x_, l3k1x), dim = 1).view(-1, 2763), x_s), dim = 1)

			l4_res_x = self.l4resfc.forward(x_)
			
			x_ = leaky_relu(self.l4ln1.forward(self.l4fc1.forward(x_)))
			x_ = leaky_relu(self.l4ln2.forward(self.l4fc2.forward(x_)))
			x_ = leaky_relu(self.l4ln3.forward(self.l4fc3.forward(x_)))
			x_ = leaky_relu(self.l4ln4.forward(self.l4fc4.forward(x_)))
			x_ = leaky_relu(self.l4ln5.forward(self.l4fc5.forward(x_)))
			x_ = leaky_relu(self.l4ln6.forward(self.l4fc6.forward(x_)))
			x_ = leaky_relu(self.l4ln7.forward(self.l4fc7.forward(x_)))
			x_ = leaky_relu(self.l4ln8.forward(self.l4fc8.forward(x_)))
			x_ = leaky_relu(self.l4ln9.forward(self.l4fc9.forward(x_)))
			x_ = leaky_relu(self.l4ln10.forward(self.l4fc10.forward(x_)) + l4_res_x)
			return x_



class critic(nn.Module):
	def __init__(self, lr:float = 0.01):
		super(critic, self).__init__()
		# self.share = share_net()
		self.resfc = nn.Linear(2384, 1)
  
		self.afc1 = nn.Linear(336, 336)
		self.afc2 = nn.Linear(336, 336)
		self.afc3 = nn.Linear(336, 336)
		self.afc4 = nn.Linear(336, 336)
		self.afc5 = nn.Linear(336, 336)
		self.afc6 = nn.Linear(336, 336)
		self.afc7 = nn.Linear(336, 336)
		self.afc8 = nn.Linear(336, 336)
		self.afc9 = nn.Linear(336, 336)
		self.afc10 = nn.Linear(336, 336)

		self.aln1 = nn.LayerNorm(336, 1e-5)
		self.aln2 = nn.LayerNorm(336, 1e-5)
		self.aln3 = nn.LayerNorm(336, 1e-5)
		self.aln4 = nn.LayerNorm(336, 1e-5)
		self.aln5 = nn.LayerNorm(336, 1e-5)
		self.aln6 = nn.LayerNorm(336, 1e-5)
		self.aln7 = nn.LayerNorm(336, 1e-5)
		self.aln8 = nn.LayerNorm(336, 1e-5)
		self.aln9 = nn.LayerNorm(336, 1e-5)
		self.aln10 = nn.LayerNorm(336, 1e-5)


		self.fc21 = nn.Linear(2384, 2145)
		self.fc22 = nn.Linear(2145, 1907)
		self.fc23 = nn.Linear(1907, 1669)
		self.fc24 = nn.Linear(1669, 1430)
		self.fc25 = nn.Linear(1430, 1192)
		self.fc26 = nn.Linear(1192, 954)
		self.fc27 = nn.Linear(954, 715)
		self.fc28 = nn.Linear(715, 477)
		self.fc29 = nn.Linear(477, 239)
		self.fc30 = nn.Linear(239, 1)
		
		self.ln21 = nn.LayerNorm(2145, 1e-5)
		self.ln22 = nn.LayerNorm(1907, 1e-5)
		self.ln23 = nn.LayerNorm(1669, 1e-5)
		self.ln24 = nn.LayerNorm(1430, 1e-5)
		self.ln25 = nn.LayerNorm(1192, 1e-5)
		self.ln26 = nn.LayerNorm(954, 1e-5)
		self.ln27 = nn.LayerNorm(715, 1e-5)
		self.ln28 = nn.LayerNorm(477, 1e-5)
		self.ln29 = nn.LayerNorm(239, 1e-5)
		
		#self.scaler = GradScaler()
		
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-5)
		

	def forward(self, x:torch.Tensor) -> torch.Tensor:
			x_ = x[:, -336:]

			x_ = leaky_relu(self.aln1.forward(self.afc1.forward(x_)) + x_)
			x_ = leaky_relu(self.aln2.forward(self.afc2.forward(x_)) + x_)
			x_ = leaky_relu(self.aln3.forward(self.afc3.forward(x_)) + x_)
			x_ = leaky_relu(self.aln4.forward(self.afc4.forward(x_)) + x_)
			x_ = leaky_relu(self.aln5.forward(self.afc5.forward(x_)) + x_)
			x_ = leaky_relu(self.aln6.forward(self.afc6.forward(x_)) + x_)
			x_ = leaky_relu(self.aln7.forward(self.afc7.forward(x_)) + x_)
			x_ = leaky_relu(self.aln8.forward(self.afc8.forward(x_)) + x_)
			x_ = leaky_relu(self.aln9.forward(self.afc9.forward(x_)) + x_)
			x_ = leaky_relu(self.aln10.forward(self.afc10.forward(x_)) + x_)

			x__ = torch.cat((x[:, :-336],x_), dim = 1)
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

			x__ = leaky_relu((self.fc30.forward(x__) + res_x) * 0.0625)
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
		
		# self.share = share_net()

		self.resfc = nn.Linear(2048, 336)

		self.fc21 = nn.Linear(2048, 1962)
		self.fc22 = nn.Linear(1962, 1876)
		self.fc23 = nn.Linear(1876, 1791)
		self.fc24 = nn.Linear(1791, 1705)
		self.fc25 = nn.Linear(1705, 1620)
		self.fc26 = nn.Linear(1620, 1534)
		self.fc27 = nn.Linear(1534, 1448)
		self.fc28 = nn.Linear(1448, 1363)
		self.fc29 = nn.Linear(1363, 1277)
		self.fc30 = nn.Linear(1277, 1192)
		self.fc31 = nn.Linear(1192, 1106)
		self.fc32 = nn.Linear(1106, 1020)
		self.fc33 = nn.Linear(1020, 935)
		self.fc34 = nn.Linear(935, 849)
		self.fc35 = nn.Linear(849, 764)
		self.fc36 = nn.Linear(764, 678)
		self.fc37 = nn.Linear(678, 592)
		self.fc38 = nn.Linear(592, 507)
		self.fc39 = nn.Linear(507, 421)
		self.fc40 = nn.Linear(421, 336)
		#torch.nn.init.uniform_(self.fc40.weight, -0.001, 0.001)


		self.ln21 = nn.LayerNorm(1962, 1e-5)
		self.ln22 = nn.LayerNorm(1876, 1e-5)
		self.ln23 = nn.LayerNorm(1791, 1e-5)
		self.ln24 = nn.LayerNorm(1705, 1e-5)
		self.ln25 = nn.LayerNorm(1620, 1e-5)
		self.ln26 = nn.LayerNorm(1534, 1e-5)
		self.ln27 = nn.LayerNorm(1448, 1e-5)
		self.ln28 = nn.LayerNorm(1363, 1e-5)
		self.ln29 = nn.LayerNorm(1277, 1e-5)
		self.ln30 = nn.LayerNorm(1192, 1e-5)
		self.ln31 = nn.LayerNorm(1106, 1e-5)
		self.ln32 = nn.LayerNorm(1020, 1e-5)
		self.ln33 = nn.LayerNorm(935, 1e-5)
		self.ln34 = nn.LayerNorm(849, 1e-5)
		self.ln35 = nn.LayerNorm(764, 1e-5)
		self.ln36 = nn.LayerNorm(678, 1e-5)
		self.ln37 = nn.LayerNorm(592, 1e-5)
		self.ln38 = nn.LayerNorm(507, 1e-5)
		self.ln39 = nn.LayerNorm(421, 1e-5)

		#self.ln40 = nn.LayerNorm(336, 1e-5)


		self.noise_std = noise_std
		self.var_range = math.sqrt(noise_std * 12)
		
		#self.way=[]

		#self.exp_replay = deque(maxlen = 2048)
		# self.state = torch.Tensor()
		# self.action = torch.Tensor()
		#self.scaler = GradScaler()
		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-9)
		
		self.loss_bais = 0.91893853320467274178032973640562 + math.log(self.noise_std)
		#self.reward_bais = 0.0
		#self.reward_l_bais = 0.0
	

	def forward(self, x:torch.Tensor) -> torch.Tensor:
			# x = self.share.forward(x)

		#with autocast(dtype = torch.float16):
			res_x = self.resfc.forward(x)

			x = leaky_relu(self.ln21.forward(self.fc21.forward(x)))
			x = leaky_relu(self.ln22.forward(self.fc22.forward(x)))
			x = leaky_relu(self.ln23.forward(self.fc23.forward(x)))
			x = leaky_relu(self.ln24.forward(self.fc24.forward(x)))
			x = leaky_relu(self.ln25.forward(self.fc25.forward(x)))
			x = leaky_relu(self.ln26.forward(self.fc26.forward(x)))
			x = leaky_relu(self.ln27.forward(self.fc27.forward(x)))
			x = leaky_relu(self.ln28.forward(self.fc28.forward(x)))
			x = leaky_relu(self.ln29.forward(self.fc29.forward(x)))
			x = leaky_relu(self.ln30.forward(self.fc30.forward(x)))
			x = leaky_relu(self.ln31.forward(self.fc31.forward(x)))
			x = leaky_relu(self.ln32.forward(self.fc32.forward(x)))
			x = leaky_relu(self.ln33.forward(self.fc33.forward(x)))
			x = leaky_relu(self.ln34.forward(self.fc34.forward(x)))
			x = leaky_relu(self.ln35.forward(self.fc35.forward(x)))
			x = leaky_relu(self.ln36.forward(self.fc36.forward(x)))
			x = leaky_relu(self.ln37.forward(self.fc37.forward(x)))
			x = leaky_relu(self.ln38.forward(self.fc38.forward(x)))
			x = leaky_relu(self.ln39.forward(self.fc39.forward(x)))

			x = torch.tanh(self.fc40.forward(x) + res_x)

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