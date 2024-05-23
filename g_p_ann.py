
#from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
import math




class critic(nn.Module):
	def __init__(self, input_size:int, output_size:int, num_layer:int, num_:int = None, lr:float = 0.01):
		super(critic, self).__init__()
		
		if num_ == None:
			num_ = max(input_size, output_size)

		self.fc = nn.ModuleList()
		self.ln = nn.ModuleList()
		for _ in range(num_layer):
			self.fc.append(nn.Linear(num_, num_))
			self.ln.append(nn.LayerNorm(num_))
			
		self.ifc = nn.Linear(input_size, num_)
		self.ofc = nn.Linear(num_, output_size)

		# self.ofc.weight.data = self.ofc.weight.data
		# self.ofc.bias.data = self.ofc.bias.data

		self.opt = optim.AdamW(self.parameters(), lr = lr, weight_decay = 1e-2)
		
	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.ifc.forward(x)
		for fc, ln in zip(self.fc, self.ln):
			x = F.elu(ln.forward(fc.forward(x))) + x
		x = self.ofc.forward(x)
		return x

class actor(nn.Module):
	def __init__(self, input_size:int, output_size:int, num_layer:int, num_:int = None, lr:float = 0.01, noise_std:float = 0.1):
		super(actor, self).__init__()
		
		if num_ == None:
			num_ = max(input_size, output_size)

		self.fc = nn.ModuleList()
		self.ln = nn.ModuleList()
		self.ln_2 = nn.ModuleList()
		for _ in range(num_layer):
			fc = nn.Linear(num_, num_)
			# fc.weight.data = fc.weight.data / 64
			# fc.bias.data = fc.bias.data / 64
			self.fc.append(fc)
			self.ln.append(nn.LayerNorm(num_))
			self.ln_2.append(nn.LayerNorm(num_))

			
		self.ifc = nn.Linear(input_size, num_)
		# self.ifc.weight.data = self.ifc.weight.data / 64
		# self.ifc.bias.data = self.ifc.bias.data / 64

		self.ofc = nn.Linear(num_, output_size)
		# self.ofc.weight.data = self.ofc.weight.data / 64
		# self.ofc.bias.data = self.ofc.bias.data / 64

		self.oln = nn.LayerNorm(num_)

		self.noise_std = noise_std
		self.opt = optim.AdamW(self.parameters(), lr = lr, weight_decay = 1e-2)
		self.p_b = 1.8378770664093453 + math.log(noise_std)
		
	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.ifc.forward(x)
		for fc, ln, ln_2 in zip(self.fc, self.ln, self.ln_2):
			x = ln_2.forward(F.leaky_relu(ln.forward(fc.forward(x)))) + x
		x = self.ofc.forward(self.oln.forward(x))
		x = x.tanh() + 0.01 * x
		return x
	
	def explor(self, x:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			self.eval()
			action = self.forward(x)				
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			#action_ = torch.clamp(action + noise, -1, 1)
			action_ = action + noise
		return action_.detach()
	
