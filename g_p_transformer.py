from pickletools import floatnl
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
		

class self_attention(nn.Module):
	def __init__(self, input_size:int, num_dim:int = None) -> None:
		super(self_attention, self).__init__()
		if num_dim == None:
			num_dim = input_size
		self.w_k = nn.Linear(input_size, num_dim, bias = False)
		self.w_q = nn.Linear(input_size, num_dim, bias = False)
		self.w_v = nn.Linear(input_size, num_dim, bias = False)

		self.dk = num_dim ** 0.5

	def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor) -> torch.Tensor:
		q = self.w_q.forward(q)
		k = self.w_k.forward(k)
		v = self.w_v.forward(v)
		z = torch.matmul(torch.softmax(torch.matmul(q / self.dk, k.transpose(1, 2)), dim = 2), v)
		return z
	
class multi_head_attention(nn.Module):
	def __init__(self, input_size:int, num_dim:int = None, num_head:int = 8) -> None:
		super(multi_head_attention, self).__init__()

		if num_dim == None:
			num_dim = input_size

		self.heads = nn.ModuleList()
		for _ in range(num_head):
			self.heads.append(self_attention(input_size,num_dim))

		self.fc = nn.Linear(num_dim * num_head, input_size)
	def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor) -> torch.Tensor:
		z:list[torch.Tensor] = []
		for i in self.heads:
			z.append(i.forward(q, k, v))
		z = torch.cat(z, dim = 2)
		return self.fc.forward(z)





class encoder(nn.Module):
	def __init__(self, input_size:int, num_dim:int = None, num_head:int = 8) -> None:
		super(encoder, self).__init__()

		self.attention = multi_head_attention(input_size, num_dim, num_head)
		
		self.fc1 = nn.Linear(input_size, input_size)
		self.fc2 = nn.Linear(input_size, input_size)

		self.ln1 = nn.LayerNorm(input_size)
		self.ln2 = nn.LayerNorm(input_size)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.ln1.forward(self.attention.forward(x, x, x) + x)
		x = self.ln2.forward(self.fc1.forward(x).relu() + x)
		return self.fc2.forward(x)


class decoder(nn.Module):
	def __init__(self, input_size:int, num_dim:int = None, num_head:int = 8) -> None:
		super(decoder, self).__init__()

		self.attention1 = multi_head_attention(input_size, num_dim, num_head)
		self.attention2 = multi_head_attention(input_size, num_dim, num_head)

		self.fc1 = nn.Linear(input_size, input_size)
		self.fc2 = nn.Linear(input_size, input_size)

		self.ln1 = nn.LayerNorm(input_size)
		self.ln2 = nn.LayerNorm(input_size)
		self.ln3 = nn.LayerNorm(input_size)

	def forward(self, x:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
		x = self.ln1.forward(self.attention1.forward(x, x, x) + x)
		x = self.ln2.forward(self.attention2.forward(x, c, c) + x)
		x = self.ln3.forward(self.fc1.forward(x).relu() + x)
		return self.fc2.forward(x)


		
class critic(nn.Module):
	def __init__(self, num_encoder:int, num_decoder:int, num_dim:int = None, num_head:int = 8, lr:float = 0.01):
		super(critic, self).__init__()

		self.encoders = nn.ModuleList()
		for _ in range(num_encoder):
			self.encoders.append(encoder(5, num_dim, num_head))

		self.decoders = nn.ModuleList()
		for _ in range(num_decoder):
			self.decoders.append(decoder(5, num_dim, num_head))

		self.ifc = nn.Linear(3, 5)
		self.ofc = nn.Linear(1540, 1)

		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-8)


	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = self.ifc.forward(x)
		x_ = x.clone()
		for i in self.encoders:
			x = i.forward(x)
		for i in self.decoders:
			x_ = i.forward(x_, x)
		x_ = x_.view(x_.size()[0], -1)

		return self.ofc.forward(x_)
		
class actor(nn.Module):
	def __init__(self, num_encoder:int, num_decoder:int, num_dim:int = None, num_head:int = 8, lr:float = 0.01, noise_std:float = 0.1):
		super(actor, self).__init__()

		self.encoders = nn.ModuleList()
		for _ in range(num_encoder):
			self.encoders.append(encoder(5, num_dim, num_head))

		self.decoders = nn.ModuleList()
		for _ in range(num_decoder):
			self.decoders.append(decoder(5, num_dim, num_head))


		self.ifc = nn.Linear(3, 5)
		self.ofc = nn.Linear(860, 272)
		self.ofc.weight.data = self.ofc.weight.data / 32
		self.ofc.bias.data = self.ofc.bias.data / 32

		self.opt = optim.AdamW(self.parameters(), lr = lr, eps = 1e-5, weight_decay = 1e-8)

		self.noise_std = noise_std




	def forward(self, x:torch.Tensor) -> torch.Tensor:

		x = self.ifc.forward(x)

		x_ = x.clone()
		for i in self.encoders:
			x = i.forward(x)
		for i in self.decoders:
			x_ = i.forward(x_, x)
		x_ = x_.view(x_.size()[0], -1)

		return self.ofc.forward(x_).tanh()
	
	def explor(self, x:torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			self.eval()
			action = self.forward(x)				
			noise = torch.normal(mean = 0, std = self.noise_std, size = action.size()).cuda()
			action_ = torch.clamp(action + noise, -1, 1)
		return action_.detach()