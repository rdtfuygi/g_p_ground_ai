from turtle import circle
import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter

import random

from neural import actor
from neural import critic
from neural import share_net

from collections import deque

from torch.cuda.amp import autocast
from numba import jit



if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter('.\\' + name + '_log')
	
	share = share_net()
	critic_net = critic(share, 5e-5)
	actor_net = actor(share, 2, 5e-5)
	
	# torch.save({
	# 	'share':share.state_dict(),
	# 	'actor':actor_net.state_dict(),
	# 	'actor_opt':actor_net.opt.state_dict(),
	# 	'critic':critic_net.state_dict(),
	# 	'critic_opt':critic_net.opt.state_dict()
	# 	},'f:\\场地\\' + name + num + '.pth')

	# cp = torch.load('f:\\场地保留\\asd00000003.pth')
	# share.load_state_dict(cp['share'])
	# actor_net.load_state_dict(cp['actor'])
	# critic_net.load_state_dict(cp['critic'])


	share.cuda()

	critic_net.share = share
	actor_net.share = share

	critic_net.cuda()
	actor_net.cuda()

	output_pipe = pipe('asd_out')
	input_pipe = pipe('asd_in')
	callback_pipe = pipe('asd_back')
	G_pipe = pipe('asd_G')

	exp_replay = deque(maxlen = 1024)

	i = 0
	
	for name_, param in share.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_share_fc/' + name_, param, i)
			#writer.add_histogram('asd_share_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_share_bn/' + name_, param, i)
			#writer.add_histogram('asd_share_bn_grad/' + name_, param.grad, i)
	for name_, param in actor_net.named_parameters():
		if 'share' not in name_:
			if 'fc' in name_:
				writer.add_histogram('asd_actor_fc/' + name_, param, i)
				#writer.add_histogram('asd_actor_fc_grad/' + name_, param.grad, i)
			else:
				writer.add_histogram('asd_actor_bn/' + name_, param, i)
				#writer.add_histogram('asd_actor_bn_grad/' + name_, param.grad, i)
	for name_, param in critic_net.named_parameters():
		if 'share' not in name_:
			if 'fc' in name_:
				writer.add_histogram('asd_critic_fc/' + name_, param, i)
				#writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
			else:
				writer.add_histogram('asd_critic_bn/' + name_, param, i)
				#writer.add_histogram('asd_critic_bn_grad/' + name_, param.grad, i)

	_, output_ = output_pipe.recept()

	net_input_ = torch.FloatTensor(list(output_)).view(-1,954).cuda()
	


	while True:
	
		net_input = net_input_.clone()		


		net_output = actor_net.explor(net_input)
			

		net_output_ = net_output.view(1,-1).tolist()[0]

		input_pipe.send(net_output_)
	
		_, callback_ = callback_pipe.recept()
		
		callback = torch.FloatTensor([callback_[0]]).cuda()
		
		_, G_ = G_pipe.recept();
		
		
		_, output_ = output_pipe.recept()
		net_input_ = torch.FloatTensor(output_).cuda()
		
		# if i == 0:
		# 	G_ = [0.0]
		


		if G_[0] == 0.0:
			exp_replay.append((net_input.view(954), callback, net_input_.view(954), net_output.view(336)))
		else:
			writer.add_scalar('g', G_[0], i)
			

			
		if len(exp_replay) >= 128:
			batch = random.sample(exp_replay, 128)
			#batch.append(exp_replay[-1])
			
			net_input, callback, net_input__, action = zip(*batch)
			
			net_input = torch.cat(net_input).view(-1, 954)
			callback = torch.cat(callback).view(-1,1)
			net_input__ = torch.cat(net_input__).view(-1, 954)
			action = torch.cat(action).view(-1, 336)

			td_error, critic_loss = critic_net.learn(net_input, callback, net_input__)
			actor_loss = actor_net.learn(td_error, net_input, action)


	
		#print(loss);
		
		# if len(callback_) != 0:
			#if(loss != 0.0):
			writer.add_scalar('actor_loss', actor_loss, i)
			writer.add_scalar('reward', callback_[0], i)
			writer.add_scalar('critic_loss', critic_loss, i)				

		if (i % 2000) == 1999:
			for name_, param in share.named_parameters():
				if 'fc' in name_:
					writer.add_histogram('asd_share_fc/' + name_, param, i)
					writer.add_histogram('asd_share_fc_grad/' + name_, param.grad, i)
				else:
					writer.add_histogram('asd_share_bn/' + name_, param, i)
					writer.add_histogram('asd_share_bn_grad/' + name_, param.grad, i)
			for name_, param in actor_net.named_parameters():
				if 'share' not in name_:
					if 'fc' in name_:
						writer.add_histogram('asd_actor_fc/' + name_, param, i)
						writer.add_histogram('asd_actor_fc_grad/' + name_, param.grad, i)
					else:
						writer.add_histogram('asd_actor_bn/' + name_, param, i)
						writer.add_histogram('asd_actor_bn_grad/' + name_, param.grad, i)
			for name_, param in critic_net.named_parameters():
				if 'share' not in name_:
					if 'fc' in name_:
						writer.add_histogram('asd_critic_fc/' + name_, param, i)
						writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
					else:
						writer.add_histogram('asd_critic_bn/' + name_, param, i)
						writer.add_histogram('asd_critic_bn_grad/' + name_, param.grad, i)
				
			
			num='{:08d}'.format(int(i / 1000))
			
			
			torch.save({
				'share':share.state_dict(),
				'actor':actor_net.state_dict(),
				'critic':critic_net.state_dict(),
				},'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	



	
