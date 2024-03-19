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


if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter('.\\' + name + '_log')
	
	share = share_net()
	critic_net = critic(share, 1e-7)
	actor_net = actor(share, 1, 1e-7)
	
	# torch.save({
	# 	'share':share.state_dict(),
	# 	'actor':actor_net.state_dict(),
	# 	'actor_opt':actor_net.opt.state_dict(),
	# 	'critic':critic_net.state_dict(),
	# 	'critic_opt':critic_net.opt.state_dict()
	# 	},'f:\\场地\\' + name + num + '.pth')

	# cp = torch.load('f:\\场地保留\\asd00000032.pth')
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

	exp_replay = deque(maxlen = 512)

	i = 0
	
	for name_, param in share.named_parameters():
		writer.add_histogram('asd_share/' + name_, param, i)
		#writer.add_histogram('asd_share_grad/' + name_, param.grad, i)
	for name_, param in actor_net.named_parameters():
		if 'share' not in name_:
			writer.add_histogram('asd_actor/' + name_, param, i)
			#writer.add_histogram('asd_actor_grad/' + name_, param.grad, i)
	for name_, param in critic_net.named_parameters():
		if 'share' not in name_:
			writer.add_histogram('asd_critic/' + name_, param, i)
			#writer.add_histogram('asd_critic_grad/' + name_, param.grad, i)

	_, output_ = output_pipe.recept()

	net_input_ = torch.FloatTensor(list(output_)).view(-1,951).cuda()
	
	while True:
	
		net_input = net_input_.clone()
	
		net_output = actor_net.explor(net_input)

		net_output_ = net_output.view(1,-1).tolist()[0]

		input_pipe.send(net_output_);
	
		_, callback_ = callback_pipe.recept();
		
		callback = torch.FloatTensor([callback_[0]]).cuda()
		
		_, G_ = G_pipe.recept();
		
		
		_, output_ = output_pipe.recept()
		net_input_ = torch.FloatTensor(list(output_)).view(-1,951).cuda()
		
		if i == 0:
			G_ = [0.0]
			
		if G_[0] == 0.0:
			exp_replay.append((net_input.view(1,-1).tolist()[0], callback.view(1,-1).tolist()[0], net_input_.view(1,-1).tolist()[0], net_output.view(1,-1).tolist()[0]))
			
		if len(exp_replay) >= 127:
			batch = random.sample(exp_replay, 127)
			batch.append(exp_replay[-1])
			
			net_input, callback, net_input__, action = zip(*batch)
			net_input = torch.Tensor(net_input).view(-1,951).cuda()
			callback = torch.Tensor(callback).view(-1,1).cuda()
			net_input__ = torch.Tensor(net_input__).view(-1,951).cuda()
			action = torch.Tensor(action).view(-1,336).cuda()

			td_error = critic_net.learn(net_input, callback, net_input__)
			loss = actor_net.learn(td_error, net_input, action)


	
		#print(loss);
		
		# if len(callback_) != 0:
			if(loss != 0.0):
				writer.add_scalar('loss', loss, i)
			writer.add_scalar('reward', callback_[0], i)
			writer.add_scalar('td_error', torch.mean(torch.square(td_error)).item(), i)
			if G_[0] != 0.0:
				writer.add_scalar('g', G_[0], i)

		if (i % 2000) == 0 and i != 0:
			for name_, param in share.named_parameters():
				writer.add_histogram('asd_share/' + name_, param, i)
				writer.add_histogram('asd_share_grad/' + name_, param.grad, i)
			for name_, param in actor_net.named_parameters():
				if 'share' not in name_:
					writer.add_histogram('asd_actor/' + name_, param, i)
					writer.add_histogram('asd_actor_grad/' + name_, param.grad, i)
			for name_, param in critic_net.named_parameters():
				if 'share' not in name_:
					writer.add_histogram('asd_critic/' + name_, param, i)
					writer.add_histogram('asd_critic_grad/' + name_, param.grad, i)
				
		if (i % 2000) == 0 and i != 0:
			
			num='{:08d}'.format(int(i / 1000))
			
			# torch.save(share.state_dict(),'f:\\场地\\share_' + name + num + '.pth')
			# torch.save(actor_net.state_dict(),'f:\\场地\\actor_' + name + num + '.pth')
			# torch.save(critic_net.state_dict(),'f:\\场地\\critic_' + name + num + '.pth')
			
			torch.save({
				'share':share.state_dict(),
				'actor':actor_net.state_dict(),
				'critic':critic_net.state_dict(),
				},'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	



	
