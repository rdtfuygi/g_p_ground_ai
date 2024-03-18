from turtle import circle
import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter

import random

from neural import actor
from neural import critic

from collections import deque


if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter('.\\' + name + '_log')
	
	critic_net = critic(1e-4)
	actor_net = actor(1, 1e-3)

	#critic_net.load_state_dict(torch.load('F:\\场地保留\\critic_asd00000082.pth'))
	#actor_net.load_state_dict(torch.load('F:\\场地保留\\actor_asd00000082.pth'))

	critic_net.cuda()
	actor_net.cuda()

	output_pipe = pipe('asd_out') 
	input_pipe = pipe('asd_in') 
	callback_pipe = pipe('asd_back') 
	G_pipe = pipe('asd_G')

	exp_replay = deque(maxlen = 128)

	i = 0
	
	for name_, param in actor_net.named_parameters():
		writer.add_histogram('asd_actor\\/' + name_, param, i)
	for name_, param in critic_net.named_parameters():
		writer.add_histogram('asd_critic\\/' + name_, param, i)

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
			
			batch = random.sample(exp_replay, min(len(exp_replay), 31))
			batch.append(exp_replay[-1])
			
			net_input, callback, net_input__, action = zip(*batch)
			net_input = torch.Tensor(net_input).view(-1,951).cuda()
			callback = torch.Tensor(callback).view(-1,1).cuda()
			net_input__ = torch.Tensor(net_input__).view(-1,951).cuda()
			action = torch.Tensor(action).view(-1,336).cuda()

			td_error = critic_net.learn(net_input, callback, net_input__)
			loss = actor_net.learn(td_error, net_input, action)


	
		#print(loss);
		
		if len(callback_) != 0:
			if(loss != 0.0):
				writer.add_scalar('loss', loss, i)
			writer.add_scalar('reward', callback_[0], i)
			writer.add_scalar('td_error', torch.sum(torch.square(td_error)).item(), i)
			if G_[0] != 0.0:
				writer.add_scalar('g', G_[0], i)

		if (i % 2000) == 0 and i != 0:
			for name_, param in actor_net.named_parameters():
				writer.add_histogram('asd_actor\\/' + name_, param, i)
			for name_, param in critic_net.named_parameters():
				writer.add_histogram('asd_critic\\/' + name_, param, i)
				
		if (i % 2000) == 0 and i != 0:
			
			num='{:08d}'.format(int(i / 1000))			

			torch.save(actor_net.state_dict(),'f:\\场地\\actor_' + name + num + '.pth')
			torch.save(critic_net.state_dict(),'f:\\场地\\critic_' + name + num + '.pth')
		
		i += 1
	
	



	
