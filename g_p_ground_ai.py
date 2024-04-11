import copy
from turtle import circle
import torch
import torch.cuda
import torch.nn.functional as F
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter

import random

from neural import actor
from neural import critic
from neural import share_net

from collections import deque

import winsound



if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter('D:\\' + name + '_log')
	
	tau = 0.001
	critic_net = critic(1e-3)
	actor_net = actor(0.5, 1e-4)
	share = share_net(1e-4)
	
	critic_target_net = critic(1e-4)
	actor_target_net = actor(0.2, 1e-3)
	target_share = share_net(1e-2)


	cp = torch.load('f:\\场地保留\\asd00000017.pth')
	actor_net.load_state_dict(cp['actor'])
	critic_net.load_state_dict(cp['critic'])
	share.load_state_dict(cp['share'])
	actor_target_net.load_state_dict(cp['actor_target'])
	critic_target_net.load_state_dict(cp['critic_target'])
	target_share.load_state_dict(cp['target_share'])
	cp.clear()

	critic_net.cuda()
	actor_net.cuda()
	share.cuda()
	
	critic_target_net.cuda()
	actor_target_net.cuda()
	target_share.cuda()

	output_pipe = pipe('asd_out')
	input_pipe = pipe('asd_in')
	callback_pipe = pipe('asd_back')
	G_pipe = pipe('asd_G')

	exp_replay = deque(maxlen = 2048)

	i = 0
	i_ = 0
	
	callback_last = 0
	
	for name_, param in actor_net.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_actor_fc/' + name_, param, i)
			#writer.add_histogram('asd_actor_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_actor_ln/' + name_, param, i)
			#writer.add_histogram('asd_actor_ln_grad/' + name_, param.grad, i)
	for name_, param in critic_net.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_critic_fc/' + name_, param, i)
			#writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_critic_ln/' + name_, param, i)
			#writer.add_histogram('asd_critic_ln_grad/' + name_, param.grad, i)
	for name_, param in share.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_share_fc/' + name_, param, i)
			#writer.add_histogram('asd_share_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_share_ln/' + name_, param, i)
			#writer.add_histogram('asd_share_ln_grad/' + name_, param.grad, i)

	_, output_ = output_pipe.recept()

	net_input_ = torch.FloatTensor(list(output_)).cuda().view(-1,955)

	r_sum = 0.0

	while True:
		with torch.no_grad():
			net_input = net_input_.clone()

			share.eval()
			x = share.forward(net_input)
			share.train()

			net_output = actor_net.explor(x)
				
			net_output_ = net_output * 2
			net_output_[:, 40::42] = (net_output[:, 40::42] * 20).round() % 20
			net_output_[:, 41::42] = (net_output[:, 41::42] * 20).round() % 20

			net_output_ = net_output_.view(1,-1).tolist()[0]

			input_pipe.send(net_output_)
		
			_, callback_ = callback_pipe.recept()
			
			callback = torch.FloatTensor([callback_[0]]).cuda()
			
			_, G_ = G_pipe.recept()
			
			
			_, output_ = output_pipe.recept()
			net_input_ = torch.FloatTensor(output_).cuda().view(-1,955)
			
			# if i == 0:
			# 	G_ = [0.0]
			

			r_sum += callback_[0]
			if G_[0] == 0.0:
				exp_replay.append([net_input.view(-1, 955), callback.view(-1,1) + (i - i_) * 0.1, net_input_.view(-1, 955), net_output.view(-1, 336)])				
			else:
				exp_replay.append([net_input.view(-1, 955), callback.view(-1,1) + (i - i_) * 0.1, net_input.view(-1, 955), net_output.view(-1, 336)])				
				try:
					writer.add_scalar('g', G_[0], i)
					writer.add_scalar('r_sum', r_sum, i)
					writer.add_scalar('last_step', i - i_, i)
				except:
					writer = SummaryWriter('.\\' + name + '_log')
					writer.add_scalar('g', G_[0], i)
					writer.add_scalar('r_sum', r_sum, i)
					writer.add_scalar('last_step', i - i_, i)
				i_ = i
				r_sum = 0.0
			

			
		if len(exp_replay) >= 1024:
			with torch.no_grad():
				batch = random.sample(exp_replay, 511)
				batch.append(exp_replay[-1])
				
				net_input, callback, net_input__, action = zip(*batch)
				
				net_input = torch.cat(net_input).view(-1, 955)
				callback = torch.cat(callback).view(-1,1)
				net_input__ = torch.cat(net_input__).view(-1, 955)
				action = torch.cat(action).view(-1, 336)

			critic_net.opt.zero_grad()
			actor_net.opt.zero_grad()
			share.opt.zero_grad()

			x = share.forward(net_input)
			q = critic_net.forward(torch.cat((x, action), dim = 1))
			q_n = q[-1,-1].item()

			with torch.no_grad():
				x_ = target_share.forward(net_input__)
				a_ = actor_target_net.forward(x_)
				q_ = critic_target_net.forward(torch.cat((x_, a_), dim = 1))

			td_e = callback * 1e-3 + 0.9977 * q_ - q
			
			critic_loss_ = td_e.square().mean() + q.square().mean()
			critic_loss = critic_loss_.item()
			
			critic_loss_.backward(retain_graph = True)
			torch.nn.utils.clip_grad_value_(critic_net.parameters(), 1)
			critic_net.opt.step()

			
			action = actor_net.forward(x)
			if torch.any(torch.isnan(action)):
				winsound.Beep(500,3000)
				continue


			q = critic_net.forward(torch.cat((x, action), dim = 1))

			# action = torch.cat([action] * 64, dim = 0)
			# with torch.no_grad():
			# 	noise = torch.normal(mean = 0, std = actor_net.noise_std, size = action.size()).cuda()
			# 	action_ = (action + noise.clamp(-actor_net.noise_std, actor_net.noise_std)).clamp(-1, 1)
			# 	q = critic_net.forward(torch.cat((torch.cat([x] * 64, dim = 0), action_), dim = 1))
			# 	q = ((q - q.mean()) / q.std()).clamp(-1, 1)
				
			# action_probs = torch.distributions.Normal(action, actor_net.noise_std).log_prob(action_).cuda() + actor_net.loss_bais
			
			# actor_loss_ = -(action_probs * q).mean()

			actor_loss_ = -torch.mean(q)
			actor_loss = actor_loss_.item()
			
			actor_loss_.backward()


			torch.nn.utils.clip_grad_value_(actor_net.parameters(), 1)
			torch.nn.utils.clip_grad_value_(share.parameters(), 1)

			
			actor_net.opt.step()
			share.opt.step()

	
		#print(loss);
		
		# if len(callback_) != 0:
			#if(loss != 0.0):

			
			with torch.no_grad():
				for param, target_param in zip(actor_net.parameters(), actor_target_net.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
	
				for param, target_param in zip(critic_net.parameters(), critic_target_net.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

				for param, target_param in zip(share.parameters(), target_share.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

				# for critic_param, actor_param in zip(critic_net.share.parameters(), actor_net.share.parameters()):
				# 	temp_param = ((critic_param + actor_param) * 0.5).clone()
				# 	critic_param.data.copy_(tau * temp_param + (1.0 - tau) * critic_param)
				# 	actor_param.data.copy_(tau * temp_param + (1.0 - tau) * actor_param)
					

			try:
				writer.add_scalar('q', q_n, i)
				writer.add_scalar('actor_loss', actor_loss, i)
				writer.add_scalar('critic_loss', critic_loss, i)
				writer.add_scalar('reward', callback[-1,-1], i)

				if (i % 2000) == 1999:
					for name_, param in actor_net.named_parameters():
						if 'fc' in name_:
							writer.add_histogram('asd_actor_fc/' + name_, param, i)
							writer.add_histogram('asd_actor_fc_grad/' + name_, param.grad, i)
						else:
							writer.add_histogram('asd_actor_ln/' + name_, param, i)
							writer.add_histogram('asd_actor_ln_grad/' + name_, param.grad, i)
					for name_, param in critic_net.named_parameters():
						if 'fc' in name_:
							writer.add_histogram('asd_critic_fc/' + name_, param, i)
							writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
						else:
							writer.add_histogram('asd_critic_ln/' + name_, param, i)
							writer.add_histogram('asd_critic_ln_grad/' + name_, param.grad, i)
					for name_, param in share.named_parameters():
						if 'fc' in name_:
							writer.add_histogram('asd_share_fc/' + name_, param, i)
							writer.add_histogram('asd_share_fc_grad/' + name_, param.grad, i)
						else:
							writer.add_histogram('asd_share_ln/' + name_, param, i)
							writer.add_histogram('asd_share_ln_grad/' + name_, param.grad, i)
			except:
				writer = SummaryWriter('.\\' + name + '_log')
			
			if (i % 2000) == 1999:
				num='{:08d}'.format(int(i / 1000))
				
				
				torch.save({
					#'share':share.state_dict(),
					'actor':actor_net.state_dict(),
					'critic':critic_net.state_dict(),
					'share':share.state_dict(),
					'actor_target':actor_target_net.state_dict(),
					'critic_target':critic_target_net.state_dict(),
					'target_share':target_share.state_dict()
					},'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	