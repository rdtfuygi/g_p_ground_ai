from turtle import circle
import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter

import random

from g_p_transformer import actor
from g_p_transformer import critic
#from neural import share_net

from collections import deque

import winsound

import os




log_path = 'D:\\'
if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter(log_path + name + '_log')
	
	tau = 0.001
	critic_net_1 = critic(3, 3, lr = 1e-4, num_head = 2)
	critic_net_2 = critic(3, 3, lr = 1e-4, num_head = 2)
	actor_net = actor(3, 5, lr = 1e-4, num_head = 2, noise_std = 0.1)
	#share = share_net(1e-3)
	
	critic_target_net_1 = critic(3, 3, lr = 1e-4, num_head = 2)
	critic_target_net_2 = critic(3, 3, lr = 1e-4, num_head = 2)
	actor_target_net = actor(3, 5, lr = 1e-4, num_head = 2, noise_std = 0.1)
	#target_share = share_net(1e-2)


	# cp = torch.load('f:\\场地保留\\asd00000060.pth')
	# critic_net_1.load_state_dict(cp['critic_1'])
	# critic_net_2.load_state_dict(cp['critic_2'])
	# critic_target_net_1.load_state_dict(cp['critic_target_1'])
	# critic_target_net_2.load_state_dict(cp['critic_target_2'])

	# actor_net.load_state_dict(cp['actor'])
	# actor_target_net.load_state_dict(cp['actor_target'])

	# cp.clear()


	critic_net_1.cuda()
	critic_net_2.cuda()
	actor_net.cuda()
	#share.cuda()
	
	critic_target_net_1.cuda()
	critic_target_net_2.cuda()
	actor_target_net.cuda()
	#target_share.cuda()

	output_pipe = pipe('asd_out')
	input_pipe = pipe('asd_in')
	callback_pipe = pipe('asd_back')
	G_pipe = pipe('asd_G')

	exp_replay = deque(maxlen = 2048)
	g_exp_replay = []

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
	for name_, param in critic_net_1.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_critic_1_fc/' + name_, param, i)
			#writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_critic_1_ln/' + name_, param, i)
			#writer.add_histogram('asd_critic_ln_grad/' + name_, param.grad, i)
	for name_, param in critic_net_2.named_parameters():
		if 'fc' in name_:
			writer.add_histogram('asd_critic_2_fc/' + name_, param, i)
			#writer.add_histogram('asd_critic_fc_grad/' + name_, param.grad, i)
		else:
			writer.add_histogram('asd_critic_2_ln/' + name_, param, i)
			#writer.add_histogram('asd_critic_ln_grad/' + name_, param.grad, i)

	_, output_ = output_pipe.recept()

	net_input_ = torch.FloatTensor(list(output_)).cuda().view(-1, 379)


	r_sum = 0.0

	c_1_loss_smooth = 0.0
	c_2_loss_smooth = 0.0

	rand_time = int.from_bytes(os.urandom(4), 'big') % 2048

###########################################################################################################################
	while True:
		if i % 2048 == rand_time:
			torch.cuda.manual_seed(int.from_bytes(os.urandom(4), 'big'))
			rand_time = int.from_bytes(os.urandom(4), 'big') % 2048
		with torch.no_grad():
			net_input = net_input_.clone()

			# share.eval()
			# x = share.forward(net_input)
			# share.train()
			actor_net.eval()
			net_output = actor_net.explor(net_input)
			actor_net.train()

			net_output = net_output.view(-1, 42)

			
			net_output_ = net_output * 2
			net_output_[:, 40:] = (net_output_[:, 40:] * 20).round() % 20

			net_output_ = net_output_.view(1,-1).tolist()[0]

			input_pipe.send(net_output_)
		
			_, callback_ = callback_pipe.recept()
			
			callback = torch.FloatTensor([callback_[0]]).cuda()
			
			_, G_ = G_pipe.recept()
			
			
			_, output_ = output_pipe.recept()
			net_input_ = torch.FloatTensor(output_).cuda().view(-1, 379)
			
			# if i == 0:
			# 	G_ = [0.0]
			
###########################################################################################################################

			dist = -(max((net_output[:, 0:40:2].square() + net_output[:, 1:40:2].square()).sum() - 80, 0) + 240) / 16 + 70
			r_sum = r_sum + callback_[0] + dist
			if G_[0] == 0.0:				
				exp_replay.append([net_input.view(-1, 379), callback.view(-1,1) + dist, net_input_.view(-1, 379), net_output.view(-1, 336), 0])		
			else:
				exp_replay.append([net_input.view(-1, 379), callback.view(-1,1) + dist + (i - i_) +  r_sum / (i - i_) * 32, net_input.view(-1, 379), net_output.view(-1, 336), 1])
				g_exp_replay = [net_input.view(-1, 379), callback.view(-1,1) + dist + (i - i_) + r_sum / (i - i_) * 32, net_input.view(-1, 379), net_output.view(-1, 336), 1]

				writer.add_scalar('r/g', G_[0], i)
				writer.add_scalar('r/sum', r_sum, i)
				writer.add_scalar('r/mean', r_sum / (i - i_) * 32, i)
				writer.add_scalar('actor/last_step', i - i_, i)
				
				i_ = i
				r_sum = 0.0
			
###########################################################################################################################
			
		if len(exp_replay) >= 32:
			with torch.no_grad():

				if len(g_exp_replay) != 0:
					batch = random.sample(exp_replay, 31)
					batch.append(g_exp_replay)
				else:
					batch = random.sample(exp_replay, 32)
				# for j in g_exp_replay:
				# 	batch.append(j)
				# batch.append(exp_replay[-1])
				
				net_input, callback, net_input__, action, done = zip(*batch)
				
				net_input = torch.cat(net_input).view(-1, 379)

				callback = torch.cat(callback).view(-1,1)
				
				reward = callback[-1, -1]

				net_input__ = torch.cat(net_input__).view(-1, 379)
				action = torch.cat(action).view(-1, 336)
				done = torch.tensor(done).view(-1,1).cuda()
				a_ = actor_target_net.forward(net_input__)

			writer.add_scalar('r/reward', reward, i)

###########################################################################################################################

			critic_net_1.opt.zero_grad()

			q_1 = critic_net_1.forward(torch.cat((net_input, action), dim = 1))
			q_n_1 = q_1[-1,-1].item()

			with torch.no_grad():
				q__1 = critic_target_net_1.forward(torch.cat((net_input__, a_), dim = 1))
				y_1 = callback + 0.99 * q__1 * (1 - done)

			td_e_1 = y_1 - q_1
			
			critic_loss__1 = td_e_1.square().mean()
			critic_loss_1 = critic_loss__1.item()
			c_1_loss_smooth = 0.1 * critic_loss_1 + 0.9 * c_1_loss_smooth
			if c_1_loss_smooth > 1:
				critic_loss__1.backward()
				# torch.nn.utils.clip_grad_value_(critic_net.parameters(), 1)
				critic_net_1.opt.step()

				if (i % 2000) == 0:
					for name_, param in critic_net_1.named_parameters():
						if 'fc' in name_:
							writer.add_histogram('asd_critic_1_fc/' + name_, param, i)
							writer.add_histogram('asd_critic_1_fc_grad/' + name_, param.grad, i)
						else:
							writer.add_histogram('asd_critic_1_ln/' + name_, param, i)
							writer.add_histogram('asd_critic_1_ln_grad/' + name_, param.grad, i)

			critic_net_1.opt.zero_grad()

			writer.add_scalar('critic/q_1', q_n_1, i)
			writer.add_scalar('critic/loss_1', critic_loss_1, i)

###########################################################################################################################

			critic_net_2.opt.zero_grad()
			
			q_2 = critic_net_2.forward(torch.cat((net_input, action), dim = 1))
			q_n_2 = q_2[-1,-1].item()

			with torch.no_grad():
				q__2 = critic_target_net_2.forward(torch.cat((net_input__, a_), dim = 1))
				y_2 = callback + 0.99 * q__2 * (1 - done)

			td_e_2 = y_2 - q_2
			
			critic_loss__2 = td_e_2.square().mean()
			critic_loss_2 = critic_loss__2.item()
			c_2_loss_smooth = 0.1 * critic_loss_2 + 0.9 * c_2_loss_smooth
			if c_2_loss_smooth > 1:
				critic_loss__2.backward()
				# torch.nn.utils.clip_grad_value_(critic_net.parameters(), 1)
				critic_net_2.opt.step()

				if (i % 2000) == 0:
					for name_, param in critic_net_2.named_parameters():
						if 'fc' in name_:
							writer.add_histogram('asd_critic_2_fc/' + name_, param, i)
							writer.add_histogram('asd_critic_2_fc_grad/' + name_, param.grad, i)
						else:
							writer.add_histogram('asd_critic_2_ln/' + name_, param, i)
							writer.add_histogram('asd_critic_2_ln_grad/' + name_, param.grad, i)

			critic_net_2.opt.zero_grad()

			writer.add_scalar('critic/q_2', q_n_2, i)
			writer.add_scalar('critic/loss_2', critic_loss_2, i)

###########################################################################################################################
#and c_1_loss_smooth < 1000 and c_2_loss_smooth < 1000:
			#if i > 2048 :

			# with torch.no_grad():
			# 	batch = random.sample(exp_replay, 512)
				
			# 	net_input, _, _, _, _ = zip(*batch)
				
			# 	net_input = torch.cat(net_input).view(-1, 379)

			#actor_net.noise_std = 0.1

			actor_net.opt.zero_grad()

			action = actor_net.forward(net_input)
			# if torch.any(torch.isnan(action)) or action.abs().min() > 0.9:
			# 	winsound.Beep(500,3000)
			# 	continue


			q_1 = critic_net_1.forward(torch.cat((net_input, action), dim = 1))
			q_2 = critic_net_2.forward(torch.cat((net_input, action), dim = 1))
			actor_loss_ = -torch.min(q_1, q_2).mean()
			actor_loss = actor_loss_.item()

			# torch.nn.utils.clip_grad_value_(actor_net.parameters(), 1)
			# torch.nn.utils.clip_grad_value_(share.parameters(), 1)

			actor_loss_.backward()
			actor_net.opt.step()

			if (i % 2000) == 0:
				for name_, param in actor_net.named_parameters():
					if 'fc' in name_:
						writer.add_histogram('asd_actor_fc/' + name_, param, i)
						writer.add_histogram('asd_actor_fc_grad/' + name_, param.grad, i)
					else:
						writer.add_histogram('asd_actor_ln/' + name_, param, i)
						writer.add_histogram('asd_actor_ln_grad/' + name_, param.grad, i)

			actor_net.opt.zero_grad()

			writer.add_scalar('actor/loss', actor_loss, i)
			# else:
			# 	actor_net.noise_std = 0.5




###########################################################################################################################


			
			with torch.no_grad():
				for param, target_param in zip(critic_net_1.parameters(), critic_target_net_1.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

				for param, target_param in zip(critic_net_2.parameters(), critic_target_net_2.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

				for param, target_param in zip(actor_net.parameters(), actor_target_net.parameters()):
					target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

			
			if (i % 2000) == 0:
				num='{:08d}'.format(int(i / 1000))				
				
				torch.save({
					#'share':share.state_dict(),
					'actor':actor_net.state_dict(),
					'critic_1':critic_net_1.state_dict(),
					'critic_2':critic_net_2.state_dict(),
					'actor_target':actor_target_net.state_dict(),
					'critic_target_1':critic_target_net_1.state_dict(),
					'critic_target_2':critic_target_net_2.state_dict(),
					},'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	