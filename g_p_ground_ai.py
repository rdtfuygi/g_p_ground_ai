#from turtle import circle
import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter

import random

from g_p_ann import actor
from g_p_ann import critic
#from neural import share_net

from collections import deque


import os





if __name__ == '__main__':

	name = sys.argv[1]

	
	writer = SummaryWriter(name + '_log')
	
	tau = 1e-3
	td_s = 0.99

	c_l = 50
	c_lr = 1e-5

	critic_net_1 = critic(1082, 1, c_l, 1024, lr = c_lr)
	critic_net_2 = critic(1083, 1, c_l, lr = c_lr)
	actor_net = actor(810, 272, 200, lr = 1e-7, noise_std = 0.1)

	
	critic_target_net_1 = critic(1082, 1, c_l, 1024, lr = c_lr)
	critic_target_net_2 = critic(1083, 1, c_l, lr = c_lr)
	actor_target_net = actor(810, 272, 200, lr = 1e-6, noise_std = 0.1)

	with torch.no_grad():
		for param, target_param in zip(critic_net_1.parameters(), critic_target_net_1.parameters()):
			target_param.data.copy_(param.data)

		for param, target_param in zip(critic_net_2.parameters(), critic_target_net_2.parameters()):
			target_param.data.copy_(param.data)

		for param, target_param in zip(actor_net.parameters(), actor_target_net.parameters()):
			target_param.data.copy_(param.data)



	# cp = torch.load('F:\\场地保留\\asd00000022.pth')
	# critic_net_1.load_state_dict(cp['critic_1'])
	# #critic_net_2.load_state_dict(cp['critic_2'])
	# critic_target_net_1.load_state_dict(cp['critic_target_1'])
	# #critic_target_net_2.load_state_dict(cp['critic_target_2'])

	# actor_net.load_state_dict(cp['actor'])
	# actor_target_net.load_state_dict(cp['actor_target'])

	# cp.clear()


	critic_net_1.cuda()
	critic_net_2.cuda()
	actor_net.cuda()
	
	critic_target_net_1.cuda()
	critic_target_net_1.eval()
	critic_target_net_2.cuda()
	critic_target_net_2.eval()
	actor_target_net.cuda()
	actor_target_net.eval()


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






	r_sum = 0.0

	c_1_loss_smooth = 0.0
	c_2_loss_smooth = 0.0
	r_smooth = 0.0

	rand_time = int.from_bytes(os.urandom(4), 'big') % 2048

	_, output_ = output_pipe.recept()
	output_ =list(output_)
	net_input_ = torch.FloatTensor(output_).cuda().view(-1, 810)

	i = 1
	actor_learn = False
	c_1_l = True
	c_2_l = True


###########################################################################################################################
	while True:
		if i % 2048 == rand_time:
			torch.cuda.manual_seed(int.from_bytes(os.urandom(4), 'big'))
			rand_time = int.from_bytes(os.urandom(4), 'big') % 2048
		with torch.no_grad():
			net_input = net_input_.clone()

			actor_net.eval()
			net_output = actor_net.explor(net_input)
			actor_net.train()

			net_output_ = net_output

			net_output_ = net_output_.view(-1, 34) * 2

			net_output_[:, -2:] = (net_output_[:, -2:] * 16).round() % 16

			net_output_ = net_output_.view(1,-1).tolist()[0]

			# done = ((net_output[:, -1].item() + 1) * 0.5) ** 3

			# if random.random() < (done + 1) * 0.5:
			# 	done = 1.0
			# else:
			# 	done = 0.0

			net_output_.append(0.0)
			input_pipe.send(net_output_)

			net_output = net_output.view(-1, 272)
		
			_, callback_ = callback_pipe.recept()
			
			callback = torch.FloatTensor([callback_[0]]).cuda()
			
			_, G_ = G_pipe.recept()
			
			
			_, output_ = output_pipe.recept()
			output_ =list(output_)
			net_input_ = torch.FloatTensor(output_).cuda().view(-1, 810)
			
			# if i == 0:
			# 	G_ = [0.0]
			
###########################################################################################################################

			#dist = -0.1 * (net_output.square().sum() - 273) + 100
			r_sum = r_sum + callback_[0]
			if G_[0] == 0.0:
				exp_replay.append([net_input, (callback.view(-1,1)) / 512, net_input_, net_output, 1])
				# if i - i_ != 1:
				# 	exp_replay.append([net_input, (callback.view(-1,1)) / 512, net_input_, net_output, 1])
				# if callback.item() < 0:
				# 	exp_replay.append([net_input, (callback.view(-1,1)) / 512, net_input_, net_output, 1])
			else:
				#exp_replay.append([net_input.view(-1, 810), callback.view(-1,1) + dist + (i - i_) + r_sum / (i - i_) * 2, net_input_.view(-1, 810), net_output.view(-1, 273), 0])
				exp_replay.append([net_input, (callback.view(-1,1) - 64) / 512, net_input_, net_output, 0])
				# if i - i_ != 1:
				# 	exp_replay.append([net_input, (callback.view(-1,1) - 64) / 512, net_input_, net_output, 0])
				# if callback.item() < 0:
				# 	exp_replay.append([net_input, (callback.view(-1,1) - 64) / 512, net_input_, net_output, 0])
				writer.add_scalar('r/g', G_[0], i)
				writer.add_scalar('r/sum', r_sum, i)
				writer.add_scalar('r/mean', r_sum / (i - i_), i)
				writer.add_scalar('actor/last_step', i - i_, i)
				
				i_ = i
				r_sum = 0.0
			
###########################################################################################################################
			
		if i > 512:
			with torch.no_grad():


				batch = random.sample(exp_replay, 511)
				# for j in g_exp_replay:
				# 	batch.append(j)
				batch.append(exp_replay[-1])
				
				net_input, callback, net_input__, action, not_done = zip(*batch)
				
				net_input = torch.cat(net_input)

				callback = torch.cat(callback).view(-1,1)

				
				reward = callback[-1, -1]

				r_smooth = 0.1 * reward.item() + 0.9 * r_smooth

				net_input__ = torch.cat(net_input__)
				action = torch.cat(action)
				not_done = torch.tensor(not_done).view(-1,1).cuda()


				critic_in = torch.cat((net_input, action), dim = 1)

				a_ = actor_target_net.forward(net_input__)
				critic_target_in = torch.cat((net_input__, a_), dim = 1)

			writer.add_scalar('r/reward', reward, i)

###########################################################################################################################


			critic_net_1.opt.zero_grad()

			q_1 = critic_net_1.forward(critic_in)
			q_n_1 = q_1[-1, -1].item()

			with torch.no_grad():
				q__1 = critic_target_net_1.forward(critic_target_in)
				y_1 = callback + td_s * q__1 * not_done

			td_e_1 = y_1 - q_1

			td_1 = td_e_1[-1, -1].item()
			
			critic_loss__1 = td_e_1.square().mean()
			critic_loss_1 = critic_loss__1.item()
			c_1_loss_smooth = 0.1 * critic_loss_1 + 0.9 * c_1_loss_smooth

			# if c_1_loss_smooth < 0.003:
			# 	c_1_l = False
			# elif c_1_loss_smooth > 0.005:
			# 	c_1_l = True

			if c_1_l:
				critic_loss__1.backward()
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
			writer.add_scalar('critic/td_1', td_1, i)
			# writer.add_scalar('critic/loss_1_smooth', c_1_loss_smooth, i)


###########################################################################################################################

			critic_net_2.opt.zero_grad()
			
			q_2 = critic_net_2.forward(critic_in)
			q_n_2 = q_2[-1,-1].item()

			with torch.no_grad():
				q__2 = critic_target_net_2.forward(critic_target_in)
				y_2 = callback + td_s * q__2 * not_done

			td_e_2 = y_2 - q_2

			td_2 = td_e_2[-1, -1].item()
			
			critic_loss__2 = td_e_2.square().mean()
			critic_loss_2 = critic_loss__2.item()
			c_2_loss_smooth = 0.1 * critic_loss_2 + 0.9 * c_2_loss_smooth

			# if c_2_loss_smooth < 500:
			# 	c_2_l = False
			# elif c_2_loss_smooth > 1000:
			# 	c_2_l = True

			if c_2_l:
				critic_loss__2.backward()
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
			writer.add_scalar('critic/td_2', td_2, i)
			# writer.add_scalar('critic/loss_2_smooth', c_2_loss_smooth, i)


###########################################################################################################################

			if i>0:
				actor_net.opt.zero_grad()
				critic_net_1.opt.zero_grad()
				critic_net_2.opt.zero_grad()


				writer.add_scalar('actor/r_smooth', r_smooth, i)
				if r_smooth < 0.125 or c_1_loss_smooth > 0.03:
					with torch.no_grad():
						callback = (callback - callback.mean()) / callback.std()
					output = actor_net.forward(net_input)
					action_probs = torch.distributions.Normal(output, actor_net.noise_std).log_prob(action) + actor_net.p_b
					actor_loss_ = -(action_probs * callback).mean()
					

					actor_loss = actor_loss_.item()
					writer.add_scalar('actor/loss_2', actor_loss, i)

				else:
					output = actor_net.forward(net_input)
					critic_in = torch.cat((net_input, output), dim = 1)

					critic_net_1.eval()
					critic_net_2.eval()
					q_1 = critic_net_1.forward(critic_in)
					q_2 = critic_net_2.forward(critic_in)
					critic_net_1.train()
					critic_net_2.train()

					actor_loss_ = -torch.min(q_1, q_2).mean()

					actor_loss = actor_loss_.item()
					writer.add_scalar('actor/loss_1', actor_loss, i)



				
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

				





###########################################################################################################################

			#, c_2_param, c_2_target_param
			#, critic_net_2.parameters(), critic_target_net_2.parameters()
			with torch.no_grad():
				for c_1_param, c_1_target_param, c_2_param, c_2_target_param, a_param, a_target_param in zip(critic_net_1.parameters(), critic_target_net_1.parameters(), critic_net_2.parameters(), critic_target_net_2.parameters(), actor_net.parameters(), actor_target_net.parameters()):
					c_1_target_param.data.copy_(tau * c_1_param.data + (1.0 - tau) * c_1_target_param.data)
					c_2_target_param.data.copy_(tau * c_2_param.data + (1.0 - tau) * c_2_target_param.data)
					a_target_param.data.copy_(tau * a_param.data + (1.0 - tau) * a_target_param.data)
					

			
			if (i % 2000) == 0:
				num='{:08d}'.format(int(i / 1000))
				
				torch.save({
					'actor':actor_net.state_dict(),
					'critic_1':critic_net_1.state_dict(),
					'critic_2':critic_net_2.state_dict(),
					'actor_target':actor_target_net.state_dict(),
					'critic_target_1':critic_target_net_1.state_dict(),
					'critic_target_2':critic_target_net_2.state_dict(),
					},'F:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	