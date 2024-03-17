from turtle import circle
import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter


from neural import actor
from neural import critic




if __name__ == '__main__':
	#print(sys.argv)

	name = sys.argv[1]

	#file = open(name + '_output.txt', 'w', encoding='utf-8')
	
	writer = SummaryWriter('.\\' + name + '_log')
	
	critic_net = critic(1e-3)
	actor_net = actor(1, 1e-3)

	#ground_net.load_state_dict(torch.load('F:\\场地保留\\asd00000012.pth'))
	critic_net.cuda()
	actor_net.cuda()

	output_pipe = pipe('asd_out') 
	input_pipe = pipe('asd_in') 
	#action_pipe = pipe('asd_act')
	callback_pipe = pipe('asd_back') 
	G_pipe = pipe('asd_G')

	i = 0

	_, output_ = output_pipe.recept()

	net_input_ = torch.FloatTensor(list(output_)).view(-1,951).cuda()
	
	while True:
	
		net_input = net_input_.clone()
	
		net_output = actor_net.explor(net_input).view(1,-1).tolist()[0]

		input_pipe.send(net_output);
		
		#_, action = action_pipe.recept();
	
		_, callback_ = callback_pipe.recept();
		
		_, G_ = G_pipe.recept();
		
		
		_, output_ = output_pipe.recept()
		net_input_ = torch.FloatTensor(list(output_)).view(-1,951).cuda()
		
		if i == 0:
			G_ = [0.0]
			
		if G_[0] == 0.0:
			td_error = critic_net.learn(net_input, torch.FloatTensor([callback_[0]]).cuda(), net_input_)
			loss = actor_net.learn(td_error)


	
		#print(loss);
		
		if len(callback_) != 0:
			if(loss != 0.0):
				writer.add_scalar('loss', loss, i)
			writer.add_scalar('reward', callback_[0], i)
			writer.add_scalar('td_error', td_error.item(), i)
			if G_[0] != 0.0:
				writer.add_scalar('g', G_[0], i)
			#writer.add_scalar('reward_b', ground_net.reward_bais, i)
			#writer.add_scalar('reward_l_b', ground_net.reward_l_bais, i)

		if (i % 2000) == 0:
			for name_, param in actor_net.named_parameters():
				writer.add_histogram('asd\\/' + name_, param, i)
				
		if (i % 2000) == 0:
			
			num='{:08d}'.format(int(i / 1000))			

			torch.save(actor_net.state_dict(),'f:\\场地\\actor_' + name + num + '.pth')
			torch.save(critic_net.state_dict(),'f:\\场地\\critic_' + name + num + '.pth')
		
		i += 1
	
	



	
