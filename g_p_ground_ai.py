import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter


from neural import actor




if __name__ == '__main__':
	#print(sys.argv)

	name = sys.argv[1]

	#file = open(name + '_output.txt', 'w', encoding='utf-8')
	
	writer = SummaryWriter('.\\' + name + '_log')
	
	ground_net=actor(1, 1e-3)

	#ground_net.load_state_dict(torch.load('F:\\场地保留\\asd00000012.pth'))
	
	ground_net.cuda()

	output_pipe = pipe('asd_out') 
	input_pipe = pipe('asd_in') 
	#action_pipe = pipe('asd_act')
	callback_pipe = pipe('asd_back') 
	G_pipe = pipe('asd_G')

	i = 0

	while True:
	
		_, output_ = output_pipe.recept()
	
		net_input = torch.tensor(list(output_)).view(-1,951).cuda()
	
		net_output = ground_net.explor(net_input).view(1,-1).tolist()[0]

		input_pipe.send(net_output);
		
		#_, action = action_pipe.recept();
	
		_, callback_ = callback_pipe.recept();
		
		_, G_ = G_pipe.recept();
		
		if i == 0:
			G_ = [0.0]

		loss = ground_net.learn(list(callback_), list(G_))
	
		#print(loss);
		
		if len(callback_) != 0:
			if(loss != 0.0):
				writer.add_scalar('loss', loss, i)
			writer.add_scalar('reward', callback_[0], i)
			if G_[0] != 0.0:
				writer.add_scalar('g', G_[0], i)
			#writer.add_scalar('reward_b', ground_net.reward_bais, i)
			#writer.add_scalar('reward_l_b', ground_net.reward_l_bais, i)

		if (i % 2000) == 0:
			for name_, param in ground_net.named_parameters():
				writer.add_histogram('asd\\/' + name_, param, i)
				
		if (i % 2000) == 0:
			
			num='{:08d}'.format(int(i / 1000))			

			torch.save(ground_net.state_dict(),'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	



	
