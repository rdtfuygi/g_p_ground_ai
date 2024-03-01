import torch
import torch.cuda
import sys
from pipe import pipe
from torch.utils.tensorboard import SummaryWriter


from neural import net




if __name__ == '__main__':
	print(sys.argv)

	name = sys.argv[1]

	file = open(name + '_output.txt', 'w', encoding='utf-8')
	
	writer = SummaryWriter('.\\' + name + '_log')
	
	ground_net=net(0.1, 0.01).cuda()

	ground_net.load_state_dict(torch.load('F:\\场地保留\\asd00000065.pth'))

	output_pipe=pipe('asd_out') 
	input_pipe=pipe('asd_in') 
	callback_pipe=pipe('asd_back') 

	i = 0	

	while True:
	
		_, output_ = output_pipe.recept()
	
		net_input = torch.tensor(list(output_)).view(-1,935).cuda()
	
		net_output = ground_net.explor(net_input).view(1,-1).tolist()[0]

		input_pipe.send(net_output);
	
		_, callback_=callback_pipe.recept();

		loss = ground_net.learn(list(callback_))
	
		#print(loss);
		
		if len(callback_) != 0:
			if(loss != 0.0):
				writer.add_scalar('loss', loss, i)
			writer.add_scalar('reward', callback_[0], i)

		if (i % 2000) == 0:
			for name_, param in ground_net.named_parameters():
				writer.add_histogram('asd\\/' + name_, param, i)
				
		if (i % 5000) == 0:
			
			num='{:08d}'.format(int(i / 1000))			

			torch.save(ground_net.state_dict(),'f:\\场地\\' + name + num + '.pth')
		
		i += 1
	
	



	
