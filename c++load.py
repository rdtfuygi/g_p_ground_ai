from g_p_ann import critic,actor
import torch


a = actor(811, 272, 60, 1024, lr = 1e-4, noise_std = 0.1)
c = critic(1083, 1, 50, 1024, lr = 1e-3)

cp = torch.load('f:\\场地保留\\asd00000072.pth')
c.load_state_dict(cp['critic_1'])
a.load_state_dict(cp['actor'])
cp.clear()

a.eval()
c.eval()

a.cpu()
c.cpu()
torch.jit.save(torch.jit.trace(a, torch.zeros(811)),'actor_cpu.pth')
torch.jit.save(torch.jit.trace(c, torch.zeros(1083)),'critic_1_cpu.pth')

a.cuda()
c.cuda()

torch.jit.save(torch.jit.trace(a, torch.zeros(811).cuda()),'actor_cuda.pth')
torch.jit.save(torch.jit.trace(c, torch.zeros(1083).cuda()),'critic_1_cuda.pth.pth')


print(a.forward(torch.zeros(811).cuda()))
print(c.forward(torch.zeros(1083).cuda()))

print(a.forward(torch.ones(811).cuda()))
print(c.forward(torch.ones(1083).cuda()))

print(a.forward(-torch.ones(811).cuda()))
print(c.forward(-torch.ones(1083).cuda()))