import torch
import torch.nn as nn
import snntorch as snn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(42, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        return x


class FCSpiking(nn.Module):
  def __init__(self, alpha, beta):
    super(FCSpiking, self).__init__()

    self.fc1 = nn.Linear(42, 10)
    self.fc2 = nn.Linear(10, 2)

    self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)

  def forward(self, x):

        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.lif1.init_synaptic()

        # (nSteps, batch, data)
        x = x.permute(1, 0, 2)

        # Record the final layer
        spk1_rec = []
        mem1_rec = []
        self.register_buffer('out_rec', torch.zeros((nSteps, x.size()[1], 2)))
        self.out_rec = torch.zeros((nSteps, x.size()[1], 2)).to(device)

        for step in range(nSteps):
            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            out = self.fc2(mem1)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            self.out_rec[step] = out

        return mem1_rec, spk1_rec, self.out_rec
