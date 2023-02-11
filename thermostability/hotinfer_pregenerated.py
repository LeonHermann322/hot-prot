from torch import nn
import torch
from collections import OrderedDict

class HotInferPregeneratedLSTM(nn.Module):
    def __init__(self, hidden_size, hidden_layers):
        super().__init__()
      
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        
        rnn_hidden_size = hidden_size
        rnn_hidden_layers = hidden_layers

        self.thermo_module_rnn = torch.nn.LSTM(input_size=1024,
            hidden_size =rnn_hidden_size, 
            num_layers =rnn_hidden_layers,
            batch_first =True,
            bidirectional=False)
        
        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(rnn_hidden_layers * rnn_hidden_size),
            nn.Linear(rnn_hidden_layers * rnn_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1))
            
      

    def forward(self, s_s: torch.Tensor):
        output, (hidden, final) = self.thermo_module_rnn(s_s)
        thermostability = self.thermo_module_regression(torch.transpose(hidden, 0,1))
    
        return thermostability
        


class HotInferPregeneratedFC(nn.Module):
    def __init__(self, input_len=700, num_hidden_layers=3, first_hidden_size=1024):
        super().__init__()
      
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        
        

        self.thermo_module_regression = torch.nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_len ),
            nn.Linear(input_len , first_hidden_size),
            nn.ReLU(),
            self.create_hidden_layers(num_hidden_layers, first_hidden_size,16),
            nn.Linear(16,1))
            
     

    def forward(self, s_s: torch.Tensor):
        thermostability = self.thermo_module_regression(s_s)
        return thermostability
    
    def create_hidden_layers(self,num:int, input_size:int, output_size:int) -> nn.Module:
        if num == 1:
            return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        
        result = []
        for i in range(num-1):
            print(input_size)
            _output_size = int(input_size/2)
            layer = nn.Linear(input_size, _output_size)
            result.append((str(i*2),layer))
            result.append((str(i*2+1),nn.ReLU()))
            input_size = _output_size
        
        result.append((str((num-1)*2),nn.Linear(input_size, output_size)))
        result.append((str((num-1)*2+1),nn.ReLU()))

        return nn.Sequential(OrderedDict(result))
        