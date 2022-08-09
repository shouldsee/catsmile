
import torch.nn as nn
from dataclasses import dataclass
import torch
@dataclass
class RNNConfig(object):
    input_size:int = -1
    hidden_size:int = -1
    num_layers:int = -1
    batch_first:bool = True
    # dtype:object = torch.float

class GRUMinimal(nn.Module):
    @staticmethod
    def get_default_config():
        return RNNConfig()

    def __init__(self, device, config, _ = None):
        super().__init__()

        # input_size, hidden_size,num_layers=1,batch_first=True, dtype=torch.float):
        self.config=  config
        EI = self.input_size = config.input_size
        EO = self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.device = device
        self.dtype = torch.float
        self.batch_first = config.batch_first
        # self.depth = config.depth
        # self.embed_dim = config.embed_dim
        # input_size
        x = nn.Linear(EI,EO)
        self.wxf = nn.Parameter( x.weight.T )
        x = nn.Linear(EI,EO)
        self.wxh = nn.Parameter( x.weight.T )
        self.bf = nn.Parameter(  x.bias)

        x = nn.Linear(EO,EO)
        self.whf = nn.Parameter( x.weight.T )
        x = nn.Linear(EO,EO)
        self.whh = nn.Parameter( x.weight.T )
        self.bh = nn.Parameter( x.bias)
        self.NOT_MUTATED = [1. ] * 8

    def mutate(self,i):
        self.NOT_MUTATED[i] = 0.
        return

    def unmutate(self,i):
        self.NOT_MUTATED[i] = 1.
        return

    def forward(self,x,h0):
        # h = h0
        B = x.size(0)
        T = x.size(1)
        EO = self.hidden_size
        outs = torch.zeros((B,T,EO),device=self.device,dtype=self.dtype)
        ht1 = h0
        UM = self.NOT_MUTATED
        # self.NO_MUTATED[0]
        # CLS_NAME = self.__class__.__name__
        for t in range(T):
            xt = x[:,t]
            ft  = (UM[0] * xt @ self.wxf + UM[1] * ht1 @ self.whf   + UM[2]*self.bf[None,:]).sigmoid()
            htp = (UM[3] * xt @ self.wxh + UM[4] * (ht1 * ft) @ self.whh  + UM[5] * self.bh[None,:]).tanh()
            # htp = (xt @ self.wxh + ht1  @ self.whh  + self.bh[None,:]).tanh()
            h   = UM[6] * (1-ft) * ht1 + UM[7]* ft * htp
            outs[:,t] = h
            ht1 = h
        return outs, ht1
