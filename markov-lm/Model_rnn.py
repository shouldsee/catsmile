
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
        B = x.size(0)
        T = x.size(1)
        EO = self.hidden_size
        outs = torch.zeros((B,T,EO),device=self.device,dtype=self.dtype)
        ht1 = h0
        UM = self.NOT_MUTATED
        for t in range(T):
            xt = x[:,t]
            ft  = (   UM[0] * xt @ self.wxf
                    + UM[1] * ht1 @ self.whf
                    + UM[2]*self.bf[None,:]).sigmoid()
            htp = (   UM[3] * xt @ self.wxh
                    + UM[4] * (ht1 * ft) @ self.whh
                    + UM[5] * self.bh[None,:]).tanh()
            h   =  UM[6] * (1-ft) * ht1 + UM[7]* ft * htp
            outs[:,t] = h
            ht1 = h
        return outs, ht1



class MGRU(nn.Module):
    '''
    0,2,4,5 mutant of GRUMinimal
    '''

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
        self.wxh = nn.Parameter( x.weight.T )


        x = nn.Linear(EO,EO)
        self.whf = nn.Parameter( x.weight.T )

        self.NOT_MUTATED = [1. ] * 10

    def mutate(self,i):
        self.NOT_MUTATED[i] = 0.
        return

    def unmutate(self,i):
        self.NOT_MUTATED[i] = 1.
        return

    def forward(self,x,h0):
        B = x.size(0)
        T = x.size(1)
        EO = self.hidden_size
        outs = torch.zeros((B,T,EO),device=self.device,dtype=self.dtype)
        ht1 = h0
        UM = self.NOT_MUTATED
        for t in range(T):
            xt = x[:,t]
            if UM[6]>0.5 or UM[7] < 0.5:
                ft  = ( UM[0] * ht1 @ self.whf ).sigmoid()
            else:
                if UM[9]>0.5:
                    ft  = ( UM[0] * (ht1 - 0.5) @ self.whf ).sigmoid()
                else:
                    ft  = ( UM[0] * (2*(ht1 - 0.5)).clip(-0.9999,0.9999).atanh() @ self.whf ).sigmoid()
                    # ft  = ( UM[0] * ( (  1/ (ht1) ) - 1).log()  @ self.whf ).sigmoid()


            gt  = ( UM[1] * xt @ self.wxh  )
            if UM[4]>0.5:
                gt = gt.tanh()
            else:
                gt = gt.sigmoid()

            h   =  UM[2] * (1-ft) * ht1 + UM[3]* ft * gt + (1 - UM[5]) * ft * ht1

            if UM[6]>0.5 or UM[8] < 0.5:
                oh = h
            else:
                oh = h-0.5
            outs[:,t] = oh
            ht1 = h

        return outs, ht1
