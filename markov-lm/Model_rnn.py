
import torch.nn as nn
from dataclasses import dataclass
import torch
@dataclass
class RNNConfig(object):
    input_size:int = -1
    hidden_size:int = -1
    num_layers:int = -1
    batch_first:bool = True
    max_step: int = -1
    head_size: int =-1
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
        self.head_size = config.head_size
        # self.depth = config.depth
        # self.embed_dim = config.embed_dim
        # input_size
        self.max_step = T = config.max_step

        self.NOT_MUTATED = [1. ] * 12

        x = nn.Linear(EI,EO)
        self.wxh = nn.Parameter( x.weight.T )

        x = nn.Linear(EO,EO)
        self.whf = nn.Parameter( x.weight.T )



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

            if UM[10]<0.5 and t!=0:
                # T = self.max_step
                prior = self.log_conn_prior[t:t+1]
                prior = prior + - self.INF * torch.arange(T,device=self.device)[None,:,None]>=t
                att = (prior[:,:t] + outs[:,:t] @ self.whj).softmax(dim=1)
                jt  = (att * outs[:,:t]).sum(dim=1)
            elif UM[11]<0.5 and t!=0:
                ### (B,T,W,1)
                W = self.head_size
                bilinear = (outs[:,:t] @ self.att_head_weight).reshape((B,t,W, EO)) @ ht1[0,:,None,:,None]
                ### (B,Ts,E)
                att = self.bi2att( bilinear[:,:,:,0] ).softmax(dim=1)
                # att = self.bi2att(bilinear.transpose(3,2)).softmax(dim=1)
                jt  = (att * outs[:,:t]).sum(dim=1)
            else:
                jt = ht1

            h   =  UM[2] * (1-ft) * jt + UM[3]* ft * gt + (1 - UM[5]) * ft * jt

            if UM[6]>0.5 or UM[8] < 0.5:
                oh = h
            else:
                oh = h-0.5
            outs = torch.scatter(outs,src=oh.transpose(0,1),index=torch.ones((B,1,EO),device=self.device).long()*t,dim=1)
            # outs[:,t] = oh
            ht1 = h

        return outs, ht1





class MGRUWithAttention(MGRU):
    '''
    0,2,4,5 mutant of GRUMinimal
    '''
    INF = 1E14
    @staticmethod
    def get_default_config():
        return RNNConfig()

    def __init__(self, device, config, _ = None):
        super().__init__(device,config)

        T= self.max_step
        EO=self.hidden_size
        W = self.head_size
        x = nn.Linear(EO,EO)
        self.whj = nn.Parameter( x.weight.T )



        x = nn.Linear(T*T,EO)
        self.log_conn_prior = nn.Parameter( x.weight.T.reshape((T,T,EO)))

        x = nn.Linear(EO,EO*W)
        self.att_head_weight = nn.Parameter( x.weight.T)
        # .reshape((T,T,EO)))


        x = nn.Linear(self.head_size, EO)
        self.bi2att = x


        # self.
