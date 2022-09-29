'''
- Author: Feng Geng
- Changelog:
  - 20220507-20220514: for (dataset=ArithmeticTest, model=AddModelWithAttention),
  tested different parameters. use_dense_relu=1/11 shows interesting capacity.
  per-position dropout greatly prevents overfitting for unknown reason.
  - 思考: 模型的内部计算经常是梯度的,但是模型参数的更新一般是基于梯度的.这不知道是否矛盾.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import os,sys

import numpy as np
from torch import autograd

_ConvNd = object
if 0:
    class Conv2d(_ConvNd):
        __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
        planes.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            # kernel_size: _size_2_t,
            # stride: _size_2_t = 1,
            # padding: Union[str, _size_2_t] = 0,
            # dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = padding if isinstance(padding, str) else _pair(padding)
            dilation_ = _pair(dilation)
            super(Conv2d, self).__init__(
                in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
                False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias)


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

import torch.nn as nn


def expand_with_dirac(xs,kv):
    '''
    Put a image/mask to different location in the image

    xs = (60,28,30)
    ks = (20,21,13)
    xs = torch.rand(xs)
    kv = torch.rand(ks)

    xconv = expand_with_dirac(xs,kv)
    print(xconv.shape)
       XY    K   X   Y
    # [840, 13, 28, 30]
    # B is lost

    #  MB    XY   K
    # [840, 840, 13]

    '''
    (_,X,Y)=xs.shape
    device = xs.device
    xdirac = torch.eye((X*Y),device=device).reshape((X*Y,X,Y))
    xconv = F.conv2d(xdirac[:,None], kv.transpose(-1,0)[:,None],padding='same')
    xconv = xconv.transpose(1,-1).reshape((X*Y,X*Y,-1))
    return xconv
    
def shift_with_component(xs,kv):
    '''
    shift an image with spatial component on k different directions
    '''
    # assert xs.shape.__len__()==2,xs.shape
    xconv = expand_with_dirac(xs,kv)
    xs = flatten(xs)
    xshift = xs[:,:,None,None] + xconv[None]
    '''
     B    MB  XY   K
    [60, 840, 840, 13]
    '''
    return xshift

def flatten(xs):
    return xs.reshape((xs.size(0),-1))

import torch.nn.functional as F
def main():
    xs = (60,28,30)
    ks = (20,21,13)
    xs = torch.rand(xs)
    kv = torch.rand(ks)
    # xconv = F.conv2d(xdirac[:,None], kv.transpose(-1,0)[:,None],padding=(10,10))
    xshift = shift_with_component( (xs), kv)
    '''
    '''
    print(xshift.shape)
    (xconv[5*28+5][0][:10,:10]*10).int()
    (kv.transpose(2,0)[0][:10,:10]*10).int()
    print(xconv.shape)
    import pdb; pdb.set_trace()
    # m = nn.Conv2d(1,1)

    m = nn.Conv2d(5, 1, (3, 5), stride=1, padding_mode='zeros',padding='same', dilation=(1,1))
    input = torch.randn(10, 5, 30, 40)
    output = m(input)
    print(input.shape)
    print(output.shape)
    import pdb; pdb.set_trace()


if __name__=='__main__':
    main()
