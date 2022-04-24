from glob import glob
import sys,os

import torch
from train import init_conf
import numpy as np
'''
20220401 Decomposition must carefully control the parameter count
to avoid overfitting. The simple KV-matrix combined with sequence
rearranger model is ill-defined since you can not allow both
parameter to be fittable... if kv is allowed to freely vary,
then any permuter model will performs equally good, by adapting kv
matrix to the permuter. The degenerate model stores tokens sequentially
and the permuter just take them out in sequential order.

if instead, you uses the worst performing permuter on a kv-matrix
then you will get a very bad sentence because it can be arbitrariliy
mismatching? Needs more thoughts here.
'''
main()
