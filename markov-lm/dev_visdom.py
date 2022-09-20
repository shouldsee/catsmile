import visdom

vis = visdom.Visdom(env='example',port='6006')

import numpy as np
vis.scatter(np.random.random((50,2)))

