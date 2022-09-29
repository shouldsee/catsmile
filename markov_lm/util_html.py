from io import BytesIO
import base64
import requests
import os
import shutil
# from tqdm import tdqm
from tqdm import tqdm
# import toml



from markov_lm.util_base import register_object_method

class my_cls():
    def __init__(self,i ):
        if i==1:
            @register_object_method(self)
            def func(self,x):
                return -x
        elif i==2:
            @register_object_method(self)
            def func(self,x):
                return x**2
        else:
            assert 0
    def __call__(self, x):
        v = self.func(x)
        return v

if __name__=='__main__':
    print(my_cls(1)(2))
    print(my_cls(2)(2))


def write_png_tag(fig,props=''):
	with BytesIO() as temp:
		fig.savefig(temp,format='png')
		temp.seek(0)
		png_string = base64.b64encode(temp.read()).decode()
		ret =  f'''
	<img {props} src="data:image/png;base64,{png_string}"></img>
		'''
		return ret


def add_legend_patch( ax, color_label_dict, cmap):
	from matplotlib.patches import Patch
	from matplotlib.lines import Line2D
	print(color_label_dict)
	legend_elements = [Line2D([0], [0], color=cmap(int(k)), lw=4, label=color_label_dict[k]) for k in color_label_dict]
	ax.legend(handles=legend_elements,)

def abline(ax,slope, intercept):
	import numpy as np
	"""Plot a line from slope and intercept"""
	# axes = plt.gca()
	x_vals = np.array(ax.get_xlim())
	y_vals = intercept + slope * x_vals
	ax.plot(x_vals, y_vals, 'r--')


class Vocab(object):
	def __init__(self,vocab,offset):
		self.offset = offset
		self.i2w = list(sorted(vocab))
		self.w2i = {w:i+offset for i,w in enumerate(self.i2w)}
	def add(self,v):
		self.i2w.append(v)
		self.w2i[v] = self.i2w.__len__()-1 + self.offset
	def __len__(self):
		return self.i2w.__len__()
	def tokenize(self,k):
		return self.w2i[k]

	def wordize(self,i):
		return self.i2w[i-self.offset]


def lazy_if_file_exists(func, argi):
	def lazy_func(*a,verbose=0):
		a = list(a)
		fname = a[argi]
		if os.path.exists(fname):
			if verbose>=1:
				print(f'[skipped]{func!r}{a!r}')
			pass
		else:
			a[argi] = fname+'.temp'
			ret = func(*a)
			shutil.move(fname+'.temp',fname)
		return fname

	return lazy_func

def get_url(url,fname):
	'''
	Adapted from SO: https://stackoverflow.com/a/35997720/8083313
	'''
	r = requests.get(url, stream=True)
	with open(fname, 'wb') as f:
		total_length = int(r.headers.get('content-length'))
		for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length/1024) + 1):
			if chunk:
				f.write(chunk)
				f.flush()
		# shutil.move(fname+'.temp', fname)

get_url_lazy = lazy_if_file_exists(get_url, 1)
from markov_lm.util_base import dict_to_argv,toml_to_argv

if __name__ == '__main__':
	fname = 'guppy-0.1.10.tar.gz'
	# get_url('https://pypi.python.org/packages/source/g/guppy/' + fname, fname)
	get_url_lazy('https://pypi.python.org/packages/source/g/guppy/' + fname, fname,verbose=1)
