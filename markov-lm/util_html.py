from io import BytesIO
import base64
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
