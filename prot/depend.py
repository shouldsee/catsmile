'''
This script build an environment as adapted from GROMACS
'''
from markov_lm.util_check import (
	# lazy_wget,
	# lazy_apt_install,
	sjoin,
	is_root,
	test_is_root,
	SEXE,
	# RWC,
	check_write_1,
	check_write_2,
	DefaultWriter,
	)
# from markov_lm.util_check import run_node_with_control as RWC
from markov_lm.util_check import Controller
from markov_lm.util_check import ShellCaller as sc

import os,sys,toml
import toml

def main():
	# ctl = prepare_run()
	ctl.run()

def prepare_run():
	ctl = Controller()
	# Controller = Controller
	RWC = ctl.register_node
	NCORE = 4
	ret = ctl.lazy_wget( "https://files.rcsb.org/download/1PGB.pdb")

	ctl.lazy_apt_install('libopenmpi-dev libfftw3-dev')

	sc(f'''
	# Set up a Python virtual environment in which to install gmxapi.
	#python3 -m venv $HOME/pygmx
	#. $HOME/pygmx/bin/activate
	#pip install --upgrade pip setuptools wheel

	MPICC=`which mpicc` MPICXX=`which mpic++` {SEXE} -m pip install --upgrade mpi4py
	''')()

	ctl.lazy_wget('ftp://ftp.gromacs.org/pub/gromacs/gromacs-2022.tar.gz')

	sc('''tar xvf gromacs-2022.tar.gz''')()
	'gromacs-tmpi'

	TARGET = f'{os.getcwd()}/gromacs-tmpi'

	RWC(check_write_1, TARGET+'.done', sc(f'''
	TARGET={TARGET}
	# Build and install thread-MPI GROMACS to your home directory.
	# Make sure the compiler toolchain matches that of mpi4py as best we can.
	mkdir -p build && pushd build
	 cmake --trace ../gromacs-2022 -DCMAKE_INSTALL_PREFIX=$TARGET -DGMX_THREAD_MPI=ON \
		 -DCMAKE_C_COMPILER=`which mpicc` -DCMAKE_CXX_COMPILER=`which mpic++`
	 make -j{NCORE} install
	popd
	'''))




	example = '''
	GMXPREFIX=/root/catsmile/prot/gromacs-tmpi
	GMXBIN=${GMXPREFIX}/bin
	GMXLDLIB=${GMXPREFIX}/lib
	GMXMAN=${GMXPREFIX}/share/man
	GMXDATA=${GMXPREFIX}/share/gromacs
	GMXTOOLCHAINDIR=${GMXPREFIX}/share/cmake
	GROMACS_DIR=${GMXPREFIX}

	LD_LIBRARY_PATH=$(replace_in_path "${LD_LIBRARY_PATH}" "${GMXLDLIB}" "${OLD_GMXLDLIB}")
	PKG_CONFIG_PATH=$(replace_in_path "${PKG_CONFIG_PATH}" "${GMXLDLIB}/pkgconfig" "${OLD_GMXLDLIB}/pkgconfig")
	PATH=$(replace_in_path "${PATH}" "${GMXBIN}" "${OLD_GMXBIN}")
	MANPATH=$(replace_in_path "${MANPATH}" "${GMXMAN}" "${OLD_GMXMAN}")
	'''

	# Activate the GROMACS installation.
	TARGET = 'genv.toml'

	RWC(check_write_2, TARGET, sc(f'''
	set -o allexport;
	source $PWD/gromacs-tmpi/bin/GMXRC.bash;
	{SEXE} -c "import toml,os;toml.dump(dict(os.environ), open('{TARGET}','w'))"
	'''))

	### loads environ from bash file
	xd = toml.loads(open(TARGET,'r').read())
	for line in example.strip().splitlines():
		k = line.split('=',1)
		if len(k)!=2:
			continue

		k =k[0].strip()
		if k:
			os.environ[k] = xd[k]


	# Build and install the latest gmxapi Python package.

	import importlib.util
	def is_pypack_installed(package_name_list,):
		# package_name = 'pandas'
		ret = True
		for package_name in package_name_list:
			spec = importlib.util.find_spec(package_name)
			ret = ret & (spec is not None)
		return ret
		# return spec is not None

	TARGET= 'gmxapi nglview ipywidgets'.split()
	RWC([is_pypack_installed,DefaultWriter],TARGET, f'''
	{SEXE} -m pip install --upgrade {sjoin(TARGET)}
	''')

	TARGET='nglview-js-widgets.js'
	import pkgutil
	def run(ctx):
		data = pkgutil.get_data('nglview', 'static/index.js')
		with open(ctx,'wb') as f:
			f.write(data)
	RWC(check_write_2, './node_modules/ngl/dist/ngl.js', 'npm install ngl',name='init_ngl')
	return ctl
ctl = prepare_run()
if __name__ == '__main__':
	main()
