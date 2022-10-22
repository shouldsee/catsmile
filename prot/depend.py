'''
Last Updated: 2022-10-22
Author: shouldsee <shouldsee@qq.com>

This script build an environment, including
  - a gromacs installation, with gmxapi
  - a nglview installation

This script depends on
  - bash
  - markov_lm.util_check
  - apt
  - git
  - python3.7

It is meant to mix dependency with runtime code in the same script,
 so as to bootstrap an environment from different machine states
'''
from markov_lm.util_check import (
	# lazy_wget,
	# lazy_apt_install,
	is_pypack_installed,
	sjoin,
	is_root,
	test_is_root,
	SEXE,
	s,
	# RWC,
	check_write_1,
	check_write_2,
	check_write_single_target,
	check_write_always,
	check_write_pypack,
	DefaultWriter,
	)
# from markov_lm.util_check import run_node_with_control as RWC
from markov_lm.util_check import Controller
# from markov_lm.util_check import ShellCaller as sc

import os,sys,toml
import toml

def main():
	# ctl = prepare_run()
	ctl.run()
	ctl.pprint_stats()

def prepare_run():
	ctl = Controller()
	# Controller = Controller
	RWC = ctl.register_node
	GROMACS_DGMX_GPU='CUDA'
	NCORE = 4

	ret = ctl.lazy_wget( "https://files.rcsb.org/download/1PGB.pdb")

	ctl.lazy_apt_install('libopenmpi-dev libfftw3-dev grace')
	TARGETS = ['mpi4py']
	RWC(check_write_pypack, TARGETS,run = f'''
	### [shouldsee] is this required before compiling GROMACS???

	MPICC=`which mpicc` MPICXX=`which mpic++` {SEXE} -m pip install --upgrade {sjoin(TARGETS)}
	''')

	ctl.lazy_wget('ftp://ftp.gromacs.org/pub/gromacs/gromacs-2022.tar.gz')

	TARGET = 'gromacs-2022.tar.gz'
	RWC(check_write_1, TARGET+'.done', run=f'''tar xvf {TARGET}''')
	'gromacs-tmpi'
	TARGET = f'{os.getcwd()}/gromacs-tmpi'
	CFLAGS = f'../gromacs-2022 -DCMAKE_INSTALL_PREFIX=$TARGET -DGMX_THREAD_MPI=ON \
	   -DGMX_GPU={GROMACS_DGMX_GPU} \
		-DCMAKE_C_COMPILER=`which mpicc` -DCMAKE_CXX_COMPILER=`which mpic++`'
	RWC(check_write_1, TARGET+'.done', (f'''
	TARGET={TARGET}
	set -e
	# Build and install thread-MPI GROMACS to your home directory.
	# Make sure the compiler toolchain matches that of mpi4py as best we can.
	mkdir -p build && pushd build
	 cmake --trace {CFLAGS}
	 make -j{NCORE} install
	popd
	'''), name ='gromacs',ctx=TARGET)

	### CMake 3.16.3 or higher is required
	if s('cmake --version')< 'cmake version 3.16.3':
		print('''
		[WARN] use with caution
		''')
		RWC(run = '''
		version=3.24
		build=1
		## don't modify from here
		limit=3.20
		os="linux"
		mkdir -p ./temp; cd ./temp
		wget https://cmake.org/files/v$version/cmake-$version.$build-$os-x86_64.sh
		sudo mkdir -p /opt/cmake
		sudo sh cmake-$version.$build-$os-x86_64.sh --prefix=/opt/cmake
		sudo ln -sf /opt/cmake/*/bin/cmake `which cmake`
		''')


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

	RWC(check_write_2, TARGET, (f'''
	set -o allexport;
	set -e;
	source $PWD/gromacs-tmpi/bin/GMXRC.bash;
	{SEXE} -c "import toml,os;toml.dump(dict(os.environ), open('{TARGET}','w'))"
	'''))

	### loads environ from bash file
	def run(ctx,TARGET=TARGET):
		xd = toml.loads(open(TARGET,'r').read())
		for line in example.strip().splitlines():
			k = line.split('=',1)
			if len(k)!=2:
				continue

			k =k[0].strip()
			if k:
				os.environ[k] = xd[k]
	RWC(run=run)


	TARGET= 'gmxapi nglview ipywidgets'.split()
	RWC([is_pypack_installed,DefaultWriter],TARGET, f'''
	{SEXE} -m pip install --upgrade {sjoin(TARGET)}
	''')

	# TARGET='nglview-js-widgets.js'
	# import pkgutil
	# def run(ctx):
	# 	data = pkgutil.get_data('nglview', 'static/index.js')
	# 	with open(ctx,'wb') as f:
	# 		f.write(data)
	RWC(check_write_2, './node_modules/ngl/dist/ngl.js', 'npm install ngl',name='init_ngl')
	return ctl
ctl = prepare_run()
if __name__ == '__main__':
	main()
