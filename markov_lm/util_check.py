
import urllib.request
import os
import shutil
import subprocess
import sys
import toml

SEXE = sys.executable
from collections import OrderedDict,namedtuple

def s(cmd,shell=True):
    ret= subprocess.check_output(cmd,shell=shell,executable='/bin/bash')
    return ret

def is_root():
    return (os.geteuid() == 0)

def test_is_root():
    if not is_root():
        sys.exit("[Exit]:You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")
    return



def check_done_file_1(fn,):
    if os.path.exists(fn) and toml.loads(open(fn).read())['DONE']==1:
        return True
    else:
        return False
def write_done_file_1(fn):
    with open(fn,'w') as f:
        f.write(toml.dumps(dict(DONE=1)))
check_write_1 = check_done_file_1, write_done_file_1
DefaultWriter = (lambda cctx:1)
DefaultChecker = (lambda cctx:False)
check_write_always= (DefaultChecker, DefaultWriter)
check_write_2 = (lambda cctx:os.path.exists(cctx)), DefaultWriter

def sjoin(x,sep=' '):
    return sep.join(x)
from pprint import pprint
import time



ControllerNode = namedtuple('ControllerNode','control check_ctx run ctx name')
class Controller(object):
    def __init__(self):
        self.state = OrderedDict()
        self.stats = {}
    def __getitem__(self,k):
        return self.state.__getitem__(k)
        #[k]
    def run_node_with_control(self, control, check_ctx, run, ctx=None,name = None):

        t0 = time.time()
        # self.state[]
        check, write = control
        if isinstance(run, str):
            run = sc(run)
        if check(check_ctx, ):
            print(f'[SKIP]{repr(run)}({repr(ctx)})')
        else:
            print(f'[RUNN]{repr(run)}({repr(ctx)})')

            run(ctx)
            write(check_ctx, )

        t1 = time.time()
        k = repr(run)[:30]
        dt = t1-t0
        self.stats[k] = (k, int(dt*10**3), int(dt *10**6) //1000)


    RWC = run_node_with_control
    # def register_node(self,*a):
    #     if len(a)==1:
    #         a = (check_write_always,None,a[0])
    #     return self._register_node(*a)
    def pprint_stats(self):
        pprint(self.stats)
    def register_node(self, control=check_write_always,
        check_ctx=None, run=None, ctx=None, name = None, run_now = False):
        assert run is not None, 'Must specify "run"'
        if name is None:
            name = '_defaul_key_%d'%(self.state.__len__())
        self.state[name]= node = ControllerNode(control, check_ctx, run, ctx, name)
        if run_now:
            self.run_node_with_control(*node)

    def run(self):
        rets = []
        for k,v in self.state.items():
            self.run_node_with_control(*v)
#            rets.append( v())
        return rets

    def build(self):
        return self.run()

    def lazy_wget(self, url,):
        target =os.path.basename(url)
        #return
        def _lazy_wget(ctx):
            ret = urllib.request.urlretrieve(url, target+'.temp',)
            shutil.move(target+'.temp',target)

        return self.register_node(check_write_2, target, _lazy_wget, run_now=True)

    def lazy_apt_install(self, PACK):
        if not isinstance(PACK,(list,tuple)):
            PACK = PACK.split()


        ret = s(f'''apt list --installed {sjoin(PACK)}''').splitlines()
        if len(ret) - 1 >= len(PACK):
            print(f'[SKIP]lazy_apt_install({PACK})')
        else:
#            test_is_root()
            s(f'''sudo apt install -y {sjoin(PACK)}''')
        return

ctrl = Controller()
# RWC = ctrl.run_node_with_control
run_node_with_control = ctrl.run_node_with_control
RWC = run_node_with_control


class ShellCaller(object):
    def __init__(self,cmd):
        if 'set -e' not in cmd:
            cmd = 'set -e; ' + cmd
        self.cmd = cmd
    def __repr__(self):
        return f'ShellCaller(cmd="{self.cmd[:30]}")'
    def __call__(self,ctx=None):
        return s(self.cmd)
sc = ShellCaller
