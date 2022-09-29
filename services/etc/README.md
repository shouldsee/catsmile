## Architecture
 
nginx listen on a port then rewrite url and dispatch to child apps.
child apps listening on different ports.
supervisor to run and log services

## Install

- supervisor: python3 -m pip install supervisor 
- nginx:1.18.0:apt install nginx. 
- `sudo ln etc/my-nginx.conf /etc/sites-enabled`
- visdom: http://github.com/shouldsee/master . this branch has autosave func, 
with modified react callbacks
- (not req) cesi: http://github.com/shouldsee/cesi/tree/add_url_prefix
- (not req)frpc: download binary from http://github.com/fatedier/frp

## Start

```bash
### this is a bash function, append to .bashrc 


sstart(){
#/init/bin/supervisord -d -c /root/catsmile/markov_lm/supervisor.conf
#/root/miniconda3/bin/
supervisord  -c /root/catsmile/services/etc/supervisord.ini # -e debug
}

```

