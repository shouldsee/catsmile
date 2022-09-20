#! https://zhuanlan.zhihu.com/p/560369284
# 8507-通过手动ssh端口转发实现远程X11程序运行

[CATSMILE-8507](http://catsmile.info/8507-x11-forwarding.html)

```{toctree}
---
maxdepth: 4
---
8507-x11-forwarding.md
```

## 前言

- 目标:
- 背景与动机:
    - 巧合折腾了一下服务器上的X11程序的运行方法,比如说`gedit`,`firefox`啥的
    - 对于做过端口转发的ssh目标，直接使用`ssh -X -p 12345`不能实现x11 forwarding，因为无法直接访问目标机的端口
- 结论: 
    - 可以通过配置本机Xorg打开6000监听，并把远程6011转发到本地6000,来指定远程程序在6011也就是本地xorg打开窗口
- 完成度: 
- 备注: 
- 关键词: 
- 展望方向:
    - [TBC,应该可以用x11-req绕开本机xorg的配置？]
- 相关篇目
- CHANGLOG:

### 简介

因为`ssh -X`搞不定，去搜了一些xserver-xclient的资料。大致意思是，用DISPLAY变量，可以控制console里X11
程序连接的目标port。

### 问题

`ssh -X`失败的时候不太好判断,用`ssh -vvvv`是打印不出来日志的，不过可以用`echo $DISPLAY`检查变量值来间接判断X11转发是否成功。

```bash
# 返回localhost表示成功
echo $DISPLAY
# localhost:12.0

# 返回空值表示失败
echo $DISPLAY
# 
```

通过`ssh -vvvv` 是看不出区别的

```
## 失败
debug2: x11_get_proto: /usr/bin/xauth  list :0 2>/dev/null
debug1: Requesting X11 forwarding with authentication spoofing.
debug2: channel 0: request x11-req confirm 1
debug3: send packet: type 98
debug2: fd 3 setting TCP_NODELAY
debug3: ssh_packet_set_tos: set IP_TOS 0x10
debug2: client_session2_setup: id 0
debug2: channel 0: request pty-req confirm 1
debug3: send packet: type 98
debug1: Sending environment.


## 成功
debug2: x11_get_proto: /usr/bin/xauth  list :0 2>/dev/null
debug1: Requesting X11 forwarding with authentication spoofing.
debug2: channel 0: request x11-req confirm 1
debug3: send packet: type 98
debug2: fd 3 setting TCP_NODELAY
debug3: ssh_packet_set_tos: set IP_TOS 0x10
debug2: client_session2_setup: id 0
debug2: channel 0: request pty-req confirm 1
debug3: send packet: type 98
debug1: Sending environment.
```

### 解决办法

根据x11的原理，可以通过设置DISPLAY变量将x11数据流发送到指定端口，
同时用remote ssh tunnel将这个端口的数据流接受到本机的window manager上。
本机的xorg一般监听的是:6000，但是ubuntu+lightdm默认关闭tcp监听，要做特殊配置才能打开

```bash
### -nolisten 表示没有监听
ps aux | grep lightdm
#root     17152  0.0  0.0 364336  6916 ?        SLsl 21:39   0:00 /usr/sbin/lightdm
#root     18227  3.7  0.2 800800 116820 tty7    Ssl+ 21:40   0:50 /usr/lib/xorg/Xorg -nolisten tcp  -seat seat0 -auth /var/run/lightdm/root/:0 -listen tcp vt7 -novtswitch
```

lightdm可以通过/etc/lightdm/lightdm.conf写入如下内容进行调整.

```conf
[Seat:*]
autologin-guest=true
#autologin-user=ubuntu
autologin-user-timeout=0
autologin-session=lightdm-autologin
xserver-allow-tcp=true
xserver-command=X -listen tcp
```

我不太确定X11的配置是否一定要更改，不过我是去掉了`/etc/X11/xinit/xserverrc`里的`-nolisten tcp`

```bash
#!/bin/sh

#exec /usr/bin/X -nolisten tcp "$@"
exec /usr/bin/X "$@"
```

重启窗口(注意保存工作)后，可以发现Xorg开始监听:6000

`sudo service lightdm restart ##会关闭所有窗口并登出`

```bash
ps aux | grep lightdm
#root     17152  0.0  0.0 364336  6916 ?        SLsl 21:39   0:00 /usr/sbin/lightdm
#root     18227  3.9  0.2 804764 117628 tty7    Ssl+ 21:40   1:01 /usr/lib/xorg/Xorg -listen tcp :0 -seat seat0 -auth /var/run/lightdm/root/:0 -listen tcp vt7 -novtswitch

sudo lsof -i:6000
#COMMAND   PID USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
#Xorg    18227 root    5u  IPv6 2667959      0t0  TCP *:x11 (LISTEN)
#Xorg    18227 root    6u  IPv4 2667960      0t0  TCP *:x11 (LISTEN)
```

这个时候我们可以把remote的数据流通过ssh接受回来.这里我们选择承接remote host上的6011端口的数据到本地的6000

```bash
sshpass -p 123456 ssh -p 65530 -X -vvvv -NR 6011:localhost:6000 root@blah.com

## 检查远程机器的端口监听
sudo lsof -i:6011
#sshd    211939 root    7u  IPv4 500511124      0t0  TCP localhost:6011 (LISTEN)
```

这个时候可以在remote shell里面指定6011,然后运行x11程序了，应当直接在localMachine的windowManager上打开窗口

```bash
### 这几种都指向6011
export DISPLAY=:11
#export DISPLAY=:11.0
#export DISPLAY=localhost:11

xclock  
#sudo apt install x11-apps #临时安装
```


## 参考

- x11原理: <https://dreamanddead.github.io/post/ssh-x11-forward/>
- lightdm: 配置xserver tcp监听 <https://askubuntu.com/questions/615139/how-to-make-x-org-listen-tcp-port-to-remote-connections>