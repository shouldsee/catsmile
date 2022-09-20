# 8508-Linux应用层启动序列

[CATSMILE-8508](http://catsmile.info/8508-systemd-vinit.html)

```{toctree}
---
maxdepth: 4
---
8508-systemd-vinit.md
```

## 前言

- 目标:
- 背景与动机:
- 结论: 
- 完成度: 
- 备注: 
- 关键词: 
- 展望方向:
- 相关篇目
- CHANGLOG:

## 例子: 用sysvinit加入自启动项目

#!/bin/bash
# chkconfig: 2345 20 80
# description: Description comes here....

# Source function library.

```
. /etc/init.d/functions
start() {
    # TODO: code to start app comes here 
}

stop() {
    # TODO: code to stop app comes here 
}

case "$1" in 
    start)
       start
       ;;
    stop)
       stop
       ;;
    restart)
       stop
       start
       ;;
    status)
       # TODO: code to check status of app comes here 
       ;;
    *)
       echo "Usage: $0 {start|stop|restart|status}"
esac

exit 0
```

## 参考

https://uace.github.io/learning/init-vs-systemd-what-is-an-init-daemon
