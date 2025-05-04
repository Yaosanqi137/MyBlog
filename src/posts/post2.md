---
cover: /assets/images/post2/head.png
icon: network-wired
date: 2025-05-04
category: 教程
tag:
  - 计算机网络
  - 组网
  - tinc
star: true
sticky: true
---

# tinc 组网教程·其一

tinc 是一款专业实用的一款组件虚拟专用网络(VPN)的工具，可以在互联网上点对点(P2P)之间创建专用网络，其工作在网络层，可以用它来搭建低延迟、高带宽、可拓展的 P2P VPN

它有如下**优点**:

- [开源](https://github.com/gsliepen/tinc)
- 分布式网状路由，无需担心单节点高负载或故障
- 高效且有效的加密和数据压缩
- 支持多个操作系统
- 单节点可以配置、加入多个 tinc VPN

## tinc 连接过程

1. tinc 主程序启动后，自动启动指定的 VPN 网络，并定位到对应的配置文件目录读取目录
2. 读取主配置文件 `tinc.conf` 后执行 `tinc-up` 启动脚本，然后连接到(ConnectTo)指定主机，并同时接收其他主机对本主机的连接
3. tinc 读取主机描述文件以获取主机信息，并且连接到和被连接的主机都有对方的配置文件拷贝
4. 连接认证通过，加入 tinc 网络
5. tinc 关闭，执行 `tinc-down` 关闭脚本

## tinc 配置文件

- **tinc.conf**
> VPN 网络的主配置文件，一般有两个配置项:
> Name $\Rightarrow$ 本主机的名字
> ConnectTo $\Rightarrow$ 此 VPN 网络启动后自动连接的主机

- **tinc-up/tinc-down**
> tinc 启动脚本/关闭脚本，分别在启动和关闭的时候调用

- **rsa_key.priv**
> 密钥文件，注意保密

- **host 目录**
> 用于存储 VPN 网络中每台主机的信息，内部的每一个文件都是一个主机，并且文件名就是主机名

拿 Windows 端的 tinc 举例，最后你的 tinc 目录可能是这样的:

```
C:\Program Files (x86)\tinc
├─company
│  │  rsa_key.priv
│  │  tinc-down.bat
│  │  tinc-up.bat
│  │  tinc.conf
│  │
│  └─hosts
│          114514
│          client
│          company_server
│
├─dorm
│  │  rsa_key.priv
│  │  tinc-down.bat
│  │  tinc-up.bat
│  │  tinc.conf
│  │
│  └─hosts
│          dorm_tinc_server
│          pc
...
```

现在我们来实操一下

## Linux 端配置

现在，我有一台 **Debian 12** 的 Linux 机器，一切都非常正常，网络是通畅的，并且我这台机器还有公网 ip (如果没有的话可以弄个 frp 什么的)，这样就可以被其他主机连接啦

但是要注意，tinc 本身没有 服务端/客户端 之说，因为他是 P2P 的，但是以目前压根没什么人有公网 ip 的情况下，就姑且认为有公网，能够收到其他主机连接的机器就是 “服务端” 吧

首先先更新一下你的 apt，如果慢的话大概率没换源，去[换个源](https://mirrors.tuna.tsinghua.edu.cn/help/debian/)吧

```shell
sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get dist-upgrade -y && sudo apt-get autoremove -y
```

然后安装 tinc

```shell
sudo apt-get install tinc -y
```

::: tip 疑问

为什么我执行这些指令，提示 `-bash: sudo: command not found`

这可能是因为你本来就是以 `root` 用户登录的 Linux，如果你本来就是的话，把这些指令的 sudo 前缀去了再执行就行

:::

安装完毕后，前往 `/etc/tinc`，这里就是它的默认主配置目录，里面只有一个 `nets.boot` 文件，对于现代的 Linux 没啥用，一会儿会提到

![](/assets/images/post2/img1.png)

先在里面创建一个目录，这个目录名字就是你的 tinc VPN 的网络名字了（因为这台机器在我的宿舍，我就叫它 `dorm` 吧），然后里面再创建一个 `host` 文件夹

```shell
mkdir -p ./dorm/host
```

进入 `/etc/tinc/dorm`，使用 vim 新建并编辑 `tinc.conf` 在里面写入你的主机在 VPN 网络中的名字（这里我取 `dorm_tinc_server`）

```yaml
Name = dorm_tinc_server
```

然后我们再分别创建 `tinc-up` 和 `tinc-down`

```yaml
#!/bin/sh

ifconfig $INTERFACE 100.64.0.254 netmask 255.255.255.0

# 100.64.0.254 是虚拟网络接口的IP，也就是 tinc server 在虚拟网络中的ip
# 255.255.255.0 是子网掩码，整个换成 cidr 表示法是 100.64.0.254/16
# 当然，这俩不是固定的，你想弄什么内网段弄什么内网端
```

```yaml
#!/bin/sh

ifconfig $INTERFACE down
```

::: tip 为什么这两个脚本无法执行？

或许是因为你的系统里面根本没有 `ifconfig` 这个指令，这是 `net-tools` 包里的，现在某些 Linux 发行版不自带这个软件包

你可以使用 `sudo apt install net-tools` 安装，安装完就有啦

:::

在创建好后，输入这个指令给予文件执行权限

```shell
chmod +x tinc-*
```

现在，我们进入 `host` 目录，用 vim 创建一个和你的主机名一样名字的**描述文件**（也就是 `dorm_tinc_server`），然后写上

```yaml
Address = # 你服务器的公网地址
Subnet = 100.64.0.254/32 
# 跟你刚刚写的 tinc-up 里面写的 ip 一样，但是子网掩码先填着 32 用，这个之后再提
```

写完了吗？写完了我们就开始生成密钥，输入这个指令

```shell
tincd -n dorm -K
```

其中，-n 后填上你的 VPN 网络的名字，-K 后可以填密钥长度，默认值是 2048，回车执行后会问你俩问题，直接无脑回车就行了

![](/assets/images/post2/img2.png)

现在，`/etc/tinc/dorm` 里面就会出现一个叫 `rsa_key.priv` 的文件，里面是你的私钥，一定要注意保存，然后再看看 `./host/dorm_tinc_server`，里面也会出现改动，会把此主机的公钥写在里面

![](/assets/images/post2/img3.png)

现在来配置一下开机自动启动此 VPN 网络吧

一般来说，你可以在 `nets.boot` 的末尾写上你 VPN 网络的名字（本文中应当写 `dorm`）

![](/assets/images/post2/img4.png)

但是更加现代的做法是，使用 systemd 设置开机自启动

```shell
systemctl enable tinc@dorm # tinc@ 后面同样接你 VPN 网络的名字
```

然后重启，不出意外的话，tinc VPN 会启动成功，输入 `ip a` 或者 `ifconfig` 看看，应该会给你创建一个和 VPN 网络同名的接口，如图所示

![](/assets/images/post2/img5.png)

但是也有可能会出现问题，比如笔者遇到的问题，见 [FAQ](https://www.gstech.fun/posts/post3.html#FAQ)

至此，Linux 端(本文中是作为服务端)配置完毕，现在开始配置 Windows 客户端

## Windows 端配置

在这里，Windows 作为客户端

先[下载安装包](https://www.tinc-vpn.org/packages/windows/tinc-1.0.36-install.exe)

或者直接去[tinc 官网](https://www.tinc-vpn.org/download/)找安装包

下载了以后打开，会弹出以下界面

![](/assets/images/post2/img6.png)

遇到用户协议就同意，基本上无脑下一步就行了，但是要注意这一步

![](/assets/images/post2/img7.png)

这一步是指定安装位置，一般不用改，改了也没啥事，记住它就好，本文就按默认位置操作

安装完毕后，我们按 `win + r` 键，输入 `powershell` 进入 powershell 界面

然后输入这个指令，创建虚拟网卡

```shell
& 'C:\Program Files (x86)\tinc\tap-win64\addtap.bat'
```

![](/assets/images/post2/img8.png)

点击安装

进入网络适配器界面，你就会发现多了一个适配器，大概长这样

![](/assets/images/post2/img9.png)

点一下它的名字，笔者这里改成了 `tinc` 方便辨认

现在，进入 tinc 的安装目录，同样在此目录下创建 `dorm` 文件夹，里面也再创建一个 `hosts` 文件夹

然后在 `dorm` 文件夹里面创建 `tinc.conf`，里面写上

```yaml
Name = # 填你这台机器想要叫的名字
ConnectTo = # 填上你想要连接的服务器的名字
Interface = # 填上 tinc 网络适配器的名字
```

笔者这里填的是

![](/assets/images/post2/img10.png)

然后再创建一个 `tinc-up.bat` 和 `tinc-down.bat` 分别写入

```bat
netsh interface ip set address "tinc" static 100.64.0.1 255.255.255.0
```

```bat
netsh interface ip set address "tinc" source=dhcp
```

然后再在 `host` 文件里面新建一个**描述文件**，同样与你的这台主机的名字一样，也就是 `tinc.conf` 里面写的那个名字（本文是 pc）

里面写上

```yaml
Subnet = # 此 VPN 网络的内网端内的 ip，也就是同刚刚配置的服务端里面的那个网段 
```

笔者这里写的是

```yaml
Subnet = 100.64.0.1/32
```

也就是说，这台机器在连上 `dorm` 这个 VPN 网络后，它在 VPN 网络中的 ip 就是 `100.64.0.1` 了

现在开始生成密钥，在 powershell 中输入

```shell
& 'C:\Program Files (x86)\tinc\tincd.exe' -n dorm -K 
# 形参跟 Linux 那边是一样的，不再解释了
```

操作与 Linux 端完全一致，这里不再赘述

然后，交换两个主机的主机描述文件，都放进对方的 `host` 文件夹里面，这样他们就可以相互确认对方的身份了

然后在 powershell 中输入这个指令，设置自启动

```shell
& 'C:\Program Files (x86)\tinc\tincd.exe' -n dorm 
# -n 后的 dorm 同样应该替换成你的 VPN 网络的名字
```

最后，你在 powershell 中输入 `ipconfig` ，如果你发现了这一项

![](/assets/images/post2/img11.png)

说明它已经成功连上了，现在这两台主机应该可以互相 ping 通对方，至此，tinc 基础设置结束

## FAQ

> **Q: 我的 tinc 机器是运行在 proxmox 上的一个 lxc 容器，我发现按正常步骤做以后，发现没有创建出对应的接口，查阅日志后发现 `Could not open /dev/net/tun: No such file or directory`**

> A: 由于 tinc、Zerotier 等软件是通过绑定 /dev/net/tun 的 tun 接口来实现组网的，而 pve 中的 lxc 容器都默认不会有 tun 接口
> 所以，解决方案就是改用虚拟机，或者给你的 lxc 添加上这个接口，具体可以看[这个文章](https://www.cnblogs.com/lynetwork/articles/17271495.html)

> **Q: 我的 VPN 服务器上有很多个内网段，比如 `100.64.1.0/10`、`10.0.0.0/24`、`192.168.1.0/24` 这些网段，我想让连上 tinc 的客户端都能访问这些网段应该怎么做？**

> A: 那就是下一节的问题了，请看 [tinc 组网教程·其二](https://www.gstech.fun/posts/post3.html)

> **Q: 为什么这里只教了 Linux 服务端 和 Windows 客户端的设置，为什么没有教 Linux 客户端 和 Windows 服务端的设置？**
 
> A: 从文章中应该也可见了，tinc 的设置非常简单，服务端、客户端，或者说不同平台的设置基本上都是换汤不换药的，学会了最基础的两个，理应其他所有的都已经会了
> 再者，tinc 这种点对点的 VPN，本身没有服务器和客户端之分，大不了就是有公网的看作服务端，有 ConnectTo 参数且没有公网的看作客户端，二者差距很小，因此不用再单独教了