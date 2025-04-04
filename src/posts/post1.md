---
cover: /assets/images/post1/head.jpg
icon: network-wired
date: 2022-01-12
category: 教程
tag:
  - 计算机网络
  - Windows
  - Linux
star: true
sticky: true
---

# 如何让串口能够通过网络访问

::: info 俳句

> 平日没事干
> 想让串口连上网
> 头发嘎嘎掉

:::

## 情景介绍

我最近花 177 大洋买了台金千的 **写字机** ，想拿这玩意来帮我写大学物理实验报告

然而，这台写字机有一个致命的缺点，那就是它只能通过 Type-B 接口连接电脑后，再在写字机软件上连接上它的串口端口才能使用

而且送的线也非常短，十分不方便，再者宿舍里的地本来就十分宝贵，尤其是我的座位，要是再放个写字机岂不是得炸？

所以，我想了一个办法让这种串口设备穿透到宿舍的内网里，就可以愉快的用网络连接写字机了！

## 基本思路

- 将写字机连接到一台 Linux 系统的小机器上，让这台小机器认到写字机
- 使用 [ser2net](https://github.com/cminyard/ser2net) 将串口穿透到网络中(串口 -> TCP)
- 在 Windows 上使用 [VSPD](https://cloud.tencent.com/developer/article/2311089) 创建**一对**虚拟串口
- 编写 python 程序将 TCP "转化"为串口
- 让写字机软件连上虚拟串口
- 愉快畅写

## 需要的材料

- Windows 电脑 x1
- 玩客云(或其他的小机器) x1
- 写字机和 Type-B 线(废话) x1
- 脑子 x1

## 具体步骤

首先，将你的写字机插上电，然后把线插在玩客云的 USB 口上。

![](/assets/images/post1/img1.png)

我们连上玩客云的 SSH ，输入 `lsusb` 指令，查看认到的 USB 设备，这个时候应该是能看到写字机设备了，如果不确定到底是不是写字机，可以拔线然后再执行一次这个指令，看一下哪个设备少了，少掉的那个设备就是你的写字机

![](/assets/images/post1/img2.png)

这时候我们再输入 `ls /dev/ttyUSB*` (如果没有的话可以试试 `ls /dev/ttyACM*` 指令)看看哪个是你的写字机的设备名，如果还是不清楚哪个是你的，和上面同理，拔线看看哪个设备少了就行了

通过这个指令，我们就可以获取到串口的名字，有了这个以后我们就可以让这个串口网上冲浪了

我们输入 `apt install ser2net` 下载安装需要的软件包，然后输入 `vim /etc/ser2net.yaml` 编辑 ser2net 的配置文件

::: tip 注意事项

Vim 是 Linux 上的一款实用的文本编辑器，如果在输入 `vim /etc/ser2net.yaml` 时提示没有 vim ，那么你用 `nano`、`vi` 这些文本编辑器也是没有问题的

如果想要下载 vim ，推荐使用 neovim ，在命令行中输入 `apt install neovim` 下载安装即可，使用时使用 `nvim` 指令

:::

进入配置文件，图中红色箭头所指的配置项从上到下依次为

端口号 启用配置项 要穿透的串口设备路径 波特率 

![](/assets/images/post1/img3.png)

假如我现在要让 `/dev/ttyUSB3` 这个串口通过本机的 `10001` 端口穿透出去，并且设置波特率为 `9600`，那么这里的配置应该是：

```yaml
connection: &con0096
    accepter: tcp,10001
    enable: on
    options:
      banner: *banner
      kickolduser: true
      telnet-brk-on-sync: true
    connector: serialdev,
              /dev/ttyUSB3,
              9600n81,local
```

配置完以后按 ESC 键，输入 `:wq` 保存退出(如果你用的是 nano ，按 ctrl X ，然后按下 Y 键即可)

最后我们输入 `systemctl restart ser2net` 重启 ser2net 服务即可(或者输入 `service ser2net reload` 重载配置也可以)

![](/assets/images/post1/img4.png)

::: center 

*这张图右侧其实还显示了一个 [ OK ] 实际操作的时候别漏看了* 

:::

至此，串口就穿透出去了，现在的问题是让 TCP 变回串口呢？该死的 Windows 非常奸诈，没办法很简单的创建一对虚拟串口，Linux 那边倒是挺简单的，如果你用的是 Linux ，可以跳过这一步(yysy，linux 上有写字机软件吗？)

笔者在此之前尝试了很多很多软件，比如 com2com 、Serial to Ethernet Connector 什么的，都失败了，古老、收费这些问题太缠人了

最后，终于找到了我认为最好的软件，[VSPD](https://cloud.tencent.com/developer/article/2311089)

在我们安装好 VSPD 后(不会有人都用 Linux 了都还不会安装软件的吧?)，打开软件进入这个界面

![](/assets/images/post1/img5.png)

点击添加端口，就会为你创建一对虚拟串口，记号它们的名字，我们会用到，为了方便理解，这边假设你创建了一对分别叫 COM1 和 COM2 的虚拟串口

::: tip 为什么串口要创建一对？

在 Windows 上，串口类似网络上的端口，串口也只能一次让一个服务占用

在这种情况下，其中一个串口需要被 **写字机软件** 占用，另外一个需要被我们编写的 **TCP转串口** 软件占用，因此要占用两个串口

同时，这两个串口是“通”的，它们内部会传递交换信息的，这点你就不用担心了

:::

这里，我们编写一个 python 程序用于实现 TCP <-> 串口

首先我们需要安装 pySerial 库，在命令行中输入 `pip install pyserial` 安装

安装完成后，开始编写程序，笔者这边已经写好程序了，这里提供给大家

```py
import serial
import socket
import threading
import time

# 配置参数
TCP_IP = '192.168.1.112' # 填玩客云的ip
TCP_PORT = 10086 # 填端口
VIRTUAL_SERIAL_PORT = 'COM2' # 填想要使用的虚拟串口
BAUDRATE = 115200 # 填波特率

# 初始化串口
try:
    ser = serial.Serial(
        port=VIRTUAL_SERIAL_PORT,
        baudrate=BAUDRATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1  # 缩短超时时间避免阻塞
    )
    print(f"[+] 串口 {VIRTUAL_SERIAL_PORT} 已打开")
except Exception as e:
    print(f"[-] 串口打开失败: {e}")
    exit()

# 初始化TCP连接
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((TCP_IP, TCP_PORT))
    print(f"[+] 已连接到TCP服务器 {TCP_IP}:{TCP_PORT}")
except Exception as e:
    print(f"[-] TCP连接失败: {e}")
    ser.close()
    exit()

running = True

def tcp_to_serial():
    """TCP → 串口"""
    while running:
        try:
            data = sock.recv(1024)
            if data:
                ser.write(data)
                # 打印十六进制和ASCII字符串
                ascii_str = data.decode('ascii', errors='replace')
                print(f"TCP -> 串口 ({len(data)}字节): {data.hex()} -> {ascii_str}")
        except Exception as e:
            print(f"TCP到串口错误: {e}")
            break

def serial_to_tcp():
    """串口 → TCP"""
    while running:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                sock.sendall(data)
                # 打印十六进制和ASCII字符串
                ascii_str = data.decode('ascii', errors='replace')
                print(f"串口 -> TCP ({len(data)}字节): {data.hex()} -> {ascii_str}")
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"串口到TCP错误: {e}")
            break

# 启动线程
t1 = threading.Thread(target=tcp_to_serial, daemon=True)
t2 = threading.Thread(target=serial_to_tcp, daemon=True)
t1.start()
t2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[!] 正在关闭...")
    running = False
    sock.close()
    ser.close()
    t1.join(timeout=1)
    t2.join(timeout=1)
    print("[+] 资源已释放")

```

如果你是 Linux 用户的话，可以试试这个程序

```py
import serial
import socket
import threading
import time

# 配置参数
TCP_IP = '192.168.1.112'
TCP_PORT = 10086
VIRTUAL_SERIAL_PORT = '/dev/ttyUSB0'  # Linux 上的虚拟串口设备
BAUDRATE = 115200

# 初始化串口
try:
    ser = serial.Serial(
        port=VIRTUAL_SERIAL_PORT,
        baudrate=BAUDRATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1  # 缩短超时时间避免阻塞
    )
    print(f"[+] 串口 {VIRTUAL_SERIAL_PORT} 已打开")
except Exception as e:
    print(f"[-] 串口打开失败: {e}")
    exit()

# 初始化TCP连接
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((TCP_IP, TCP_PORT))
    print(f"[+] 已连接到TCP服务器 {TCP_IP}:{TCP_PORT}")
except Exception as e:
    print(f"[-] TCP连接失败: {e}")
    ser.close()
    exit()

running = True

def tcp_to_serial():
    """TCP → 串口"""
    while running:
        try:
            data = sock.recv(1024)
            if data:
                ser.write(data)
                # 打印十六进制和ASCII字符串
                ascii_str = data.decode('ascii', errors='replace')
                print(f"TCP -> 串口 ({len(data)}字节): {data.hex()} -> {ascii_str}")
        except Exception as e:
            print(f"TCP到串口错误: {e}")
            break

def serial_to_tcp():
    """串口 → TCP"""
    while running:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                sock.sendall(data)
                # 打印十六进制和ASCII字符串
                ascii_str = data.decode('ascii', errors='replace')
                print(f"串口 -> TCP ({len(data)}字节): {data.hex()} -> {ascii_str}")
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"串口到TCP错误: {e}")
            break

# 启动线程
t1 = threading.Thread(target=tcp_to_serial, daemon=True)
t2 = threading.Thread(target=serial_to_tcp, daemon=True)
t1.start()
t2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[!] 正在关闭...")
    running = False
    sock.close()
    ser.close()
    t1.join(timeout=1)
    t2.join(timeout=1)
    print("[+] 资源已释放")

```

Linux 上的程序笔者没有测试环境(毕竟我没有 Linux 版本的写字机软件)，看起来是正确的，可以试试看，不行的话可以联系我改改

现在，我们可以开始运行程序了，在这里假设你使用了 **COM2** 串口让 python 程序使用，那么在写字机上你就要选择 **COM1** 串口来连接写字机

现在，看着程序运行的日志，如果写字机软件点击连接后，终端类似如下有输出(如图)，说明连接成功，可以愉快的使用写字机了

![](/assets/images/post1/img6.png)

## 大胆的想法

既然在内网里可以随意连接了，那如果搭配 frp 穿透什么的，岂不是可以在地球的任何地方都可以使用这台写字机？

并且，本文的应用场景还是局限了，还可以开发更多用途，看你脑洞了

## FAQ

Q: 为什么我打开写字机连接或者打开 python 程序后报错了或者没反应？

A: 有可能是你的串口被占用了，本文提过不要让这两个软件用同一个串口，而是要用一对串口，甚至你压根就没成功创建虚拟串口