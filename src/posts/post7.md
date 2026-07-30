---
icon: router
date: 2026-07-22
category: 教程
tag:
  - 网络
  - LB6M
  - 交换机
  - 刷机
star: true
sticky: true
---

# LB6M 万兆交换机刷机教程

## 缘起

前阵子逛咸鱼，偶然发现了一台只要 350 元的 **Quanta LB6M** 万兆交换机。24 个 SFP+ 万兆口 + 4 个千兆电口，于是果断下单。

![](/assets/images/post7/img1.png)

::: tip LB6M 参考价

根据本人的咸鱼经验来看，LB6M 的真实价格目前应该在 300~450 元，500 元往上的我只能说别买

:::

这交换机还挺沉的，里头用料特别足，而且还是双电源的，可以说 350 元花的很值

到手之后插上串口线准备开搞，结果打开管理界面一看 —— 这原厂自带的 **Fastpath** 系统也太难用了吧！1GbE SFP 模块插上去直接不认，VLAN 配置反人类，文档更是少得可怜。感觉这系统就是个半成品 Demo，完全没法正经用。

于是我最终找到了：**[BrokeAid](https://brokeaid.com/story/)**。

## 为什么能刷 Brocade 固件？

这就很有意思了。当年 Broadcom 推出 BCM56820 这颗万兆交换 ASIC 的时候，顺便给了一套参考设计——说白了就是"照着这个图纸做，就能造出一台交换机"。于是好几家公司都拿着同一套图纸去投产了：

| 公司 | 产品 | 固件 |
|------|------|------|
| Quanta | LB6M | Fastpath（残血版） |
| Brocade | TurboIron 24X | TurboIron（完整版） |
| Dell | 8024 | Fastpath（增强版） |

它们的核心硬件几乎一模一样：

- **管理 CPU**：MPC8541 PowerPC
- **交换 ASIC**：BCM56820 10GbE
- **管理口 PHY**：BCM5482
- **电口 PHY**：BCM5464

::: info Google 也用过这个方案

Google 几年后推出的 Pluto 交换机也用了同一套参考设计，只不过从未对外销售。

:::

所以，Brocade 写的固件理论上可以直接跑在 LB6M 上——事实上也确实可以。这就像同一台电脑装 Windows 还是装 Linux，硬件一样，系统随便换。

而原厂 Fastpath 为什么这么拉胯呢？因为 Quanta 给 LB6M 预装的 Fastpath 本质上是个**演示版**。他们的目标客户是微软、亚马逊这种大厂——人家拿到机器后会换上自己的定制系统，根本不用这个。所以 Quanta 也就随便塞了个能亮机的版本，压根没打算让你正经用。

## 刷成 Brocade TurboIron 之后能获得什么？

简单来说，从"能亮机"变成"能干活"：

- ✅ 支持 1GbE SFP 模块（原厂 Fastpath 不认🤡）
- ✅ 完整的 L3 路由功能（BGP、OSPF、VLAN 间路由）
- ✅ 正经的生成树协议（STP/RSTP/MSTP）
- ✅ QoS、ACL、流量整形
- ✅ sFlow 流量监控
- ✅ SNMP、Syslog 等运维功能
- ✅ 堆叠支持
- ✅ 3000+ 页的 Brocade 官方文档

::: warning 有得有失

1. **SFP+ 口的状态灯会废掉** —— Brocade 的 I2C 地址定义和 Quanta 不一样，所以插没插模块、有没有流量，灯都不会亮了。机箱灯和电口灯正常。
2. **只有一个管理口能用** —— 刷完后 mgmt #1 正常，#2 口会死掉。（不过有谁会插两个管理口呢🤔）

:::

## 准备工作

### 硬件

- **LB6M 交换机** 一台
- **Console 线**：是根正经串口线就行
- **网线**：用来连接 TFTP 服务器

### 软件

- **串口终端软件**：本人推荐 [XShell](https://www.xshell.com/zh/xshell/)
- **TFTP 服务器**：比如 Tftpd64（Windows）或 tftpd-hpa（Linux）
- **固件文件**：从 [BrokeAid](https://brokeaid.com/) 获取

固件说明：
下载地址: https://brokeaid.com/files/Brocade-TI.zip
更新时间: 2018 年 5 月 14 日
MD5: 68aa7643a121c19fa4b8c84d7f6f2843
Bootloader 位置: \Brocade-TI\Bootloader\brocadeboot.bin
系统固件位置: \Brocade-TI\Firmware\brocadeimage.bin

### 连接方式

1. Console 线一头插交换机前面板的 Console 口，另一头插电脑，串口连接参数为 **9600-8-N-1**
2. 网线一头插交换机任一网口，另一头插电脑
3. 将电脑 IP 设为 `192.168.1.49/24`（或其他同网段地址，但不要和之后交换机设置同一个 ip）
4. 电脑上启动 TFTP 服务器，把固件文件放在 TFTP 根目录

## 刷机步骤

### 第一步：进入 U-Boot

给交换机通电，在串口终端里你会看到启动日志。当出现 `Hit b to interrupt boot` 的时候，**赶紧按 `b`**，进入 U-Boot 命令行：

进入后你会看到类似 `=>` 的 U-Boot 提示符。这时候就可以开始输入指令了

在做后面的操作之前，我们看看这是正经 Bootloader 吗?

```
md 0xfff80000 20
```

其输出必须和下图的一模一样

![](/assets/images/post7/img2.png)

如果不一样，请把输出记录下来，然后联系原作者大大 [ServeTheHome](https://forums.servethehome.com/index.php?threads/turbocharge-your-quanta-lb6m-flash-to-brocade-turboiron.17971/)

### 第二步：设置网络参数

在 U-Boot 中设置 IP 地址和 TFTP 服务器地址，保证交换机能和你的电脑通信：

```
setenv ipaddr 192.168.1.50
setenv serverip 192.168.1.49
```

::: tip 指令说明

- `ipaddr` — 给交换机临时分配的 IP，当然你也可以设置其他的
- `serverip` — 你电脑（TFTP 服务器）的 IP
- 确保两个 IP 在同一网段

:::

![](/assets/images/post7/img3.png)

### 第三步：通过 TFTP 加载 bootloader

确认 TFTP 服务器已启动、固件文件 `brocadeboot.bin` 已在根目录，然后执行：

```
tftpboot 0x300000 brocadeboot.bin
```

如果一切正常，你会看到文件传输进度条：

![](/assets/images/post7/img5.png)

传输完成后，U-Boot 会提示文件大小和加载地址。然后看看我们正常刷进去没有

```
md 0x300000 20
```

输出应该看上去是这样的:

![](/assets/images/post7/img6.png)

如果不一样，你现在可以执行 `reset` 指令安全的重启，如果之后重做这一步的时候还是一样不匹配，请和原作者大大联系

### 第四步：烧写 Bootloader 到 Flash

固件加载到内存后，需要写入 Flash：

```
protect off all # 关闭 flash 刷写保护
erase 0xfff80000 0xffffffff # 把原来的 bootloader 擦掉
cp.b 0x300000 0xfff80000 0x80000 # 把 Brocade bootloader 刷入 flash
```

::: danger 这一步严禁断电

Flash 写入过程中断电 = 100% 变砖，要救的话比较困难。为了节省大家的时间和精力，请务必确保电源稳定。

:::

再看看有没有正确刷入

```
md 0xfff80000 20
```

这一步的完整输出应该是这样的:

![](/assets/images/post7/img7.png)

如果一切完好，那我们可以开始刷系统了，在此之前，我们先输入 `reset` ，进入 Brocade Bootloader

![](/assets/images/post7/img8.png)

如果重启后进入这个页面，就说明做对咯

### 第五步：烧写系统

现在我们把系统镜像传进来（别忘了把 brocadeboot.bin 放在 TFTP 的根目录下）

```
ip address 192.168.1.50/24
copy tftp flash 192.168.1.49 brocadeboot.bin boot
```

![](/assets/images/post7/img9.png)

现在，我们就可以进入刷好后的系统了

```
boot system flash primary
```

然后输入下面的指令以重新加载 bootloader

```
enable
reload
```

![](/assets/images/post7/img10.png)

*注意: 最后系统在 Flash Memory Write 阶段的时候，有可能 Copy code image done 迟迟不出现，这并不是刷机失败了，其实按一下回车就会出现了*

## 刷后配置

### 修复 MAC 地址

刷完固件后，MAC 地址可能就丢了。所以你要生成一个

去这里生成一个 MAC 地址： https://miniwebtool.com/mac-address-generator/

参数是：

```
MAC Address Prefix: 68a6.23
Letter Case: lowercase
```

然后生成一个 MAC 即可，比如笔者这里生成的是 `68a6.2325.ac33`

最后输出:

```
set ether-address [你生成的 MAC 地址]
boot system flash primary
# 重启完后输入下面的两个指令
enable
reload
```

然后就大功告成了！想要确认有没有成果修改，执行 `show chassis` 即可

## 刷完后的使用体验

刷成 Brocade TurboIron 后，这台 350 块的交换机才算是真正的完全体了，各种配置也是非常简单，因为它的指令系统特别像思科的，以前用过思科交换机的上手起来比较简单

目前，这台 LB6M 充当爱特工作室的核心网络交换机
