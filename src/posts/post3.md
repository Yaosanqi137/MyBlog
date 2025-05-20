---
cover: /assets/images/post3/head.png
icon: network-wired
date: 2025-05-20
category: 教程
tag:
  - OpenWRT
  - 磁盘管理
  - Linux
star: true
sticky: true
---

# OpenWRT overlay 分区扩容

有时候在用 OpenWRT 的时候，因为某些固件的一些奇葩原因，或者自己安装固件的方法问题，可能会发现自己的的空余磁盘空间很小很小，根本不够用，但是自己的设备显然不可能只有这么一点磁盘空间的时候，很可能是你的很多空间都没有被分配，或者全部分配到其他分区去了，而不是 overlay 分区

所以，我们需要对 overlay 分区进行扩容操作

![](/assets/images/post3/img1.png)

## 操作磁盘分区

首先，我们先在 OpenWRT 上安装磁盘管理工具，这里使用 cfdisk 作为演示。cfdisk 是一个图形化的磁盘操作软件包，用起来非常方便，使用这个命令进行安装

```shell
sudo opkg install cfdisk
```

安装完毕后，我们输入

```shell
block info
```

或者如果有 fdisk 软件包的话，可以用这个指令

```shell
fdisk -l
```

查看自己的磁盘名称和对应的磁盘设备名，笔者设备如图所示

![](/assets/images/post3/img2.png)

发现我们的硬盘设备是 mmcblk0 此时我们进入 `/dev` 文件夹，发现确实有这个设备，说明这个磁盘设备确实是存在的

::: info 为什么是 mmcblk ?

由于笔者用的是 `京东云亚瑟AX1800 pro` ，因此他的存储设备是闪存(嵌入式存储 eMMC)

而在 Linux 中，eMMC/SD卡 设备就是 mmcblk 表示 `MMC/SD Block Device`

而对于 SATA/USB 设备，在 Linux 中则是 sd 开头的，一般是叫 sda 之类的

还有一种 NVMe SSD 设备，在 Linux 中则是 nvme 开头的，比如 nvme0 之类的，其他更多设备类型可以自己上网查查

:::

那我们现在就可以开始编辑分区了，输入指令

```shell
cfdisk /dev/mmcblk0
```

就会进入一个图形化的界面，还是非常美观的

![](/assets/images/post3/img3.png)

注意箭头指向的绿色那一行，这就是你设备的空闲空间，我们用方向键，切到空闲空间哪一行，然后选择 new 新建分区

之后它会让你填 `Partition size`，这是让你填写新分区的空间，笔者这里把所有空间都用上，默认也是全部空间

创建完以后，你就会看见新的分区，因为笔者这边上一个分区是 mmcblk0p28 (p表示分区)，那么就可以看到后面新追加的新分区 mmcblk0p29 了

::: danger 严重警告！！！

一定要看清楚哪个是你的新分区，因为等一下还涉及到格式化分区，千万别把 /overlay 分区或者引导分区什么的给格式化了，不然你就等着重新刷固件吧

:::

确认已经创建好新分区以后，我们退出 cfdisk 输入

```shell
mkfs.ext4 /dev/mmcblk0p29 # /dev/后面接你 新分区 的名字，千万别搞错了
```

对新分区进行格式化操作，操作完成后，我们挂载分区

```shell
mkdir /mnt/mmcblk0p29 # 先新建一个挂载用的文件夹，名字你随意，后面别输错了就行了

mount /dev/mmcblk0p29 /mnt/mmcblk0p29 # 挂载新分区到刚刚创建的文件夹里面
```

然后迁移 /overlay 分区

```shell
cp -r /overlay/* /mnt/mmcblk0p29
```

等待迁移完毕并确认正常迁移后，我们进入 OpenWRT 的管理页面（笔者这里用 argon 主题作为演示）

![](/assets/images/post3/img4.png)

分别点击 系统-挂载点，然后下滑，在 **挂载点** 那一栏点击添加，分别点击启用、选择你创建的新分区、选择作为外部 overlay 使用

点击保存，然后重启你的路由器，重启完毕后你就发现你的 /overlay 分区已经成功扩容了