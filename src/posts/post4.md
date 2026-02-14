---
icon: file
date: 2025-07-17
category: 教程
tag:
  - Linux
star: true
sticky: true
---

# Linux 下解压和打包文件的正确姿势

Linux 中的文件打包和解压总是特别复杂，与解压有关的软件包、指令及其对应的选项多种多样，非常非常难记忆，这里总结了一篇文章方便大家记忆和使用

先介绍一下 Linux 下常见的压缩文件类型

- `.tar`
- `.7z`
- `.zip`
- `.rar`
- `.tar.gz` 或 `.tgz`
- `.bz2`
- `.tar.bz2` 或 `.tbz2`
- `.xz`
- `.tar.xz`
- `.zstd`

其中 `.tar` 仅仅只是归档文件而已，事实上并没有压缩

这里给出一个不同压缩文件类型的对比表格，给大家参考

| 压缩类型       | 压缩速度  | 解压速度  |  压缩率  | 对应的压缩软件包 |
|:-----------|:-----:|:-----:|:-----:|:--------:|
| `.tar.gz`  | ★★★★★ | ★★★★★ | ★★☆☆☆ |   gzip   |
| `.tar.bz2` | ★★☆☆☆ | ★☆☆☆☆ | ★★★☆☆ |  bzip2   |
| `.tar.xz`  | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |    xz    |
| `.zip`     | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |   zip    |
| `7z`       | ★☆☆☆☆ | ★★☆☆☆ | ★★★★★ |    7z    |
| `rar`      | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ |   rar    |

现在，为大家一一介绍常见的软件包的使用方法

## 1. Tar 指令（最核心、最常用）

`tar` 是 Linux 中最常用的归档工具。**请注意：** `tar` 本身只是将多个文件“打包”成一个 `.tar` 文件，并不进行压缩。但它可以通过参数调用 gzip、bzip2 或 xz 等工具进行压缩。

**记忆口诀：**
*   **c** (Create): 打包/压缩
*   **x** (Extract): 解压
*   **v** (Verbose): 显示详细过程（看着舒服）
*   **f** (File): 指定文件名（**注意：这个参数必须放在所有参数的最后**）

### 解压文件

现代的 `tar` 命令非常智能，通常只需要记住万能组合 `-xvf`，它会自动识别压缩格式。

```bash
# 解压 .tar
tar -xvf FileName.tar

# 解压 .tar.gz
tar -zxvf FileName.tar.gz

# 解压 .tar.bz2
tar -jxvf FileName.tar.bz2

# 解压 .tar.xz
tar -Jxvf FileName.tar.xz

# 【推荐】万能解压（自动识别格式）
tar -xvf FileName.tar.gz
```

### 打包/压缩文件

打包时需要指定对应的压缩算法参数：
*   `-z`: 对应 gzip
*   `-j`: 对应 bzip2
*   `-J`: 对应 xz

```bash
# 仅打包，不压缩 (.tar)
tar -cvf FileName.tar DirName/

# 打包并使用 gzip 压缩 (.tar.gz) - 速度快，最常用
tar -zcvf FileName.tar.gz DirName/

# 打包并使用 bzip2 压缩 (.tar.bz2) - 压缩率稍好
tar -jcvf FileName.tar.bz2 DirName/

# 打包并使用 xz 压缩 (.tar.xz) - 压缩率极高，但慢
tar -Jcvf FileName.tar.xz DirName/
```

## 2. Zip 指令

`.zip` 是 Windows 下最常见的格式，在 Linux 中通常需要安装 `zip` 和 `unzip` 软件包。

### 解压

```bash
# 解压到当前目录
unzip FileName.zip

# 解压到指定目录 (-d)
unzip FileName.zip -d /path/to/directory
```

### 压缩

```bash
# 压缩目录（注意要加 -r 表示递归，否则只打包空文件夹）
zip -r FileName.zip DirName/

# 压缩单个文件
zip FileName.zip FileName
```

## 3. 7z 指令

`.7z` 格式拥有极高的压缩率。在 Linux 下通常需要安装 `p7zip` 或 `p7zip-full` 包。

### 解压

注意这里使用 `x` 而不是 `e`，`x` 会保持原有的目录结构，而 `e` 会把所有文件都解压到同一层级（会很乱）。

```bash
# 解压
7z x FileName.7z
```

### 压缩

`a` 代表 Add（添加）。

```bash
# 压缩文件或目录
7z a FileName.7z DirName/
```

## 4. Rar 指令

Linux 下处理 `.rar` 稍微麻烦一点，因为它是专有格式。通常需要安装 `rar` (用于压缩) 和 `unrar` (用于解压)。大多数情况我们只需要解压。

### 解压

```bash
# 解压到当前文件夹
unrar x FileName.rar
```

### 压缩

```bash
# 压缩目录
rar a FileName.rar DirName/
```

## 5. 单文件压缩指令 (gzip, bzip2, xz)

有时候我们会看到单独的 `.gz`, `.bz2`, `.xz` 文件（没有 `.tar`），这通常表示只压缩了一个单独的文件。

```bash
# gzip
gzip -d FileName.gz      # 解压
gzip FileName            # 压缩（原文件会被替换）

# bzip2
bzip2 -d FileName.bz2    # 解压
bzip2 -z FileName        # 压缩

# xz
xz -d FileName.xz        # 解压
xz -z FileName           # 压缩
```

## 6. Zstd 指令 (新秀)

`.zstd` 是 Facebook 开源的一种实时压缩算法，速度和压缩率都非常优秀，Arch Linux 等发行版已经大量采用。

### 单独使用

```bash
# 解压
zstd -d FileName.zstd

# 压缩
zstd FileName
```

### 配合 Tar 使用

```bash
# 解压 .tar.zst 或 .tzst
tar --zstd -xvf FileName.tar.zst

# 压缩为 .tar.zst
tar --zstd -cvf FileName.tar.zst DirName/
```

## 总结一张表

| 场景 | 核心指令 | 常用参数 | 示例 |
| :--- | :--- | :--- | :--- |
| **通用解压** | `tar` | `-xvf` | `tar -xvf file.tar.gz` |
| **通用压缩** | `tar` | `-zcvf` | `tar -zcvf file.tar.gz dir/` |
| **Windows 格式** | `unzip` | 无 | `unzip file.zip` |
| **高压缩率** | `7z` | `x` | `7z x file.7z` |
| **解压 RAR** | `unrar` | `x` | `unrar x file.rar` |

希望本文章能帮你搞定 Linux 下的文件解压和打包 :)


