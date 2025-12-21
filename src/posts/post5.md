---
icon: laptop-code
date: 2025-12-18
category: 教程
tag:
  - AI
  - AnyRouter
  - ClaudeCode
star: true
sticky: true
---

# Windows 下 AnyRouter 等公益站的食用方法

AI 相信大家并不陌生，从之前的 `ChatGPT3`、`ChatGPT4` ，到如今的 `Claude 4.5`、`Gemini 3 pro`、`ChatGPT 5.2`，人类的 AI 已经进入了一个黄金时代。如今的 AI，显然已经从之前的彩云小梦续写智障小说图一笑，到天下万事事事用 AI 的程度。

作为一名开发者，自然也不例外，AI 写出的代码，如果提示词写的好，质量是很好的，而且很快。然而，AI 当然不是免费给你使用的，如果你原价买 `Cursor` 的会员，或者原价买 `Anthropic` 的套餐，价格是非常昂贵的，AI 输出的每个 Token 都是实实在在的美刀。

但极客圈从来不缺好人，诞生了 `AnyRouter`、`AgentRouter` 等优秀的公益站，给大家免费使用这些 AI，你说你还没有注册？那就赶紧注册一个账号吧！

*注：AnyRouter 需要 Linux.do 账户注册，AgentRouter 需要 Linux.do 或 GitHub 账户注册*

::: info 通过邀请码注册有更多初始金额

> AnyRouter: 初始可获得 100\$，每日签到可获得 25\$
> 注册码: https://anyrouter.top/register?aff=gnbe
> AgentRouter: 初始可获得 200\$，每日签到可获得 25\$
> 注册码：https://agentrouter.org/register?aff=uqdv

:::

## 安装 Claude Code

首先安装 `Node.js`

1. 进入 https://nodejs.org/en/download ，点击下面的 `Windows Installer (.msi)` 按钮下载安装包
2. 打开安装包安装 Node.js
3. 安装完成后打开命令行，输入 `npm -v` 和 `node -v` 验证安装，如果没有报错就说明安装成功

然后安装 `ClaudeCode`

在命令行中输入以下指令

```shell
# 如果用 pnpm 包管理器，把这里的 npm 写成 pnpm 即可

npm install -g @anthropic-ai/claude-code
```

安装完成后，在命令行输入 `claude` 即可开始使用 ClaudeCode，但是你现在并没有配置 `API_KEY` 和 `BASE_URL`，所以你还是用不了

此时，你要先前往 AnyRouter 或 AgentRouter 网站，点击 `API密钥`->`添加密钥` 来新增密钥，并得到一个形如 `sk-xxxxxxxxxxxxxxxxxxxxxxx` 的 API_KEY

然后在 PowerShell 中输入以下指令导入配置

```powershell
# 这里输入 BASE_URL
$env:ANTHROPIC_BASE_URL="https://anyrouter.org/"

# 这里输入 API_KEY 两个都要填，都填上面拿到的 API_KEY 即可
$env:ANTHROPIC_AUTH_TOKEN="sk-xxx"
$env:ANTHROPIC_API_KEY="sk-xxx"
```

有时候，可能会由于一些原因，AnyRouter 和 AgentRouter 可能连不上，这个时候你可以尝试更换 BASE_URL

::: info BASE_URL

> AnyRouter:
> https://c.cspok.cn
> https://anyrouter.top (主要)
> https://pmpjfbhq.cn-nb1.rainapp.top
> https://a-ocnfniawgw.cn-shanghai.fcapp.run
> 
> AgentRouter:
> https://agentrouter.org (主要)

:::

当然，你可以不输入 PowerShell 指令，只需要修改一个配置文件即可，并且在身边没有文档时是更方便的

打开路径 `C:\Users\你的用户名\.claude` ，里面有一个叫 `settings.json` 的配置文件（初次启动完成以后才会出现，此前 ClaudeCode 不认这个配置文件），里面是长这样的

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://anyrouter.top",
    "ANTHROPIC_AUTH_TOKEN": "sk-xxxxx",
    "ANTHROPIC_API_KEY": "sk-xxxxx",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "model": "claude-sonnet-4-5-20250929"
}
```

填写方法参考 PowerShell 指令，修改完文件同样重新打开 ClaudeCode 即可开始使用，这种方法非常适合快速切换 API，现在开始你就可以开始享受你的 AI Coding 了

## 可能想问的问题

> Q: 可以联动 IDE 吗？
> A: JetBrains IDE 和 VSCode 的插件商城中可见，搜索 Claude Code 即可找到

> Q: 使用过程中报 500 520 等错误怎么办？
> A: 这大概率是公益站的节点炸了，开关魔法上网或修改 BASE_URL 即可，如果不行，你可以尝试不断开关 ClaudeCode 然后做一些让机魂大悦的事情让它正常工作