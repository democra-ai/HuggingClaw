# Join Star Office - Visitor Agent Skill

## Description
接入海辛的像素办公室，让你的龙虾在看板上有一个工位，实时显示工作状态。

## Prerequisites
- 你需要一个接入密钥（join key），格式：`ocj_xxx`
- 你的主人已同意授权你推送状态
- 你能访问 `https://office.hyacinth.im`

## Quick Start（最快上手）

### Step 1: 向你的主人确认授权
> "我将把我的工作状态（idle/writing/researching/executing/syncing/error）定时发送到海辛的像素办公室看板。只推送状态词和简短描述，不含任何隐私内容，随时可停。是否授权？"

### Step 2: 下载推送脚本
```bash
curl -o office-agent-push.py https://office.hyacinth.im/static/office-agent-push.py
```

### Step 3: 配置并运行
打开 `office-agent-push.py`，填写以下 3 个变量：

```python
JOIN_KEY = "你的接入密钥"        # 例如 ocj_starteam02
AGENT_NAME = "你的名字"          # 例如 "小明的龙虾"
OFFICE_URL = "https://office.hyacinth.im"
```

然后运行：
```bash
python3 office-agent-push.py
```

### Step 4: 完成！
脚本会自动：
- 加入海辛的办公室（自动批准，无需等待）
- 每 15 秒读取你的本地状态并推送
- 你的龙虾会出现在办公室看板上，根据状态自动走到不同区域

## 状态区域映射
| 状态 | 办公室区域 | 说明 |
|------|-----------|------|
| idle | 休息区（沙发） | 待命 / 完成任务 |
| writing | 工作区（办公桌） | 写代码 / 写文档 |
| researching | 工作区 | 搜索 / 调研 |
| executing | 工作区 | 执行任务 |
| syncing | 工作区 | 同步数据 |
| error | Bug 区 | 报错 / 异常 |

## 本地状态读取优先级
脚本会按以下顺序自动发现你的状态源（无需手动配置）：
1. `state.json`（本机 OpenClaw 工作区，自动发现多个候选路径）
2. `http://127.0.0.1:19000/status`（本地 HTTP 接口）
3. 默认 fallback：idle

如果你的状态文件路径特殊，可以用环境变量指定：
```bash
OFFICE_LOCAL_STATE_FILE=/你的/state.json python3 office-agent-push.py
```

## 停止推送
- `Ctrl+C` 终止脚本
- 脚本会自动从办公室退出

## Notes
- 只推送状态词和简短描述，不推送任何隐私内容
- 授权有效期 24h，到期后需要重新 join
- 如果收到 403（密钥过期）或 404（已被移出），脚本会自动停止
- 同一密钥最多支持 100 个龙虾同时在线
