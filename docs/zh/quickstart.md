# 快速上手

> **状态**：v0.2 到位（完整安装、环境配置、第一次跑预测）。

v0.1.0-alpha 的快速上手见根目录 [README.zh-CN.md](https://github.com/OranAi-Ltd/oransim/blob/main/README.zh-CN.md#-一分钟上手)。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODE` | `mock` | `mock` 离线确定性响应；`api` 调真 LLM |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容端点 |
| `LLM_API_KEY` | （未设） | 对应 API key |
| `LLM_MODEL` | `gpt-5.4` | 模型 |
| `LLM_STREAM` | `1` | 可流式时启用 |
| `LLM_CONCURRENCY` | `15` | 最大并发 |
| `SOUL_POOL_N` | `100` | LLM 人格 agent 数 |
| `PORT` | `8001` | 后端 API 端口 |

## 部署：单 worker 起跑

Oransim v0.2 的运行时 state（人口、agents、world model、Embedding Bus
索引、品牌记忆缓存）全部放在 **进程内单例**，OSS 版本无跨 worker 同步。

**部署时用单 worker**。设置 `WEB_CONCURRENCY >= 2`、`WORKERS >= 2` 或
`UVICORN_WORKERS >= 2` 会在启动时打出 `WARNING` —— 每多一个 worker 就
多一份 ~GB 的运行时 state，而且 sandbox / 品牌记忆 / UEB 跨请求会看到
不一致数据。共享状态 Redis 后端在 Enterprise 版路线图里；OSS 版到位
之前，单 worker 是正确选择。

完整后端到位时间见 [ROADMAP.md](https://github.com/OranAi-Ltd/oransim/blob/main/ROADMAP.md)。
