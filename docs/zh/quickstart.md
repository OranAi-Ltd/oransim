# 快速上手

> **状态**：v0.2 到位（完整安装、环境配置、第一次跑预测）。

v0.1.0-alpha 的快速上手见根目录 [README.zh-CN.md](https://github.com/deterministically/oransim/blob/main/README.zh-CN.md#-一分钟上手)。

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

完整后端到位时间见 [ROADMAP.md](https://github.com/deterministically/oransim/blob/main/ROADMAP.md)。
