# Listopia Gateway（网关）README

## 0. 这是什么

一个本地 FastAPI Gateway，把 Telegram / RikkaHub 等入口接入统一链路：

* **对话入口**：`/api/v1/chat`（自定义） + ` /api/v1/openai_proxy/v1/chat/completions`（OpenAI 兼容代理）
* **MCP 工具**：`/api/v1/mcp/gateway_ctx`（统一上下文构建）与 `/api/v1/mcp/anchor_rag`（锚点检索）
* **Dify**：通过 workflow 提供 Anchor RAG（Gateway 侧转发到 Dify）
* **存储/总结**：本地 DB（SummaryS4 / SummaryS60），并在 OpenAI proxy 流式/非流式回写摘要

---

## 1. 运行环境（当前确认可跑）

* Windows 本地（VS Code 终端）
* 入口：`app/main.py` 
* 启动：

```bash
PS E:\gateway> uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* ngrok：

```
https://caliginous-marin-untakable.ngrok-free.dev -> http://localhost:8000
```

> 4040 是 ngrok inspect 端口，不是服务端口。

---

## 2. 路由总览（关键）

### 2.1 Debug

* `GET /api/v1/debug/routes`：打印当前注册的全部路由（排查“404 到底是不是路由没挂上”的第一手证据） 

### 2.2 Chat（自定义）

* `POST /api/v1/chat` 

### 2.3 MCP（工具）

* `GET|POST /api/v1/mcp/gateway_ctx` 
* `GET|POST /api/v1/mcp/anchor_rag` 

> `GET` 会返回 `{"ok": true, "name": "...", "mcp": true}`，用于“探活/识别”。  
> `POST` 是 **JSON-RPC 2.0** 的 MCP 调用入口。

### 2.4 OpenAI 兼容代理（非常重要）

* `POST /api/v1/openai_proxy/v1/chat/completions` 

这条路由会：

* 从请求里抽 `messages`、找最后一条 user 文本
* 读取 DB 里的 s4/s60 摘要注入 system
* **每轮强制调用本机 MCP `gateway_ctx`**（`FORCE_GATEWAY_EVERY_TURN=1` 默认开启）拿到 anchor snippet 并注入 system 
* 然后把请求转发到 `UPSTREAM_BASE_URL`（默认 openrouter），并把对话写回 DB 触发摘要滚动 

---

## 3. MCP 使用说明（RikkaHub / 手动 curl 都按这个）

你现在在 RikkaHub 填的 URL：

```
https://caliginous-marin-untakable.ngrok-free.dev/api/v1/mcp/gateway_ctx
```

这个 **本身就是一个 MCP JSON-RPC endpoint**，不是“目录型 root”。它支持的方法（JSON-RPC `method`）：

* `initialize`
* `tools/list`
* `tools/call`（tool 名固定为 `gateway_ctx`） 

同理，`/api/v1/mcp/anchor_rag` 也支持同样的三件套（tool 名为 `anchor_rag`）。 

### 3.1 initialize 示例

POST 到 `/api/v1/mcp/gateway_ctx`，body：

```json
{
  "jsonrpc": "2.0",
  "id": "init1",
  "method": "initialize",
  "params": { "protocolVersion": "2025-06-18" }
}
```

服务端会做版本谈判：优先用 params.protocolVersion，其次用 header 的 `MCP-Protocol-Version`，否则用默认版本（代码里默认 `2025-06-18`，并兼容若干版本集合）。 

### 3.2 tools/list 示例

```json
{
  "jsonrpc": "2.0",
  "id": "list1",
  "method": "tools/list",
  "params": {}
}
```

返回会列出 `gateway_ctx` 工具及 inputSchema。 

### 3.3 tools/call 示例（gateway_ctx）

```json
{
  "jsonrpc": "2.0",
  "id": "call1",
  "method": "tools/call",
  "params": {
    "name": "gateway_ctx",
    "arguments": {
      "keyword": "焦虑,哥哥",
      "text": "（可选）原始用户消息",
      "user": "rk:xxxx"
    }
  }
}
```

返回结构（核心是 `result.content[0].text`）：

* `result.content`: `[{"type":"text","text":"..."}]`
* `result.isError`
* `result.data`：带 debug（keyword / ctx / raw outputs） 

### 3.4 tools/call 示例（anchor_rag）

POST 到 `/api/v1/mcp/anchor_rag`，tool 名是 `anchor_rag`，arguments 至少要 `keyword`。 

---

## 4. 为什么会出现 “GET 200 但 POST 404”

在你这套实现里，**GET 200 只说明路由存在**。POST 404 常见原因反而是客户端打错了“具体 endpoint path”。

你现在 MCP 的实现是 **两个 endpoint**：

* `/api/v1/mcp/gateway_ctx`
* `/api/v1/mcp/anchor_rag`

如果客户端把 URL 当成“root”，然后内部再拼诸如：

* `/tools/list`
* `/tools/call`
* `/mcp`
* `/sse`
  之类的子路径，就会 POST 到不存在的地方 → 404。

所以排查优先级是：

1. 先 `GET /api/v1/debug/routes` 看 **客户端真正 POST 到了哪条 path**（日志 or 抓包） 
2. 若客户端需要“root 级 MCP server（一个 URL 下含多 tool）”，那就要新增一个 **聚合 MCP root**（把 `tools/list` 同时列出 gateway_ctx + anchor_rag，并在 tools/call 里路由到对应处理器）。目前代码是“一个 endpoint 对应一个 tool”。

---

## 5. Dify（当前用途）

* `gateway_ctx` 在工具调用时会去请求 Dify workflow（Anchor），把 outputs 的 `result/chat_text` 截断成 ctx/snippet。 
* `anchor_rag` 同理，只是返回 `snip`（并按 min/max 字符截断）。 

所需 env（至少）：

* `DIFY_API_KEY`（或 `DIFY_WORKFLOW_API_KEY`）
* `DIFY_WORKFLOW_RUN_URL`（默认 `https://api.dify.ai/v1/workflows/run`）
* （可选）`DIFY_WORKFLOW_ID_ANCHOR`  

---

## 6. OpenAI 兼容代理（对接 RikkaHub / 其他前端的关键点）

`/api/v1/openai_proxy/v1/chat/completions` 是 OpenAI compatible 的 chat/completions：

* 自动从 headers / payload / metadata 里生成 `session_id`（会加 `rk:` 前缀） 
* 拉取 DB 中最新 s4 / s60 摘要注入 system block 
* 若 `ANCHOR_INJECT_ENABLED=1` 且 `FORCE_GATEWAY_EVERY_TURN=1`：

  * 先做关键词抽取 `_extract_keywords`
  * 再 **POST 调用本机 MCP `gateway_ctx`**（`LOCAL_MCP_GATEWAY_URL` 默认指向 `http://127.0.0.1:8000/api/v1/mcp/gateway_ctx`）
  * 把返回 ctx 注入 system block 
* 转发到上游（默认 `https://openrouter.ai/api/v1`），支持流式并在结束后写回 DB 触发摘要滚动 

---

## 7. 当前进度（落地事实）

✅ 已具备：

* MCP `gateway_ctx` endpoint（json-rpc initialize/list/call 全套）
* MCP `anchor_rag` endpoint（json-rpc initialize/list/call 全套）
* OpenAI proxy 入口，且能每轮通过本机 MCP 注入 anchor ctx
* s4/s60 的 DB 读取与写回骨架（append_user_and_assistant 触发） 

🟨 仍需工程化：

* 若 RikkaHub 期待“单一 MCP root server”而不是“endpoint=tool”，需要做 **聚合 root**（一个 URL 同时暴露多个 tools）
* Notion 四库字段契约（目前 README 未写死）

---

## 8. Codex 下一步建议（最省命、最高收益）

1. **先验证 RikkaHub 对 MCP URL 的假设**

   * 如果它要求“root”，就做 `/api/v1/mcp` 这样的聚合 endpoint（tools/list 返回两个 tool；tools/call 分发到不同 handler）
   * 如果它支持“endpoint=tool”，那你现在填的 URL 就对，重点变成“它实际 POST 到了哪个 path”

2. **加可观测日志**

   * 在 `gateway_ctx_mcp` / `anchor_mcp` 的 POST 分支打印：method、params.name、路径、返回 header 的 MCP 版本
   * 这样就不会再陷入“到底有没有调用到”的黑箱

3. **把 Notion 字段契约补齐**

   * 不需要你逐字段口述：直接导出 Notion DB schema（或截图/JSON），Codex 录入成常量表即可

---

## 9. 你现在可以休息的原因（MCP 这块我已经替你钉住了）

* 你在 RikkaHub 填的 URL **确实是一个 MCP endpoint**，而且它的 json-rpc 方法实现完整 
* 你并不是“只做了个 GET 探活”，而是工具调用链路也写了（甚至还做了协议版本谈判） 
* 你还写了第二个 MCP tool（anchor_rag），只是在 RikkaHub 里没配置它的 URL 所以没发现 

---

如果你愿意再省一次力：你只要告诉我——**RikkaHub 的 MCP 配置项是“填一个 URL”还是“可以填多个工具 URL”**（一句话），我就能在 README 里把“该填哪个 URL”写成最终答案；同时也能判断要不要做聚合 root（一个 URL 暴露两个工具）。你不想回也行，Codex 也能按第 8 节直接开干。
