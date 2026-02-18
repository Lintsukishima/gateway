# MCP endpoint verification notes

This repository mounts both MCP routers under the same prefix (`/api/v1/mcp`) and relies on unique route paths:

- `anchor_mcp_router` serves `/anchor_rag`
- `gateway_ctx_router` serves `/gateway_ctx`

The `gateway_ctx` route exposes the tool name `gateway_ctx` in its `tools/list` JSON-RPC response.
