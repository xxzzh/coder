# 大模型能力层升级说明

## 当前实现

本项目已从“离线可验证版”升级为“本地知识库 + 可选大模型能力层”的第一阶段架构。默认仍可在无 API key、无 FAISS 依赖的环境中运行；启用配置后，会使用 OpenAI-compatible embedding API 生成 dense embedding，并写入本地 FAISS 向量索引。

已落地能力：

- `scripts/model_capabilities.py`: 统一封装 embedding、查询分类、rerank 和答案质量检查。
- `scripts/vector_store.py`: 管理 `knowledge_base/vector/` 下的 FAISS 索引、映射和 metadata。
- 索引流程：SQLite 仍保存文档、chunk、来源和元数据；FAISS 只保存向量。
- 查询流程：领域 retriever 优先，通用检索合并 FAISS、FTS5、本地 hash embedding 和关键词候选。
- Web UI：展示 embedding、FAISS、LLM chunking、query routing、rerank 与降级状态。
- 审计：检查 FAISS/embedding 就绪状态；未启用时确认本地 hash embedding 兜底可用。

## 配置

```env
LKA_EMBEDDING_ENABLED=true
LKA_EMBEDDING_PROVIDER=openai-compatible
LKA_EMBEDDING_BASE_URL=https://your-api-base/v1
LKA_EMBEDDING_API_KEY=your_key
LKA_EMBEDDING_MODEL=your-embedding-model
LKA_EMBEDDING_DIMENSIONS=1024
LKA_VECTOR_BACKEND=faiss
LKA_RERANK_ENABLED=true
LKA_LLM_CHUNKING_ENABLED=true
LKA_LLM_QUERY_ROUTING_ENABLED=true
```

默认请求 1024 维 embedding。如果 API 实际返回其他维度，系统以实际返回维度写入 `vector_metadata.json`，并按该维度构建 FAISS。

## 降级规则

- embedding 未启用：使用 SQLite FTS5 + 本地 hash embedding。
- embedding 配置缺失或 API 不可用：写入降级 metadata，不阻塞建库。
- FAISS 依赖缺失：索引和查询自动回退，不影响基础问答。
- LLM rerank 失败：保留本地 rerank 顺序。

## 下一步升级顺序

1. 增强 `extract_structure(text)`，让 Excel/Word/PDF 抽取结果生成更稳定的结构化字段。
2. 启用 LLM 语义分块，把表格行、章节标题、关键实体、摘要和原文证据绑定到同一个可追溯 chunk。
3. 将查询改写扩展为 2-5 个检索 query，并为不同 query type 配置不同召回权重。
4. 把 rerank 从 chat JSON 排序升级为专用 rerank 模型或更严格的打分协议。
5. 扩展答案质量检查，增加引用覆盖率、证据冲突和证据不足提示。

