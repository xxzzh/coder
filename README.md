# Local Knowledge Base Agent

一个可本地运行的知识库问答 Agent。用户把原始资料放入 `knowledge_base/raw`，运行索引更新命令后，就可以通过命令行直接提问。

当前定位：可交付试用的本地 RAG 工具。它已经具备安装配置、健康检查、多格式资料导入、语义分块、混合检索、缓存、增量索引和降级兜底能力；但它不是带权限管理、Web 控制台和分布式向量数据库的生产级平台。

## 支持格式

- Markdown: `.md`
- Word: `.docx`
- PDF: `.pdf`
- Excel: `.xlsx`

资料可以按类型放入子目录，也可以放在 `knowledge_base/raw` 的任意递归子目录中：

```text
knowledge_base/raw/
  markdown/
  word/
  pdf/
  excel/
```

## 快速开始

普通用户推荐直接双击：

```bat
start.bat
```

菜单会提供初始化、打开资料目录、严格建库、健康检查、提问、查看来源和 OCR 测试。

首次使用：

```bat
.\run-agent.bat setup
```

安装配置会询问是否有大模型 API：

- 如果有 API key，会写入 `.env`，后续答案会在本地检索证据基础上调用 API 做受控总结。
- 如果没有 API key，会保持纯本地命令行回答，不影响索引、检索和引用输出。

检查环境：

```bat
.\run-agent.bat verify
```

更新索引：

```bat
.\run-agent.bat update
```

提问：

```bat
.\run-agent.bat "What is Ohm's law?"
```

兼容旧式 ask 命令：

```bat
.\run-agent.bat ask "加强针有什么作用？"
```

只使用本地知识库、不联网：

```bat
.\run-agent.bat ask "加强针有什么作用？" --no-web
```

彻底重建索引：

```bat
.\run-agent.bat rebuild
```

查看索引来源：

```bat
.\run-agent.bat sources
```

## 命令说明

| 命令 | 作用 |
|---|---|
| `setup` | 交互式初始化，创建 `.env`，可选择配置 API |
| `verify` / `healthcheck` | 检查 Python、SQLite FTS5、raw 文件、索引、API 配置 |
| `update` | 增量更新知识库，只处理新增、修改、删除的资料 |
| `rebuild` | 清理生成索引并从 raw 目录重新导入 |
| `sources` | 列出当前索引中的资料来源 |
| 直接输入问题 | 查询本地知识库；本地不足时默认允许联网兜底 |
| `ask "问题" --no-web` | 只查本地知识库 |

## API 配置

`.env` 示例：

```env
LKA_USE_API=true
LKA_API_PROVIDER=openai-compatible
LKA_API_BASE_URL=https://api.openai.com/v1
LKA_API_KEY=your_api_key_here
LKA_API_MODEL=gpt-4o-mini
LKA_API_TIMEOUT_SECONDS=20
```

不使用 API：

```env
LKA_USE_API=false
```

API 只用于“基于已召回引用的答案总结”。检索、索引、缓存和引用都在本地完成。没有 API 时，系统会返回本地抽取式答案。

临时禁用 API：

```bat
python scripts\query_knowledge_base.py "What is ACID?" --no-api
```

## 当前检索技术

- SQLite FTS5 全文检索
- 本地 hash n-gram embedding
- 语义分块
- FTS + semantic cosine + keyword score 混合 rerank
- 热门 Query 缓存
- raw 文件变化检测和异步增量更新
- 索引不可用时 FAQ 兜底

当前没有接入独立向量数据库。对于小中型本地资料库，SQLite + 本地 embedding 更轻量；当 chunk 数量达到几万以上时，可以再迁移到 Qdrant、Milvus、Chroma 或 FAISS。

## 依赖

核心功能只依赖 Python 标准库。

可选增强依赖列在 `requirements.txt`：

```bat
python -m pip install -r requirements.txt
```

这些依赖用于改善 PDF 抽取和中文繁简转换；没有安装时系统会走内置轻量逻辑。

## 扫描件、图片表格和复杂版式

为了避免资料内容被静默漏掉，导入流程会生成抽取质量报告：

```text
knowledge_base/processed/extraction_report.jsonl
```

`verify` 会统计报告中的问题：

```bat
.\run-agent.bat verify
```

报告字段含义：

- `warnings`: 抽取过程中的风险，例如 PDF 似乎包含图片、DOCX 内嵌图片、XLSX 有图表/绘图。
- `requires_review`: 该文件可能存在未完整抽取的内容，需要人工处理或安装 OCR 后重建索引。
- `method`: 实际使用的抽取方法，例如 `pdfplumber`、`pypdf_or_pypdf2`、`ocr_tesseract`。

扫描 PDF 和图片表格需要 OCR 引擎。推荐安装：

1. Python 可选依赖：

```bat
python -m pip install -r requirements.txt
```

2. Tesseract OCR，并确保 `tesseract.exe` 在 PATH 中。

3. `pdftoppm`。如果安装了 TeX Live、Poppler 或 Xpdf，通常会提供该命令。

安装 OCR 后重新构建：

```bat
.\run-agent.bat rebuild
.\run-agent.bat verify
```

如果希望“发现可能漏抽就不要入库”，使用严格重建：

```bat
.\run-agent.bat rebuild-strict
```

当 `verify` 中 `extraction_report.requires_review` 为 `0` 时，说明当前导入没有发现明显的未抽取风险。若大于 `0`，不要把该索引当作完整知识库使用；应先处理报告中的文件。

## 交付给别人使用

对方只需要：

1. 安装 Python 3.10 或更高版本。
2. 解压/拉取本项目。
3. 双击 `start.bat`。
4. 选择 `1` 完成初始化。
5. 选择 `2` 打开资料目录，把资料放进 `knowledge_base/raw`。
6. 选择 `3` 严格构建索引。
7. 选择 `5` 健康检查，看到 `"ok": true` 后即可使用。
8. 选择 `6` 输入问题。

如果资料更新了，只需要再次运行：

```bat
.\run-agent.bat update
```

或者直接提问，系统会检测 raw 文件变化并异步触发更新。

## 限制

- 当前没有 Web UI、用户权限、多租户隔离和监控面板。
- 当前没有独立向量数据库，超大规模知识库需要升级检索后端。
- API 总结只基于本地召回引用，不会替代资料质量和索引质量。
