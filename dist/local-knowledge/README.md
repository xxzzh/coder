# Local Knowledge Base Agent

一个可本地运行的知识库问答 Agent。用户把原始资料放入 `knowledge_base/raw`，运行索引更新命令后，就可以通过命令行直接提问。

当前定位：可交付给普通用户试用的本地 RAG 工具。它已经具备一键菜单、本地 Web UI、状态面板、安装配置、健康检查、多格式资料导入、OCR 抽取、语义分块、混合检索、缓存、增量索引和降级兜底能力；同时开始接入“大模型能力层”，用于 embedding、FAISS 向量召回、查询分类、rerank 和答案质量检查。它仍不是带多租户权限隔离和分布式向量数据库的生产级平台。

## 支持格式

- 文本: `.md`、`.txt`
- Word: `.docx`、`.doc`
- PDF: `.pdf`
- Excel: `.xlsx`

旧版 Word `.doc` 会通过本机 Microsoft Word 静默转换后提取，不会修改原文件。处理 `.doc` 的电脑需要安装 Microsoft Word；未安装时索引更新会明确提示提取失败。

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

菜单会提供初始化、Web UI、打开资料目录、严格建库、健康检查、提问、查看来源和 OCR 测试。

最简单的使用方式：

1. 双击 `start.bat`。
2. 选择 `1` 完成首次配置。
3. 选择 `3` 打开资料目录，把原始资料放入 `knowledge_base/raw`。
4. 选择 `4` 严格重建索引。
5. 选择 `2` 启动 Web UI，在浏览器里查看状态并提问。

也可以直接启动浏览器界面：

```bat
.\run-agent.bat web --open
```

默认地址是：

```text
http://127.0.0.1:8765/
```

Web UI 当前提供：流式提问进度、答案逐段输出、引用来源展示、健康状态、OCR 状态、embedding/FAISS/rerank 状态、抽取待复核数量、增量更新、严格重建、向量索引重建、embedding API 验证、向量索引验证、检索诊断、OCR 评测、来源列表和 API 精炼配置。

如果资料或联网搜索结果包含英文，安装一次离线英文到中文模型：

```bat
.\run-agent.bat install-deps
.\run-agent.bat translation-setup
```

安装后，无论资料和搜索结果使用什么语言，最终回答都会以中文输出；引用来源仍保留原始文字，便于核对。当前离线模型支持英文到中文转换，其他外语在无法可靠翻译时会返回中文提示并保留原始引用。

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
| `install-deps` | 在项目内创建 `.venv`，并把依赖下载缓存放入 `tools/cache` |
| `setup` | 交互式初始化，创建 `.env`，可选择配置 API |
| `web` | 启动本地浏览器界面，默认只监听 `127.0.0.1` |
| `translation-setup` | 下载并安装英文到中文离线翻译模型 |
| `api-setup` | 识别 API 提供商、读取可用模型、选择模型并验证调用 |
| `embedding-setup` | 配置高质量 embedding API，默认 OpenAI `text-embedding-3-large` |
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
LKA_API_MODEL=gpt-5.2
LKA_API_TIMEOUT_SECONDS=20
```

不使用 API：

```env
LKA_USE_API=false
```

API 默认用于“基于已召回引用的答案总结”。如果启用大模型能力层，系统还可以调用 OpenAI-compatible embedding API 生成 dense embedding，并把结果写入本地 FAISS 索引；查询时优先合并 FAISS、SQLite FTS5、领域 retriever 和本地 hash embedding 候选，再进行 rerank。联网兜底不会让 API 脱离来源直接回答：系统会先搜索网页、抓取公开网页正文、对正文分块并做轻量 RAG 排序，再把整理后的引用交给 API 去重、精炼和生成中文答案。没有 API 时，系统会优先使用本地离线翻译生成中文答案。

### 大模型能力层

可选配置：

```env
LKA_EMBEDDING_ENABLED=true
LKA_EMBEDDING_PROVIDER=openai
LKA_EMBEDDING_BASE_URL=https://api.openai.com/v1
LKA_EMBEDDING_API_KEY=your_key
LKA_EMBEDDING_MODEL=text-embedding-3-large
LKA_EMBEDDING_DIMENSIONS=3072
LKA_VECTOR_BACKEND=faiss
LKA_RERANK_ENABLED=true
LKA_LLM_CHUNKING_ENABLED=true
LKA_LLM_QUERY_ROUTING_ENABLED=true
```

行为规则：

- Embedding 接口使用 OpenAI-compatible `/embeddings`，支持批量请求、重试、维度读取和失败回退。
- 默认推荐 OpenAI `text-embedding-3-large`，请求 3072 维，优先质量和后续兼容性。
- 如果 API 实际返回其他维度，会以实际维度写入 `knowledge_base/vector/vector_metadata.json`。
- SQLite 继续保存文档、chunk、来源和元数据；FAISS 只保存向量索引文件。
- 向量文件位于 `knowledge_base/vector/`，包含 `chunks.faiss`、`chunks_map.json` 和 `vector_metadata.json`。
- 当前 FAISS 默认使用归一化向量 + `IndexFlatIP`，即精确 cosine 检索，不用近似算法牺牲召回质量。
- 如果 embedding API、FAISS 或配置不可用，系统自动回退到 SQLite FTS5 + 本地 hash embedding，并在 healthcheck/Web UI 中显示降级原因。

联网兜底会区分两种失败情况：

- 搜索源暂时无法访问：检查网络连接或稍后重试。
- 搜索已完成但没有可信相关结果：补充关键词或换一种更具体的问法。

联网检索会并行尝试百度、Bing 中国、Bing RSS、DuckDuckGo、Wikipedia 和 Jina Bing，并对结果统一做相关性过滤、质量排序和去重。随后，联网研究 Agent 会并发抓取前几个公开网页，只允许公网 `http/https` 地址，抽取正文后复用本地 hash n-gram embedding 和关键词评分筛选相关片段。单个搜索源或网页不可用时不会阻塞其他来源返回答案。

支持的 API 提供商：

- 小米 MiMo：默认地址 `https://api.xiaomimimo.com/v1`
- DeepSeek：默认地址 `https://api.deepseek.com`
- OpenAI：默认地址 `https://api.openai.com/v1`
- 自定义 OpenAI 兼容接口

配置方式：

```bat
.\run-agent.bat api-setup
```

也可以在 Web UI 的 `API 精炼` 面板中选择提供商、粘贴 API key、读取可用模型、选择模型并保存。系统会在保存前执行一次真实模型调用验证。

MiMo Token Plan 的 `tp-` key 仅限编程工具使用，不能配置为本应用后端。需要调用大模型总结时，请使用允许应用后端调用的 API key；没有合规 API 时，系统会使用本地离线翻译。

临时禁用 API：

```bat
python scripts\query_knowledge_base.py "What is ACID?" --no-api
```

## 当前检索技术

- SQLite FTS5 全文检索
- 本地 hash n-gram embedding
- OpenAI-compatible dense embedding，可选
- FAISS 本地向量索引，可选
- 查询分类：缩写、Excel 表格、站位测试、代码/文档说明、普通 RAG
- 语义分块
- FAISS + FTS + semantic cosine + keyword score 混合 rerank
- 可选大模型 rerank 和答案质量检查
- 联网网页正文抽取、分块和轻量 RAG rerank
- 热门 Query 缓存
- raw 文件变化检测和异步增量更新
- 索引不可用时 FAQ 兜底

FAISS 是本地文件型向量索引，不需要 Docker 或外部服务。当前默认使用精确 `IndexFlatIP`，优先召回质量；当 chunk 规模明显增大时，再有意识地切换到 HNSW/IVF 这类近似索引。对于没有 API key 或没有安装 FAISS 的环境，当前 SQLite + 本地 hash embedding 仍是稳定兜底路径。

## 项目本地依赖和下载目录

推荐用项目命令安装依赖：

```bat
.\run-agent.bat install-deps
```

该命令会创建项目内虚拟环境：

```text
.venv/
```

并把下载缓存和模型缓存默认放到：

```text
tools/cache/
  pip/
  npm/
  playwright-browsers/
  huggingface/
  torch/
  argos/
```

`run-agent.bat` 和 `start.bat` 会优先使用 `.venv\Scripts\python.exe`，并设置 `PIP_CACHE_DIR`、`HF_HOME`、`PLAYWRIGHT_BROWSERS_PATH`、`NPM_CONFIG_CACHE`、`ARGOS_PACKAGE_DIR` 等环境变量，避免依赖和模型缓存散落到用户目录。

## 依赖

核心功能只依赖 Python 标准库。

可选增强依赖列在 `requirements.txt`：

```bat
.\run-agent.bat install-deps
```

这些依赖用于改善 PDF 抽取、中文繁简转换、OCR、FAISS 向量检索和联网请求；没有安装时系统会走内置轻量逻辑或降级检索。

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
.\run-agent.bat install-deps
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

## 项目内本地审计

本项目不依赖外部审计目录即可完成日常审计。解压后在项目根目录执行：

```bat
.\audit-local.bat
```

该命令会在当前项目目录内检查必需文件、raw 资料类型、健康状态、检索护栏和索引来源。若刚解压还没有放资料或尚未构建索引，审计会提示索引未就绪；先按下面的交付流程放入资料并构建索引即可。

## 交付给别人使用

对方只需要：

1. 安装 Python 3.10 或更高版本。
2. 解压/拉取本项目。
3. 双击 `start.bat`。
4. 选择 `1` 完成初始化。
5. 选择 `3` 打开资料目录，把资料放进 `knowledge_base/raw`。
6. 选择 `4` 严格构建索引。
7. 选择 `6` 健康检查，看到 `"ok": true` 后即可使用。
8. 如需完整本地审计，运行 `.\audit-local.bat`。
9. 选择 `2` 启动 Web UI，通过浏览器提问；或选择 `7` 在命令行输入问题。

如果资料更新了，只需要再次运行：

```bat
.\run-agent.bat update
```

或者直接提问，系统会检测 raw 文件变化并异步触发更新。

## 限制

- 当前 Web UI 是本机单用户界面，不包含账号、用户权限和多租户隔离。
- 当前使用本地 FAISS 文件索引，不是多节点向量数据库；超大规模知识库仍需要升级检索后端。
- API 生成和大模型增强只基于可追溯证据，不会替代资料质量和索引质量。
