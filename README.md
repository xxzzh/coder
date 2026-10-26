# Local Knowledge Base Agent

一个可本地运行的知识库问答 Agent。用户把原始资料放入 `knowledge_base/raw`，运行索引更新命令后，就可以通过命令行直接提问。

当前定位：可交付给普通用户试用的本地 RAG 工具。它已经具备一键菜单、本地 Web UI、状态面板、安装配置、健康检查、多格式资料导入、OCR 抽取、语义分块、混合检索、缓存、增量索引和降级兜底能力；本地 `BAAI/bge-m3` 向量模型 + FAISS 负责语义召回。配置 DeepSeek 等大模型 API 后，由智能检索 Agent 判断证据够不够回答、必要时换成文档的说法再检索，并生成带引用的答案；不配置 API 也能纯本地运行。它仍不是带多租户权限隔离和分布式向量数据库的生产级平台。

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

### 环境要求

- Windows 10/11（命令脚本是 `.bat`）。
- Python 3.10 或更高版本，安装时勾选 “Add python.exe to PATH”。
- 磁盘预留约 4 GB：依赖约 1 GB，本地向量模型 `BAAI/bge-m3` 约 2.2 GB，都放在项目目录内。
- 安装依赖和下载模型时需要联网；之后本地检索可以离线运行（大模型 API 和联网兜底除外）。
- 资料里有扫描件或图片时，另装 Tesseract OCR，见「扫描件、图片表格和复杂版式」。

### 第一次安装

获取代码（当前功能在 `codex/agent-delivery-hardening` 分支）：

```bat
git clone -b codex/agent-delivery-hardening https://github.com/xxzzh/coder.git
cd coder
```

不用 git 的话，可以在 GitHub 仓库的 Releases 页面下载打包好的 `local-knowledge.zip`，解压后进入 `local-knowledge` 目录，从下面第 1 步开始。

1. 安装依赖：在项目内创建 `.venv`，下载缓存放在 `tools\cache`。

   ```bat
   .\run-agent.bat install-deps
   ```

2. 生成配置文件 `.env`。会询问是否配置大模型 API，可以先选否，第 5 步再配。

   ```bat
   .\run-agent.bat setup
   ```

3. 启用本地向量检索。首次运行会把 `BAAI/bge-m3`（约 2.2 GB）下载到 `tools\cache\huggingface`，提示输入时直接回车用默认值。

   ```bat
   .\run-agent.bat embedding-setup
   ```

4. 把资料放进 `knowledge_base\raw`（支持格式和子目录见上文），然后严格建库并检查。`verify` 输出 `"ok": true` 即可使用；如果 `requires_review` 大于 0，见下文「扫描件、图片表格和复杂版式」。

   ```bat
   .\run-agent.bat rebuild-strict
   .\run-agent.bat verify
   ```

5. （推荐）配置大模型 API，开启智能检索 Agent，详见「API 配置」。

   ```bat
   .\run-agent.bat api-setup
   ```

6. 启动 Web UI，在浏览器里提问。默认地址 `http://127.0.0.1:8765/`，只监听本机。

   ```bat
   .\run-agent.bat web --open
   ```

不习惯命令行可以双击 `start.bat`，菜单编号对应：`13` 安装依赖、`1` 初始化、`14` 配置向量模型、`3` 打开资料目录、`4` 严格建库、`6` 健康检查、`12` 配置 API、`2` 启动 Web UI、`7` 命令行提问。

### 日常使用

Web UI 当前提供：流式提问进度、答案逐段输出、引用来源展示、健康状态、OCR 状态、embedding/FAISS/rerank 状态、抽取待复核数量、增量更新、严格重建、向量索引重建、embedding API 验证、向量索引验证、检索诊断、OCR 评测、来源列表和 API 精炼配置。

如果资料或联网搜索结果包含英文、又没有配置大模型 API，安装一次离线英文到中文模型（配置了 API 时由模型直接用中文作答，不需要这一步）：

```bat
.\run-agent.bat install-deps
.\run-agent.bat translation-setup
```

`translation-setup` 会从 Argos 官方包索引下载英译中模型（约 71 MB，放在 `tools\cache\argos`），然后在联网状态下试译一句，让句子切分组件 Stanza 补齐所需文件（约 1.6 MB）。之后翻译完全离线运行，不再访问网络；Web UI 状态栏会显示“英译中：可用（离线）”。

安装后，无论资料和搜索结果使用什么语言，最终回答都会以中文输出；引用来源仍保留原始文字，便于核对。当前离线模型支持英文到中文转换，其他外语在无法可靠翻译时会返回中文提示并保留原始引用。代码片段不做机器翻译（翻译后只会变成乱码），只翻译英文正文；离线模型是通用机器翻译，专业术语可能译得不准，需要更好的译文时请配置大模型 API。

命令行提问（本地资料不足时默认允许联网兜底）：

```bat
.\run-agent.bat "What is Ohm's law?"
```

只查本地知识库、不联网：

```bat
.\run-agent.bat ask "加强针有什么作用？" --no-web
```

资料增删改后增量更新（直接提问时也会检测 `knowledge_base\raw` 的变化并在后台更新）：

```bat
.\run-agent.bat update
```

彻底重建索引、查看索引来源、检查环境：

```bat
.\run-agent.bat rebuild
.\run-agent.bat sources
.\run-agent.bat verify
```

没有配置 API 时是纯本地回答，不影响索引、检索和引用输出；配置后答案由智能检索 Agent 生成。

## 命令说明

| 命令 | 作用 |
|---|---|
| `install-deps` | 在项目内创建 `.venv`，并把依赖下载缓存放入 `tools/cache` |
| `setup` | 交互式初始化，创建 `.env`，可选择配置 API |
| `web` | 启动本地浏览器界面，默认只监听 `127.0.0.1` |
| `translation-setup` | 下载并安装英文到中文离线翻译模型 |
| `api-setup` | 识别 API 提供商、读取可用模型、选择模型并验证调用 |
| `embedding-setup` | 配置高质量 embedding，默认本地 `BAAI/bge-m3` |
| `verify` / `healthcheck` | 检查 Python、SQLite FTS5、raw 文件、索引、API 配置 |
| `update` | 增量更新知识库，只处理新增、修改、删除的资料 |
| `rebuild` | 清理生成索引并从 raw 目录重新导入 |
| `rebuild-strict` | 同 `rebuild`，但发现可能漏抽的文件就不入库 |
| `sources` | 列出当前索引中的资料来源 |
| `eval` | 运行检索与问答评测，见「检索与问答评测」 |
| 直接输入问题 | 查询本地知识库；本地不足时默认允许联网兜底 |
| `ask "问题" --no-web` | 只查本地知识库 |

## API 配置

推荐用命令配置：输入 Key 时不回显，会读取可用模型，并在真实调用验证通过后才保存。也可以在 Web UI 的 `API 精炼` 面板中配置，或直接编辑 `.env`。

```bat
.\run-agent.bat api-setup
```

DeepSeek 的 `.env` 示例（Key 在 https://platform.deepseek.com/api_keys 申请）：

```env
LKA_USE_API=true
LKA_API_PROVIDER=deepseek
LKA_API_BASE_URL=https://api.deepseek.com
LKA_API_KEY=你的 Key
LKA_API_MODEL=deepseek-flash
LKA_API_TIMEOUT_SECONDS=60
```

`deepseek-flash` 便宜、快；需要更强推理时可以改成 `deepseek-v4-pro`。用 OpenAI 或其他 OpenAI 兼容接口时，把 `LKA_API_PROVIDER`、`LKA_API_BASE_URL`、`LKA_API_MODEL` 换成对应值即可（例如 `openai-compatible`、`https://api.openai.com/v1`）。

`.env` 已在 `.gitignore` 中，不会被提交。Key 只写在 `.env` 里，不要写进其他文件或截图分享。

不使用 API：

```env
LKA_USE_API=false
```

> 数据外发说明：启用 API 后，每次提问会把问题和检索到的资料片段（最多约 10 段完整章节或表格）发给 API 提供商。资料涉密时请保持 `LKA_USE_API=false`，或改用自己部署的 OpenAI 兼容服务。

参考成本与速度：2026 年 10 月在本项目 79 题评测上实测，`deepseek-flash`、思考模式关闭，中位延迟约 1.7 秒，每千次提问约 0.3–1.2 美元（命中提供商缓存时低）。价格以提供商官网为准。

启用 API 后，本地检索到的证据交给智能检索 Agent（见下文）判断并生成带引用的答案。如果启用大模型能力层，系统可以使用本地 `BAAI/bge-m3` 或 OpenAI-compatible embedding API 生成 dense embedding，并把结果写入本地 FAISS 索引；查询时优先合并 FAISS、SQLite FTS5、领域 retriever 和本地 hash embedding 候选，再进行 rerank。联网兜底不会让 API 脱离来源直接回答：系统会先搜索网页、抓取公开网页正文、对正文分块并做轻量 RAG 排序，再把整理后的引用交给 API 去重、精炼和生成中文答案。没有 API 时，系统会优先使用本地离线翻译生成中文答案。

### 大模型能力层

可选配置（`embedding-setup` 会写好前几项）：

```env
LKA_EMBEDDING_ENABLED=true
LKA_EMBEDDING_PROVIDER=local-bge-m3
LKA_EMBEDDING_BASE_URL=
LKA_EMBEDDING_API_KEY=
LKA_EMBEDDING_MODEL=BAAI/bge-m3
LKA_EMBEDDING_DIMENSIONS=1024
LKA_VECTOR_BACKEND=faiss
LKA_RERANK_ENABLED=false
LKA_LLM_CHUNKING_ENABLED=false
LKA_LLM_QUERY_ROUTING_ENABLED=false
```

`LKA_RERANK_ENABLED`、`LKA_LLM_CHUNKING_ENABLED`、`LKA_LLM_QUERY_ROUTING_ENABLED` 这三项大模型功能还没有经过评测验证，建议保持 `false`；智能检索 Agent 已经由模型挑选证据，不需要再开模型 rerank。

行为规则：

- Embedding 默认使用本地 `BAAI/bge-m3`，1024 维，不需要额外 embedding API key。
- 也支持 OpenAI-compatible `/embeddings`，支持批量请求、重试、维度读取和失败回退。
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

MiMo Token Plan 的 `tp-` key 仅限编程工具使用，不能配置为本应用后端。需要调用大模型总结时，请使用允许应用后端调用的 API key；没有合规 API 时，系统会使用本地离线翻译。

临时禁用 API：

```bat
python scripts\query_knowledge_base.py "What is ACID?" --no-api
```

### 智能检索 Agent

配置了可用的答案 API 后，由模型而不是关键词门槛决定能不能答（`scripts/agentic_retrieval.py`）：

1. 按问题做混合检索取前 10 条，每条展开成完整章节或表格（每条最多 1800 字）交给模型。
2. 模型读完证据后三选一：直接作答；给出 1–3 个按文档写法改写的检索语句再查一轮；或拒答并说明缺什么。最多两轮，最后一轮只能作答或拒答。
3. 答案每条要点带引用编号。模型用 `[#n]` 标注证据，系统再换成来源编号 `[n]`，门限里的 `[10,20]` 这类原文不会被误当成引用。
4. 问题里出现知识库从未提到的英文名称或“名称+版本”（如某产品的 3.0 版）时，不调用模型直接拒答。
5. 模型回复不可用（不是 JSON、引用了不存在的证据、超时、网络错误）时，自动退回本地答案，结果里记录 `agent_error`。

`.env` 配置项：

- `LKA_AGENT_ENABLED`：默认 `true`，设为 `false` 时只用模型润色本地答案。
- `LKA_AGENT_MAX_ROUNDS`：检索轮数上限，默认 `2`，范围 1–3。
- `LKA_API_THINKING`：DeepSeek 默认开启思考模式，这里默认 `disabled` 以控制延迟；设为 `enabled` 可换取更强推理。

## 当前检索技术

- SQLite FTS5 全文检索
- 本地 hash n-gram embedding
- 本地 BGE-M3 dense embedding，可选
- OpenAI-compatible dense embedding，可选
- FAISS 本地向量索引，可选
- 查询分类：缩写、Excel 表格、站位测试、代码/文档说明、普通 RAG
- 语义分块
- FAISS + FTS + semantic cosine + keyword score 混合 rerank
- 智能检索 Agent：大模型判断证据是否足够、按文档写法再检索、生成带引用的答案（配置 API 后启用）
- 可选大模型 rerank 和答案质量检查
- 联网网页正文抽取、分块和轻量 RAG rerank
- 热门 Query 缓存
- raw 文件变化检测和异步增量更新
- 索引不可用时 FAQ 兜底

FAISS 是本地文件型向量索引，不需要 Docker 或外部服务。当前默认使用精确 `IndexFlatIP`，优先召回质量；当 chunk 规模明显增大时，再有意识地切换到 HNSW/IVF 这类近似索引。对于没有 API key 或没有安装 FAISS 的环境，当前 SQLite + 本地 hash embedding 仍是稳定兜底路径。

## 代码结构

问答主流程在 `scripts/query_knowledge_base.py`：答案缓存、索引状态、大模型作答与智能检索 Agent 的衔接、找不到时的引导、`query()` 和命令行入口。各环节的实现按功能放在 `scripts/kb/`：

| 模块 | 内容 |
|---|---|
| `kb/text.py` | 文本清洗、繁简转换、句子切分、Markdown 小工具 |
| `kb/terms.py` | 问题归一化、停用词、检索关键词、问题类型判断 |
| `kb/config.py` | 路径、`.env` 读取、大模型 API 与 Agent 开关 |
| `kb/retrieval.py` | 向量/全文/关键词候选、章节扩展、混合重排、`search()` |
| `kb/relevance.py` | 相关性门槛：证据是否覆盖问题、知识库中没有的名称直接拒答 |
| `kb/citations.py` | 引用选取（整节、表格聚焦行、句子）与引用校验 |
| `kb/answer_text.py` | 不调用模型时的答案：润色、离线英译中、本地答案拼装 |
| `kb/structured.py` | 直接读表格的结构化答案：缩写词典、站位测试项、Excel 字段、FFT 清单 |
| `kb/web.py` | 联网搜索、公开网页抓取与正文提取、联网来源筛选去重 |

`scripts/agentic_retrieval.py` 是智能检索 Agent 的循环，`scripts/ingest_knowledge_base.py` 负责导入和建索引。`kb/` 中的名称都会从 `query_knowledge_base` 重新导出，原有调用方式不变。写测试替换某个函数时，要替换调用方所在的模块：例如联网搜索的各个来源在 `kb.web` 里被调用，就替换 `kb.web` 中的函数。

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

扫描 PDF、Word/Excel 里的图片和图片表格靠本机 OCR 识别，全程离线运行，开不开联网都能用。需要：

1. Python 依赖（含把 PDF 页面转成图片的 `pypdfium2`，随 `pdfplumber` 一起安装）：

```bat
.\run-agent.bat install-deps
```

2. Tesseract OCR 引擎。Windows 可从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装包，装在默认位置 `C:\Program Files\Tesseract-OCR` 即可自动找到，装在别处时把 `tesseract.exe` 所在目录加入 PATH。中文和英文识别数据已随仓库放在 `tools/tessdata`，不需要另外下载。

3. 可选：Poppler 的 `pdftoppm`。在 PATH 中时优先用它渲染 PDF 页面，没有时自动改用 `pypdfium2`，识别效果不受影响。

`verify` 输出里的 `optional_extractors.ocr_ready` 为 `true`（Web UI 状态栏显示“OCR：可用（离线/online）”）即表示 OCR 可用，`pdf_renderer` 显示实际使用的 PDF 渲染方式。用随仓库附带的测试资料检查识别效果（结果写入 `test_materials/ocr/ocr_eval_results.json`）：

```bat
.venv\Scripts\python.exe scripts\evaluate_ocr_test_materials.py
```

每页会依次尝试几种版面分析模式，保留 Tesseract 自评置信度最高的结果。

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

## 检索与问答评测

改动检索、门槛或答案生成逻辑后，用评测集量化效果，并把结果追加到 `eval/experiment_log.md`：

```bat
.\run-agent.bat eval --log "这次改了什么"
```

评测不联网、不调用 API、关闭查询缓存，判分只用规则不用模型。指标定义、评测集格式和出题要求见 `eval/README.md`。

评测集 `eval/eval_set.jsonl` 依赖具体资料内容，不随仓库提供（已在 `.gitignore` 中）。换成自己的资料后，按 `eval/README.md` 的格式为它出题，并先运行 `.venv\Scripts\python.exe eval\validate_set.py` 检查题目和资料是否对得上。

评测智能检索 Agent 时加 `--allow-api`，会按 `.env` 调用模型（产生少量 API 费用），检索指标不变，只有作答、通过和拒答会变化：

```bat
.\run-agent.bat eval --allow-api --log "这次改了什么"
```

Web UI 端到端验证：先启动 Web UI，再运行下面的命令。17 道提问、3 项页面检查和 3 项“同一问题不同问法引用相同证据”的一致性检查，默认通过 Web UI 自己的查询接口执行，不需要安装 Playwright：

```bat
.venv\Scripts\python.exe scripts\run_webui_question_suite.py
.venv\Scripts\python.exe scripts\run_webui_question_suite.py --allow-api
```

`--allow-api` 按配置了大模型 API 时 Web UI 的实际方式提问。想用真实浏览器检查时加 `--transport playwright`（需要 Node.js 和 Playwright 浏览器）。

## 项目内本地审计

本项目不依赖外部审计目录即可完成日常审计。解压后在项目根目录执行：

```bat
.\audit-local.bat
```

该命令会在当前项目目录内检查必需文件、raw 资料类型、健康状态、检索护栏和索引来源。若刚解压还没有放资料或尚未构建索引，审计会提示索引未就绪；先按下面的交付流程放入资料并构建索引即可。

## 交付给别人使用

对方按上文「快速开始」操作即可。仓库里不包含以下内容，需要各自准备：

| 内容 | 位置 | 怎么得到 |
|---|---|---|
| 依赖环境 | `.venv/` | `.\run-agent.bat install-deps` |
| 配置和 API Key | `.env` | `.\run-agent.bat setup`，再 `api-setup` |
| 向量模型 | `tools/cache/huggingface/` | `.\run-agent.bat embedding-setup` 首次自动下载 |
| OCR 引擎（有扫描件或图片时） | 系统安装的 Tesseract | 见「扫描件、图片表格和复杂版式」；识别数据已在 `tools/tessdata` |
| 原始资料 | `knowledge_base/raw/` | 自己放入 |
| 索引和向量 | `knowledge_base/index/`、`knowledge_base/vector/` | `.\run-agent.bat rebuild-strict` |
| 评测集 | `eval/eval_set.jsonl` | 按 `eval/README.md` 自己出题，可选 |

建好后运行 `.\audit-local.bat` 可以做一次完整的本地审计。资料更新后再运行 `.\run-agent.bat update`，或者直接提问，系统会检测 raw 文件变化并在后台更新。

打包发布（维护者）：运行 `.\package-dist.bat` 生成 `dist/local-knowledge/` 和 `dist/local-knowledge.zip`。`dist/` 不提交到仓库，zip 作为附件发布到 GitHub Releases。包里不含 `.env`、索引、模型缓存和本地评测集。

## 常见问题

- **命令行提问每次要等 15–20 秒**：每条命令都要重新加载本地向量模型。Web UI 常驻运行，只有第一次提问慢，之后每题一般 1–4 秒（含 API 调用），日常使用推荐 Web UI。
- **回答“本地知识库中没有出现「某名称」”**：问题里的英文名称或“名称+版本”在资料中一次都没出现，系统直接拒答。确认名称写法，或把相关资料放进 `knowledge_base\raw` 后更新索引。
- **回答“本地知识库没有找到足够依据：……”**：智能检索 Agent 读过检索到的资料，认为其中没有这项信息，冒号后面写了缺什么。
- **API 失败或超时**：系统会自动退回本地答案，返回结果里的 `agent_error` 字段记录原因。请检查 Key、余额和网络；这类退回的答案不会写入缓存，恢复后重新提问即可。
- **`embedding-setup` 下载很慢或失败**：模型来自 Hugging Face。网络受限时可以先设置镜像再运行，cmd 用 `set HF_ENDPOINT=https://hf-mirror.com`，PowerShell 用 `$env:HF_ENDPOINT="https://hf-mirror.com"`；下载完成后不再需要联网。
- **`verify` 显示 `requires_review` 大于 0**：有文件可能没有完整抽取（扫描件、图片表格等），见上文「扫描件、图片表格和复杂版式」。

## 限制

- 当前 Web UI 是本机单用户界面，不包含账号、用户权限和多租户隔离。
- 当前使用本地 FAISS 文件索引，不是多节点向量数据库；超大规模知识库仍需要升级检索后端。
- API 生成和大模型增强只基于可追溯证据，不会替代资料质量和索引质量。
