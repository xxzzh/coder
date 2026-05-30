# Local Knowledge Base Agent

一个本地知识库 Agent MVP，用于把本地文档导入 SQLite FTS5 索引，并通过命令行查询。当前已完成第一阶段、第二阶段和第三阶段：本地知识库检索、本地答不了时可选择联网搜索，以及本地语义检索与准确性增强。

## 当前能力

- 从 `knowledge_base/raw` 读取本地知识库文件。
- 支持 `.md`、`.docx`、`.pdf`、`.xlsx`。
- 提取文本并写入 SQLite FTS5 索引。
- 通过 `run-agent.bat` 查询知识库。
- 输出结构化 JSON，包含 `answer`、`source_type`、`sources`、`need_web_search`、`web_search_used`。
- 本地知识库答不了时，不加 `--web` 只提示是否需要联网。
- 加 `--web` 时允许联网搜索，并标识 `source_type: "web_search"`。
- 输出内容会清洗为自然语言文本，避免 Markdown 标题、列表符号和换行残留。
- 联网搜索结果中的中文内容会尽量转成简体中文。
- 使用本地 hash n-gram embedding 做语义检索，不依赖外部 API。
- 对候选 chunk 进行 rerank，综合语义分、关键词分和 FTS 分。
- 输出引用片段 `citations`、索引更新时间 `index_updated_at` 和 `embedding_model`。
- 导入时检测重复文件，输出 `duplicates` 和 `duplicate_files`。

## 目录结构

```text
.
├── knowledge_base/
│   └── raw/                 # 原始知识库文件
├── scripts/
│   ├── ingest_knowledge_base.py
│   └── query_knowledge_base.py
├── run-agent.bat            # Windows 命令入口
├── install_kb_phase1.py     # 示例知识库生成脚本
└── 阶段验证命令.md           # 分阶段验证命令
```

运行导入后会生成：

```text
knowledge_base/index/        # SQLite 索引，已被 .gitignore 排除
knowledge_base/processed/    # 提取后的中间文件，已被 .gitignore 排除
```

## 快速开始

在 Windows PowerShell 或 CMD 中运行：

```bat
.\run-agent.bat ingest
```

查询本地知识库：

```bat
.\run-agent.bat ask "加强针有什么作用？"
```

本地答不了时只提示，不联网：

```bat
.\run-agent.bat ask "量子纠缠如何用于加密？"
```

允许联网搜索：

```bat
.\run-agent.bat ask "量子纠缠如何用于加密？" --web
```

## 示例输出

本地知识库命中时：

```json
{
  "answer": "根据本地知识库，影响光合作用速率的因素包括光照强度、二氧化碳浓度、温度和水分条件。",
  "source_type": "knowledge_base",
  "sources": [
    {
      "file": "photosynthesis_carbon_cycle.md",
      "file_path": "markdown/photosynthesis_carbon_cycle.md",
      "file_type": "md",
      "chunk_id": "doc_0002_0001",
      "excerpt": "影响光合作用速率的因素包括光照强度、二氧化碳浓度、温度和水分条件。"
    }
  ],
  "need_web_search": false,
  "web_search_used": false
}
```

本地知识库答不了且未启用联网时：

```json
{
  "answer": "本地知识库没有找到足够依据。",
  "source_type": "none",
  "sources": [],
  "need_web_search": true,
  "web_search_used": false,
  "message": "本地知识库没有足够依据。如需联网搜索，请追加 --web 参数。"
}
```

## 分阶段计划

### 第一阶段：本地知识库 MVP

已完成：

- 新建 `knowledge_base/raw`
- 支持 `.md`、`.docx`、`.pdf`、`.xlsx`
- 提取文本
- 用 SQLite FTS5 建索引
- Agent 查询知识库
- 输出 `answer`、`source_type`、`sources`

### 第二阶段：增加“答不了就询问联网”

已完成：

- 增加 `need_web_search`
- 增加 `--web` 参数
- 不加 `--web` 时只提示，不联网
- 加 `--web` 时联网搜索并标识 `source_type = web_search`

### 第三阶段：增强准确性

已完成：

- 加 embedding 语义检索
- 加 chunk rerank
- 加引用片段
- 加重复文件检测
- 加索引更新时间

### 第四阶段：做成更方便的使用方式

待实现：

- `.\run-agent.bat ingest`
- `.\run-agent.bat ask "问题"`
- `.\run-agent.bat ask "问题" --web`
- `.\run-agent.bat sources`
- `.\run-agent.bat rebuild`

## 验证命令

完整验证流程见：

[阶段验证命令.md](./阶段验证命令.md)

## 依赖说明

当前实现主要使用 Python 标准库：

- `sqlite3`
- `zipfile`
- `xml.etree.ElementTree`
- `html.parser`
- `urllib`

PDF 提取会优先尝试 `pypdf` 或 `PyPDF2`，如果未安装，会回退到内置的简单 PDF 文本提取逻辑。
