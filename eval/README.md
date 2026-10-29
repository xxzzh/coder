# 检索与问答评测

把“命中率/准确率不高”变成可复现、可量化的数字。每次改动检索或问答逻辑后跑一遍，结果追加到 `experiment_log.md`，改好还是改坏一目了然。

## 运行

先建好索引（`.\run-agent.bat rebuild`），然后：

```bat
.\run-agent.bat eval --log "这次改了什么"
```

等价于 `python eval/run_eval.py --label <名字> --log "<说明>"`。默认不联网、不调用 API、关闭查询缓存；加 `--allow-api` 可评测带大模型总结的效果。

结果写入 `eval/results/<时间>-<label>/`（不提交到 Git）：

| 文件 | 内容 |
|---|---|
| `validation.json` | 评测集校验结果 |
| `raw_responses.jsonl` | 系统原始输出：检索 top-10 chunk 原文 + 最终答案 |
| `scores.jsonl` | 每道题的得分明细 |
| `summary.md` / `summary.json` | 汇总表、未通过题目、负样本误答 |
| `run_meta.json` | 代码版本、索引状态、运行参数 |

## 按需 API 评测（只测证据变了的题）

一次全量 API 评测会把评测集的每道题都发给 API：每道题一次回答调用，再加上按意思判分（`eval/judge.py`）的调用。API 的措辞每次都略有不同，**证据没变的题，判分升降只是措辞波动，说明不了改动的好坏**。所以小改动只把证据变了的题发给 API：

1. 改动后先跑本地评测（不调用 API）：`.\run-agent.bat eval --log "这次改了什么"`
2. 和改动前的本地评测比较每道题检索到的证据：

   ```bat
   python eval\api_subset.py eval\results\<改动前的本地结果> eval\results\<改动后的本地结果>
   ```

   证据都没变时直接提示“不需要跑 API 评测”。否则把证据变了的题和几道固定的负样本（检查新的误答）写入改动后结果目录下的 `api_subset.jsonl`，并打印下一步的命令。
3. 用 API 只跑这些题，判分后与上一次全量 API 评测逐题比较：

   ```bat
   .\run-agent.bat eval --allow-api --set eval\results\<改动后的本地结果>\api_subset.jsonl --label <名字>-api-subset
   .venv\Scripts\python.exe eval\judge.py eval\results\<子集结果目录>
   .venv\Scripts\python.exe eval\compare_runs.py eval\results\<上一次全量 API 结果> eval\results\<子集结果目录>
   ```

   子集运行不要加 `--log`：子集的平均数不能和全量比较。`compare_runs.py` 只比较两次都有的题，并标出每道判分变化的题证据是否相同，证据相同的是措辞波动。

以下情况仍跑全量 API 评测：

- 改了 API 一侧（提示词、润色规则、智能检索 Agent 的流程）：这类改动不改变证据，挑不出题；
- 发布新版本之前。

比较的是用问题本身检索到的证据，也就是 Agent 第一轮拿到的证据；Agent 自己追加搜索得到的证据没有比较。

## 调混合检索权重

```bat
python eval\run_eval.py ...    REM 先确认当前基线
.venv\Scripts\python.exe eval\tune_weights.py --step 0.05
```

`tune_weights.py` 对每道题只跑一次检索，记录完整候选池及各项分数（语义、关键词、全文），然后离线按不同权重重新排序，结果与 `search()` 用这些权重时完全一致。为了防止在小评测集上过拟合：

- 题目按类别分层，分成两折；
- 只接受在两折上 MRR 都提高、且任一折的 Hit@1 和 Hit@5 都不下降的权重；
- 在符合条件的权重里，选两折提升中较小的那个最大的一组；
- 同时报告"在一折上最优的权重放到另一折上的表现"，用来判断泛化。

脚本只给出建议，不会改代码；采用时修改 `scripts/query_knowledge_base.py` 里的 `RERANK_WEIGHTS`，再跑一遍完整评测确认。

## 三个角色互相独立

| 脚本 | 职责 | 是否用模型 |
|---|---|---|
| `validate_set.py` | 用 openpyxl / 纯文本**独立读取原始文件**，确认每个标准关键词真的在原文里，负样本的关键词真的不在任何资料里；顺带报告“原文有但索引里丢失”的关键词 | 否 |
| `collect.py` | 只负责运行系统、落盘原始输出，不判断对错 | 只调用被测系统 |
| `score.py` | 只用子串匹配和集合运算打分；每次都会把标准关键词替换成乱码再判一遍，若仍有题被判命中则自检失败 | 否 |

## 指标

正样本（A–D 类）：

- **Hit@k**：top-k 中存在来自标准文档、且包含至少一半标准关键词的 chunk
- **MRR@k**：第一个命中 chunk 排名的倒数
- **KwCov@k**：top-k 标准文档 chunk 合起来覆盖了多少标准关键词
- **作答率**：系统给出了答案而不是“没有找到足够依据”
- **答案覆盖 / 通过率**：答案文本里出现的标准关键词比例；覆盖 ≥ 50% 记为通过

负样本（E 类）：**正确拒答率**。

检索层指标测的是通用混合检索 `search()`；作答相关指标测的是完整 `query()`，包含缩写、站位等领域 retriever。

## 评测集 `eval_set.jsonl`

**评测集只保存在本地，不提交到 Git**（已写入 `.gitignore`）。题目和标准关键词逐字摘录自内部资料，而这个仓库是公开的。换电脑时请从本机备份复制 `eval/eval_set.jsonl`；文件不存在时，评测命令会给出提示后退出。

每行一道题，格式如下（示例内容为虚构）：

```json
{"id": "B01", "category": "B", "question": "如何新增一个设备", "gold_doc_ids": ["设备手册.md"], "gold_chunk_keywords": ["新增设备步骤", "注册设备"], "gold_answer_summary": "人工对照原文写的标准答案摘要", "source_ref": "第三章 / 新增设备步骤", "origin": "raw-doc", "review_status": "draft"}
{"id": "E01", "category": "E", "question": "公司年假有几天", "gold_doc_ids": [], "absent_terms": ["年假"], "gold_answer_summary": "知识库无此内容，应明确表示没有依据。", "origin": "raw-doc", "review_status": "draft"}
```

当前本地评测集的构成：

| 类别 | 含义 | 题数 |
|---|---|---|
| A | 精确匹配：问法与原文一致 | 12 |
| B | 同义改写：问法与原文用词不同 | 19 |
| C | 表格/结构化：需要从表格取具体字段 | 15 |
| D | 多片段综合：答案分布在多个片段 | 9 |
| E | 负样本：知识库里没有答案 | 9 |

每道题都有 `source_ref` 指向原文位置。`origin: "wenti.txt"` 是 wenti.txt 里实测过的问题；`origin: "raw-doc"` 是对照原文起草的题。

**所有题目目前都是 `review_status: "draft"`**：标准答案是对照原文写的，没有参考系统输出，但仍需要人工逐条核对。核对无误后改为 `"verified"`。新增题目时：

1. 不运行系统，直接读原文出题，故意混入与原文用词不同的问法；
2. `gold_chunk_keywords` 从原文**逐字**摘录（单个表格单元格内的片段，不要跨单元格）；
3. 负样本填 `absent_terms`，即能证明资料里没有答案的词；
4. 运行 `python eval/validate_set.py`，有 ERROR 先修正再评测。

评测集依赖 `knowledge_base/raw` 下的资料；标准文档不存在的题会被跳过并在结果里列出。
