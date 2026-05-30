# 客服工单 Agent 示例

这个示例展示一个小而完整的 agent 应该包含哪些部分：

1. 明确任务：对客户工单做分诊，并选择下一步处理动作。
2. 定义输入和输出：输入是 JSON 工单，输出是结构化 JSON 决策。
3. 定义可调用工具：知识库查询、退款工单草稿、升级处理。
4. 实现运行循环：观察输入、制定计划、调用工具、生成回复。
5. 设置边界条件：稳定分类，高风险工单自动升级。
6. 用演示工单和样例工单验证结果。

运行命令：

```powershell
python .\examples\customer-support-agent\agent.py
python .\examples\customer-support-agent\agent.py --ticket .\examples\customer-support-agent\sample_ticket.json
```

查看当前可见的 skills：

```powershell
python .\examples\customer-support-agent\agent.py --list-skills
```

直接在命令里输入工单内容：

```powershell
python .\examples\customer-support-agent\agent.py --subject "紧急退款" --body "我今天被扣费了，需要退款，否则会申请拒付。"
```

默认只输出工具结果和下一步动作。如果需要查看分类、计划和完整工具调用细节，添加 `--verbose`：

```powershell
python .\examples\customer-support-agent\agent.py --subject "紧急退款" --body "我今天被扣费了，需要退款，否则会申请拒付。" --verbose
```

也可以把输入内容写到 `sample_ticket.json`，然后继续使用 `--ticket` 运行。

更方便的运行方式是在项目根目录使用：

```powershell
.\run-agent.bat
```

它会提示你输入工单标题和正文。也可以这样传参：

```powershell
.\run-agent.bat "紧急退款" "我今天被扣费了，需要退款，否则会申请拒付。"
.\run-agent.bat ticket .\examples\customer-support-agent\sample_ticket.json
.\run-agent.bat skills
```

本地知识库导入和查询：

```powershell
.\run-agent.bat ingest
.\run-agent.bat ask "光合作用受哪些因素影响？"
.\run-agent.bat ask "加强针有什么作用？"
.\run-agent.bat ask "P waves 和 S waves 有什么区别？"
```

如果知识库没有找到依据，会返回 `need_web_search: true`，提示是否允许联网搜索。

如果你想使用 PowerShell 参数风格，也可以用：

```powershell
PowerShell -ExecutionPolicy Bypass -File .\run-agent.ps1 -Subject "紧急退款" -Body "我今天被扣费了，需要退款，否则会申请拒付。"
```

这个 agent 示例适合用来理解“能自己推进任务的程序”通常怎么组织：它不是只保存提示词，而是包含输入、状态判断、工具调用、决策规则和最终输出。
