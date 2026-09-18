# Gemma Agent Tool-Use Lab

> 面向任务型 Agent 的小模型工具学习、受控执行与可靠性评测框架。

本项目研究的不是“再搭一个聊天机器人”，而是 Agent 系统中更难验证的一层：**怎样把通用小模型训练成稳定的工具决策器，并用确定性运行时约束它何时调用、何时追问、何时停止。**

项目以 Gemma 3-1B 为基座，完成从多轮 Agent Trace 构造、4bit QLoRA 后训练，到 Tool Registry、Policy Engine、受控 Runtime 和分场景 Eval Harness 的完整闭环。CrossWOZ 只是可复现的数据来源；核心交付物是可迁移的 Tool-Use 后训练与验证方法。

在 SFT 基线之上，项目使用 DPO 进行第二阶段对齐：只在训练集 Prompt 上收集 SFT 模型真实生成的错误输出，将标准 JSON 作为 `chosen`、错误 JSON 作为 `rejected` 构造偏好对；验证集只用于 checkpoint 选择，test 与 challenge 不参与偏好对构造或训练。DPO Adapter 继续使用既有评测器，输出 JSON 的 `metrics` 与逐样本 `records` 格式与 SFT 基线一致，可直接横向比较。

## 已完成的闭环

```text
CrossWOZ 多轮对话
       │  状态归一化 / 标签溯源 / 缺参消融
       ▼
Canonical Agent Trace ───────► 4bit QLoRA SFT
       │                           │
       │ history / state / tools   ▼
       └────────────────────► Gemma 3-1B
                                   │
                    belief_state + decision + tool_call
                                   │
                     Contract / Registry / Policy
                        │          │           │
                    自动执行     继续追问     人工确认
                                   │
                   Eval Harness + Error Analysis
```

这条链路对应四类真实工作：

- **数据工程**：把原始多轮对话转换为带上下文、状态、工具 Schema、目标决策和来源标记的 Agent Trace；
- **模型后训练**：用 4bit NF4 QLoRA 训练结构化状态维护和 Tool-Use 决策，训练损失只覆盖目标 JSON；
- **偏好对齐**：从 SFT 模型在训练集上的真实错误构造 `chosen / rejected` 偏好对，使用 DPO 强化工具选择、参数生成与拒调边界；
- **运行时治理**：模型只负责提议，代码负责 Schema、白名单、必填参数、风险分级和执行授权；
- **可靠性评测**：比较 Base 与 QLoRA，并按调用、追问、拒调、多轮状态、长状态和风险动作切片分析。

## 核心结果

SFT 基线使用 1,968 条训练样本完成首轮 QLoRA；DPO 训练仅使用训练集上的模型真实错误构造偏好对，验证集仅用于 checkpoint 选择，test 与 challenge 不参与偏好对构造或训练。两阶段模型在固定 440 条 test 与 80 条 challenge 上使用相同 Prompt、4bit 精度和贪心解码评测。

| 指标 | SFT 基线（test） | DPO（test） | DPO（challenge） |
| --- | ---: | ---: | ---: |
| 严格 JSON 合法率 | 96.36% | **98.18%** | 95.00% |
| Schema 合法率 | **98.41%** | 97.95% | 93.75% |
| 状态 Slot F1 | 89.17% | **92.96%** | 91.16% |
| 决策准确率 | 91.82% | **97.27%** | 92.50% |
| 工具选择准确率 | 86.93% | **95.17%** | 92.11% |
| 参数 Slot F1 | 68.63% | **79.02%** | 79.65% |
| 参数完全匹配率 | 31.82% | **42.61%** | 42.11% |
| 非工具轮误调用率 ↓ | 11.36% | **3.41%** | 2.38% |

DPO 阶段进一步改善了工具选择、参数生成和拒调边界：固定 test 中工具选择为 335/352，非工具场景误调用为 3/88；参数 Slot F1 的 matched/predicted/target 分别为 851/1091/1063。参数完全匹配仍只有 42.61%，因此项目不宣称已经解决复杂参数精确生成。SFT 基线的分析见 [实验报告](reports/EXPERIMENT_REPORT.md) 与 [分场景基准](reports/BENCHMARK_REPORT.md)。

## 1. Canonical Agent Trace

每条训练样本先被解释为统一 Trace，而不是散落的 Prompt 拼接：

```json
{
  "schema_version": "1.0",
  "trace_id": "controlled_test_xxx",
  "context": {
    "history": [],
    "previous_state": [],
    "current_user": "我从故宫出发，帮我叫辆出租车。"
  },
  "tools": [{"type": "function", "function": {"name": "request_taxi"}}],
  "events": [
    {"stage": "model_target", "payload": {"decision": "ask_user"}},
    {"stage": "policy_target", "payload": {"status": "needs_input"}}
  ],
  "provenance": {"type": "controlled_required_slot_ablation"}
}
```

Trace 统一承载上下文、工具声明、目标状态转移、预期策略结果与数据来源。训练脚本通过 `render_sft_pair` 将 Trace 渲染为原有的 Gemma Prompt/JSON target，因此不改变已完成实验的监督目标；导出脚本则可生成便于审计和迁移的 JSONL。

```powershell
.\run_local.ps1 -Mode trace
```

数据来自 [CrossWOZ](https://github.com/thu-coai/CrossWOZ) 官方划分。项目保留最近 6 轮对话、上一轮累计状态、当前输入以及下一系统轮的数据库查询标签；另外使用 split 内真实地点值构造显式标记的单槽位消融样本，用于测试缺参时能否停止。

## 2. QLoRA 后训练

| 配置 | 实际值 |
| --- | --- |
| 基座 | `google/gemma-3-1b-it` |
| 量化 | 4bit NF4 + double quant，FP16 计算 |
| LoRA | r=16，alpha=32，dropout=0.05 |
| 目标层 | q/k/v/o + gate/up/down projection |
| 最大长度 | 1,024 tokens |
| batch | micro batch 1，gradient accumulation 8 |
| 可训练参数 | 13,045,760，约 1.29% |
| 实际训练 | 1 epoch，约 3 小时 17 分钟 |

Base 权重被冻结，Prompt token 的 label 设为 `-100`，只对目标 JSON 计算 Causal LM loss。序列过长时优先截断最早的 Prompt token，尽量保留完整监督答案。

## 3. Guarded Agent Runtime

模型输出不会直接触发工具。运行时依次经过：

1. `contracts.py`：严格 JSON、字段类型、decision 不变量；
2. `tool_registry.py`：工具白名单、参数 Schema、风险级别和执行器元数据；
3. `policy.py`：必填参数检查、只读/副作用授权；
4. `local_tools.py`：可复现的 SQLite 查询与模拟副作用执行；
5. `runtime.py`：串联各阶段并生成可审计事件 Trace。

| 模型提议 | Policy 结果 | 系统行为 |
| --- | --- | --- |
| `query_hotel_db` 且参数合法 | `auto_execute` | 执行只读查询 |
| 路线缺少目的地 | `needs_input` | 停止并追问 |
| `request_taxi` 参数完整 | `pending_confirmation` | 不自动执行，等待确认 |
| `no_tool` | `no_tool` | 不触发任何工具 |

风险来自注册表而不是模型字段，因此模型不能通过生成“操作安全”来提升自己的权限。

## 4. 错误驱动的 DPO 对齐

SFT 负责让模型学习正确的结构化工具调用；DPO 使用同一 Prompt 下的标准 JSON 与模型真实错误输出组成偏好对，进一步强化“正确调用优于典型错误调用”的相对偏好。训练偏好对仅从 `train.jsonl` 生成；`validation.jsonl` 的偏好对只用于 checkpoint 选择，不参与梯度更新；`test.jsonl` 和 `challenge.jsonl` 始终只用于最终评测。

```text
SFT Adapter + train Prompt
             │
             ▼
  正确 JSON / 模型错误 JSON
             │
             ▼
       DPO preference pairs
             │
             ▼
        QLoRA DPO Adapter
             │
             ▼
  evaluate_tool_use.py（相同报告格式）
```

执行以下命令会依次构造训练与验证偏好对、训练 DPO Adapter，并评测 validation、test、challenge：

```powershell
.\run_local.ps1 -Mode pipeline-dpo
```

偏好对、Adapter 和评测报告默认写入：

```text
data/tool_use/dpo_train.jsonl
data/tool_use/dpo_validation.jsonl
artifacts/tool_use_dpo_adapter/
reports/tool_use_dpo_validation_4bit.json
reports/tool_use_dpo_4bit.json
reports/tool_use_dpo_challenge_4bit.json
```

## 5. Eval Harness

评测不是只看一个 accuracy，而是拆成：

- 格式层：严格 JSON、Schema 合法率；
- 状态层：JGA、Slot Precision / Recall / F1；
- 决策层：`call_tool / ask_user / no_tool` 混淆矩阵；
- 工具层：工具准确率、参数 Exact Match 与参数 Slot F1；
- 安全层：缺参追问、非工具轮误调用、副作用工具拦截；
- 泛化层：指代、修正、跨领域、长状态等 challenge 切片。

```powershell
.\run_local.ps1 -Mode analyze
.\run_local.ps1 -Mode benchmark
```

本地 Benchmark 的设计受 BFCL/When2Call 的分层思路启发，但它使用本项目固定数据，不冒充 BFCL 官方分数。

## 快速运行

环境要求：Python 3.10+；训练与 4bit 推理需要 NVIDIA GPU。首次运行需要在 Hugging Face 接受 [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) 许可，并通过 `hf auth login` 或本地 `.env` 提供只读 Token。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

本机已配置 `C:\anaconda\envs\gemma-workorder` 时，使用统一入口：

```powershell
.\run_local.ps1 -Mode check
.\run_local.ps1 -Mode trace
.\run_local.ps1 -Mode smoke
.\run_local.ps1 -Mode train
.\run_local.ps1 -Mode build-dpo-train
.\run_local.ps1 -Mode build-dpo-validation
.\run_local.ps1 -Mode train-dpo
.\run_local.ps1 -Mode pipeline-dpo
.\run_local.ps1 -Mode eval-dpo-validation
.\run_local.ps1 -Mode eval-dpo
.\run_local.ps1 -Mode eval-dpo-challenge
.\run_local.ps1 -Mode eval-base
.\run_local.ps1 -Mode eval-qlora
.\run_local.ps1 -Mode eval-challenge
.\run_local.ps1 -Mode analyze
.\run_local.ps1 -Mode benchmark
.\run_local.ps1 -Mode demo
```

已有 Adapter 和三份原始预测时，无需重训；运行 `analyze → benchmark → demo` 即可复核结果。

## 目录

```text
data/tool_use/                     固定数据划分、challenge 与来源清单
src/gemma_eval/contracts.py        模型输出契约与严格校验
src/gemma_eval/tool_registry.py    工具 Schema、风险、必填参数
src/gemma_eval/policy.py           确定性执行策略
src/gemma_eval/local_tools.py      可复现本地工具
src/gemma_eval/runtime.py          受控编排与运行 Trace
src/gemma_eval/traces.py           Canonical Agent Trace 与 SFT 渲染
src/gemma_eval/benchmark.py        分场景 Benchmark 切片
scripts/build_tool_use_dataset.py  CrossWOZ → 训练样本
scripts/build_dpo_preferences.py   SFT训练错误 → DPO偏好对
scripts/export_agent_traces.py     训练样本 → Agent Trace JSONL
scripts/train_tool_use_qlora.py    4bit QLoRA SFT
scripts/train_tool_use_dpo.py      基于SFT Adapter的4bit QLoRA DPO
scripts/evaluate_tool_use.py       逐样本模型评测
scripts/run_agent_benchmark.py     场景级可靠性报告
scripts/analyze_tool_use_results.py 错误分析与实验报告
tests/test_tool_use.py             契约、策略、工具、Trace、Benchmark 测试
```

`decoding.py` 与 `gemma3_core.py` 是早期手写生成和 Gemma Decoder 对齐实验，保留为模型原理附录，不参与当前主线指标。

## 设计来源

本项目没有照搬单一仓库，主要吸收以下公开路线：

- [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)：工具选择、参数、格式和多轮场景分层；
- [NVIDIA When2Call](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/develop/packages/nvidia-nat-eval/src/nat/eval/evaluator/evaluate_when2call)：把“不调用/先追问”作为独立能力；
- [Open Trajectory Gym](https://github.com/vecna-labs/open-trajectory-gym)：用 Trace 串联采集、转换、训练和评测；
- [qwen35-agent-post-training](https://github.com/CHEN2003-CHIP/qwen35-agent-post-training)：社区项目中对 Tool-Use SFT、策略安全和本地 Benchmark 的工程组织；
- [Llama3-FunctionCalling](https://github.com/michaelnny/Llama3-FunctionCalling)：工具元数据、LoRA 训练与本地推理的完整链路。

详细取舍见 [系统设计](docs/系统设计与求职定位.md)。第一次学习和面试准备请直接看 [面试口语学习手册](docs/项目学习与面试手册.md)；需要按代码链路理解时查 [技术主线参考](docs/项目学习主线_技术参考.md)，需要补全部原理、指标和实验计划时查 [完整参考手册](docs/项目学习与面试手册_完整参考.md)。简历表述见 [简历与面试表达](docs/简历与面试表达.md)。

## 边界

- 本地工具使用 CrossWOZ 裁剪数据，不代表实时业务数据库；
- 打车是副作用策略演示，不接入真实派车服务；
- challenge 是项目内可解释切片，不是 BFCL 官方测试成绩；
- Adapter、基础模型、Token 和逐样本原始预测不提交到 Git；
- 当前结论是“小模型在固定任务协议上的后训练与可靠性实验”，不宣称通用或生产级 Agent。
