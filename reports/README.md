# 模型评测结果

已验证的第二轮训练指标见 [V2 定向训练与评测摘要](V2_RUN_SUMMARY.md)。这些数字来自训练集定向重采样后的 SFT，不是 DPO。仓库目前没有公开可复核的 DPO 训练与评测产物；取得独立 DPO 结果后再更新 README 和简历。

运行 `scripts/evaluate_tool_use.py` 后，Base 与 QLoRA 的完整 JSON 报告会写到这里，但默认不提交，避免仓库被逐样本原始输出撑大。

三份评测完成后运行：

```powershell
.\run_local.ps1 -Mode analyze
.\run_local.ps1 -Mode benchmark
```

脚本会重新关联固定数据集：`analyze` 生成总体对照、决策混淆矩阵和错误分类；`benchmark` 按调用、追问、拒调、多轮、长状态和 challenge 标签生成 [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md)。两份 Markdown 都可提交，逐样本 JSON 仍留在本地。

正式更新 README 或简历前，请至少核对：

- 两次评测使用同一模型版本、测试集、Prompt、精度和贪心解码设置；
- 报告中的 Adapter 路径正确；
- 测试集共 440 条，其中 40 条为 manifest 明示的受控缺参样本；
- 随机抽查原始输出，确认指标提升不是解析器放宽造成的；
- 记录 GPU、依赖版本、训练参数和失败案例。

建议文件名：

```text
tool_use_base_4bit.json
tool_use_qlora_4bit.json
tool_use_qlora_challenge_4bit.json
```
