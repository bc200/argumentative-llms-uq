# original_qbaf 与上游代码及论文的一致性复核

复核日期：2026-09-25

## 结论

`original_qbaf` **在给定同一论证图和同一模型回复时，与上游代码的「Direct Prompting + 根节点 0.5 + DF-QuAD」路径一致**。它对应论文的 **0.5 BS** 配置，不对应论文另列的 **Est. BS** 配置。本次实验使用用户指定的 DeepSeek 模型及共享缓存图，因此不能把数值结果称为对论文表格的原样复现。

## 逐项核对

| 项目 | 论文与上游代码 | 当前实现 | 判断 |
|---|---|---|---|
| 论证结构 | D=1 对根论点生成一个支持、一个攻击；D=2 再向下一层扩展。上游 `ArgumentMiner.generate_arguments` 保留 `N/A` 节点、给其基础分 0，并不扩展第一层的 `N/A`。 | `ArgumentMiner.generate_graph` 和 `extend_graph` 使用相同的生成提示及命名、关系规则；D=2 从 D=1 缓存图扩展。 | 给定相同生成回复时一致。 |
| 非根论点基础分 | 上游 `UncertaintyEstimator` 使用 `UncertaintyEvaluatorPrompts.analyst`，结合论点正文、父论点和既定支持/攻击类型，以百分比衡量论点的正确性及论证说服力。 | `original_argument_scores` 使用同一估计器、提示正文、解析器和生成参数；`N/A` 分数为 0。 | 方法口径一致。这里评估的不只是论点本身的事实正确性，还包括其对父论点的说服力。 |
| 根论点基础分 | 上游同时生成根论点的提示估计分，但其 `t_base` 分支最终仍把根基础分保持为 0.5；论文将此称为 0.5 BS。 | `qbaf_prediction` 固定根基础分为 0.5。 | 最终评分一致；当前实现省略了不会被使用的根估计调用。 |
| 聚合与判定 | 上游 `ProductAggregation`、`LinearInfluence(conservativeness=1)` 和 `computeStrengthValues` 实现 DF-QuAD，根最终强度严格大于 0.5 判真。 | `original_qbaf` 直接调用这些未改动的上游模块。 | 一致。 |

上述论文依据分别见 [Zhou 等，EMNLP 2025，第 2.2、3.4 节及附录 B](https://aclanthology.org/2025.findings-emnlp.1184.pdf) 和 [Freedman 等，AAAI 2025，DF-QuAD 定义与 ArgLLM 变体](https://ojs.aaai.org/index.php/AAAI/article/download/33637/35792)。本仓库的 `main.py`、`prompt.py`、`uncertainty_estimator.py` 及标准 DF-QuAD 模块与 `upstream/main` 无差异。

## 与「逐次运行原论文代码」不同的地方

1. **模型与接口不同。** 论文报告 Gemma 2、Llama 3.1 和 GPT-4o-mini；本次按用户配置使用 DeepSeek-v4.1-flash。上游 OpenAI 适配器不执行本地约束解码，当前适配器为打分请求另外加入百分比输出的系统格式指令，并关闭 API 思考模式。这些改变可能影响模型给出的具体分数，虽然用户提示正文及分数解析器没有改动。
2. **图与调用顺序不同。** 为让五方法比较同图，当前程序先完成图生成、再分别估分；D=2 复用 D=1 的图与第一层分数。上游程序在生成各层时穿插估分，独立运行 D=1、D=2。对于非确定性 API，即使提示正文相同，这些执行安排也不保证生成完全相同的文本与分数。
3. **省略未使用的根估分。** 上游 `generate_arguments` 即使计算 0.5 BS 分支，也会先请求一次根论点的提示估计，再仅用于另一条 Est. BS 分支。当前 `original_qbaf` 不发送该调用。因此当前报告中的该方法 API 成本与延迟，比逐字执行上游代码少一笔根估分请求；对 0.5 BS 的最终预测没有影响。
4. **上游 API 参数映射需要区分于论文设置。** 论文写的是重复惩罚 1.0；上游 `OpenAiLlmManager` 把 `repetition_penalty=1.0` 作为 `presence_penalty=1.0` 发送，当前适配器继承了这一映射。它保留了上游 API 路径的行为，但两项参数并非同一设置。当前 API 路径也没有把代码里的默认 `seed=42` 发给服务端；论文说明其实验使用种子 42。故不能据此声称当前 API 调用严格复现了论文的采样设置。

## 验证

- 新增 `tests/test_original_qbaf_parity.py`：在 D=1、D=2 及第一层攻击论点为 `N/A`/正常文本的四种组合中，固定模型回复后逐项比较上游 `ArgumentMiner.generate_arguments` 的 0.5 分支与当前缓存图路径。图中的论点正文、支持/攻击关系、所有非根基础分和根最终强度全部相同。
- 只读取完整实验的缓存与逐样本结果，重新经上游打分提示解析器计算六组各 500 条、共 3000 条样本的 `original_qbaf` 基础分及最终强度；全部与已保存结果一致，过程中没有补发 API 请求。
- 这两项验证检查了给定回复后的方法实现与结果保存；不能证明不同模型、不同采样顺序会产生相同回复。
