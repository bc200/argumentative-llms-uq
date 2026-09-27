# Jev × QBAF 置信度实验

本实验在上游仓库的三个 Experiment 数据集上比较七种方法：
direct_llm、direct_jev、original_qbaf、jev_qbaf、jev_ew_qbaf、
jev_prior_qbaf 和 jev_prior_ew_qbaf。
默认将 TruthfulQA、StrategyQA、MedQA 的 Experiment 目录分别称为
TruthfulClaim、StrategyClaim、MedClaim；若使用另行整理的 Claim 数据集，
请通过命令行参数替换路径。

## 复现口径

- 论点文本始终由上游 ArgumentMiner 和 ArgumentMiningPrompts.new_sup_att
  生成。每个样本、每个深度只生成一张图；五种 QBAF 方法从同一份图缓存重建各自的计分图。
- D=1 和 D=2 分别缓存；D=2 从同一样本的 D=1 缓存图继续生成，
  因而第一层论点完全一致；其节点分数和边权也直接复用。默认 breadth=1。
  N/A 保留为上游图中的节点；
  其节点基础分为 0，既有边仍保留。
- original_qbaf、jev_qbaf、jev_ew_qbaf 的根节点基础分固定为 0.5；
  jev_prior_qbaf 和 jev_prior_ew_qbaf 使用 direct_jev 对根论点的概率作为基础分。
- direct_llm 是上游 UncertaintyEstimator 对根论点的 Direct Prompting 概率；
  original_qbaf 用同一估计器给生成论点打分。API 模型通过系统格式要求遵循
  上游的“Likelihood: N%”约束，估计器、提示正文与解析规则保持不变。
  模型偶有未遵循百分比格式的回复；实验沿用上游解析器的 0.5 回退，中文报告记录回退次数。
- direct_jev 用 Jev Noul 对根论点真假打分。jev_qbaf 用 Jev Noul
  估计生成论点本身的真确性，与它和父论点的关系分开。
  jev_ew_qbaf 进一步对每条已有 support/attack 边询问该指定关系是否有效；
  不新增、删除或改写边的类型。
- jev_prior_qbaf 使用 Jev 根先验、Jev 论点基础分与单位边权，采用标准 DF-QuAD；
  jev_prior_ew_qbaf 保持相同根先验和论点基础分，对已有边采用 Jev 关系有效性权重。
  两者复用 direct_jev 的根概率和现有响应缓存。
- EW-DF-QuAD 把进入目标论点的每个父论点强度 σ(β) 替换成
  σ(β) × w(β,α)，然后调用仓库原有的 ProductAggregation 与
  LinearInfluence(conservativeness=1)。标准 DF-QuAD 的源码保持原样。
- Accuracy 采用上游的 p > 0.5 阈值。Brier 是 (p-y)² 的平均值。
  ECE 对正类概率做 10 个等宽分箱，计算各箱平均概率与正例率之差的样本占比加权和。
- 表中 QBAF 的 API 成本包含共享图生成成本，所以不同方法的行不能相加。
  DMX 费用以人民币计，Jev 费用以美元计，分别列示，不进行汇率换算。
  报告另外给出按缓存文件去重后的实际调用成本。若使用本地 LLM，
  API 成本为 0，推理耗时仍计入延迟。Jev 按其返回的输入 token 数与
  配置单价估算；若使用 OpenAI 生成模型，必须传入该模型的输入和输出单价，
  才能给出完整的美元成本。

## 运行

先配置 Jev API key。可设置当前进程环境变量 TYPESAFE_API_KEY，
也可在工作区根目录的 .env 文件中写入 TYPESAFE_API_KEY=你的密钥。
.env 已被 Git 忽略，脚本只读取所需的 API key 字段。
默认使用 TypeSafe 官方 https://api.typesafe.ai/v1/systemone 和 jev-latest。
Jev 的接口格式见 https://api.typesafe.ai/docs，默认输入价格
$0.042 / 百万 token 见 https://typesafe.ai/blog/introducing-system-one-models-and-jev。

上游默认论点生成模型是 mistralai/Mistral-7B-Instruct-v0.2。
本地模型需先安装上游所需的 PyTorch、Transformers 等依赖和对应模型权重。
只使用 openai/ 生成模型时，可以安装较轻的
pyarrow==14.0.0、openai==1.60.0；若已安装上游的 datasets 包，
程序会优先沿用其 load_from_disk 读取方式。
并设置相应的 API key。当前 DMX / DeepSeek-V4.1-Flash 实验配置：

    python benchmark_jev_qbaf.py --generator-model openai/deepseek-v4.1-flash --llm-base-url https://www.dmxapi.cn/v1 --llm-key-env DMX_API_KEY --llm-thinking disabled --llm-currency CNY --llm-input-price 1.9 --llm-output-price 7.6 --workers 6

DMX_API_KEY 可放在已忽略的 .env 文件中。DeepSeek 的思考模式默认开启；
在上游 128-token 限额下，测试调用的输出额度被推理内容用尽，因而本实验显式关闭思考，
保持上游的论点生成提示和 token 限额。
相关参数见 https://api-docs.deepseek.com/api/create-chat-completion/。

为比较生成模型的影响，Pro 0813 实验仅更换论点生成和 Direct Prompting 模型，
请求名为 `deepseek-v4-pro-0813`，保持 thinking 关闭及其余实验口径不变。
使用单独的图与响应缓存：

    python benchmark_jev_qbaf.py --generator-model openai/deepseek-v4-pro-0813 --llm-base-url https://www.dmxapi.cn/v1 --llm-key-env DMX_API_KEY --llm-thinking disabled --llm-currency CNY --cache-dir experiment_cache/pro_0813 --output-dir experiment_results/pro_0813_full --workers 16

为让两次实验的 Direct Jev 基线逐样本相同，Pro 缓存中的
`jev/<数据集>/sample_<序号>/direct.json` 从 Flash 缓存复制；生成论点、
其 Jev 分数及边权在 Pro 缓存中重新生成和请求。若 DMX 单价稍后确定，
使用相同参数并补充 `--llm-input-price`、`--llm-output-price` 与
`--cache-only`，可不发出新请求而补算人民币成本。

两次全量实验完成后，可生成逐样本配对比较与 Pro 置信度分桶报告：

    python compare_generator_models.py
    python analyze_jev_confidence_bins.py --input-dir experiment_results/pro_0813_full/data --report reports/2026-09-27_pro0813_置信度分桶分析.md --metrics reports/2026-09-27_pro0813_置信度分桶指标.json --plot reports/2026-09-27_pro0813_置信度分桶图.png

配对脚本核对每条论点、标签与 Direct Jev 根概率一致，再分别计算两个深度的
Accuracy、Brier、ECE 差异及 Rescue、Harm。生成模型每次只采样一次且 API
没有固定随机种子，因此两模型差值同时包含生成随机性；并发数不同也会影响
跨模型延迟的可比性。

使用本地模型：

    python benchmark_jev_qbaf.py --quantization 4bit --input-device cuda:0

默认处理三个数据集的全部样本、D=1 与 D=2。可用
--max-samples 10 先作小样本试跑。自定义数据路径使用
--truthfulclaim-path、--strategyclaim-path 和 --medclaim-path。
模型、温度、token 上限、breadth、Jev 版本或接口地址变化时，
建议指定新的 --cache-dir，以保持实验配置独立。
若仅调整成本单价，可以复用缓存。

缓存保存在 experiment_cache/，逐样本结果和指标 JSON 保存在
experiment_results/。完成后会生成中文 experiment_results/实验报告.md。
中断后重复相同命令可读取缓存继续；已缓存的 Jev 回答也可在无 API key
时离线重算报告。
使用 `--cache-only` 时，任一缓存缺失会停止运行，不发起新的 API 请求。

运行测试：

    python -m unittest discover -s tests -v

## 本次全量结果

三个数据集各 500 条、D=1 与 D=2 的五方法结果见 [中文报告](reports/2026-09-24_jev_qbaf_实验报告.md)。报告包含中文实验设置、指标、结果解读、格式回退及成本说明。逐样本输出和原始 API 响应缓存保存在本地已忽略的 experiment_results/ 与 experiment_cache/。

[original_qbaf 与上游代码及论文的一致性复核](reports/2026-09-25_original_qbaf_一致性复核.md)。

两种 Jev 根先验方法的全量结果、与 direct_jev 的配对修正分析，见
[Jev 根先验实验中文报告](reports/2026-09-26_jev_prior_qbaf_实验报告.md)。

按 Direct Jev 决策置信度 `max(p, 1-p)` 对现有逐样本结果分桶，
比较 Jev-Prior-EW-QBAF 的 Rescue、Harm、Brier 和判断翻转率，见
[置信度分桶中文报告](reports/2026-09-27_jev_prior_ew_置信度分桶分析.md)。
离线复算命令：

    python analyze_jev_confidence_bins.py --plot reports/2026-09-27_jev_prior_ew_置信度分桶图.png

不需要图片时省略 `--plot`；生成图片需安装 Matplotlib。

DeepSeek V4 Pro 0813 的七方法全量结果见
[Pro 0813 中文实验报告](reports/2026-09-27_pro0813_七方法全量实验.md)，
与 V4.1 Flash 的逐样本配对结果见
[生成模型影响中文报告](reports/2026-09-27_生成模型Flash与Pro0813对比.md)。
新模型下的根先验置信度分桶结果见
[Pro 0813 分桶中文报告](reports/2026-09-27_pro0813_置信度分桶分析.md)。
本次 DMX Pro 折后单价尚未确认，报告先保存实际 token 用量；
人民币成本栏在确认单价前标为“未计价”。

Qwen 3.8 Max 的七方法全量结果见
[Qwen 中文实验报告](reports/2026-09-27_qwen38max_七方法全量实验.md)，
与 Flash、Pro 0813 的逐样本配对比较见
[三生成模型中文对比报告](reports/2026-09-27_生成模型三方对比.md)，
按 Direct Jev 决策置信度分桶的结果见
[Qwen 分桶中文报告](reports/2026-09-27_qwen38max_置信度分桶分析.md)。
Qwen 使用 `qwen3.8-max`，通过 `enable_thinking=false` 关闭思考；
三次实验各自缓存图，同一模型内七方法共用图，根节点 Direct Jev 回答完全相同。
本轮对三个 500 条全集均发起了实验；DMX 持续拒绝 TruthfulClaim 第 442 条的原始
论点生成提示，Qwen 最终完成 2998/3000 条样本深度结果。
三模型配对比较在每个深度的共同 1499 条样本上进行，并在报告中标明缺失。
Qwen 成本按 2026-09-27 DMX 公布的折后价输入 ¥9.48、输出 ¥28.44／百万 token
估算，尚未扣除可能的输入缓存优惠。全量缓存完成后可以离线重新生成三模型对比：

    python compare_three_generators.py

Qwen 的 Jev 置信度分桶分析命令：

    python analyze_jev_confidence_bins.py --input-dir experiment_results/qwen38_max_full/data --report reports/2026-09-27_qwen38max_置信度分桶分析.md --metrics reports/2026-09-27_qwen38max_置信度分桶指标.json --plot reports/2026-09-27_qwen38max_置信度分桶图.png

## Qwen 3.8 27B 复现实验

新一轮仅将论点生成和 Direct Prompting 模型改为 `qwen3.8-27b`，其余三个
500 条 Experiment 全集、D=1/D=2、七方法、图生成提示、Jev 和 QBAF 设置
保持一致。每个模型使用独立图与图相关响应缓存；Direct Jev 根响应沿用此前的
同一批 1500 条缓存，使模型间逐样本配对比较使用相同根概率。

DMX 此模型用 `reasoning_effort=none` 关闭思考，参数依据
[Qwen OpenAI 兼容接口说明](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)。
试跑中通用的
`enable_thinking=false` 仍耗尽 128-token 输出额度；因此适配器针对这一
模型发送 `reasoning_effort=none`，仍保留原有输出限额。请求的模型名为
`qwen3.8-27b`。人民币成本按
[DMX 公开价格](https://rmb.dmxapi.cn/?api=model_prices)的输入 ¥3、输出 ¥12／百万 token 估算，
未扣除可能的输入缓存优惠。

    python benchmark_jev_qbaf.py --generator-model openai/qwen3.8-27b --llm-base-url https://www.dmxapi.cn/v1 --llm-key-env DMX_API_KEY --llm-thinking disabled --llm-currency CNY --llm-input-price 3 --llm-output-price 12 --cache-dir experiment_cache/qwen38_27b --output-dir experiment_results/qwen38_27b_full --workers 16 --record-provider-rejections

    python analyze_jev_confidence_bins.py --input-dir experiment_results/qwen38_27b_full/data --report reports/2026-09-27_qwen38_27b_置信度分桶分析.md --metrics reports/2026-09-27_qwen38_27b_置信度分桶指标.json --plot reports/2026-09-27_qwen38_27b_置信度分桶图.png

    python compare_four_generators.py

本轮 3000/3000 条样本深度结果全部完成，无服务商拒绝或百分比解析回退。
七方法的 Accuracy、Brier、ECE、API 成本和延迟见
[27B 中文全量报告](reports/2026-09-27_qwen38_27b_七方法全量实验.md)；
与 Flash、Pro 0813、Qwen Max 在共同 2998 条结果上的配对比较见
[四模型中文对比报告](reports/2026-09-27_qwen38_27b_与三模型对比.md)；
Jev 决策置信度、Rescue−Harm 与翻转率见
[27B 中文分桶报告](reports/2026-09-27_qwen38_27b_置信度分桶分析.md)。
在共同样本上，27B 的 direct_llm 比 Max 低 7.47 个百分点；
Jev-Prior-EW-QBAF 相对 Max 在 D=1/D=2 分别高 0.40/1.13 个百分点，
配对区间均跨 0。27B 的这两种深度下 Jev-Prior-EW-QBAF Brier 均高于
Direct Jev；接近 0.5 的 Jev 样本没有稳定的净纠错收益。
生成模型估算成本为 ￥15.31，Jev 估算成本为 $0.265897，分别计价。
使用 `--cache-only` 在独立输出目录复算后，全部指标与在线运行相同。
