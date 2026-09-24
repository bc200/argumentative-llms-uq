# Jev × QBAF 置信度实验

本实验在上游仓库的三个 Experiment 数据集上比较五种方法：
direct_llm、direct_jev、original_qbaf、jev_qbaf 和 jev_ew_qbaf。
默认将 TruthfulQA、StrategyQA、MedQA 的 Experiment 目录分别称为
TruthfulClaim、StrategyClaim、MedClaim；若使用另行整理的 Claim 数据集，
请通过命令行参数替换路径。

## 复现口径

- 论点文本始终由上游 ArgumentMiner 和 ArgumentMiningPrompts.new_sup_att
  生成。每个样本、每个深度只生成一张图；三种 QBAF 方法从同一份图缓存重建各自的计分图。
- D=1 和 D=2 分别缓存，默认 breadth=1。N/A 保留为上游图中的节点；
  其节点基础分为 0，既有边仍保留。
- 所有 QBAF 根节点基础分固定为 0.5。
- direct_llm 是上游 UncertaintyEstimator 对根论点的 Direct Prompting 概率；
  original_qbaf 用同一估计器给生成论点打分。
- direct_jev 用 Jev Noul 对根论点真假打分。jev_qbaf 用 Jev Noul
  估计生成论点本身的真确性，与它和父论点的关系分开。
  jev_ew_qbaf 进一步对每条已有 support/attack 边询问该指定关系是否有效；
  不新增、删除或改写边的类型。
- EW-DF-QuAD 把进入目标论点的每个父论点强度 σ(β) 替换成
  σ(β) × w(β,α)，然后调用仓库原有的 ProductAggregation 与
  LinearInfluence(conservativeness=1)。标准 DF-QuAD 的源码保持原样。
- Accuracy 采用上游的 p > 0.5 阈值。Brier 是 (p-y)² 的平均值。
  ECE 对正类概率做 10 个等宽分箱，计算各箱平均概率与正例率之差的样本占比加权和。
- 表中 QBAF 的 API 成本包含共享图生成成本，所以不同方法的行不能相加。
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
并设置 OPENAI_KEY 或 OPENAI_API_KEY。示例命令：

    python benchmark_jev_qbaf.py --generator-model openai/gpt-4o-mini --llm-input-price <每百万输入token美元价格> --llm-output-price <每百万输出token美元价格>

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

运行测试：

    python -m unittest discover -s tests -v
