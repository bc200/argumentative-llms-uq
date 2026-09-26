"""Run five confidence methods on shared, cached argument graphs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import random

import Uncertainpy.src.uncertainpy.gradual as grad

from argument_miner import ArgumentMiner
from ew_dfquad import compute_ew_dfquad, existing_edges
from experiment_io import CachedJev, CachedLlmManager, read_json, write_json
from llm_managers import HuggingFaceLlmManager, OpenAiLlmManager
from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts
from uncertainty_estimator import UncertaintyEstimator


DATASETS = {
    "TruthfulClaim": "Datasets/TruthfulQA/Experiment",
    "StrategyClaim": "Datasets/StrategyQA/Experiment",
    "MedClaim": "Datasets/MedQA/Experiment",
}
METHODS = (
    "direct_llm",
    "direct_jev",
    "original_qbaf",
    "jev_qbaf",
    "jev_ew_qbaf",
    "jev_prior_qbaf",
    "jev_prior_ew_qbaf",
)
PRIOR_METHODS = ("jev_prior_qbaf", "jev_prior_ew_qbaf")


def dataset_rows(path, max_samples=None):
    """Read the bundled Hugging Face dataset, with a small Arrow-only fallback."""
    try:
        from datasets import load_from_disk
    except ImportError:
        try:
            import pyarrow.ipc as ipc
        except ImportError as error:
            raise RuntimeError(
                "Dataset loading requires 'datasets' or 'pyarrow'"
            ) from error
        files = sorted(Path(path).glob("data-*.arrow"))
        if len(files) != 1:
            raise ValueError("Expected one Arrow data file in " + str(path))
        with ipc.open_stream(str(files[0])) as reader:
            dataset = reader.read_all().select(["claim", "valid"]).to_pylist()
    else:
        dataset = load_from_disk(str(path))
    length = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(length):
        row = dataset[index]
        yield index, str(row["claim"]), int(row["valid"])


class LazyModelManager:
    """Keep a model client per worker and load it only on a cache miss."""

    def __init__(self, args):
        from threading import local

        self.args = args
        self.local = local()

    @property
    def last_usage(self):
        return getattr(self.local, "last_usage", None)

    def chat_completion(self, message, **kwargs):
        if not hasattr(self.local, "delegate"):
            if self.args.generator_model.startswith("openai/"):
                self.local.delegate = OpenAiLlmManager(
                    self.args.generator_model,
                    base_url=self.args.llm_base_url,
                    api_key_env=self.args.llm_key_env,
                    thinking_mode=self.args.llm_thinking,
                )
            else:
                self.local.delegate = HuggingFaceLlmManager(
                    model_name=self.args.generator_model,
                    quantization=self.args.quantization,
                    cache_dir=self.args.model_cache_dir,
                    input_device=self.args.input_device,
                )
        response = self.local.delegate.chat_completion(message, **kwargs)
        self.local.last_usage = self.local.delegate.last_usage
        return response


def model_manager(args):
    return LazyModelManager(args)


def llm_cache(delegate, args, directory):
    return CachedLlmManager(
        delegate,
        args.generator_model,
        directory,
        input_price=args.llm_input_price,
        output_price=args.llm_output_price,
        currency=args.llm_currency,
        cache_config={
            "quantization": getattr(args, "quantization", None)
            if not args.generator_model.startswith("openai/") else None,
            "base_url": args.llm_base_url,
            "thinking_mode": args.llm_thinking,
            "api_constraint_format": "v1" if args.generator_model.startswith("openai/") else None,
        },
        cache_only=getattr(args, "cache_only", False),
    )


def graph_request(args, dataset_name, index, claim, depth):
    return {
        "dataset": dataset_name,
        "sample_index": index,
        "claim": claim,
        "depth": depth,
        "breadth": args.breadth,
        "generator_model": args.generator_model,
        "quantization": getattr(args, "quantization", None),
        "llm_base_url": args.llm_base_url,
        "llm_thinking": args.llm_thinking,
        "generation_args": args.generation_args,
        "argument_prompt": "ArgumentMiningPrompts.new_sup_att",
        "pipeline_version": 1 if depth == 1 else 2,
    }


def reprice_graph_calls(calls, args):
    for event in calls:
        usage = event.get("usage")
        if not args.generator_model.startswith("openai/"):
            cost = 0.0
        elif usage is None or args.llm_input_price is None or args.llm_output_price is None:
            cost = None
        else:
            cost = (
                usage["input_tokens"] * args.llm_input_price
                + usage["output_tokens"] * args.llm_output_price
            ) / 1000000
        event["cost_usd"] = cost if args.llm_currency == "USD" else 0.0
        event["cost_cny"] = cost if args.llm_currency == "CNY" else 0.0
    return calls


def load_or_generate_graph(args, dataset_name, index, claim, depth, delegate):
    graph_group = "graphs" if depth == 1 else "graphs_v2"
    path = args.cache_dir / graph_group / dataset_name / ("D%d" % depth) / (
        "sample_%05d.json" % index
    )
    request = graph_request(args, dataset_name, index, claim, depth)
    if path.exists():
        record = read_json(path)
        if record["request"] != request:
            raise ValueError("Cached graph was generated under different settings: " + str(path))
        return record["graph"], reprice_graph_calls(record["calls"], args), path

    prior_graph = None
    prior_calls = []
    if depth > 1:
        prior_graph, prior_calls, _ = load_or_generate_graph(
            args, dataset_name, index, claim, depth - 1, delegate
        )
    call_group = "graph" if depth == 1 else "graph_v2"
    call_dir = args.cache_dir / "llm" / call_group / dataset_name / (
        "D%d" % depth
    ) / ("sample_%05d" % index)
    recorder = llm_cache(delegate, args, call_dir)
    miner = ArgumentMiner(
        generate_prompt_am=ArgumentMiningPrompts.new_sup_att,
        generate_prompt_ue=UncertaintyEvaluatorPrompts.analyst,
        llm_manager=recorder,
        depth=depth,
        breadth=args.breadth,
        generation_args=args.generation_args,
    )
    graph = (
        miner.generate_graph(claim).to_dict()
        if prior_graph is None else miner.extend_graph(prior_graph).to_dict()
    )
    if graph["arguments"]["db0"]["initial_weight"] != 0.5:
        raise ValueError("Root base score must be 0.5")
    calls = prior_calls + recorder.records
    write_json(path, {"request": request, "graph": graph, "calls": calls})
    return graph, calls, path


def parent_relations(graph):
    relations = {}
    for kind, pairs in (("support", graph["supports"]), ("attack", graph["attacks"])):
        for source, target in pairs:
            if source in relations:
                raise ValueError("Generated argument has multiple parent edges: " + source)
            relations[source] = (kind, target)
    if set(relations) != set(graph["arguments"]) - {"db0"}:
        raise ValueError("Generated graph does not match the expected argument tree")
    return relations


def direct_llm_score(args, dataset_name, index, claim, delegate):
    directory = args.cache_dir / "llm" / "direct_constrained_v1" / dataset_name / (
        "sample_%05d" % index
    )
    recorder = llm_cache(delegate, args, directory)
    estimator = UncertaintyEstimator(
        llm_manager=recorder,
        generate_prompt=UncertaintyEvaluatorPrompts.analyst,
        verbal=False,
        generation_args=args.generation_args,
    )
    return float(estimator(claim, topic=True)), recorder.records


def original_argument_scores(args, dataset_name, index, depth, graph, delegate,
                             prior_graph=None):
    if prior_graph is None:
        scores, prior_calls = {}, []
    else:
        scores, prior_calls = original_argument_scores(
            args, dataset_name, index, 1, prior_graph, delegate
        )
    score_group = ("original_constrained_v1" if depth == 1
                   else "original_d2_extension_v1")
    directory = args.cache_dir / "llm" / score_group / dataset_name / (
        "D%d" % depth
    ) / ("sample_%05d" % index)
    recorder = llm_cache(delegate, args, directory)
    estimator = UncertaintyEstimator(
        llm_manager=recorder,
        generate_prompt=UncertaintyEvaluatorPrompts.analyst,
        verbal=False,
        generation_args=args.generation_args,
    )
    relations = parent_relations(graph)
    for name, argument in graph["arguments"].items():
        if name == "db0" or name in scores:
            continue
        if argument["argument"] == "N/A":
            scores[name] = 0.0
            continue
        kind, target = relations[name]
        scores[name] = float(estimator(
            argument["argument"],
            claim=graph["arguments"][target]["argument"],
            support=(kind == "support"),
        ))
    return scores, prior_calls + recorder.records


def direct_jev_score(jev, dataset_name, index, claim):
    return jev.ask(
        Path("jev") / dataset_name / ("sample_%05d" % index) / "direct.json",
        {"claim": claim},
        "Is the claim correct, accurate, and truthful?",
        {"true": "The claim is true.", "false": "The claim is false."},
    )


def jev_argument_scores(jev, dataset_name, index, depth, graph,
                        prior_graph=None):
    if prior_graph is None:
        scores, prior_calls = {}, []
    else:
        scores, prior_calls = jev_argument_scores(
            jev, dataset_name, index, 1, prior_graph
        )
    relations = parent_relations(graph)
    calls = []
    for number, (name, argument) in enumerate(graph["arguments"].items()):
        if name == "db0" or name in scores:
            continue
        if argument["argument"] == "N/A":
            scores[name] = 0.0
            continue
        _, target = relations[name]
        score, record = jev.ask(
            Path("jev" if depth == 1 else "jev_d2_extension_v1") / dataset_name
            / ("D%d" % depth) / ("sample_%05d" % index)
            / ("node_%03d.json" % number),
            {
                "parent_claim": graph["arguments"][target]["argument"],
                "argument": argument["argument"],
            },
            "Is the argument factually correct, accurate, and truthful in this context? "
            "Judge the argument itself, independently of whether it supports or attacks "
            "the parent claim.",
            {"true": "The argument is factually valid.",
             "false": "The argument is factually invalid."},
        )
        scores[name] = score
        calls.append(record)
    return scores, prior_calls + calls


def jev_edge_weights(jev, dataset_name, index, depth, graph,
                     prior_graph=None):
    if prior_graph is None:
        weights, prior_calls = {}, []
    else:
        weights, prior_calls = jev_edge_weights(
            jev, dataset_name, index, 1, prior_graph
        )
    calls = []
    for number, (kind, source, target) in enumerate(
        list(existing_edges(grad.BAG.from_dict(graph))), start=1
    ):
        if (kind, source, target) in weights:
            continue
        relation = "support" if kind == "support" else "attack"
        score, record = jev.ask(
            Path("jev" if depth == 1 else "jev_d2_extension_v1") / dataset_name
            / ("D%d" % depth) / ("sample_%05d" % index)
            / ("edge_%03d.json" % number),
            {
                "parent_claim": graph["arguments"][target]["argument"],
                "argument": graph["arguments"][source]["argument"],
                "specified_relation": relation,
            },
            (
                "Does the specified argument form a valid %s relation to the parent "
                "claim? Assess only whether this specified relation holds. "
                "Do not change its relation type."
            ) % relation,
            {
                "true": "The specified %s relation is valid." % relation,
                "false": "The specified %s relation is invalid." % relation,
            },
        )
        weights[(kind, source, target)] = score
        calls.append(record)
    return weights, prior_calls + calls


def qbaf_prediction(graph, argument_scores, edge_weights=None, root_score=0.5):
    bag = grad.BAG.from_dict(graph)
    for name, argument in bag.arguments.items():
        argument.reset_initial_weight(
            root_score if name == "db0" else argument_scores[name]
        )
    if edge_weights is None:
        grad.algorithms.computeStrengthValues(
            bag,
            grad.semantics.modular.ProductAggregation(),
            grad.semantics.modular.LinearInfluence(conservativeness=1),
        )
    else:
        compute_ew_dfquad(bag, edge_weights)
    return float(bag.arguments["db0"].strength)


def has_conflicting_root_evidence(graph, argument_scores):
    """Both sides must contain a non-N/A root argument with positive Jev score."""
    def active(kind):
        return any(
            target == "db0"
            and graph["arguments"][source]["argument"] != "N/A"
            and argument_scores[source] > 0
            for source, target in graph[kind]
        )

    return active("supports") and active("attacks")


def event_summary(events):
    def total(currency):
        costs = [event.get("cost_" + currency, 0.0) for event in events]
        return None if any(cost is None for cost in costs) else sum(costs)

    return {
        "api_cost_usd": total("usd"),
        "api_cost_cny": total("cny"),
        "latency_seconds": sum(event["elapsed_seconds"] for event in events),
        "calls": len(events),
    }


def evaluate_sample(args, dataset_name, index, claim, label, depth, delegate, jev):
    graph, graph_calls, graph_path = load_or_generate_graph(
        args, dataset_name, index, claim, depth, delegate
    )
    prior_graph = (
        read_json(args.cache_dir / "graphs" / dataset_name / "D1"
                  / ("sample_%05d.json" % index))["graph"]
        if depth == 2 else None
    )
    llm_probability, direct_llm_calls = direct_llm_score(
        args, dataset_name, index, claim, delegate
    )
    jev_probability, direct_jev_call = direct_jev_score(
        jev, dataset_name, index, claim
    )
    original_scores, original_calls = original_argument_scores(
        args, dataset_name, index, depth, graph, delegate, prior_graph
    )
    jev_scores, jev_node_calls = jev_argument_scores(
        jev, dataset_name, index, depth, graph, prior_graph
    )
    edge_weights, jev_edge_calls = jev_edge_weights(
        jev, dataset_name, index, depth, graph, prior_graph
    )

    predictions = {
        "direct_llm": llm_probability,
        "direct_jev": jev_probability,
        "original_qbaf": qbaf_prediction(graph, original_scores),
        "jev_qbaf": qbaf_prediction(graph, jev_scores),
        "jev_ew_qbaf": qbaf_prediction(graph, jev_scores, edge_weights),
        "jev_prior_qbaf": qbaf_prediction(
            graph, jev_scores, root_score=jev_probability
        ),
        "jev_prior_ew_qbaf": qbaf_prediction(
            graph, jev_scores, edge_weights, root_score=jev_probability
        ),
    }
    events = {
        "direct_llm": direct_llm_calls,
        "direct_jev": [direct_jev_call],
        "original_qbaf": graph_calls + original_calls,
        "jev_qbaf": graph_calls + jev_node_calls,
        "jev_ew_qbaf": graph_calls + jev_node_calls + jev_edge_calls,
        "jev_prior_qbaf": graph_calls + [direct_jev_call] + jev_node_calls,
        "jev_prior_ew_qbaf": (
            graph_calls + [direct_jev_call] + jev_node_calls + jev_edge_calls
        ),
    }
    return {
        "dataset": dataset_name,
        "depth": depth,
        "index": index,
        "claim": claim,
        "valid": label,
        "graph_cache": str(graph_path),
        "argument_count": len(graph["arguments"]),
        "edge_count": len(graph["supports"]) + len(graph["attacks"]),
        "root_conflicting_evidence": has_conflicting_root_evidence(
            graph, jev_scores
        ),
        "predictions": predictions,
        "root_counterevidence": (
            abs(jev_probability - 0.5) > 1e-12
            and abs(predictions["jev_qbaf"] - 0.5) > 1e-12
            and ((jev_probability > 0.5) !=
                 (predictions["jev_qbaf"] > 0.5))
        ),
        "argument_base_scores": {
            "original_qbaf": original_scores,
            "jev_qbaf": jev_scores,
        },
        "edge_weights": [
            {"type": kind, "source": source, "target": target, "weight": weight}
            for (kind, source, target), weight in edge_weights.items()
        ],
        "usage": {method: event_summary(calls) for method, calls in events.items()},
        "_events": events,
    }


def metrics(labels, probabilities, bins=10):
    length = len(labels)
    accuracy = sum((probability > 0.5) == bool(label)
                   for label, probability in zip(labels, probabilities)) / length
    brier = sum((probability - label) ** 2
                for label, probability in zip(labels, probabilities)) / length
    groups = defaultdict(list)
    for label, probability in zip(labels, probabilities):
        groups[min(int(probability * bins), bins - 1)].append((label, probability))
    ece = sum(
        len(group) / length * abs(
            sum(probability for _, probability in group) / len(group)
            - sum(label for label, _ in group) / len(group)
        )
        for group in groups.values()
    )
    return {"accuracy": accuracy, "brier": brier, "ece": ece}


def summarize(rows):
    summary = []
    for dataset_name in DATASETS:
        for depth in (1, 2):
            subset = [row for row in rows
                      if row["dataset"] == dataset_name and row["depth"] == depth]
            if not subset:
                continue
            labels = [row["valid"] for row in subset]
            for method in METHODS:
                probabilities = [row["predictions"][method] for row in subset]
                quality = metrics(labels, probabilities)
                usd_costs = [row["usage"][method]["api_cost_usd"] for row in subset]
                cny_costs = [row["usage"][method]["api_cost_cny"] for row in subset]
                summary.append({
                    "dataset": dataset_name,
                    "depth": depth,
                    "method": method,
                    "n": len(subset),
                    **quality,
                    "api_cost_usd": (
                        None if any(cost is None for cost in usd_costs) else sum(usd_costs)
                    ),
                    "api_cost_cny": (
                        None if any(cost is None for cost in cny_costs) else sum(cny_costs)
                    ),
                    "mean_latency_seconds": sum(
                        row["usage"][method]["latency_seconds"] for row in subset
                    ) / len(subset),
                })
    return summary


def paired_interval(differences, rounds=2000):
    """Paired percentile bootstrap interval for the mean post-minus-prior change."""
    rng = random.Random(42)
    n = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(rounds)
    )
    return [samples[int(0.025 * (rounds - 1))],
            samples[int(0.975 * (rounds - 1))]]


def prior_comparisons(rows):
    """Measure how QBAF revises Jev's root belief on paired samples."""
    comparisons = []
    for dataset_name in DATASETS:
        for depth in (1, 2):
            group = [row for row in rows
                     if row["dataset"] == dataset_name and row["depth"] == depth]
            for subset_name, subset in (
                ("全部", group),
                ("根节点双向证据", [row for row in group
                              if row["root_conflicting_evidence"]]),
                ("先验反向证据", [row for row in group
                             if row["root_counterevidence"]]),
            ):
                if not subset:
                    continue
                labels = [row["valid"] for row in subset]
                prior = [row["predictions"]["direct_jev"] for row in subset]
                prior_quality = metrics(labels, prior)
                for method in PRIOR_METHODS:
                    posterior = [row["predictions"][method] for row in subset]
                    posterior_quality = metrics(labels, posterior)
                    prior_right = [(p > 0.5) == bool(y)
                                   for p, y in zip(prior, labels)]
                    posterior_right = [(p > 0.5) == bool(y)
                                       for p, y in zip(posterior, labels)]
                    accuracy_changes = [int(after) - int(before)
                                        for before, after in zip(
                                            prior_right, posterior_right)]
                    brier_changes = [(after - y) ** 2 - (before - y) ** 2
                                     for before, after, y in zip(
                                         prior, posterior, labels)]
                    comparisons.append({
                        "dataset": dataset_name,
                        "depth": depth,
                        "subset": subset_name,
                        "method": method,
                        "n": len(subset),
                        "prior_accuracy": prior_quality["accuracy"],
                        "posterior_accuracy": posterior_quality["accuracy"],
                        "accuracy_change": (posterior_quality["accuracy"]
                                            - prior_quality["accuracy"]),
                        "accuracy_change_ci95": paired_interval(accuracy_changes),
                        "prior_brier": prior_quality["brier"],
                        "posterior_brier": posterior_quality["brier"],
                        "brier_change": (posterior_quality["brier"]
                                         - prior_quality["brier"]),
                        "brier_change_ci95": paired_interval(brier_changes),
                        "prior_ece": prior_quality["ece"],
                        "posterior_ece": posterior_quality["ece"],
                        "changed_probability": sum(
                            abs(after - before) > 1e-12
                            for before, after in zip(prior, posterior)
                        ),
                        "mean_absolute_revision": sum(
                            abs(after - before)
                            for before, after in zip(prior, posterior)
                        ) / len(subset),
                        "decision_flips": sum(
                            (before > 0.5) != (after > 0.5)
                            for before, after in zip(prior, posterior)
                        ),
                        "corrected": accuracy_changes.count(1),
                        "spoiled": accuracy_changes.count(-1),
                    })
    return comparisons


def direct_prompt_fallbacks(events):
    counts = {
        name: {"direct_llm": 0, "original_qbaf": 0}
        for name in DATASETS
    }
    for event in events:
        if event["request"].get("kwargs", {}).get("constraint_prefix") != "Likelihood:":
            continue
        output = event["response"]
        value = output.replace("Likelihood:", "").strip()
        value = value.replace("is", "").strip()
        value = value.replace("%", "").strip()
        value = value.replace(".", "").strip()
        value = value.split("\n")[0]
        try:
            int(value)
        except ValueError:
            path = event["_cache_path"].replace("\\", "/")
            for name in DATASETS:
                if "/" + name + "/" in path:
                    method = ("direct_llm" if "/direct_constrained_v1/" in path
                              else "original_qbaf")
                    counts[name][method] += 1
                    break
    return counts


def chinese_report(args, summary, total_unique_cost, rows, jev_versions=None,
                   parse_fallbacks=None, prior_stats=None):
    lines = [
        "# Jev 根先验与 QBAF 的七方法对比实验报告",
        "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "",
        "## 实验设置",
        "",
        "- 论点生成模型：" + args.generator_model,
        "- 论点生成接口：" + (args.llm_base_url or "本地模型或默认服务"),
        "- 生成模型思考模式：" + (args.llm_thinking or "服务默认"),
        "- Jev 请求模型：" + args.jev_model,
        "- Jev 实际响应版本：" + (
            "、".join(jev_versions) if jev_versions else "未记录"
        ),
        "- 本次运行方式：" + (
            "仅读取既有响应缓存，不发起 API 请求。"
            if getattr(args, "cache_only", False) else "缓存优先，缺失时请求 API。"
        ),
        "- 数据：上游仓库三个 Experiment 数据集；D=1、D=2；breadth="
        + str(args.breadth) + "。",
        "- original_qbaf、jev_qbaf、jev_ew_qbaf 的根节点基础分为 0.5；"
        "两种 jev_prior 方法将 direct_jev 的根概率作为基础分。"
        "所有 QBAF 方法复用同一深度、同一样本的缓存图。",
        "- D=2 在 D=1 图上继续生成，共有的第一层论点、基础分和边权均直接复用。",
        "- API 模型额外接收与上游 Direct Prompting 一致的百分比格式要求，"
        "估计器、提示正文及解析规则保持原样。",
        "- original_qbaf 的论点基础分来自上游 Direct Prompting 估计器；"
        "jev_qbaf 使用 Jev Noul 论点真确性概率；jev_ew_qbaf "
        "进一步使用每条既有边的 Jev Noul 关系有效性概率。",
        "- jev_prior_qbaf 使用 Jev 根先验、Jev 论点分数和单位边权；"
        "jev_prior_ew_qbaf 再使用已缓存的 Jev 关系边权。"
        "两者均复用 direct_jev 的根概率，没有新增根打分请求。",
        "- EW-DF-QuAD 对每条边先计算 σ(父论点) × 边权，"
        "再使用原有乘积聚合及线性影响函数。",
        "- Accuracy 以概率 > 0.5 为真；Brier 为平均平方误差；"
        "ECE 为 10 个等宽概率区间的平均置信度与正例率差的加权和。",
        "- 生成模型按配置币种 " + args.llm_currency +
        " 计价，Jev 按美元计价；报告分别列示美元和人民币，不进行汇率换算。"
        "表中各 QBAF 方法都计入相同的共享论点生成成本，因此方法行的成本不可相加。"
        "两种 jev_prior 方法还计入同一条已缓存的根论点 Jev 请求。"
        "延迟是每样本顺序执行相应方法调用的平均耗时；"
        "缓存命中仍使用原始调用的记录耗时。",
        "",
        "## 结果",
        "",
        "| 数据集 | 深度 | 方法 | 样本数 | Accuracy | Brier | ECE | Jev/API 成本 (USD) | LLM/API 成本 (CNY) | 平均延迟 (秒/样本) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        usd_cost = "未计价" if item["api_cost_usd"] is None else (
            "%.6f" % item["api_cost_usd"]
        )
        cny_cost = "未计价" if item["api_cost_cny"] is None else (
            "%.6f" % item["api_cost_cny"]
        )
        lines.append(
            "| {dataset} | {depth} | {method} | {n} | {accuracy:.4f} | "
            "{brier:.4f} | {ece:.4f} | {usd_cost} | {cny_cost} | "
            "{latency:.3f} |".format(
                dataset=item["dataset"], depth=item["depth"],
                method=item["method"], n=item["n"],
                accuracy=item["accuracy"], brier=item["brier"],
                ece=item["ece"], usd_cost=usd_cost, cny_cost=cny_cost,
                latency=item["mean_latency_seconds"],
            )
        )
    if prior_stats is not None:
        lines += [
            "", "## Jev 根先验修正分析", "",
            "每个后验均与同一样本的 direct_jev 根先验配对比较。"
            "差值 = 后验指标 − direct_jev 指标；Accuracy 差值为正、"
            "Brier 差值为负表示改善。95% 区间采用 2000 次样本内配对重抽样。",
            "“根节点双向证据”指根论点同时有非 N/A 的支持和攻击论点，"
            "且两者的 Jev 论点分数均大于 0。"
            "“先验反向证据”指将根基础分临时置为 0.5 时，标准 DF-QuAD "
            "从同一图与 Jev 论点分数得到的方向，与非中性的 direct_jev 判断相反。"
            "“纠错”是先验判断错误、后验判断正确；“误改”方向相反。",
            "", "| 数据集 | 深度 | 子集 | 方法 | n | ΔAccuracy [95%区间] | "
            "ΔBrier [95%区间] | 翻转 | 纠错 | 误改 | 平均绝对概率修正 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in prior_stats:
            accuracy_ci = item["accuracy_change_ci95"]
            brier_ci = item["brier_change_ci95"]
            lines.append(
                "| {dataset} | {depth} | {subset} | {method} | {n} | "
                "{accuracy:+.4f} [{accuracy_low:+.4f}, {accuracy_high:+.4f}] | "
                "{brier:+.4f} [{brier_low:+.4f}, {brier_high:+.4f}] | "
                "{flips} | {corrected} | {spoiled} | {shift:.4f} |".format(
                    dataset=item["dataset"], depth=item["depth"],
                    subset=item["subset"], method=item["method"], n=item["n"],
                    accuracy=item["accuracy_change"],
                    accuracy_low=accuracy_ci[0], accuracy_high=accuracy_ci[1],
                    brier=item["brier_change"],
                    brier_low=brier_ci[0], brier_high=brier_ci[1],
                    flips=item["decision_flips"],
                    corrected=item["corrected"], spoiled=item["spoiled"],
                    shift=item["mean_absolute_revision"],
                )
            )
    if parse_fallbacks is not None:
        lines += ["", "## 格式回退", "", "缓存中的 Direct Prompting 百分比回复经上游解析器检查，回退次数如下："]
        for name in DATASETS:
            counts = parse_fallbacks[name]
            lines.append(
                name + "：根论点直评 %d 次，生成论点评分 %d 次。" % (
                    counts["direct_llm"], counts["original_qbaf"]
                )
            )
        lines += ["", "解析失败的回复按上游原规则赋值为 0.5；这些样本保留在指标中。"]
    lines += [
        "",
        "## 成本说明",
        "",
        "本次缓存记录中去重后的 API 成本：美元 " + (
            "未计价" if total_unique_cost["usd"] is None else
            "$%.6f" % total_unique_cost["usd"]
        ) + "；人民币 " + (
            "未计价" if total_unique_cost["cny"] is None else
            "￥%.6f" % total_unique_cost["cny"]
        ) + "。本地生成模型的 API 成本按 0 计；其运行时间仍计入延迟。",
        "成本以服务端返回的 token 用量乘配置单价估算。"
        "若服务商账单包含其他费用，以实际账单为准。",
        "",
        "共完成 %d 个数据集—深度—样本组合。逐样本概率和调用记录存于输出目录。" % len(rows),
        "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="8bit")
    parser.add_argument("--input-device", default="cuda:0")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("experiment_cache"))
    parser.add_argument("--cache-only", action="store_true",
                        help="Read responses from cache without making API calls")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--breadth", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel samples; use 1 for a local GPU model")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-key-env", default=None)
    parser.add_argument("--llm-thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--llm-currency", choices=["USD", "CNY"], default="USD")
    parser.add_argument("--llm-input-price", type=float, default=None,
                        help="Price per million input tokens in --llm-currency")
    parser.add_argument("--llm-output-price", type=float, default=None,
                        help="Price per million output tokens in --llm-currency")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--jev-endpoint", default="https://api.typesafe.ai/v1/systemone")
    parser.add_argument("--jev-key-env", default="TYPESAFE_API_KEY")
    parser.add_argument("--jev-input-price", type=float, default=0.042)
    for name, path in DATASETS.items():
        parser.add_argument("--" + name.lower() + "-path", type=Path, default=Path(path))
    args = parser.parse_args()
    if (args.breadth < 1 or args.workers < 1 or
            args.max_samples is not None and args.max_samples < 1):
        parser.error("breadth, workers and max-samples must be positive")
    if not args.generator_model.startswith("openai/") and args.workers != 1:
        parser.error("Local GPU models require --workers 1")
    args.generation_args = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }
    return args


def load_local_keys(path, names):
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in names and not os.environ.get(name):
            os.environ[name] = value.strip().strip('"').strip("'")


def main():
    args = parse_args()
    load_local_keys(
        Path(".env"),
        {args.jev_key_env, args.llm_key_env, "OPENAI_KEY", "OPENAI_API_KEY"},
    )
    delegate = model_manager(args)
    jev = CachedJev(
        args.cache_dir, args.jev_model, args.jev_endpoint,
        args.jev_input_price, args.jev_key_env, cache_only=args.cache_only,
    )
    from concurrent.futures import ThreadPoolExecutor

    rows = []
    unique_events = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for dataset_name in DATASETS:
            path = getattr(args, dataset_name.lower() + "_path")
            dataset = list(dataset_rows(path, args.max_samples))
            for depth in (1, 2):
                def process(item):
                    index, claim, label = item
                    return evaluate_sample(
                        args, dataset_name, index, claim, label, depth, delegate, jev
                    )

                for count, row in enumerate(pool.map(process, dataset), start=1):
                    for event_list in row.pop("_events").values():
                        for event in event_list:
                            unique_events[event["_cache_path"]] = event
                    result_path = args.output_dir / "data" / dataset_name / (
                        "D%d" % depth
                    ) / ("sample_%05d.json" % row["index"])
                    write_json(result_path, row)
                    rows.append(row)
                    if count % 25 == 0 or count == len(dataset):
                        print("%s D%d %d/%d 完成" % (
                            dataset_name, depth, count, len(dataset)
                        ), flush=True)
    summary = summarize(rows)
    prior_stats = prior_comparisons(rows)

    def unique_cost(currency):
        costs = [
            event.get("cost_" + currency, 0.0)
            for event in unique_events.values()
        ]
        return None if any(cost is None for cost in costs) else sum(costs)

    total_unique_cost = {
        "usd": unique_cost("usd"),
        "cny": unique_cost("cny"),
    }
    jev_versions = sorted({
        event["response"]["model"] for event in unique_events.values()
        if "payload" in event["request"] and "model" in event["response"]
    })
    parse_fallbacks = direct_prompt_fallbacks(unique_events.values())
    write_json(args.output_dir / "metrics.json", {
        "summary": summary,
        "prior_comparisons": prior_stats,
        "unique_api_cost_usd": total_unique_cost["usd"],
        "unique_api_cost_cny": total_unique_cost["cny"],
        "unique_api_calls": len(unique_events),
        "jev_response_models": jev_versions,
        "direct_prompt_fallbacks": parse_fallbacks,
        "settings": {
            "generator_model": args.generator_model,
            "llm_base_url": args.llm_base_url,
            "llm_thinking": args.llm_thinking,
            "llm_currency": args.llm_currency,
            "jev_model": args.jev_model,
            "breadth": args.breadth,
            "max_samples": args.max_samples,
            "workers": args.workers,
            "cache_only": args.cache_only,
        },
    })
    report = chinese_report(
        args, summary, total_unique_cost, rows, jev_versions, parse_fallbacks,
        prior_stats,
    )
    report_path = args.output_dir / "实验报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("中文报告：" + str(report_path))


if __name__ == "__main__":
    main()
