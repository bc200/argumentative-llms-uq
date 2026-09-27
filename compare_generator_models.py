"""Compare paired seven-method results from two argument-generation models."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analyze_jev_confidence_bins import BIN_LABELS, POOLED, analyze
from benchmark_jev_qbaf import DATASETS, METHODS, metrics, paired_interval
from experiment_io import read_json, write_json


def load_results(directory):
    directory = Path(directory)
    settings = read_json(directory / "metrics.json")
    rows = {}
    for path in sorted((directory / "data").glob("*/*/sample_*.json")):
        row = read_json(path)
        key = row["dataset"], row["depth"], row["index"]
        if key in rows:
            raise ValueError("Duplicate result: " + str(key))
        rows[key] = row
    if not rows:
        raise ValueError("No per-sample results in " + str(directory))
    return settings, rows


def aligned_pairs(flash, pro):
    if flash.keys() != pro.keys():
        raise ValueError("The two runs have different sample-depth keys")
    pairs = []
    for key in sorted(flash):
        before, after = flash[key], pro[key]
        if (before["claim"], before["valid"]) != (after["claim"], after["valid"]):
            raise ValueError("Dataset or label changed for " + str(key))
        if before["predictions"]["direct_jev"] != after["predictions"]["direct_jev"]:
            raise ValueError("Direct Jev baseline changed for " + str(key))
        if set(before["predictions"]) != set(METHODS):
            raise ValueError("Flash run is missing a method for " + str(key))
        if set(after["predictions"]) != set(METHODS):
            raise ValueError("Pro run is missing a method for " + str(key))
        pairs.append((before, after))
    return pairs


def compare_subset(pairs, dataset, depth, method):
    subset = [(before, after) for before, after in pairs
              if (dataset == POOLED or before["dataset"] == dataset)
              and before["depth"] == depth]
    if not subset:
        return None
    labels = [before["valid"] for before, _ in subset]
    old = [before["predictions"][method] for before, _ in subset]
    new = [after["predictions"][method] for _, after in subset]
    old_quality = metrics(labels, old)
    new_quality = metrics(labels, new)
    changes = [int((p_new > 0.5) == bool(y))
               - int((p_old > 0.5) == bool(y))
               for y, p_old, p_new in zip(labels, old, new)]
    brier_changes = [(p_new - y) ** 2 - (p_old - y) ** 2
                     for y, p_old, p_new in zip(labels, old, new)]
    return {
        "dataset": dataset,
        "depth": depth,
        "method": method,
        "n": len(subset),
        "flash": old_quality,
        "pro": new_quality,
        "accuracy_change": new_quality["accuracy"] - old_quality["accuracy"],
        "accuracy_change_ci95": paired_interval(changes, rounds=1000),
        "brier_change": new_quality["brier"] - old_quality["brier"],
        "brier_change_ci95": paired_interval(brier_changes, rounds=1000),
        "ece_change": new_quality["ece"] - old_quality["ece"],
        "corrected": changes.count(1),
        "spoiled": changes.count(-1),
        "prediction_flips": sum((a > 0.5) != (b > 0.5)
                                for a, b in zip(old, new)),
        "mean_absolute_probability_change": sum(abs(a - b)
                                                 for a, b in zip(old, new)) / len(old),
    }


def summarize_graphs(rows):
    result = []
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            subset = [row for row in rows.values()
                      if (dataset == POOLED or row["dataset"] == dataset)
                      and row["depth"] == depth]
            result.append({
                "dataset": dataset,
                "depth": depth,
                "n": len(subset),
                "mean_arguments": sum(row["argument_count"] for row in subset) / len(subset),
                "mean_edges": sum(row["edge_count"] for row in subset) / len(subset),
                "root_conflict_rate": sum(row["root_conflicting_evidence"]
                                          for row in subset) / len(subset),
            })
    return result


def number(value, digits=4):
    return "未计价" if value is None else ("%%.%df" % digits) % value


def render_report(result, flash_dir, pro_dir):
    lookup = {(row["dataset"], row["depth"], row["method"]): row
              for row in result["comparisons"]}
    old_graphs = {(row["dataset"], row["depth"]): row
                  for row in result["graph_summary"]["flash"]}
    new_graphs = {(row["dataset"], row["depth"]): row
                  for row in result["graph_summary"]["pro"]}
    old_bins = {(row["depth"], row["confidence_bin"]): row
                for row in result["confidence_bins"]["flash"]
                if row["dataset"] == POOLED}
    new_bins = {(row["depth"], row["confidence_bin"]): row
                for row in result["confidence_bins"]["pro"]
                if row["dataset"] == POOLED}
    lines = [
        "# DeepSeek V4.1 Flash 与 V4 Pro 0813 的 QBAF 全量配对比较", "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"), "",
        "## 对照设置", "",
        "- Flash 结果：`" + str(flash_dir).replace("\\", "/") + "`；Pro 结果：`"
        + str(pro_dir).replace("\\", "/") + "`。两组各含三个数据集的全部 500 条论点，"
        "各在 D=1、D=2 评估七种方法。",
        "- 两组使用同一批数据、标签、Jev 根论点概率、论点生成流程、QBAF 实现和指标口径。"
        "Pro 的请求模型名是 `deepseek-v4-pro-0813`，thinking 为 disabled。"
        "Pro 图及图相关 Jev 响应独立缓存；其根论点 Jev 响应沿用 Flash 缓存。",
        "- Δ 均为 Pro − Flash。Accuracy 越高越好，Brier 与 ECE 越低越好。"
        "准确率和 Brier 差值的 95% 区间对同一论点的差值进行 1000 次配对重抽样；"
        "这些区间没有做多重比较校正。",
        "- 两次生成均为单次随机采样，temperature=0.7、top_p=0.95，"
        "API 请求未固定随机种子；下列差异包含生成随机性，不能单独归因为模型权重。",
        "- Flash 全量实验用 {flash_workers} 个并发样本，Pro 全量实验用 "
        "{pro_workers} 个；两个实验各自报告实测延迟，跨模型延迟差还受并发负载影响。".format(
            flash_workers=result["run_metrics"]["flash"]["settings"]["workers"],
            pro_workers=result["run_metrics"]["pro"]["settings"]["workers"]),
        "- Direct Jev 逐样本完全相同，因此是模型切换时的固定参照。"
        "D=1、D=2 共用根论点，每个深度单独汇总；不把两个深度当作独立样本合并。", "",
        "## 三数据集合计", "",
        "| 深度 | 方法 | n | Flash Acc | Pro Acc | ΔAcc [95% CI] | Flash Brier | Pro Brier | ΔBrier | Flash ECE | Pro ECE | ΔECE | 改对/改错 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for method in METHODS:
            row = lookup[POOLED, depth, method]
            ci = row["accuracy_change_ci95"]
            lines.append(
                "| D={depth} | {method} | {n} | {old_acc:.4f} | {new_acc:.4f} | "
                "{acc_delta:+.4f} [{lo:+.4f},{hi:+.4f}] | {old_brier:.4f} | "
                "{new_brier:.4f} | {brier_delta:+.4f} | {old_ece:.4f} | "
                "{new_ece:.4f} | {ece_delta:+.4f} | {corrected}/{spoiled} |".format(
                    depth=depth, method=method, n=row["n"],
                    old_acc=row["flash"]["accuracy"],
                    new_acc=row["pro"]["accuracy"],
                    acc_delta=row["accuracy_change"], lo=ci[0], hi=ci[1],
                    old_brier=row["flash"]["brier"],
                    new_brier=row["pro"]["brier"],
                    brier_delta=row["brier_change"],
                    old_ece=row["flash"]["ece"],
                    new_ece=row["pro"]["ece"],
                    ece_delta=row["ece_change"],
                    corrected=row["corrected"], spoiled=row["spoiled"]))
    original_d1 = lookup[POOLED, 1, "original_qbaf"]
    original_d2 = lookup[POOLED, 2, "original_qbaf"]
    direct_d1 = lookup[POOLED, 1, "direct_llm"]
    prior_ew_d1 = lookup[POOLED, 1, "jev_prior_ew_qbaf"]
    prior_ew_d2 = lookup[POOLED, 2, "jev_prior_ew_qbaf"]
    root_prior = lookup[POOLED, 1, "direct_jev"]["pro"]
    lines += [
        "", "## 主要观察", "",
        "- original_qbaf 的 Accuracy 在 D=1 从 {old1:.4f} 变为 {new1:.4f} "
        "（{delta1:+.4f}），D=2 从 {old2:.4f} 变为 {new2:.4f} "
        "（{delta2:+.4f}）；Brier 在两个深度均下降。"
        "相比之下，direct_llm 的 Accuracy 仅变化 {direct_delta:+.4f}。".format(
            old1=original_d1["flash"]["accuracy"],
            new1=original_d1["pro"]["accuracy"],
            delta1=original_d1["accuracy_change"],
            old2=original_d2["flash"]["accuracy"],
            new2=original_d2["pro"]["accuracy"],
            delta2=original_d2["accuracy_change"],
            direct_delta=direct_d1["accuracy_change"]),
        "- Jev-Prior-EW-QBAF 相对 Flash 的 Accuracy 在 D=1 变化 "
        "{delta1:+.4f}，D=2 变化 {delta2:+.4f}；"
        "但 Pro 运行中它相对固定的 Direct Jev（Accuracy {prior:.4f}、"
        "Brier {prior_brier:.4f}）仅在 D=1 的 Accuracy 略高，"
        "两个深度的 Brier 均较高。".format(
            delta1=prior_ew_d1["accuracy_change"],
            delta2=prior_ew_d2["accuracy_change"],
            prior=root_prior["accuracy"],
            prior_brier=root_prior["brier"]),
        "- 根节点同时含支持与攻击的样本比例在 D=1 从 {old_conf:.3f} "
        "变为 {new_conf:.3f}。这与模型切换后的图结构变化同时发生，"
        "不能仅凭该比例解释评分改善。".format(
            old_conf=old_graphs[POOLED, 1]["root_conflict_rate"],
            new_conf=new_graphs[POOLED, 1]["root_conflict_rate"]),
        "", "## 各数据集", "",
    ]
    for dataset in DATASETS:
        lines += ["### " + dataset, "",
                  "| 深度 | 方法 | Flash Acc | Pro Acc | ΔAcc | Flash Brier | Pro Brier | ΔBrier | ΔECE | 改对/改错 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for depth in (1, 2):
            for method in METHODS:
                row = lookup[dataset, depth, method]
                lines.append(
                    "| D={depth} | {method} | {old_acc:.4f} | {new_acc:.4f} | "
                    "{acc_delta:+.4f} | {old_brier:.4f} | {new_brier:.4f} | "
                    "{brier_delta:+.4f} | {ece_delta:+.4f} | {corrected}/{spoiled} |".format(
                        depth=depth, method=method,
                        old_acc=row["flash"]["accuracy"],
                        new_acc=row["pro"]["accuracy"],
                        acc_delta=row["accuracy_change"],
                        old_brier=row["flash"]["brier"],
                        new_brier=row["pro"]["brier"],
                        brier_delta=row["brier_change"],
                        ece_delta=row["ece_change"],
                        corrected=row["corrected"], spoiled=row["spoiled"]))
        lines.append("")
    lines += [
        "## 图结构变化", "",
        "| 数据集 | 深度 | Flash 平均论点数 | Pro 平均论点数 | Flash 平均边数 | Pro 平均边数 | Flash 根节点双向证据率 | Pro 根节点双向证据率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            old, new = old_graphs[dataset, depth], new_graphs[dataset, depth]
            lines.append(
                "| {dataset} | D={depth} | {old_args:.2f} | {new_args:.2f} | "
                "{old_edges:.2f} | {new_edges:.2f} | {old_conf:.3f} | {new_conf:.3f} |".format(
                    dataset=dataset, depth=depth,
                    old_args=old["mean_arguments"], new_args=new["mean_arguments"],
                    old_edges=old["mean_edges"], new_edges=new["mean_edges"],
                    old_conf=old["root_conflict_rate"],
                    new_conf=new["root_conflict_rate"]))
    lines += [
        "", "## Jev 根先验被论证修正的情况：Prior-EW-QBAF", "",
        "| 深度 | Jev 置信度桶 | n | Flash Rescue−Harm | Pro Rescue−Harm | Pro Rescue/Harm | Flash Brier 变化 | Pro Brier 变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for label in BIN_LABELS:
            old, new = old_bins[depth, label], new_bins[depth, label]
            lines.append(
                "| D={depth} | {label} | {n} | {old_net:+d} | {new_net:+d} | "
                "{rescue}/{harm} | {old_brier:+.4f} | {new_brier:+.4f} |".format(
                    depth=depth, label=label, n=new["n"],
                    old_net=old["rescue_minus_harm"],
                    new_net=new["rescue_minus_harm"],
                    rescue=new["rescue"], harm=new["harm"],
                    old_brier=old["brier_change"],
                    new_brier=new["brier_change"]))
    lines += [
        "", "## API 用量与耗时", "",
        "| 运行 | 唯一 API 调用数 | 生成模型输入/输出 token | Jev/API 成本 (USD) | 生成模型/API 成本 (CNY) | Jev 响应版本 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, key in (("Flash", "flash"), ("Pro 0813", "pro")):
        run = result["run_metrics"][key]
        lines.append(
            "| {label} | {calls} | {tokens} | ${usd} | {cny} | {models} |".format(
                label=label, calls=run["unique_api_calls"],
                tokens=(str(run["unique_api_tokens"]["llm_input"]) + " / "
                        + str(run["unique_api_tokens"]["llm_output"]))
                if "unique_api_tokens" in run else "未记录",
                usd=number(run["unique_api_cost_usd"], 6),
                cny=("未计价" if run["unique_api_cost_cny"] is None else
                     "￥" + number(run["unique_api_cost_cny"], 6)),
                models=", ".join(run["jev_response_models"])))
    lines += [
        "", "| 深度 | 方法 | Flash 平均延迟 (秒/样本) | Pro 平均延迟 (秒/样本) |",
        "|---|---|---:|---:|",
    ]
    for depth in (1, 2):
        for method in METHODS:
            values = []
            for key in ("flash", "pro"):
                group = [row for row in result["run_metrics"][key]["summary"]
                         if row["depth"] == depth and row["method"] == method]
                values.append(sum(row["n"] * row["mean_latency_seconds"]
                                  for row in group) / sum(row["n"] for row in group))
            lines.append("| D={depth} | {method} | {flash:.3f} | {pro:.3f} |".format(
                depth=depth, method=method, flash=values[0], pro=values[1]))
    lines += [
        "", "费用按各运行完整执行一次的逻辑用量统计；两组共享的 Direct Jev 根请求"
        "在两个运行中各计一次，实际第二次执行时命中缓存。"
        "生成模型费用以实验指定的 DMX 单价乘服务端 token 用量估算；"
        "若未获得该模型的确认单价，暂列为未计价。"
        "详细的每方法 API 费用和平均延迟见各自的实验报告与指标文件。", "",
        "## 解释边界", "",
        "同一组方法在各深度内使用完全相同的生成图；Flash 与 Pro 之间的图、论点文本"
        "和论点估分都可能改变。Jev 论点及边概率随文本变化而重新请求，"
        "因此结果衡量的是完整生成模型替换对流水线的影响，"
        "不能识别其中某一个环节的单独因果贡献。", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-dir", type=Path,
                        default=Path("experiment_results/prior_full"))
    parser.add_argument("--pro-dir", type=Path,
                        default=Path("experiment_results/pro_0813_full"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/2026-09-27_生成模型Flash与Pro0813对比.md"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("reports/2026-09-27_生成模型Flash与Pro0813对比.json"))
    args = parser.parse_args()
    flash_metrics, flash = load_results(args.flash_dir)
    pro_metrics, pro = load_results(args.pro_dir)
    for field in ("llm_base_url", "llm_thinking", "llm_currency", "jev_model",
                  "jev_input_price_per_million",
                  "breadth", "temperature", "top_p", "max_new_tokens",
                  "max_samples"):
        old = flash_metrics["settings"].get(field)
        new = pro_metrics["settings"].get(field)
        if old != new:
            raise ValueError("Experiment setting changed: " + field)
    pairs = aligned_pairs(flash, pro)
    comparisons = [
        compare_subset(pairs, dataset, depth, method)
        for dataset in (POOLED,) + tuple(DATASETS)
        for depth in (1, 2)
        for method in METHODS
    ]
    result = {
        "flash_dir": str(args.flash_dir),
        "pro_dir": str(args.pro_dir),
        "paired_sample_depth_rows": len(pairs),
        "comparisons": comparisons,
        "graph_summary": {
            "flash": summarize_graphs(flash),
            "pro": summarize_graphs(pro),
        },
        "confidence_bins": {
            "flash": analyze(list(flash.values()))["buckets"],
            "pro": analyze(list(pro.values()))["buckets"],
        },
        "run_metrics": {
            "flash": {key: flash_metrics[key] for key in (
                "unique_api_calls", "unique_api_cost_usd", "unique_api_cost_cny",
                "jev_response_models", "settings", "summary")},
            "pro": {key: pro_metrics[key] for key in (
                "unique_api_calls", "unique_api_cost_usd", "unique_api_cost_cny",
                "jev_response_models", "settings", "summary")},
        },
    }
    for key, run in (("flash", flash_metrics), ("pro", pro_metrics)):
        if "unique_api_tokens" in run:
            result["run_metrics"][key]["unique_api_tokens"] = run["unique_api_tokens"]
    write_json(args.metrics, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result, args.flash_dir, args.pro_dir),
                           encoding="utf-8")
    print("中文模型对比报告：" + str(args.report))


if __name__ == "__main__":
    main()
