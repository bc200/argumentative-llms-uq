"""Compare qwen3.8-27b with Flash, Pro 0813 and Qwen3.8 Max in Chinese."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analyze_jev_confidence_bins import BIN_LABELS, POOLED, analyze
from benchmark_jev_qbaf import DATASETS, METHODS
from compare_generator_models import (
    aligned_pairs, compare_subset, load_results, summarize_graphs,
)
from compare_three_generators import COMPARABLE_SETTINGS
from experiment_io import write_json


RUN_NAMES = ("Flash", "Pro 0813", "Qwen 3.8 Max", "Qwen 3.8 27B")
TARGET = RUN_NAMES[-1]


def make_comparison(run_dirs):
    runs = {name: load_results(run_dirs[name]) for name in RUN_NAMES}
    reference = runs["Flash"][0]["settings"]
    for name in RUN_NAMES[1:]:
        for field in COMPARABLE_SETTINGS:
            if reference.get(field) != runs[name][0]["settings"].get(field):
                raise ValueError("实验设置不一致：%s (%s)" % (field, name))
    universe = set(runs["Flash"][1])
    if len(universe) != 3000 or set(runs["Pro 0813"][1]) != universe:
        raise ValueError("Flash 与 Pro 必须各有相同的 3000 条样本深度结果")
    for name in RUN_NAMES[2:]:
        keys = set(runs[name][1])
        if not keys <= universe:
            raise ValueError(name + " 有对照数据集以外的样本")
        recorded = {
            (r["dataset"], r["depth"], r["index"])
            for r in runs[name][0].get("rejected_samples", [])
        }
        if universe - keys != recorded:
            raise ValueError(name + " 缺失样本与服务商拒绝记录不一致")
    common = set.intersection(*(set(rows) for _, rows in runs.values()))
    if not common:
        raise ValueError("没有可配对的四模型共同样本")
    matched = {
        name: {key: row for key, row in runs[name][1].items() if key in common}
        for name in RUN_NAMES
    }
    pairings = {
        name: aligned_pairs(matched[name], matched[TARGET])
        for name in RUN_NAMES[:-1]
    }
    return {
        "run_dirs": {name: str(run_dirs[name]) for name in RUN_NAMES},
        "matched_sample_depth_rows": len(common),
        "matched_n_by_depth": {
            depth: sum(key[1] == depth for key in common) for depth in (1, 2)
        },
        "coverage": {
            name: {
                "successful_rows": len(runs[name][1]),
                "rejected_samples": runs[name][0].get("rejected_samples", []),
            } for name in RUN_NAMES
        },
        "paired_vs_27b": {
            name: [
                compare_subset(pairings[name], dataset, depth, method)
                for dataset in (POOLED,) + tuple(DATASETS)
                for depth in (1, 2)
                for method in METHODS
            ] for name in RUN_NAMES[:-1]
        },
        "graph_summary": {
            name: summarize_graphs(matched[name]) for name in RUN_NAMES
        },
        "confidence_bins": {
            name: analyze(list(matched[name].values()))["buckets"]
            for name in RUN_NAMES
        },
        "run_metrics": {
            name: {key: runs[name][0].get(key) for key in (
                "unique_api_calls", "unique_api_tokens", "unique_api_cost_usd",
                "unique_api_cost_cny", "jev_response_models",
                "direct_prompt_fallbacks", "settings", "summary",
            )} for name in RUN_NAMES
        },
    }


def _lookup(result, name):
    return {(r["dataset"], r["depth"], r["method"]): r
            for r in result["paired_vs_27b"][name]}


def _latency(run, depth, method):
    rows = [row for row in run["summary"]
            if row["depth"] == depth and row["method"] == method]
    return sum(r["n"] * r["mean_latency_seconds"] for r in rows) / sum(
        r["n"] for r in rows
    )


def render_report(result):
    lookup = {name: _lookup(result, name) for name in RUN_NAMES[:-1]}
    graphs = {
        name: {(r["dataset"], r["depth"]): r
               for r in result["graph_summary"][name]}
        for name in RUN_NAMES
    }
    bins = {
        name: {(r["depth"], r["confidence_bin"]): r
               for r in result["confidence_bins"][name]
               if r["dataset"] == POOLED}
        for name in RUN_NAMES
    }
    lines = [
        "# Qwen 3.8 27B 与三种生成模型的七方法配对比较", "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"), "",
        "## 对照设置与覆盖范围", "",
        "- 四组均使用三个上游 Experiment 全集，每个数据集 500 条、D=1/D=2，"
        "七种方法及相同的数据、标签、论点生成提示、DF-QuAD/EW-DF-QuAD 实现。"
        "同一模型内的 QBAF 方法共用缓存图；四组使用逐样本相同的 Direct Jev 根概率。",
        "- 生成模型均经 DMX，关闭思考、temperature=0.7、top_p=0.95、"
        "输出上限 128 token。27B 使用 `reasoning_effort=none`；"
        "Qwen Max 使用 `enable_thinking=false`。每个提示只抽样一次，"
        "接口没有固定采样种子。",
        "- 表中 Δ 为 27B 减去对照模型。Accuracy 越高越好；Brier、ECE 越低越好。"
        "准确率差的 95% 区间在相同论点上做 1000 次配对重抽样，"
        "未作多重比较校正。D=1 与 D=2 共用根论点，各深度分别统计。",
    ]
    for name in RUN_NAMES:
        coverage = result["coverage"][name]
        lines.append("- {name}：成功 {n}/3000 条样本深度结果。".format(
            name=name, n=coverage["successful_rows"]
        ))
        for row in coverage["rejected_samples"]:
            lines.append("  - 服务商拒绝：{dataset} D={depth} 样本 {index}。".format(**row))
    lines += [
        "- 四模型共同样本：D=1 为 n={n1}，D=2 为 n={n2}；"
        "下列表格均在这一共同集合上重算，以保持配对口径一致。".format(
            n1=result["matched_n_by_depth"].get(1, result["matched_n_by_depth"].get("1")),
            n2=result["matched_n_by_depth"].get(2, result["matched_n_by_depth"].get("2")),
        ), "",
        "## 三数据集合计", "",
        "| 深度 | 方法 | Flash Acc | Pro Acc | Max Acc | 27B Acc | "
        "27B−Max ΔAcc [95% CI] | 27B−Pro ΔAcc [95% CI] | 27B Brier | 27B ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for method in METHODS:
            f, p, m = (lookup[name][POOLED, depth, method]
                       for name in RUN_NAMES[:-1])
            q = m["pro"]
            mci, pci = m["accuracy_change_ci95"], p["accuracy_change_ci95"]
            lines.append(
                "| D={d} | {method} | {flash:.4f} | {pro:.4f} | {max:.4f} | "
                "{qwen:.4f} | {dm:+.4f} [{m0:+.4f},{m1:+.4f}] | "
                "{dp:+.4f} [{p0:+.4f},{p1:+.4f}] | {brier:.4f} | {ece:.4f} |".format(
                    d=depth, method=method,
                    flash=f["flash"]["accuracy"], pro=p["flash"]["accuracy"],
                    max=m["flash"]["accuracy"], qwen=q["accuracy"],
                    dm=m["accuracy_change"], m0=mci[0], m1=mci[1],
                    dp=p["accuracy_change"], p0=pci[0], p1=pci[1],
                    brier=q["brier"], ece=q["ece"],
                )
            )
    lines += ["", "## 主要观察", ""]
    direct = lookup["Qwen 3.8 Max"][POOLED, 1, "direct_llm"]
    lines.append(
        "- 27B direct_llm 的 Accuracy 为 {small:.4f}，Max 为 {large:.4f}，"
        "差值 {delta:+.4f}；27B Brier 为 {brier:.4f}，Max 为 {large_brier:.4f}。".format(
            small=direct["pro"]["accuracy"], large=direct["flash"]["accuracy"],
            delta=direct["accuracy_change"], brier=direct["pro"]["brier"],
            large_brier=direct["flash"]["brier"],
        )
    )
    for depth in (1, 2):
        original = lookup["Qwen 3.8 Max"][POOLED, depth, "original_qbaf"]
        prior = lookup["Qwen 3.8 Max"][POOLED, depth, "jev_prior_ew_qbaf"]
        jev = lookup["Qwen 3.8 Max"][POOLED, depth, "direct_jev"]["pro"]
        lines.append(
            "- D={d}：27B original_qbaf 为 {orig:.4f}（相对 Max {orig_delta:+.4f}）；"
            "Jev-Prior-EW-QBAF 为 {prior:.4f}（相对 Max {prior_delta:+.4f}），"
            "相对固定 Direct Jev {jev:.4f} 为 {jev_delta:+.4f}。".format(
                d=depth, orig=original["pro"]["accuracy"],
                orig_delta=original["accuracy_change"],
                prior=prior["pro"]["accuracy"],
                prior_delta=prior["accuracy_change"],
                jev=jev["accuracy"],
                jev_delta=prior["pro"]["accuracy"] - jev["accuracy"],
            )
        )
    originals = [lookup["Qwen 3.8 Max"][POOLED, depth, "original_qbaf"]
                 for depth in (1, 2)]
    priors = [lookup["Qwen 3.8 Max"][POOLED, depth, "jev_prior_ew_qbaf"]
              for depth in (1, 2)]
    lines.append(
        "- 在共同样本上，27B 的 direct_llm 相对 Max 差 {direct:+.2f} 个百分点；"
        "original_qbaf 在 D=1/D=2 分别差 {one:+.2f}/{two:+.2f} 个百分点。"
        "这些准确率差的配对区间均未跨 0。".format(
            direct=100 * direct["accuracy_change"],
            one=100 * originals[0]["accuracy_change"],
            two=100 * originals[1]["accuracy_change"],
        )
    )
    lines.append(
        "- Jev-Prior-EW-QBAF 相对 Max 在 D=1/D=2 分别差 {one:+.2f}/{two:+.2f} "
        "个百分点，但区间都跨 0；27B 该方法与共享的 Direct Jev 基线接近，"
        "且 Brier 在两个深度都高于 Direct Jev。现有数据未证明结构化论证能稳定改善 Jev 先验。".format(
            one=100 * priors[0]["accuracy_change"],
            two=100 * priors[1]["accuracy_change"],
        )
    )
    lines.append(
        "- 27B 的生成模型估算费用为 ￥{small:.2f}，Max 为 ￥{large:.2f}；"
        "两组图与响应产生的 token 和调用数不同，不能将费用差仅解释为单价差。".format(
            small=result["run_metrics"][TARGET]["unique_api_cost_cny"],
            large=result["run_metrics"]["Qwen 3.8 Max"]["unique_api_cost_cny"],
        )
    )
    small_graph = graphs[TARGET][POOLED, 2]
    max_graph = graphs["Qwen 3.8 Max"][POOLED, 2]
    lines.append(
        "- D=2 的平均论点数为 27B {small_args:.2f}、Max {max_args:.2f}；"
        "根节点双向证据率为 {small_conflict:.1%}、{max_conflict:.1%}。"
        "生成模型改变了可用证据的结构，不能将方法差异仅归于概率打分。".format(
            small_args=small_graph["mean_arguments"],
            max_args=max_graph["mean_arguments"],
            small_conflict=small_graph["root_conflict_rate"],
            max_conflict=max_graph["root_conflict_rate"],
        )
    )
    lines += ["", "## 各数据集准确率", ""]
    for dataset in DATASETS:
        lines += ["### " + dataset, "",
                  "| 深度 | 方法 | Flash | Pro | Max | 27B | 27B−Max | 27B Brier | 27B ECE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for depth in (1, 2):
            for method in METHODS:
                f, p, m = (lookup[name][dataset, depth, method]
                           for name in RUN_NAMES[:-1])
                lines.append(
                    "| D={d} | {method} | {f:.4f} | {p:.4f} | {m:.4f} | "
                    "{q:.4f} | {delta:+.4f} | {brier:.4f} | {ece:.4f} |".format(
                        d=depth, method=method,
                        f=f["flash"]["accuracy"],
                        p=p["flash"]["accuracy"],
                        m=m["flash"]["accuracy"],
                        q=m["pro"]["accuracy"], delta=m["accuracy_change"],
                        brier=m["pro"]["brier"], ece=m["pro"]["ece"],
                    )
                )
        lines.append("")
    lines += [
        "## 图结构", "",
        "| 数据集 | 深度 | 模型 | 平均论点数 | 平均边数 | 根节点双向证据率 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            for name in RUN_NAMES:
                row = graphs[name][dataset, depth]
                lines.append("| {ds} | D={d} | {name} | {a:.2f} | {e:.2f} | "
                             "{c:.3f} |".format(
                                 ds=dataset, d=depth, name=name,
                                 a=row["mean_arguments"], e=row["mean_edges"],
                                 c=row["root_conflict_rate"],
                             ))
    lines += [
        "", "## Jev 决策置信度分桶：Prior-EW 的净纠错", "",
        "| 深度 | 置信度桶 | n | Flash Rescue−Harm | Pro | Max | 27B | "
        "27B Rescue/Harm | 27B 翻转率 | 27B Brier 变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for label in BIN_LABELS:
            a, b, c, d = (bins[name][depth, label] for name in RUN_NAMES)
            lines.append(
                "| D={depth} | {label} | {n} | {fa:+d} | {pr:+d} | "
                "{ma:+d} | {sm:+d} | {rescue}/{harm} | {flip:.3f} | "
                "{brier:+.4f} |".format(
                    depth=depth, label=label, n=d["n"],
                    fa=a["rescue_minus_harm"], pr=b["rescue_minus_harm"],
                    ma=c["rescue_minus_harm"], sm=d["rescue_minus_harm"],
                    rescue=d["rescue"], harm=d["harm"],
                    flip=d["flip_rate"], brier=d["brier_change"],
                )
            )
    lines += [
        "", "## API 用量与平均方法延迟", "",
        "| 模型 | 唯一 API 调用 | LLM 输入/输出 token | Jev 成本 USD | "
        "LLM 成本 CNY | 并发样本数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in RUN_NAMES:
        run = result["run_metrics"][name]
        tokens = run["unique_api_tokens"]
        cny = run["unique_api_cost_cny"]
        lines.append(
            "| {name} | {calls} | {inp:,}/{out:,} | ${usd:.6f} | {cny} | "
            "{workers} |".format(
                name=name, calls=run["unique_api_calls"],
                inp=tokens["llm_input"], out=tokens["llm_output"],
                usd=run["unique_api_cost_usd"],
                cny="未计价" if cny is None else "￥%.6f" % cny,
                workers=run["settings"]["workers"],
            )
        )
    lines += ["", "| 深度 | 方法 | Flash 秒/样本 | Pro 秒/样本 | "
              "Max 秒/样本 | 27B 秒/样本 |",
              "|---|---|---:|---:|---:|---:|"]
    for depth in (1, 2):
        for method in METHODS:
            values = [_latency(result["run_metrics"][name], depth, method)
                      for name in RUN_NAMES]
            lines.append("| D={d} | {m} | {f:.3f} | {p:.3f} | "
                         "{mx:.3f} | {s:.3f} |".format(
                             d=depth, m=method, f=values[0], p=values[1],
                             mx=values[2], s=values[3],
                         ))
    lines += [
        "", "27B 人民币成本按 [DMX 价格表](https://rmb.dmxapi.cn/?api=model_prices) 的"
        "输入 ¥3、输出 ¥12／百万 token 估算，未扣除可能的输入缓存优惠；"
        "拒绝请求无 token 回执，未计价。Jev 成本以美元单独列示，未换算汇率。"
        "共享 Direct Jev 根请求在各运行的逻辑用量中均计一次，"
        "27B 实际从已有缓存复用这些根响应。Pro 0813 单价未确认，"
        "继续保留 token 用量并标记未计价。",
        "", "## 解释边界", "",
        "生成模型变化会同时改变论点文本、图结构以及由此触发的 Jev 评分。"
        "这些结果测量完整流水线的模型替换效果，不能单独识别其中一个环节的贡献。"
        "各模型仅采样一次；跨模型延迟比较还受并发数和服务负载影响。", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-dir", type=Path,
                        default=Path("experiment_results/prior_full"))
    parser.add_argument("--pro-dir", type=Path,
                        default=Path("experiment_results/pro_0813_full"))
    parser.add_argument("--max-dir", type=Path,
                        default=Path("experiment_results/qwen38_max_full"))
    parser.add_argument("--small-dir", type=Path,
                        default=Path("experiment_results/qwen38_27b_full"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/2026-09-27_qwen38_27b_与三模型对比.md"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("reports/2026-09-27_qwen38_27b_与三模型对比.json"))
    args = parser.parse_args()
    result = make_comparison(dict(zip(RUN_NAMES, (
        args.flash_dir, args.pro_dir, args.max_dir, args.small_dir,
    ))))
    write_json(args.metrics, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print("中文四模型对比报告：" + str(args.report))


if __name__ == "__main__":
    main()
