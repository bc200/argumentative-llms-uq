"""Compare the five generator runs on identical sample-depth rows in Chinese."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analyze_jev_confidence_bins import BIN_LABELS, POOLED, analyze
from benchmark_jev_qbaf import DATASETS, METHODS, metrics
from compare_generator_models import (
    aligned_pairs, compare_subset, load_results, summarize_graphs,
)
from compare_three_generators import COMPARABLE_SETTINGS
from experiment_io import write_json


RUN_NAMES = ("Flash", "Pro 0813", "Qwen 3.8 Max", "Qwen 3.8 27B", "Qwen3 8B")
TARGET = RUN_NAMES[-1]
PAIRED_BASELINES = ("Qwen 3.8 Max", "Qwen 3.8 27B")


def _quality(rows):
    result = []
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            subset = [row for row in rows.values()
                      if row["depth"] == depth
                      and (dataset == POOLED or row["dataset"] == dataset)]
            labels = [row["valid"] for row in subset]
            for method in METHODS:
                result.append({"dataset": dataset, "depth": depth,
                               "method": method, "n": len(subset),
                               **metrics(labels, [row["predictions"][method]
                                                  for row in subset])})
    return result


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
        recorded = {(r["dataset"], r["depth"], r["index"])
                    for r in runs[name][0].get("rejected_samples", [])}
        if not keys <= universe or universe - keys != recorded:
            raise ValueError(name + " 的缺失样本与服务商拒绝记录不一致")
    common = set.intersection(*(set(rows) for _, rows in runs.values()))
    if not common:
        raise ValueError("没有五模型共同样本")
    matched = {name: {key: row for key, row in runs[name][1].items()
                      if key in common} for name in RUN_NAMES}
    # This also checks that claims, labels, seven methods and Direct Jev agree.
    pairs = {name: aligned_pairs(matched[name], matched[TARGET])
             for name in RUN_NAMES[:-1]}
    return {
        "run_dirs": {name: str(run_dirs[name]) for name in RUN_NAMES},
        "matched_sample_depth_rows": len(common),
        "matched_n_by_depth": {depth: sum(key[1] == depth for key in common)
                               for depth in (1, 2)},
        "coverage": {name: {"successful_rows": len(runs[name][1]),
                            "rejected_samples": runs[name][0].get(
                                "rejected_samples", [])} for name in RUN_NAMES},
        "quality": {name: _quality(matched[name]) for name in RUN_NAMES},
        "paired_vs_8b": {
            name: [compare_subset(pairs[name], POOLED, depth, method)
                   for depth in (1, 2) for method in METHODS]
            for name in PAIRED_BASELINES
        },
        "graph_summary": {name: summarize_graphs(matched[name])
                          for name in RUN_NAMES},
        "confidence_bins": {name: analyze(list(matched[name].values()))["buckets"]
                            for name in RUN_NAMES},
        "run_metrics": {name: {key: runs[name][0].get(key) for key in (
            "unique_api_calls", "unique_api_tokens", "unique_api_cost_usd",
            "unique_api_cost_cny", "jev_response_models",
            "direct_prompt_fallbacks", "settings", "summary",
        )} for name in RUN_NAMES},
    }


def _latency(run, depth, method):
    rows = [row for row in run["summary"]
            if row["depth"] == depth and row["method"] == method]
    return sum(r["n"] * r["mean_latency_seconds"] for r in rows) / sum(
        r["n"] for r in rows)


def render_report(result):
    quality = {name: {(r["dataset"], r["depth"], r["method"]): r
                      for r in result["quality"][name]} for name in RUN_NAMES}
    paired = {name: {(r["depth"], r["method"]): r
                     for r in result["paired_vs_8b"][name]}
              for name in PAIRED_BASELINES}
    graphs = {name: {(r["dataset"], r["depth"]): r
                     for r in result["graph_summary"][name]}
              for name in RUN_NAMES}
    bins = {name: {(r["depth"], r["confidence_bin"]): r
                   for r in result["confidence_bins"][name]
                   if r["dataset"] == POOLED} for name in RUN_NAMES}
    counts = result["matched_n_by_depth"]
    n1, n2 = (counts.get(d, counts.get(str(d))) for d in (1, 2))
    lines = [
        "# Qwen3 8B 与四种生成模型的七方法配对比较", "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"), "",
        "## 设置与配对范围", "",
        "- 五组均使用 TruthfulClaim、StrategyClaim、MedClaim 的上游 Experiment "
        "全集各 500 条和 D=1/D=2，七种方法、相同生成提示与 QBAF 实现。"
        "同一模型内所有 QBAF 方法共用缓存图，五组复用逐样本完全相同的 Direct Jev 根概率。",
        "- 生成模型经 DMX 调用，均关闭思考，temperature=0.7、top_p=0.95、"
        "max_new_tokens=128。Qwen3 8B 和 Max 使用 `enable_thinking=false`；"
        "Qwen 3.8 27B 使用 `reasoning_effort=none`。每个提示仅采样一次，"
        "接口未固定采样种子。",
        "- 准确率以概率严格大于 0.5 判真。Brier 为平方误差均值；ECE 为"
        "10 个等宽分桶的校准误差。配对准确率差的 95% 区间对同一论点做 1000 次"
        "重抽样，未经多重比较校正。D=1/D=2 共享根论点，分别统计。",
    ]
    for name in RUN_NAMES:
        row = result["coverage"][name]
        lines.append("- {name}：成功 {n}/3000 条样本深度结果。".format(
            name=name, n=row["successful_rows"]))
        for rejected in row["rejected_samples"]:
            lines.append("  - 服务商拒绝：{dataset} D={depth} 样本 {index}。".format(
                **rejected))
    lines += [
        "- 五模型共同样本：D=1 为 n=%d，D=2 为 n=%d；以下模型间比较均在共同样本上重算。"
        % (n1, n2), "",
        "## 三数据集合计", "",
        "| 深度 | 方法 | Flash Acc | Pro Acc | Max Acc | 27B Acc | 8B Acc | "
        "8B−27B ΔAcc [95% CI] | 8B−Max ΔAcc [95% CI] | 8B Brier | 8B ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for method in METHODS:
            q = [quality[name][POOLED, depth, method] for name in RUN_NAMES]
            small = paired["Qwen 3.8 27B"][depth, method]
            large = paired["Qwen 3.8 Max"][depth, method]
            sc, lc = small["accuracy_change_ci95"], large["accuracy_change_ci95"]
            lines.append(
                "| D={d} | {method} | {f:.4f} | {p:.4f} | {mx:.4f} | "
                "{b:.4f} | {s:.4f} | {ds:+.4f} [{s0:+.4f},{s1:+.4f}] | "
                "{dl:+.4f} [{l0:+.4f},{l1:+.4f}] | {br:.4f} | {ece:.4f} |".format(
                    d=depth, method=method, f=q[0]["accuracy"],
                    p=q[1]["accuracy"], mx=q[2]["accuracy"],
                    b=q[3]["accuracy"], s=q[4]["accuracy"],
                    ds=small["accuracy_change"], s0=sc[0], s1=sc[1],
                    dl=large["accuracy_change"], l0=lc[0], l1=lc[1],
                    br=q[4]["brier"], ece=q[4]["ece"],
                ))
    lines += ["", "## 主要观察", ""]
    for depth in (1, 2):
        direct = quality[TARGET][POOLED, depth, "direct_llm"]
        baseline = quality[TARGET][POOLED, depth, "direct_jev"]
        prior = quality[TARGET][POOLED, depth, "jev_prior_ew_qbaf"]
        dp = paired["Qwen 3.8 27B"][depth, "direct_llm"]
        pp = paired["Qwen 3.8 27B"][depth, "jev_prior_ew_qbaf"]
        lines.append(
            "- D={d}：8B direct_llm Accuracy {direct:.4f}，相对 27B {dd:+.4f}；"
            "Jev-Prior-EW-QBAF Accuracy {prior:.4f}，相对 27B {pd:+.4f}、"
            "相对共享 Direct Jev {base:.4f} 为 {db:+.4f}。"
            "其 Brier {brier:.4f}，Direct Jev Brier {base_brier:.4f}。".format(
                d=depth, direct=direct["accuracy"], dd=dp["accuracy_change"],
                prior=prior["accuracy"], pd=pp["accuracy_change"],
                base=baseline["accuracy"],
                db=prior["accuracy"] - baseline["accuracy"],
                brier=prior["brier"], base_brier=baseline["brier"],
            ))
    direct_27 = paired["Qwen 3.8 27B"][1, "direct_llm"]
    direct_max = paired["Qwen 3.8 Max"][1, "direct_llm"]
    orig_27 = [paired["Qwen 3.8 27B"][d, "original_qbaf"] for d in (1, 2)]
    prior_27 = [paired["Qwen 3.8 27B"][d, "jev_prior_ew_qbaf"]
                for d in (1, 2)]
    prior_8 = [quality[TARGET][POOLED, d, "jev_prior_ew_qbaf"]
               for d in (1, 2)]
    root_8 = [quality[TARGET][POOLED, d, "direct_jev"] for d in (1, 2)]
    lines.append(
        "- 8B direct_llm 相对 27B 的准确率差为 {d27:+.4f} "
        "[{d27lo:+.4f},{d27hi:+.4f}]，相对 Max 为 {dm:+.4f} "
        "[{dmlo:+.4f},{dmhi:+.4f}]；original_qbaf 相对 27B 在 D=1/D=2 "
        "分别差 {o1:+.4f}/{o2:+.4f}。".format(
            d27=direct_27["accuracy_change"],
            d27lo=direct_27["accuracy_change_ci95"][0],
            d27hi=direct_27["accuracy_change_ci95"][1],
            dm=direct_max["accuracy_change"],
            dmlo=direct_max["accuracy_change_ci95"][0],
            dmhi=direct_max["accuracy_change_ci95"][1],
            o1=orig_27[0]["accuracy_change"],
            o2=orig_27[1]["accuracy_change"],
        ))
    lines.append(
        "- Jev-Prior-EW-QBAF 相对 27B 的差值在 D=1/D=2 为 "
        "{one:+.4f}/{two:+.4f}，对应配对区间分别为 "
        "[{o0:+.4f},{o1:+.4f}]、[{t0:+.4f},{t1:+.4f}]。"
        "8B 的该方法相对共享 Direct Jev 的 Accuracy 差为 "
        "{a1:+.4f}/{a2:+.4f}，Brier 差为 {b1:+.4f}/{b2:+.4f}。".format(
            one=prior_27[0]["accuracy_change"],
            two=prior_27[1]["accuracy_change"],
            o0=prior_27[0]["accuracy_change_ci95"][0],
            o1=prior_27[0]["accuracy_change_ci95"][1],
            t0=prior_27[1]["accuracy_change_ci95"][0],
            t1=prior_27[1]["accuracy_change_ci95"][1],
            a1=prior_8[0]["accuracy"] - root_8[0]["accuracy"],
            a2=prior_8[1]["accuracy"] - root_8[1]["accuracy"],
            b1=prior_8[0]["brier"] - root_8[0]["brier"],
            b2=prior_8[1]["brier"] - root_8[1]["brier"],
        ))
    small_graph = graphs[TARGET][POOLED, 2]
    large_graph = graphs["Qwen 3.8 27B"][POOLED, 2]
    lines.append(
        "- D=2 的 8B 图平均有 {small:.2f} 个论点，27B 为 {large:.2f}；"
        "根节点双向证据率分别是 {sc:.1%} 和 {lc:.1%}。"
        "这表示生成的图结构也发生了变化，不代表论点质量提高。".format(
            small=small_graph["mean_arguments"],
            large=large_graph["mean_arguments"],
            sc=small_graph["root_conflict_rate"],
            lc=large_graph["root_conflict_rate"],
        ))
    lines += ["", "## 各数据集准确率", ""]
    for dataset in DATASETS:
        lines += ["### " + dataset, "",
                  "| 深度 | 方法 | Flash | Pro | Max | 27B | 8B | 8B−27B | 8B Brier | 8B ECE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for depth in (1, 2):
            for method in METHODS:
                q = [quality[name][dataset, depth, method] for name in RUN_NAMES]
                lines.append(
                    "| D={d} | {m} | {f:.4f} | {p:.4f} | {mx:.4f} | "
                    "{b:.4f} | {s:.4f} | {delta:+.4f} | {br:.4f} | {ece:.4f} |".format(
                        d=depth, m=method, f=q[0]["accuracy"],
                        p=q[1]["accuracy"], mx=q[2]["accuracy"],
                        b=q[3]["accuracy"], s=q[4]["accuracy"],
                        delta=q[4]["accuracy"] - q[3]["accuracy"],
                        br=q[4]["brier"], ece=q[4]["ece"],
                    ))
        lines.append("")
    lines += ["## 图结构", "",
              "| 数据集 | 深度 | 模型 | 平均论点数 | 平均边数 | 根节点双向证据率 |",
              "|---|---|---|---:|---:|---:|"]
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            for name in RUN_NAMES:
                row = graphs[name][dataset, depth]
                lines.append("| {ds} | D={d} | {name} | {a:.2f} | {e:.2f} | {c:.3f} |".format(
                    ds=dataset, d=depth, name=name,
                    a=row["mean_arguments"], e=row["mean_edges"],
                    c=row["root_conflict_rate"]))
    lines += ["", "## Jev 决策置信度分桶：Prior-EW 的净纠错", "",
              "| 深度 | 置信度桶 | n | Flash Rescue−Harm | Pro | Max | 27B | 8B | "
              "8B Rescue/Harm | 8B 翻转率 | 8B Brier 变化 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for depth in (1, 2):
        for label in BIN_LABELS:
            q = [bins[name][depth, label] for name in RUN_NAMES]
            lines.append(
                "| D={d} | {label} | {n} | {f:+d} | {p:+d} | {mx:+d} | "
                "{b:+d} | {s:+d} | {rescue}/{harm} | {flip:.3f} | {br:+.4f} |".format(
                    d=depth, label=label, n=q[4]["n"],
                    f=q[0]["rescue_minus_harm"],
                    p=q[1]["rescue_minus_harm"],
                    mx=q[2]["rescue_minus_harm"],
                    b=q[3]["rescue_minus_harm"],
                    s=q[4]["rescue_minus_harm"],
                    rescue=q[4]["rescue"], harm=q[4]["harm"],
                    flip=q[4]["flip_rate"], br=q[4]["brier_change"],
                ))
    lines += ["", "## API 用量与方法延迟", "",
              "本节取各模型原始全量运行的计费与延迟记录；上方准确率表则在"
              "五模型共同样本上重算。",
              "",
              "| 模型 | 唯一 API 调用 | LLM 输入/输出 token | Jev 成本 USD | "
              "LLM 成本 CNY | 并发样本数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in RUN_NAMES:
        run = result["run_metrics"][name]
        tokens = run["unique_api_tokens"]
        cny = run["unique_api_cost_cny"]
        lines.append(
            "| {name} | {calls} | {inp:,}/{out:,} | ${usd:.6f} | {cny} | {workers} |".format(
                name=name, calls=run["unique_api_calls"],
                inp=tokens["llm_input"], out=tokens["llm_output"],
                usd=run["unique_api_cost_usd"],
                cny="未计价" if cny is None else "￥%.6f" % cny,
                workers=run["settings"]["workers"],
            ))
    lines += ["", "| 深度 | 方法 | Flash 秒/样本 | Pro 秒/样本 | Max 秒/样本 | "
              "27B 秒/样本 | 8B 秒/样本 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for depth in (1, 2):
        for method in METHODS:
            v = [_latency(result["run_metrics"][name], depth, method)
                 for name in RUN_NAMES]
            lines.append("| D={d} | {m} | {f:.3f} | {p:.3f} | {mx:.3f} | "
                         "{b:.3f} | {s:.3f} |".format(
                             d=depth, m=method, f=v[0], p=v[1],
                             mx=v[2], b=v[3], s=v[4]))
    lines += [
        "", "8B 人民币成本按 [DMX 公开价格表]"
        "(https://rmb.dmxapi.cn/?api=model_prices) 的输入 ¥0.5、输出 ¥2／百万"
        " token 估算，未扣除可能的优惠。Jev 成本单独以美元列示，未换算汇率。"
        "共享 Direct Jev 根请求在各运行的逻辑用量中均计一次；8B 实际从已有缓存"
        "复用这些根响应。Pro 0813 单价未确认，保留 token 用量并标记未计价。",
        "", "## 解释边界", "",
        "模型变化同时改变论点文本、图结构和随之触发的 Jev 评分。"
        "这些差异衡量完整流水线的模型替换效果。每个模型仅采样一次，"
        "跨模型延迟也受服务负载影响。配对区间未作多重比较校正。", "",
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
    parser.add_argument("--b27-dir", type=Path,
                        default=Path("experiment_results/qwen38_27b_full"))
    parser.add_argument("--b8-dir", type=Path,
                        default=Path("experiment_results/qwen3_8b_full"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/2026-09-27_qwen3_8b_与四模型对比.md"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("reports/2026-09-27_qwen3_8b_与四模型对比.json"))
    args = parser.parse_args()
    result = make_comparison(dict(zip(RUN_NAMES, (
        args.flash_dir, args.pro_dir, args.max_dir, args.b27_dir, args.b8_dir,
    ))))
    write_json(args.metrics, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print("中文五模型对比报告：" + str(args.report))


if __name__ == "__main__":
    main()
