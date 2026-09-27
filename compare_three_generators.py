"""Generate a paired Chinese comparison of Flash, Pro 0813 and Qwen 3.8 Max."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from analyze_jev_confidence_bins import BIN_LABELS, POOLED, analyze
from benchmark_jev_qbaf import DATASETS, METHODS
from compare_generator_models import (
    aligned_pairs, compare_subset, load_results, summarize_graphs,
)
from experiment_io import write_json


RUN_NAMES = ("Flash", "Pro 0813", "Qwen 3.8 Max")
COMPARABLE_SETTINGS = (
    "llm_base_url", "llm_thinking", "jev_model", "jev_input_price_per_million",
    "breadth", "temperature", "top_p", "max_new_tokens", "max_samples",
)


def make_comparison(run_dirs):
    runs = {name: load_results(run_dirs[name]) for name in RUN_NAMES}
    reference_settings = runs["Flash"][0]["settings"]
    for name in RUN_NAMES[1:]:
        settings = runs[name][0]["settings"]
        for field in COMPARABLE_SETTINGS:
            if reference_settings.get(field) != settings.get(field):
                raise ValueError("实验设置不一致：%s (%s)" % (field, name))
    reference_keys = set(runs["Flash"][1])
    if len(reference_keys) != 3000 or set(runs["Pro 0813"][1]) != reference_keys:
        raise ValueError("Flash 与 Pro 必须各有相同的 3000 条样本深度结果")
    qwen_keys = set(runs["Qwen 3.8 Max"][1])
    if not qwen_keys <= reference_keys:
        raise ValueError("Qwen 存在对照实验之外的样本")
    missing = sorted(reference_keys - qwen_keys)
    refused = runs["Qwen 3.8 Max"][0].get("rejected_samples") or []
    if {(r["dataset"], r["depth"], r["index"]) for r in refused} != set(missing):
        raise ValueError("Qwen 缺失的样本与服务商拒绝记录不一致")
    if not qwen_keys:
        raise ValueError("Qwen 没有可配对的逐样本结果")
    common_rows = {
        name: {key: row for key, row in runs[name][1].items()
               if key in qwen_keys}
        for name in RUN_NAMES
    }
    pairings = {}
    for name in RUN_NAMES[:-1]:
        pairings[name] = aligned_pairs(common_rows[name],
                                       common_rows["Qwen 3.8 Max"])

    result = {
        "run_dirs": {name: str(run_dirs[name]) for name in RUN_NAMES},
        "paired_sample_depth_rows": len(qwen_keys),
        "excluded_provider_refusals": refused,
        "paired_n_by_depth": {
            depth: sum(key[1] == depth for key in qwen_keys)
            for depth in (1, 2)
        },
        "comparisons": {
            name: [
                compare_subset(pairings[name], dataset, depth, method)
                for dataset in (POOLED,) + tuple(DATASETS)
                for depth in (1, 2)
                for method in METHODS
            ] for name in RUN_NAMES[:-1]
        },
        "graph_summary": {
            name: summarize_graphs(common_rows[name]) for name in RUN_NAMES
        },
        "confidence_bins": {
            name: analyze(list(common_rows[name].values()))["buckets"]
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
    return result


def _number(value, digits=4):
    return "未计价" if value is None else ("%%.%df" % digits) % value


def _signed(value, digits=4):
    return ("%+.%df" % digits) % value


def _comparison_lookup(result, name):
    return {(r["dataset"], r["depth"], r["method"]): r
            for r in result["comparisons"][name]}


def _weighted_latency(run, depth, method):
    rows = [row for row in run["summary"]
            if row["depth"] == depth and row["method"] == method]
    return sum(row["n"] * row["mean_latency_seconds"] for row in rows) / sum(
        row["n"] for row in rows
    )


def render_report(result):
    flash = _comparison_lookup(result, "Flash")
    pro = _comparison_lookup(result, "Pro 0813")
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
        "# 三种生成模型的七方法 QBAF 全量配对比较", "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"), "",
        "## 实验设计", "",
        "- Flash、Pro 0813、Qwen 3.8 Max 分别调用 `deepseek-v4.1-flash`、"
        "`deepseek-v4-pro-0813`、`qwen3.8-max`；三者均通过 DMX，均关闭思考。"
        "Qwen 的关闭思考参数为 `enable_thinking=false`。",
        "- 三个上游 Experiment 全集各 500 条，D=1 与 D=2 分别尝试；"
        "每个模型评估七种方法。数据、标签、Direct Jev 根概率、提示、抽样参数、"
        "生成流水线、DF-QuAD 和 EW-DF-QuAD 实现一致。每个模型独立生成和缓存论证图，"
        "同一模型内的 QBAF 方法共用图。",
        "- 表中 Δ 为 Qwen 减去对应模型。Accuracy 越高越好；Brier、ECE 越低越好。"
        "准确率差的 95% 区间对相同论点做 1000 次配对重抽样，未作多重比较校正。",
        "- 生成模型每个提示只抽样一次，temperature=0.7、top_p=0.95、输出上限 128 token；"
        "DMX 未固定采样种子。模型差异与采样随机性共同作用。",
        "- D=1、D=2 共用同一批根论点，两个深度不是独立样本；"
        "配对比较仅使用三个模型均有完整七方法结果的样本：D=1 为 n={n1}，"
        "D=2 为 n={n2}。Direct Jev 在这些样本上逐样本完全相同。".format(
            n1=result["paired_n_by_depth"][1],
            n2=result["paired_n_by_depth"][2],
        ), "",
        "## 覆盖范围", "",
        "Flash 与 Pro 各完成 3000 条样本深度结果；Qwen 完成 {n} 条。".format(
            n=result["paired_sample_depth_rows"]
        ),
    ]
    if result["excluded_provider_refusals"]:
        lines += [
            "Qwen 的下列原始数据样本被 DMX 返回 HTTP 400 内容拒绝。"
            "没有替换提示、模型或生成图；配对表在三组中均排除这些样本：",
            "",
        ]
        for row in result["excluded_provider_refusals"]:
            lines.append("- {dataset} D={depth} 样本 {index}。".format(**row))
    lines += [
        "",
        "## 三数据集合计：七方法", "",
        "| 深度 | 方法 | Flash Acc | Pro Acc | Qwen Acc | Qwen−Flash ΔAcc [95% CI] | "
        "Qwen−Pro ΔAcc [95% CI] | Flash Brier | Pro Brier | Qwen Brier | "
        "Flash ECE | Pro ECE | Qwen ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in (1, 2):
        for method in METHODS:
            old, new = flash[POOLED, depth, method], pro[POOLED, depth, method]
            cf, cp = old["accuracy_change_ci95"], new["accuracy_change_ci95"]
            q = old["pro"]
            lines.append(
                "| D={d} | {m} | {fa:.4f} | {pa:.4f} | {qa:.4f} | "
                "{df:+.4f} [{f0:+.4f},{f1:+.4f}] | "
                "{dp:+.4f} [{p0:+.4f},{p1:+.4f}] | "
                "{fb:.4f} | {pb:.4f} | {qb:.4f} | "
                "{fe:.4f} | {pe:.4f} | {qe:.4f} |".format(
                    d=depth, m=method,
                    fa=old["flash"]["accuracy"], pa=new["flash"]["accuracy"],
                    qa=q["accuracy"], df=old["accuracy_change"],
                    f0=cf[0], f1=cf[1], dp=new["accuracy_change"],
                    p0=cp[0], p1=cp[1],
                    fb=old["flash"]["brier"], pb=new["flash"]["brier"],
                    qb=q["brier"], fe=old["flash"]["ece"],
                    pe=new["flash"]["ece"], qe=q["ece"],
                )
            )

    direct = flash[POOLED, 1, "direct_jev"]["pro"]
    lines += ["", "## 关键结果", ""]
    qwen_direct = flash[POOLED, 1, "direct_llm"]["pro"]
    qwen_direct_vs_pro = pro[POOLED, 1, "direct_llm"]
    lines.append(
        "- Qwen 的 direct_llm 在共同样本上的 Accuracy 为 {acc:.4f}、Brier 为 "
        "{brier:.4f}；相对 Pro 的 Accuracy 高 {delta:+.4f}。"
        "它也高于固定 Direct Jev 的 {jev:.4f}。".format(
            acc=qwen_direct["accuracy"], brier=qwen_direct["brier"],
            delta=qwen_direct_vs_pro["accuracy_change"],
            jev=direct["accuracy"],
        )
    )
    for depth in (1, 2):
        original_f = flash[POOLED, depth, "original_qbaf"]
        original_p = pro[POOLED, depth, "original_qbaf"]
        prior_f = flash[POOLED, depth, "jev_prior_ew_qbaf"]
        prior_p = pro[POOLED, depth, "jev_prior_ew_qbaf"]
        qprior = prior_f["pro"]
        lines.append(
            "- D={d}：Qwen 的 original_qbaf Accuracy 为 {orig:.4f}，"
            "相对 Flash {df:+.4f}、相对 Pro {dp:+.4f}；"
            "Jev-Prior-EW-QBAF 为 {prior:.4f}，相对 Flash {pf:+.4f}、"
            "相对 Pro {pp:+.4f}，相对固定 Direct Jev {jev:.4f} 为 {pj:+.4f}。"
            "该 Prior-EW 的 Brier 为 {prior_brier:.4f}，"
            "Direct Jev 为 {jev_brier:.4f}。".format(
                d=depth, orig=original_f["pro"]["accuracy"],
                df=original_f["accuracy_change"], dp=original_p["accuracy_change"],
                prior=qprior["accuracy"], pf=prior_f["accuracy_change"],
                pp=prior_p["accuracy_change"], jev=direct["accuracy"],
                pj=qprior["accuracy"] - direct["accuracy"],
                prior_brier=qprior["brier"], jev_brier=direct["brier"],
            )
        )
    lines.append(
        "- Qwen 的 direct_llm Accuracy 高于其 original_qbaf（D=1 为 "
        "{d1:+.4f}、D=2 为 {d2:+.4f}）。"
        "生成模型的直接判断提升，没有转化为 QBAF 对 Jev 根先验的净纠错优势。"
        "Qwen 根节点双向证据率为 {qconf:.3f}，Pro 为 {pconf:.3f}；"
        "图结构差异可见，但不能据此单独解释性能变化。".format(
            d1=qwen_direct["accuracy"] - flash[POOLED, 1, "original_qbaf"]["pro"]["accuracy"],
            d2=qwen_direct["accuracy"] - flash[POOLED, 2, "original_qbaf"]["pro"]["accuracy"],
            qconf=graphs["Qwen 3.8 Max"][POOLED, 1]["root_conflict_rate"],
            pconf=graphs["Pro 0813"][POOLED, 1]["root_conflict_rate"],
        )
    )
    lines += ["", "## 各数据集：Accuracy 与 Brier", ""]
    for dataset in DATASETS:
        lines += ["### " + dataset, "",
                  "| 深度 | 方法 | Flash Acc | Pro Acc | Qwen Acc | Qwen−Flash ΔAcc | "
                  "Qwen−Pro ΔAcc | Flash Brier | Pro Brier | Qwen Brier |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for depth in (1, 2):
            for method in METHODS:
                old, new = flash[dataset, depth, method], pro[dataset, depth, method]
                lines.append(
                    "| D={d} | {m} | {fa:.4f} | {pa:.4f} | {qa:.4f} | "
                    "{df:+.4f} | {dp:+.4f} | {fb:.4f} | {pb:.4f} | {qb:.4f} |".format(
                        d=depth, m=method,
                        fa=old["flash"]["accuracy"],
                        pa=new["flash"]["accuracy"],
                        qa=old["pro"]["accuracy"],
                        df=old["accuracy_change"], dp=new["accuracy_change"],
                        fb=old["flash"]["brier"],
                        pb=new["flash"]["brier"],
                        qb=old["pro"]["brier"],
                    )
                )
        lines.append("")

    lines += ["## 图结构变化", "",
              "| 数据集 | 深度 | 模型 | 平均论点数 | 平均边数 | 根节点双向证据率 |",
              "|---|---|---|---:|---:|---:|"]
    for dataset in (POOLED,) + tuple(DATASETS):
        for depth in (1, 2):
            for name in RUN_NAMES:
                row = graphs[name][dataset, depth]
                lines.append("| {ds} | D={d} | {name} | {args:.2f} | {edges:.2f} | "
                             "{conf:.3f} |".format(
                                 ds=dataset, d=depth, name=name,
                                 args=row["mean_arguments"],
                                 edges=row["mean_edges"],
                                 conf=row["root_conflict_rate"],
                             ))
    lines += ["", "## Jev 置信度分桶：Prior-EW 的净纠错", "",
              "| 深度 | Direct Jev 置信度 | n | Flash Rescue−Harm | "
              "Pro Rescue−Harm | Qwen Rescue−Harm | Qwen Rescue/Harm | "
              "Qwen Flip 比例 | Qwen Brier 变化 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for depth in (1, 2):
        for label in BIN_LABELS:
            old, mid, new = (bins[name][depth, label] for name in RUN_NAMES)
            lines.append(
                "| D={d} | {label} | {n} | {f:+d} | {p:+d} | {q:+d} | "
                "{rescue}/{harm} | {flip:.3f} | {brier:+.4f} |".format(
                    d=depth, label=label, n=new["n"],
                    f=old["rescue_minus_harm"], p=mid["rescue_minus_harm"],
                    q=new["rescue_minus_harm"], rescue=new["rescue"],
                    harm=new["harm"], flip=new["flip_rate"],
                    brier=new["brier_change"],
                )
            )
    lines += ["", "## API 用量与平均方法延迟", "",
              "| 模型 | 唯一 API 调用 | LLM 输入/输出 token | Jev 成本 USD | "
              "LLM 成本 CNY | 并发样本数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in RUN_NAMES:
        run = result["run_metrics"][name]
        tokens = run["unique_api_tokens"]
        lines.append(
            "| {name} | {calls} | {inp:,}/{out:,} | ${usd} | {cny} | {workers} |".format(
                name=name, calls=run["unique_api_calls"],
                inp=tokens["llm_input"], out=tokens["llm_output"],
                usd=_number(run["unique_api_cost_usd"], 6),
                cny=("未计价" if run["unique_api_cost_cny"] is None else
                     "￥" + _number(run["unique_api_cost_cny"], 6)),
                workers=run["settings"]["workers"],
            )
        )
    lines += ["", "| 深度 | 方法 | Flash 秒/样本 | Pro 秒/样本 | Qwen 秒/样本 |",
              "|---|---|---:|---:|---:|"]
    for depth in (1, 2):
        for method in METHODS:
            values = [_weighted_latency(result["run_metrics"][name], depth, method)
                      for name in RUN_NAMES]
            lines.append("| D={d} | {m} | {a:.3f} | {b:.3f} | {c:.3f} |".format(
                d=depth, m=method, a=values[0], b=values[1], c=values[2]
            ))
    lines += [
        "", "Qwen 的人民币成本按 [DMX 公开价格表](https://rmb.dmxapi.cn/) "
        "中 `qwen3.8-max` 折后输入 ¥9.48/百万 token、输出 ¥28.44/百万 token 估算。"
        "未扣除可能发生的输入缓存优惠；最终账单以 DMX 为准。"
        "Pro 0813 未确认人民币单价，因此其成本继续标为未计价。"
        "表中是各运行逻辑上完整执行一次的用量，共享的 Direct Jev 根请求"
        "在三个运行中各计一次，Qwen 实际复用了已缓存的根响应。"
        "Flash 与 Pro 的用量覆盖全部 3000 条样本深度结果，Qwen 用量仅覆盖成功样本；"
        "本次拒绝请求没有 token 回执，未计入估算。",
        "", "## 解释边界", "",
        "不同生成模型得到的图结构和论点文本不同，随之触发的 Jev 节点及边概率也不同。"
        "这里测量的是替换生成模型后的完整流水线表现，"
        "不能将差异单独归因于论点文本、图拓扑或 Jev 评分某一环节。"
        "表中方法延迟为记录的顺序 API 调用耗时之和；"
        "跨模型比较还受并发数与服务负载影响。", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-dir", type=Path,
                        default=Path("experiment_results/prior_full"))
    parser.add_argument("--pro-dir", type=Path,
                        default=Path("experiment_results/pro_0813_full"))
    parser.add_argument("--qwen-dir", type=Path,
                        default=Path("experiment_results/qwen38_max_full"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/2026-09-27_生成模型三方对比.md"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("reports/2026-09-27_生成模型三方对比.json"))
    args = parser.parse_args()
    result = make_comparison(dict(zip(
        RUN_NAMES, (args.flash_dir, args.pro_dir, args.qwen_dir)
    )))
    write_json(args.metrics, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print("中文三模型对比报告：" + str(args.report))


if __name__ == "__main__":
    main()
