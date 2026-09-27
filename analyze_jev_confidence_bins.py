"""Analyze cached Direct Jev versus Jev-Prior-EW-QBAF by decision confidence."""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

from benchmark_jev_qbaf import paired_interval
from experiment_io import read_json, write_json


DATASETS = ("TruthfulClaim", "StrategyClaim", "MedClaim")
EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BIN_LABELS = (
    "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1]",
)
POOLED = "三数据集合计"


def confidence_bin(probability):
    """Jev confidence in its chosen class, with exact boundaries on the right."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Direct Jev probability must be in [0, 1]")
    confidence = round(max(probability, 1.0 - probability), 10)
    return min(bisect_right(EDGES, confidence) - 1, len(BIN_LABELS) - 1)


def bucket_summary(rows):
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "direct_accuracy": None,
            "prior_ew_accuracy": None,
            "rescue": 0,
            "harm": 0,
            "rescue_minus_harm": 0,
            "accuracy_change": None,
            "accuracy_change_ci95": None,
            "direct_brier": None,
            "prior_ew_brier": None,
            "brier_change": None,
            "brier_change_ci95": None,
            "prediction_flips": 0,
            "flip_rate": None,
        }

    direct_correct = []
    prior_ew_correct = []
    direct_brier_values = []
    prior_ew_brier_values = []
    for row in rows:
        label = int(row["valid"])
        direct = row["predictions"]["direct_jev"]
        prior_ew = row["predictions"]["jev_prior_ew_qbaf"]
        direct_correct.append((direct > 0.5) == bool(label))
        prior_ew_correct.append((prior_ew > 0.5) == bool(label))
        direct_brier_values.append((direct - label) ** 2)
        prior_ew_brier_values.append((prior_ew - label) ** 2)

    accuracy_changes = [int(after) - int(before)
                        for before, after in zip(direct_correct, prior_ew_correct)]
    brier_changes = [after - before for before, after in zip(
        direct_brier_values, prior_ew_brier_values
    )]
    rescue = accuracy_changes.count(1)
    harm = accuracy_changes.count(-1)
    flips = rescue + harm
    return {
        "n": n,
        "direct_accuracy": sum(direct_correct) / n,
        "prior_ew_accuracy": sum(prior_ew_correct) / n,
        "rescue": rescue,
        "harm": harm,
        "rescue_minus_harm": rescue - harm,
        "accuracy_change": (rescue - harm) / n,
        "accuracy_change_ci95": paired_interval(accuracy_changes),
        "direct_brier": sum(direct_brier_values) / n,
        "prior_ew_brier": sum(prior_ew_brier_values) / n,
        "brier_change": sum(brier_changes) / n,
        "brier_change_ci95": paired_interval(brier_changes),
        "prediction_flips": flips,
        "flip_rate": flips / n,
    }


def analyze(rows):
    grouped = {}
    for row in rows:
        index = confidence_bin(row["predictions"]["direct_jev"])
        for dataset in (POOLED, row["dataset"]):
            grouped.setdefault((dataset, row["depth"], index), []).append(row)

    buckets = []
    bands = []
    for depth in (1, 2):
        for dataset in (POOLED,) + DATASETS:
            for index, label in enumerate(BIN_LABELS):
                result = bucket_summary(grouped.get((dataset, depth, index), []))
                buckets.append({
                    "dataset": dataset, "depth": depth,
                    "confidence_bin": label, "bin_index": index,
                    **result,
                })
        for label, indices in (
            ("接近阈值 [0.5,0.7)", (0, 1)),
            ("高置信 [0.9,1]", (4,)),
        ):
            subset = [row for index in indices
                      for row in grouped.get((POOLED, depth, index), [])]
            if not subset:
                continue
            bands.append({"depth": depth, "confidence_band": label,
                          **bucket_summary(subset)})
    return {"buckets": buckets, "bands": bands}


def load_rows(input_dir):
    return [read_json(path) for path in sorted(
        Path(input_dir).glob("*/*/sample_*.json")
    )]


def numeric(value, digits=4):
    return "—" if value is None else ("%%.%df" % digits) % value


def render_report(analysis, input_dir, plot_path=None, metrics_path=None):
    buckets = analysis["buckets"]
    bands = analysis["bands"]
    lines = [
        "# Direct Jev 置信度分桶与 Prior-EW-QBAF 修正分析", "",
        "生成时间：" + datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ), "",
        "## 定义", "",
        "- 使用现有逐样本结果：`" + str(input_dir).replace("\\", "/") + "`；"
        "没有发起 API 请求。",
        "- `direct_jev` 给出论点为真的概率 p。分桶所用置信度为 "
        "c=max(p,1−p)，即 Jev 对其所选真假类别的置信度。"
        "因此真假两侧被对称分桶；恰好位于边界的样本进入右侧桶，1.0 进入末桶。",
        "- 两种方法均以概率严格大于 0.5 判真。Rescue 指 Direct Jev 判错而 "
        "Prior-EW-QBAF 判对；Harm 指反方向。"
        "Rescue−Harm 除以桶内样本数，恰好等于后者减前者的 Accuracy。",
        "- Brier 是真类概率与 0/1 标签的平方误差均值；"
        "翻转率为两方法真假判断不同的样本占比。"
        "95% 区间对同一桶的逐样本差值做 2000 次配对重抽样，未经多重比较校正。",
        "- D=1 与 D=2 共享同一批根论点；以下按深度分开计算，"
        "不会将两个深度当作独立样本合并。", "",
        "## 三数据集合计", "",
        "| 深度 | Jev 置信度桶 | n | Direct Acc | Prior-EW Acc | "
        "Rescue | Harm | Rescue−Harm | Direct Brier | Prior-EW Brier | 翻转率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in buckets:
        if item["dataset"] != POOLED:
            continue
        lines.append(
            "| D={depth} | {confidence_bin} | {n} | {direct_accuracy} | "
            "{prior_ew_accuracy} | {rescue} | {harm} | {rescue_minus_harm:+d} | "
            "{direct_brier} | {prior_ew_brier} | {flip_rate} |".format(
                depth=item["depth"], confidence_bin=item["confidence_bin"],
                n=item["n"], direct_accuracy=numeric(item["direct_accuracy"]),
                prior_ew_accuracy=numeric(item["prior_ew_accuracy"]),
                rescue=item["rescue"], harm=item["harm"],
                rescue_minus_harm=item["rescue_minus_harm"],
                direct_brier=numeric(item["direct_brier"]),
                prior_ew_brier=numeric(item["prior_ew_brier"]),
                flip_rate=numeric(item["flip_rate"]),
            )
        )
    if plot_path:
        lines += ["", "![各置信度桶的净纠错与判断翻转率](" +
                  Path(plot_path).name + ")"]

    lines += [
        "", "## 接近阈值与高置信对照", "",
        "| 深度 | 范围 | n | Rescue | Harm | Rescue−Harm | "
        "ΔAccuracy [95%区间] | ΔBrier [95%区间] | 翻转率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bands:
        accuracy_ci = item["accuracy_change_ci95"]
        brier_ci = item["brier_change_ci95"]
        lines.append(
            "| D={depth} | {band} | {n} | {rescue} | {harm} | {net:+d} | "
            "{accuracy:+.4f} [{accuracy_low:+.4f}, {accuracy_high:+.4f}] | "
            "{brier:+.4f} [{brier_low:+.4f}, {brier_high:+.4f}] | "
            "{flip_rate:.4f} |".format(
                depth=item["depth"], band=item["confidence_band"], n=item["n"],
                rescue=item["rescue"], harm=item["harm"],
                net=item["rescue_minus_harm"], accuracy=item["accuracy_change"],
                accuracy_low=accuracy_ci[0], accuracy_high=accuracy_ci[1],
                brier=item["brier_change"],
                brier_low=brier_ci[0], brier_high=brier_ci[1],
                flip_rate=item["flip_rate"],
            )
        )

    lines += [
        "", "## 各数据集分桶", "",
        "| 数据集 | 深度 | Jev 置信度桶 | n | Direct Acc | Prior-EW Acc | "
        "Rescue | Harm | Rescue−Harm | Direct Brier | Prior-EW Brier | 翻转率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in buckets:
        if item["dataset"] == POOLED:
            continue
        lines.append(
            "| {dataset} | D={depth} | {confidence_bin} | {n} | "
            "{direct_accuracy} | {prior_ew_accuracy} | {rescue} | {harm} | "
            "{rescue_minus_harm:+d} | {direct_brier} | {prior_ew_brier} | "
            "{flip_rate} |".format(
                dataset=item["dataset"], depth=item["depth"],
                confidence_bin=item["confidence_bin"], n=item["n"],
                direct_accuracy=numeric(item["direct_accuracy"]),
                prior_ew_accuracy=numeric(item["prior_ew_accuracy"]),
                rescue=item["rescue"], harm=item["harm"],
                rescue_minus_harm=item["rescue_minus_harm"],
                direct_brier=numeric(item["direct_brier"]),
                prior_ew_brier=numeric(item["prior_ew_brier"]),
                flip_rate=numeric(item["flip_rate"]),
            )
        )
    pooled = {(item["depth"], item["bin_index"]): item
              for item in buckets if item["dataset"] == POOLED}
    near = {item["depth"]: item for item in bands
            if item["confidence_band"].startswith("接近阈值")}
    depths = sorted(near)
    lines += ["", "## 对假设的判断", ""]
    for depth in depths:
        first = pooled[depth, 0]
        high = pooled[depth, 4]
        band = near[depth]
        lines.append(
            "- D={depth}：最接近 0.5 的桶翻转 {first_flips}/{first_n} 条，"
            "Rescue/Harm 为 {first_rescue}/{first_harm}，"
            "Brier 从 {first_direct_brier:.4f} 变为 {first_prior_brier:.4f}；"
            "合并的 [0.5,0.7) 区间 Rescue−Harm 为 {near_net:+d}。"
            "高置信 [0.9,1] 桶翻转 {high_flips}/{high_n} 条，"
            "Rescue/Harm 为 {high_rescue}/{high_harm}。".format(
                depth=depth, first_flips=first["prediction_flips"],
                first_n=first["n"], first_rescue=first["rescue"],
                first_harm=first["harm"],
                first_direct_brier=first["direct_brier"],
                first_prior_brier=first["prior_ew_brier"],
                near_net=band["rescue_minus_harm"],
                high_flips=high["prediction_flips"], high_n=high["n"],
                high_rescue=high["rescue"], high_harm=high["harm"],
            )
        )
    if all(item["rescue_minus_harm"] <= 0 for item in pooled.values()):
        lines.append(
            "- 三数据集合计的十个深度—置信度桶中，"
            "Rescue−Harm 均未为正；接近 0.5 时虽有更多判断翻转，"
            "本次 Prior-EW-QBAF 没有由此获得净纠错优势。"
        )
    if depths and all(near[depth]["brier_change_ci95"][0] > 0
                      for depth in depths):
        lines.append(
            "- [0.5,0.7) 区间两个深度的 Brier 都升高，"
            "其逐样本配对 95% 区间下界均大于 0。"
        )
    lines += [
        "- 高置信桶翻转极少，现有数据更直接支持“结构化论证几乎不改变"
        "高置信判断”；仅凭稀少的翻转，不能断言它在该桶有稳定的有害或有益效果。"
        "低置信桶未见总体正价值，但不排除其他图生成或论点评分设置得到不同结果。",
        "", "本分析按已观察到的置信度分组，属于探索性条件比较。"
        "同一根论点的两个深度不可当作独立重复，单个数据集的小桶也应结合样本数解读。",
    ]
    if metrics_path:
        lines += ["", "全部逐桶数值及配对区间见 [指标 JSON](" +
                  Path(metrics_path).name + ")。"]
    lines.append("")
    return "\n".join(lines)


def save_plot(analysis, path):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    pooled = {(item["depth"], item["bin_index"]): item
              for item in analysis["buckets"] if item["dataset"] == POOLED}
    positions = list(range(len(BIN_LABELS)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {1: "#3274a1", 2: "#dc6b43"}
    for depth, offset in ((1, -0.19), (2, 0.19)):
        values = [pooled[depth, index]["rescue_minus_harm"]
                  for index in positions]
        axes[0].bar([x + offset for x in positions], values, width=0.36,
                    label="D=" + str(depth), color=colors[depth])
        flips = [100 * (pooled[depth, index]["flip_rate"] or 0)
                 for index in positions]
        axes[1].plot(positions, flips, marker="o", linewidth=2,
                     label="D=" + str(depth), color=colors[depth])
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Rescue−Harm（例）")
    axes[0].set_title("净纠错")
    axes[1].set_ylabel("判断翻转率（%）")
    axes[1].set_title("先验判断被改变的比例")
    labels = [label + "\nn=" + str(pooled[1, index]["n"])
              for index, label in enumerate(BIN_LABELS)]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Direct Jev 决策置信度")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("experiment_results/prior_full/data"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/2026-09-27_jev_prior_ew_置信度分桶分析.md"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("reports/2026-09-27_jev_prior_ew_置信度分桶指标.json"))
    parser.add_argument("--plot", type=Path, default=None,
                        help="Optional PNG chart; requires matplotlib")
    args = parser.parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        raise ValueError("No per-sample results found in " + str(args.input_dir))
    analysis = analyze(rows)
    result = {
        "input_dir": str(args.input_dir),
        "sample_depth_rows": len(rows),
        "confidence_definition": "max(p_true, 1-p_true)",
        "bin_edges": list(EDGES),
        "baseline": "direct_jev",
        "comparison": "jev_prior_ew_qbaf",
        **analysis,
    }
    write_json(args.metrics, result)
    if args.plot:
        save_plot(analysis, args.plot)
    report = render_report(analysis, args.input_dir, args.plot, args.metrics)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print("中文报告：" + str(args.report))


if __name__ == "__main__":
    main()
