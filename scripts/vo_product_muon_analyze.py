"""
Parse VO Product Muon experiment logs into CSV summaries and plots.

Example:
python -m scripts.vo_product_muon_analyze --results-dir "$NANOCHAT_BASE_DIR/vo_product_muon_results"
"""

import argparse
import csv
import os
import re
from collections import defaultdict


TRAIN_RE = re.compile(
    r"step\s+(\d+)/(\d+).*tok/sec:\s+([\d,]+).*total time:\s+([0-9.]+)m"
)
VAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)")
CORE_RE = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([0-9.]+)")
TOTAL_TIME_RE = re.compile(r"Total training time:\s+([0-9.]+)m")


def read_manifest(results_dir):
    manifest_path = os.path.join(results_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    logs_dir = os.path.join(results_dir, "logs")
    rows = []
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            if name.endswith(".log"):
                tag = name[:-4]
                rows.append({
                    "tag": tag,
                    "phase": "unknown",
                    "optimizer_kind": "unknown",
                    "seed": "",
                    "matrix_lr": "",
                    "matrix_adamw_beta1": "",
                    "matrix_adamw_beta2": "",
                    "matrix_adamw_wd_mult": "",
                    "log_file": os.path.join(logs_dir, name),
                })
    return rows


def parse_log(path):
    val_history = []
    core_history = []
    tok_per_sec_values = []
    current_time_sec = 0.0
    total_time_sec = None

    if not os.path.exists(path):
        return {
            "val_history": val_history,
            "core_history": core_history,
            "avg_tok_per_sec": "",
            "total_training_time_sec": "",
        }

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            train_match = TRAIN_RE.search(line)
            if train_match:
                tok_per_sec_values.append(int(train_match.group(3).replace(",", "")))
                current_time_sec = float(train_match.group(4)) * 60.0
                continue

            val_match = VAL_RE.search(line)
            if val_match:
                val_history.append({
                    "step": int(val_match.group(1)),
                    "wall_time_sec": current_time_sec,
                    "val_bpb": float(val_match.group(2)),
                })
                continue

            core_match = CORE_RE.search(line)
            if core_match:
                core_history.append({
                    "step": int(core_match.group(1)),
                    "core_metric": float(core_match.group(2)),
                })
                continue

            total_match = TOTAL_TIME_RE.search(line)
            if total_match:
                total_time_sec = float(total_match.group(1)) * 60.0

    avg_tok_per_sec = ""
    if tok_per_sec_values:
        avg_tok_per_sec = sum(tok_per_sec_values) / len(tok_per_sec_values)

    return {
        "val_history": val_history,
        "core_history": core_history,
        "avg_tok_per_sec": avg_tok_per_sec,
        "total_training_time_sec": total_time_sec if total_time_sec is not None else "",
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path, history_rows, x_key, y_key, title, xlabel):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot {path}: matplotlib import failed: {exc}")
        return

    by_tag = defaultdict(list)
    for row in history_rows:
        if row[y_key] != "":
            by_tag[row["tag"]].append(row)

    if not by_tag:
        return

    plt.figure(figsize=(10, 6))
    for tag, rows in sorted(by_tag.items()):
        rows = sorted(rows, key=lambda r: float(r[x_key]))
        xs = [float(r[x_key]) for r in rows]
        ys = [float(r[y_key]) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=tag)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Validation bpb")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _float_or_none(value):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_markdown_summary(path, summary_rows):
    headers = ["optimizer", "seed", "wall-clock sec", "final val_bpb", "final CORE", "avg tok/sec"]
    lines = [
        "# VO Product Muon Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join([
                row.get("optimizer_kind", ""),
                row.get("seed", ""),
                str(row.get("total_training_time_sec", "")),
                str(row.get("final_val_bpb", "")),
                str(row.get("final_core", "")),
                str(row.get("avg_tok_per_sec", "")),
            ])
            + " |"
        )

    compare_rows = [row for row in summary_rows if row.get("phase") == "compare"]
    grouped = defaultdict(list)
    for row in compare_rows:
        grouped[row.get("optimizer_kind", "")].append(row)

    def mean_metric(rows, key):
        values = [_float_or_none(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        return None if not values else sum(values) / len(values)

    lines.extend(["", "## Conclusion", ""])
    muon_val = mean_metric(grouped.get("muon", []), "final_val_bpb")
    muon_toks = mean_metric(grouped.get("muon", []), "avg_tok_per_sec")
    vo_val = mean_metric(grouped.get("vo_product_muon", []), "final_val_bpb")
    vo_toks = mean_metric(grouped.get("vo_product_muon", []), "avg_tok_per_sec")

    if muon_val is None or vo_val is None:
        lines.append("No complete `compare` phase with both `muon` and `vo_product_muon` was found. Run the comparison phase before making a keep/revise/discard decision.")
    else:
        throughput_ok = muon_toks is None or vo_toks is None or vo_toks >= 0.90 * muon_toks
        quality_ok = vo_val <= muon_val + 0.001
        if quality_ok and throughput_ok:
            decision = "keep for further study"
        elif not quality_ok and not throughput_ok:
            decision = "revise or discard"
        else:
            decision = "revise"
        lines.append(
            f"Automated preliminary decision: **{decision}**. "
            f"Mean Muon val_bpb={muon_val:.6f}; mean VO val_bpb={vo_val:.6f}. "
            "Confirm with the full seed set and CORE before treating this as final."
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze VO Product Muon experiment logs")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.results_dir)
    summary_rows = []
    history_rows = []

    for manifest in manifest_rows:
        parsed = parse_log(manifest["log_file"])
        val_history = parsed["val_history"]
        core_history = parsed["core_history"]
        final_val = val_history[-1]["val_bpb"] if val_history else ""
        final_core = core_history[-1]["core_metric"] if core_history else ""

        summary_rows.append({
            **manifest,
            "final_val_bpb": final_val,
            "final_core": final_core,
            "avg_tok_per_sec": parsed["avg_tok_per_sec"],
            "total_training_time_sec": parsed["total_training_time_sec"],
        })

        for point in val_history:
            history_rows.append({
                **{k: manifest.get(k, "") for k in manifest.keys()},
                "step": point["step"],
                "wall_time_sec": point["wall_time_sec"],
                "val_bpb": point["val_bpb"],
            })

    summary_fields = [
        "tag", "phase", "optimizer_kind", "seed",
        "matrix_lr", "matrix_adamw_beta1", "matrix_adamw_beta2", "matrix_adamw_wd_mult",
        "final_val_bpb", "final_core", "avg_tok_per_sec", "total_training_time_sec", "log_file",
    ]
    history_fields = [
        "tag", "phase", "optimizer_kind", "seed",
        "matrix_lr", "matrix_adamw_beta1", "matrix_adamw_beta2", "matrix_adamw_wd_mult",
        "step", "wall_time_sec", "val_bpb", "log_file",
    ]

    summary_path = os.path.join(args.results_dir, "summary.csv")
    history_path = os.path.join(args.results_dir, "val_history.csv")
    markdown_path = os.path.join(args.results_dir, "benchmark_summary.md")
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(history_path, history_rows, history_fields)
    write_markdown_summary(markdown_path, summary_rows)

    plot_history(
        os.path.join(args.results_dir, "val_bpb_vs_step.png"),
        history_rows,
        x_key="step",
        y_key="val_bpb",
        title="Validation BPB vs Step",
        xlabel="Step",
    )
    plot_history(
        os.path.join(args.results_dir, "val_bpb_vs_wall_clock.png"),
        history_rows,
        x_key="wall_time_sec",
        y_key="val_bpb",
        title="Validation BPB vs Wall Clock",
        xlabel="Wall-clock seconds",
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
