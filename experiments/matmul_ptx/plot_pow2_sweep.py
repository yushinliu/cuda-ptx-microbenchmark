from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


COLORS = {
    "pytorch_mm_out_fp32": "#111111",
    "ptx_mma_cpasync": "#0b84f3",
    "ptx_mma_ldmatrix_block_cpasync_a": "#f06418",
}

LABELS = {
    "pytorch_mm_out_fp32": "PyTorch mm(fp32 out)",
    "ptx_mma_cpasync": "PTX mma cp.async",
    "ptx_mma_ldmatrix_block_cpasync_a": "PTX ldmatrix block + cp.async(A)",
}


def load_series(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        grouped[row["impl"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["shape"][0])
    return grouped


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(grouped: dict[str, list[dict]], output: Path) -> None:
    width = 980
    height = 620
    left = 96
    right = 28
    top = 56
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    sizes = sorted({row["shape"][0] for rows in grouped.values() for row in rows})
    times = [row["median_ms"] for rows in grouped.values() for row in rows]
    log_y_min = math.log10(min(times))
    log_y_max = math.log10(max(times))
    y_pad = (log_y_max - log_y_min) * 0.08
    log_y_min -= y_pad
    log_y_max += y_pad

    def x_pos(size: int) -> float:
        idx = sizes.index(size)
        if len(sizes) == 1:
            return left + plot_w / 2
        return left + plot_w * idx / (len(sizes) - 1)

    def y_pos(ms: float) -> float:
        t = (math.log10(ms) - log_y_min) / (log_y_max - log_y_min)
        return top + plot_h * (1.0 - t)

    y_ticks = []
    for exp in range(math.floor(log_y_min), math.ceil(log_y_max) + 1):
        for base in (1.0, 2.0, 5.0):
            value = base * (10 ** exp)
            if min(times) * 0.8 <= value <= max(times) * 1.25:
                y_ticks.append(value)
    y_ticks = sorted(set(y_ticks))

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#fffdf8"/>')
    parts.append(f'<text x="{left}" y="28" font-size="24" font-family="monospace" fill="#111111">Power-of-Two GEMM Sweep</text>')
    parts.append(
        f'<text x="{left}" y="48" font-size="12" font-family="monospace" fill="#4a4a4a">'
        'Square fp16 x fp16 -> fp32, median CUDA-event time, y-axis in log scale'
        "</text>"
    )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="#fffaf0" stroke="#d8cfc0" stroke-width="1"/>'
    )

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ece5d8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" font-family="monospace" fill="#555555">{tick:.3g} ms</text>'
        )

    for size in sizes:
        x = x_pos(size)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f2ecdf" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-size="12" font-family="monospace" fill="#333333">{size}</text>'
        )

    for impl, rows in grouped.items():
        points = " ".join(f"{x_pos(row['shape'][0]):.2f},{y_pos(row['median_ms']):.2f}" for row in rows)
        color = COLORS[impl]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" '
            f'stroke-linejoin="round" points="{points}"/>'
        )
        for row in rows:
            x = x_pos(row["shape"][0])
            y = y_pos(row["median_ms"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')

    legend_x = left + 12
    legend_y = top + 14
    for idx, impl in enumerate(("pytorch_mm_out_fp32", "ptx_mma_cpasync", "ptx_mma_ldmatrix_block_cpasync_a")):
        y = legend_y + idx * 22
        color = COLORS[impl]
        label = svg_escape(LABELS[impl])
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-size="12" font-family="monospace" fill="#222222">{label}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" '
        'font-size="13" font-family="monospace" fill="#222222">Matrix size N for N x N x N GEMM</text>'
    )
    parts.append(
        f'<text transform="translate(24 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" '
        'font-size="13" font-family="monospace" fill="#222222">Median runtime (ms, log scale)</text>'
    )
    parts.append("</svg>")

    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SVG line plot for power-of-two sweep")
    parser.add_argument("--input-json", type=Path, default=Path("results/pow2_sweep.json"))
    parser.add_argument("--output-svg", type=Path, default=Path("results/pow2_sweep_median_ms.svg"))
    args = parser.parse_args()

    grouped = load_series(args.input_json)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    render_svg(grouped, args.output_svg)
    print(f"saved {args.output_svg}")


if __name__ == "__main__":
    main()
