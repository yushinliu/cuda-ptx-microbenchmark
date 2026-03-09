from __future__ import annotations

import json
import math
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SIZES = [256, 512, 1024, 2048, 4096, 8192]
BENCH_TEMPLATE = "bench_stage8_sweep_{size}.json"
OUTPUT_JSON = THIS_DIR / "pow2_sweep.json"
OUTPUT_SVG = THIS_DIR / "pow2_sweep_median_ms.svg"


def load_bench(size: int) -> list[dict]:
    path = THIS_DIR / BENCH_TEMPLATE.format(size=size)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pick_impl(rows: list[dict], impl: str) -> dict:
    for row in rows:
        if row["impl"] == impl:
            return row
    raise KeyError(f"missing impl {impl}")


def pick_best_wmma(rows: list[dict]) -> dict:
    candidates = [
        row
        for row in rows
        if row["impl"] in {
            "ptx_mma_cpasync",
            "ptx_mma_cpasync_128",
            "ptx_mma_cpasync_128_k32",
            "ptx_mma_cpasync_k32",
        }
    ]
    return min(candidates, key=lambda row: row["median_ms"])


def build_series() -> list[dict]:
    series: list[dict] = []
    for size in SIZES:
        rows = load_bench(size)
        pytorch = pick_impl(rows, "pytorch_mm_out_fp32").copy()
        best_wmma = pick_best_wmma(rows).copy()
        hybrid = pick_impl(rows, "ptx_mma_ldmatrix_block_cpasync_a").copy()

        pytorch["plot_impl"] = "pytorch_mm_out_fp32"
        best_wmma["plot_impl"] = "ptx_wmma_cpasync_best"
        best_wmma["selected_impl"] = best_wmma["impl"]
        best_wmma["impl"] = "ptx_wmma_cpasync_best"
        hybrid["plot_impl"] = "ptx_mma_ldmatrix_block_cpasync_a"

        series.extend([pytorch, best_wmma, hybrid])
    return series


def linear_ticks(max_value: float) -> list[float]:
    upper = max_value * 1.08
    magnitude = 10 ** math.floor(math.log10(max(upper, 1e-6)))
    normalized = upper / magnitude
    if normalized <= 2:
        step = 0.25 * magnitude
    elif normalized <= 5:
        step = 0.5 * magnitude
    else:
        step = 1.0 * magnitude
    tick_count = math.ceil(upper / step)
    return [step * index for index in range(tick_count + 1)]


def fmt_ms(value: float) -> str:
    if value < 0.1:
        return f"{value:.02f}"
    if value < 1:
        return f"{value:.1f}"
    if value < 10:
        return f"{value:.1f}"
    return f"{value:.0f}"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(series_rows: list[dict]) -> str:
    width = 980
    height = 620
    left = 96
    right = 28
    top = 56
    bottom = 92
    plot_width = width - left - right
    plot_height = height - top - bottom

    grouped: dict[str, list[dict]] = {}
    for row in series_rows:
        grouped.setdefault(row["impl"], []).append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: row["shape"][0])

    max_ms = max(row["median_ms"] for row in series_rows)
    ticks = linear_ticks(max_ms)
    max_tick = ticks[-1]

    def x_of(size: int) -> float:
        if len(SIZES) == 1:
            return left
        index = SIZES.index(size)
        return left + plot_width * index / (len(SIZES) - 1)

    def y_of(value: float) -> float:
        return top + plot_height - (value / max_tick) * plot_height

    colors = {
        "pytorch_mm_out_fp32": "#111111",
        "ptx_wmma_cpasync_best": "#0b84f3",
        "ptx_mma_ldmatrix_block_cpasync_a": "#f06418",
    }
    labels = {
        "pytorch_mm_out_fp32": "PyTorch mm(fp32 out)",
        "ptx_wmma_cpasync_best": "PTX WMMA cp.async best",
        "ptx_mma_ldmatrix_block_cpasync_a": "PTX handwritten hybrid",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="96" y="28" font-size="24" font-family="monospace" fill="#111111">Current Power-of-Two GEMM Sweep</text>',
        '<text x="96" y="48" font-size="12" font-family="monospace" fill="#4a4a4a">Square fp16 x fp16 -&gt; fp32, median CUDA-event time, linear y-axis in milliseconds</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fffaf0" stroke="#d8cfc0" stroke-width="1"/>',
    ]

    for tick in ticks:
        y = y_of(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ece5d8" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="monospace" fill="#555555">{fmt_ms(tick)} ms</text>'
        )

    for size in SIZES:
        x = x_of(size)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f2ecdf" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="monospace" fill="#333333">{size}</text>'
        )

    for impl, rows in grouped.items():
        points = " ".join(f"{x_of(row['shape'][0]):.2f},{y_of(row['median_ms']):.2f}" for row in rows)
        color = colors[impl]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{points}"/>'
        )
        for row in rows:
            x = x_of(row["shape"][0])
            y = y_of(row["median_ms"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="10" font-family="monospace" fill="{color}">{fmt_ms(row["median_ms"])}</text>'
            )

    legend_y = 70
    for index, impl in enumerate(["pytorch_mm_out_fp32", "ptx_wmma_cpasync_best", "ptx_mma_ldmatrix_block_cpasync_a"]):
        y = legend_y + 22 * index
        color = colors[impl]
        parts.append(f'<line x1="108" y1="{y}" x2="132" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="120" cy="{y}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(
            f'<text x="142" y="{y + 4}" font-size="12" font-family="monospace" fill="#222222">{svg_escape(labels[impl])}</text>'
        )

    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="13" font-family="monospace" fill="#222222">Matrix size N for N x N x N GEMM</text>',
            f'<text transform="translate(24 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="monospace" fill="#222222">Median runtime (ms)</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    series_rows = build_series()
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(series_rows, handle, indent=2)
        handle.write("\n")
    OUTPUT_SVG.write_text(render_svg(series_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
