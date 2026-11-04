"""绘制阈值电压漂移 CSV：上排 Vg，下排相对预平衡的阈值漂移。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_threshold_voltage_drift(
    csv_path: Path | str,
    output_path: Path | str,
    *,
    dpi: int = 160,
) -> Path:
    """按阶段绘制 Vg 和阈值漂移，x 轴为阶段内采样点序号。"""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0:
        raise ValueError("threshold drift CSV contains no data rows")

    phase_ids = data[:, 1].astype(int)
    phase_count = int(phase_ids.max()) + 1
    fig, axes = plt.subplots(
        2, phase_count, figsize=(12, 5), dpi=dpi,
        sharey="row", squeeze=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.18},
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.97)

    for phase in range(phase_count):
        phase_data = data[phase_ids == phase]
        if len(phase_data) == 0:
            continue
        index = np.arange(len(phase_data), dtype=float)
        voltage = phase_data[:, 4]
        if phase == 0:
            previous_voltage = voltage[0]
        else:
            previous_phase = data[phase_ids == phase - 1]
            previous_voltage = previous_phase[0, 4]
        duration = phase_data[-1, 2]

        ax_vg, ax_shift = axes[:, phase]
        ax_vg.plot(
            np.concatenate(([index[0]], index)),
            np.concatenate(([previous_voltage], voltage)),
            color="#1f77b4", linewidth=2.5,
        )
        shift_uv = phase_data[:, 11] * 1e3
        ax_shift.plot(index, shift_uv, color="#1f77b4", linewidth=2.5)
        ax_shift.scatter(index, shift_uv, color="#2eaf72", s=15)
        ax_shift.set_yscale("linear")

        for ax in (ax_vg, ax_shift):
            ax.set_xlim(0, max(index[-1], 1))
            ax.set_xticks([max(index[-1], 1) / 2], [f"{duration:g}s"])
            ax.set_xticks(index, minor=True)
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="x", labelsize=7)
        if phase > 0:
            ax_vg.tick_params(labelleft=False)
            ax_shift.tick_params(labelleft=False)

    axes[0, 0].set_ylabel("Vg [V]")
    axes[1, 0].set_ylabel("Vth - Vth(pre) [µV]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="绘制 threshold_voltage_drift.csv 的 Vg 与阈值漂移图。",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=repo_root / "results" / "20261030_172017" / "threshold_voltage_drift.csv",
        help="阈值电压漂移 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 路径；默认与 CSV 同目录，文件名为 threshold_voltage_drift.png",
    )
    parser.add_argument("--dpi", type=int, default=160, help="输出分辨率，默认 160")
    args = parser.parse_args()
    output = args.output or args.csv.with_name("threshold_voltage_drift.png")
    saved = plot_threshold_voltage_drift(args.csv, output, dpi=args.dpi)
    print(f"阈值电压漂移图像: {saved.resolve()}")


if __name__ == "__main__":
    main()