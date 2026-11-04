#!/usr/bin/env python3
"""独立绘制阈值电压漂移：python plot_Vth.py <file> [-o output.png]。

仅依赖 numpy 和 matplotlib，不导入模拟源码。
默认在 CSV 所在目录保存同名 PNG；横轴为真实累计时间，纵轴为带符号漂移。
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_vth(source: Path, output: Path) -> None:
    data = np.genfromtxt(source, delimiter=",", names=True, encoding="utf-8-sig",
                         ndmin=1)
    names = data.dtype.names or ()
    if "time_s" not in names:
        raise ValueError("CSV 缺少 time_s 列")
    if "threshold_voltage_drift_mV" in names:
        drift_mv = data["threshold_voltage_drift_mV"]
    elif "threshold_voltage_drift_V" in names:
        drift_mv = data["threshold_voltage_drift_V"] * 1000.0
    else:
        raise ValueError("CSV 缺少 threshold_voltage_drift_mV 或 threshold_voltage_drift_V 列")
    time_s = data["time_s"]
    if (not time_s.size or not np.all(np.isfinite(time_s))
            or not np.all(np.isfinite(drift_mv))):
        raise ValueError("时间和漂移数据必须非空且全部有限")
    if np.any(np.diff(time_s) < 0):
        raise ValueError("time_s 必须按累计时间递增排列，允许阶段边界重复")

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.plot(time_s, drift_mv, color="#2166ac", linewidth=1.8)
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
    # 标记阶段切换；保留原始时间和全部采样点，不按点序号等距绘制。
    if "phase" in names:
        boundaries = np.flatnonzero(np.diff(data["phase"]) != 0) + 1
        for boundary in boundaries:
            ax.axvline(time_s[boundary], color="0.7", linewidth=0.8,
                       linestyle=":", zorder=0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$V_{\mathrm{th}}(t)-V_{\mathrm{th}}(\mathrm{pre})$ [mV]")
    ax.set_title("Threshold voltage drift")
    ax.grid(alpha=0.2)
    if time_s[-1] > time_s[0]:
        ax.set_xlim(time_s[0], time_s[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=200)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="threshold_voltage_drift.csv 的路径")
    parser.add_argument("-o", "--output", type=Path,
                        help="输出图片路径，默认在 CSV 旁保存同名 PNG")
    args = parser.parse_args()
    source = args.file.expanduser()
    output = args.output.expanduser() if args.output else source.with_suffix(".png")
    if source.resolve() == output.resolve():
        parser.error("输出路径不能与输入 CSV 相同")
    try:
        plot_vth(source, output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"绘图失败：{exc}\n")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
