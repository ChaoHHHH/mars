"""按固定帧间隔导出紧凑占据历史的能带/占据图。"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from src.io.compact_history import plot_compact_history


def plot_every_n_frames(
    history_path: Path | str,
    output_dir: Path | str,
    *,
    stride: int = 10,
    include_last: bool = True,
) -> list[Path]:
    """将 HDF5 中的每 stride 帧绘制成一张 PNG，返回生成文件列表。"""
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    with h5py.File(history_path, "r") as archive:
        frame_count = int(archive.attrs["complete_frames"])
        if frame_count < 1:
            raise ValueError("occupation history contains no complete frames")
        times = archive["frames"][:frame_count, 0]

    frame_indices = list(range(0, frame_count, stride))
    if include_last and frame_indices[-1] != frame_count - 1:
        frame_indices.append(frame_count - 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    total = len(frame_indices)
    for ordinal, frame_index in enumerate(frame_indices, start=1):
        time_s = float(times[frame_index])
        output_path = output_dir / (
            f"occupation_frame_{frame_index:04d}_t_{time_s:.6g}s.png"
        )
        print(
            f"[{ordinal}/{total}] frame={frame_index}, t={time_s:g}s -> "
            f"{output_path.name}",
            flush=True,
        )
        plot_compact_history(history_path, output_path, frame=frame_index)
        saved.append(output_path)

    print(f"共生成 {len(saved)} 张图: {output_dir.resolve()}", flush=True)
    return saved


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="按固定帧间隔绘制 occupation_history.h5 的能带/占据图。",
    )
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=repo_root / "results" / "20261030_172017" / "plotting_data" / "occupation_history.h5",
        help="CompactOccupationHistory HDF5 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "20261030_172017",
        help="PNG 输出目录",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="每隔多少帧绘制一张，默认 10",
    )
    parser.add_argument(
        "--no-final",
        action="store_true",
        help="不额外绘制最后一帧（默认在末帧不是 stride 整数倍时补画）",
    )
    args = parser.parse_args()
    plot_every_n_frames(
        args.history,
        args.output_dir,
        stride=args.stride,
        include_last=not args.no_final,
    )


if __name__ == "__main__":
    main()