"""Run the original uniform-channel impedance model with portable paths."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Type

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "research_code" / "theory_and_sensitivity" / "MeanImped.py"


def load_impedance_class() -> Type[object]:
    """Load the historical class without changing its source or directory layout."""
    spec = importlib.util.spec_from_file_location("historical_mean_impedance", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model module: {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Impedeance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the theoretical absorption curve for a uniform labyrinth."
    )
    parser.add_argument("--frequency-min", type=int, default=135, help="First frequency in Hz.")
    parser.add_argument("--frequency-max", type=int, default=695, help="Last frequency in Hz.")
    parser.add_argument("--frequency-step", type=int, default=1, help="Frequency step in Hz.")
    parser.add_argument("--channel-width-mm", type=float, default=2.0)
    parser.add_argument("--height-mm", type=float, default=28.0)
    parser.add_argument("--labyrinth-width-mm", type=float, default=28.0)
    parser.add_argument("--channels", type=int, default=7)
    parser.add_argument(
        "--partition-thickness-mm",
        type=float,
        default=1.0,
        help="Solid partition thickness used in the total aperture width.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "generated" / "theoretical-absorption.png",
        help="Output PNG path. A CSV with the same stem is also written.",
    )
    parser.add_argument("--show", action="store_true", help="Also open the Matplotlib window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frequency_step <= 0 or args.frequency_max < args.frequency_min:
        raise ValueError("Frequency bounds and step are inconsistent.")
    if args.channels < 1:
        raise ValueError("--channels must be at least 1.")

    millimeter = 1e-3
    channel_width = args.channel_width_mm * millimeter
    height = args.height_mm * millimeter
    labyrinth_width = args.labyrinth_width_mm * millimeter
    partition = args.partition_thickness_mm * millimeter
    total_width = args.channels * channel_width + (args.channels - 1) * partition

    model_class = load_impedance_class()
    model = model_class(
        args.frequency_min,
        args.frequency_max + 1,
        channel_width,
        labyrinth_width,
        total_width,
        height,
        args.channels,
        args.frequency_step,
    )
    absorption = np.asarray(model.A(), dtype=float)
    frequency = np.arange(
        args.frequency_min,
        args.frequency_max + 1,
        args.frequency_step,
        dtype=int,
    )
    if absorption.shape != frequency.shape or not np.isfinite(absorption).all():
        raise RuntimeError("The theoretical model returned an invalid response array.")

    peak_index = int(np.argmax(absorption))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = args.output.with_suffix(".csv")
    np.savetxt(
        csv_path,
        np.column_stack((frequency, absorption)),
        delimiter=",",
        header="frequency_hz,absorption_coefficient",
        comments="",
    )

    figure_status = "not written (Matplotlib is not installed)"
    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(frequency, absorption, linewidth=2, label="Theoretical response")
        axis.scatter(
            frequency[peak_index],
            absorption[peak_index],
            color="#c23b22",
            s=30,
            zorder=3,
            label="Peak",
        )
        axis.set(xlabel="Frequency (Hz)", ylabel="Absorption coefficient")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output, dpi=180)
        if args.show:
            plt.show()
        plt.close(figure)
        figure_status = str(args.output)
    except ImportError:
        pass

    print(f"Peak: {frequency[peak_index]} Hz, alpha={absorption[peak_index]:.6f}")
    print(f"Figure: {figure_status}")
    print(f"Data: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
