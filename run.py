"""Run a simulation: python run.py --config config/simulation.json."""
import argparse
from pathlib import Path
from src.io.config import load_config
from src.io.simulation import run_simulation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/simulation.json")
    parser.add_argument("--output", type=Path, help="Override the result root directory")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output if args.output is not None else Path(config["output"].get("directory", "results"))
    if not output.is_absolute():
        output = ROOT / output
    result_dir = run_simulation(config, output)
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
