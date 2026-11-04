"""JSON simulation settings; units and parameter names follow the models."""
from pathlib import Path
import json


def load_config(path: str | Path) -> dict:
    """Load simulation and relative defect files into a self-contained configuration."""
    '''
    逆天的GPT防御性编程如下，不过这里也无所谓了~
    '''
    source = Path(path)
    with source.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("simulation config must be a JSON object")
    if "defect_files" in config:
        if "defects_parameters" in config:
            raise ValueError("use defect_files or defects_parameters, not both")
        files = config.pop("defect_files")
        if (not isinstance(files, list) or not files
                or any(not isinstance(name, str) or not name.strip() for name in files)):
            raise ValueError("defect_files must be a non-empty array of file paths")
        paths = [(source.parent / name).resolve() for name in files]
        if len(set(paths)) != len(paths):
            raise ValueError("defect_files must not contain duplicate files")
        defects = []
        for defect_path in paths:
            with defect_path.open(encoding="utf-8-sig") as stream:
                parameters = json.load(stream)
            if not isinstance(parameters, dict):
                raise ValueError(f"{defect_path}: defect parameters must be a JSON object")
            defects.append(parameters)
        config["defects_parameters"] = defects
    elif isinstance(config.get("defects_parameters"), dict):
        # Historical all-in-one files remain readable.
        config["defects_parameters"] = [config["defects_parameters"]]
    defects = config.get("defects_parameters")
    if (not isinstance(defects, list) or not defects
            or any(not isinstance(parameters, dict) for parameters in defects)):
        raise ValueError("defects_parameters must be a non-empty array of JSON objects")
    for section in ("channel_parameters", "gate_parameter",
                    "simulation", "solver", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a JSON object")
    for section in ("insulating_layer_parameters", "insulating_band_parameters"):
        if not isinstance(config.get(section), list):
            raise ValueError(f"{section} must be a JSON array")
    simulation = config["simulation"]
    if simulation["period"] <= 0:
        raise ValueError("simulation.period must be positive")
    cycles = simulation["n_cycles"]
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("simulation.n_cycles must be a positive integer")
    initial = simulation["initial_f"]
    values = initial if isinstance(initial, list) else [initial] * len(defects)
    if (len(values) != len(defects)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not 0 <= value <= 1 for value in values)):
        raise ValueError("simulation.initial_f must be in [0, 1], as a scalar or one value per defect")
    labels = config["output"].get("defect_labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) != len(defects)
                               or any(not isinstance(label, str) for label in labels)):
        raise ValueError("output.defect_labels must contain one string per defect")
    return config


def save_config(config: dict, path: str | Path) -> None:
    """Save the complete input alongside simulation results."""
    Path(path).write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
                          + "\n", encoding="utf-8")
