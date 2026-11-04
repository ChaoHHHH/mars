from src.physics.solver.advance import advance
from src.io.band_occupation import save_band_occupation
from src.physics.solver.pre_equilibrate import pre_equilibrate
from src.models.device_channel import Channel
from src.models.device_gate import Gate
from src.models.device_insulatinglayer import InsulatingLayer
from src.models.device_mis1d import MOS1D
from src.models.defect import Defects
import numpy as np
from contextlib import closing, nullcontext

from pathlib import Path
from .config import save_config

def run_simulation(config: dict, output_dir: str | Path) -> Path:
    """Run the configured waveform and return its result directory."""
    channel = Channel(**config["channel_parameters"])
    layers = [InsulatingLayer(**p) for p in config["insulating_layer_parameters"]]
    mos = MOS1D(gate=Gate(**config["gate_parameter"]), channel=channel, layers=layers)
    parameters = config["defects_parameters"]
    if isinstance(parameters, dict):
        parameters = [parameters]
    for defect_parameters in parameters:
        mos.add_defects(Defects(**defect_parameters))
    simulation, output = config["simulation"], config["output"]
    T = simulation["T"]
    equilibration_Vg = simulation["equilibration_Vg"]
    initial_f = simulation["initial_f"]
    period, n_cycles = simulation["period"], simulation["n_cycles"]
    Vmin, Vmax = simulation["Vmin"], simulation["Vmax"]
    save_compact_history = output["save_compact_history"]
    occupation_energy_bins = output["occupation_energy_bins"]
    insulating_band_parameters = config["insulating_band_parameters"]
    dt_initial_s = config["solver"]["dt_initial_s"]
    dt_max_s = config["solver"]["dt_max_s"]
    electrostatics_recalc_threshold_V = config["solver"]["electrostatics_recalc_threshold_V"]
    half_period = period / 2.0
    phases = [(Vmin, half_period), (Vmax, half_period)] * n_cycles

    initial_values = initial_f if isinstance(initial_f, list) else [initial_f] * len(mos.defects)
    if len(initial_values) != len(mos.defects):
        raise ValueError("initial_f must provide one value per defect")
    for defects, value in zip(mos.defects, initial_values):
        defects.data["f"].fill(value)
    # 在指定偏压下预平衡，以收敛状态为漂移参考；不计入波形时间。
    equilibrium = pre_equilibrate(
        mos,
        Vg=equilibration_Vg, T=T,
    )
    print(f"Pre-equilibrated: steps={equilibrium['steps']}, "
          f"Vth={equilibrium['Vth']:+.6e} V, dVth={equilibrium['dVth']:+.6e} V")
    from datetime import datetime

    result_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plotting_dir = result_dir / "plotting_data"
    plotting_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, result_dir / "parameters.json")
    if output.get("save_pre_equilibrium_plot", True):
        save_band_occupation(
            mos, T, insulating_band_parameters,
            plotting_dir / "pre_equilibrium" / "band_occupation.png",
            defect_labels=output.get("defect_labels"),
            title=f"Pre-equilibrium: Vg={mos.Vg:+.3f} V, T={T:g} K",
        )

    dVth_reference = mos.get_delta_Vth()
    Vth0 = mos.get_Vth0(T)
    history = []
    elapsed = 0.0

    if save_compact_history:
        from src.io.compact_history import CompactOccupationHistory
        archive_context = closing(CompactOccupationHistory(
            plotting_dir / "occupation_history.h5", mos, insulating_band_parameters,
            bins=occupation_energy_bins, defect_labels=output.get("defect_labels"),
        ))
    else:
        archive_context = nullcontext(None)
    with archive_context as archive:
        for phase, (Vg, duration) in enumerate(phases):
            mos.set_Vg(Vg)
            def record_step(device, phase_time):
                archive.append(device, T, elapsed + phase_time, phase=phase)

            phase_result = advance(
                mos, duration, T,
                on_step=record_step if archive is not None else None,
                dt_initial_s=dt_initial_s, dt_max_s=dt_max_s,
                electrostatics_recalc_threshold_V=electrostatics_recalc_threshold_V,
            )
            phase_history = phase_result["history"]
            for phase_time, dt, psi_s, mean_f, dVth in phase_history:
                history.append([elapsed + phase_time, phase, phase_time, dt, Vg,
                                psi_s, mean_f, Vth0, dVth, Vth0 + dVth])
            elapsed += duration
            print(f"phase={phase}, t={elapsed:g} s, Vg={Vg:+.2f} V, "
                  f"steps={phase_result['steps']}, "
                  f"environment_refreshes={phase_result['environment_refreshes']}, "
                  f"mean_f={mean_f:.6f}, dVth={dVth:+.6e} V")

    result = np.asarray(history, dtype=float)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Simulation produced non-finite results")

    threshold_voltage_drift = np.column_stack((
        result,
        result[:, 8] - dVth_reference,
        (result[:, 8] - dVth_reference) * 1e3,
    ))
    drift_path = result_dir / "threshold_voltage_drift.csv"
    np.savetxt(
        drift_path,
        threshold_voltage_drift,
        delimiter=",",
        header=(
            "time_s,phase,phase_time_s,dt_s,Vg_V,psi_s_environment_V,mean_f,"
            "Vth0_V,delta_Vth_V,Vth_V,threshold_voltage_drift_V,"
            "threshold_voltage_drift_mV"
        ),
        comments="",
    )
    return result_dir
