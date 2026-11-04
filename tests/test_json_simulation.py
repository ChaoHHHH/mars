"""Exercise JSON input through the public command line and result files."""
from pathlib import Path
import json
from copy import deepcopy
import subprocess
import sys
import h5py
import numpy as np
import pytest
from src.io.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('defect_count', [1, 2])
def test_json_cli(tmp_path, record, defect_count):
    config = load_config(ROOT/'config/simulation.json')
    config['defects_parameters'][0].update(Et_sigma=0., Er12_sigma=0., Er21_sigma=0., dx=2.1e-9,
        Vc_gate=1e-7,decay_gate=1e9,temperature_vbm=.1,temperature_cbm=.2,temperature_gate=.3)
    config['simulation'].update(period=.02, n_cycles=1)
    config['output'].update(save_compact_history=record, occupation_energy_bins=8)
    if defect_count == 2:
        second = deepcopy(config['defects_parameters'][0])
        second.update(Et=-4.8, Nt=1e24, type='acceptor', dx=1.05e-9)
        config['defects_parameters'].append(second)
        config['simulation']['initial_f'] = [0., 1.]
    config['output']['defect_labels'] = [f'Species {i+1}' for i in range(defect_count)]
    external = deepcopy(config)
    parameters = external.pop('defects_parameters')
    external['defect_files'] = [f'defect_{i+1}.json' for i in range(defect_count)]
    for name, values in zip(external['defect_files'], parameters):
        (tmp_path/name).write_text(json.dumps(values), encoding='utf-8-sig')
    # A spare file must not silently add another species.
    (tmp_path/'defect_99.json').write_text('{}')
    source = tmp_path/'simulation.json'
    source.write_text(json.dumps(external), encoding='utf-8-sig')
    output = tmp_path/'results'
    subprocess.run([sys.executable, str(ROOT/'run.py'), '--config', str(source),
                    '--output', str(output)], cwd=tmp_path, check=True, capture_output=True)
    folder, = output.iterdir()
    np.testing.assert_equal(json.loads((folder/'parameters.json').read_text()), config)
    np.testing.assert_equal(load_config(folder/'parameters.json'), config)
    data = np.genfromtxt(folder/'threshold_voltage_drift.csv', delimiter=',', names=True)
    np.testing.assert_allclose(data['time_s'][-1], .02)
    np.testing.assert_equal(np.all(np.isfinite(data.view(float))), True)
    np.testing.assert_allclose(data['threshold_voltage_drift_mV'], data['threshold_voltage_drift_V']*1000)
    np.testing.assert_equal((folder/'plotting_data/pre_equilibrium/band_occupation.png').exists(), True)
    with np.load(folder/'plotting_data/pre_equilibrium/band_occupation.npz') as snapshot:
        for i in range(defect_count):
            assert f'defect_{i}_f' in snapshot
    archive = folder/'plotting_data/occupation_history.h5'
    np.testing.assert_equal(archive.exists(), record)
    if record:
        with h5py.File(archive) as stream:
            np.testing.assert_equal(stream.attrs['complete_frames'], len(data))
            assert len(stream['defects']) == defect_count
            for i in range(defect_count):
                group = stream[f'defects/{i}']
                assert group.attrs['label'] == f'Species {i+1}'
                assert group.attrs['Nt_m3'] == config['defects_parameters'][i]['Nt']
                assert group['f'].shape[0] == len(data)


def test_invalid_cycle_count(tmp_path):
    config = load_config(ROOT/'config/simulation.json')
    config['simulation']['n_cycles'] = 0
    path = tmp_path/'settings.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='n_cycles'):
        load_config(path)


def test_legacy_single_defect_config(tmp_path):
    config = load_config(ROOT/'config/simulation.json')
    parameters = config['defects_parameters'][0]
    config['defects_parameters'] = parameters
    source = tmp_path/'legacy.json'
    source.write_text(json.dumps(config))
    assert load_config(source)['defects_parameters'] == [parameters]


@pytest.mark.parametrize('files', [[], 'defect_1.json', [None], [''],
                                   ['defect_1.json', './defect_1.json']])
def test_invalid_defect_file_list(tmp_path, files):
    config = json.loads((ROOT/'config/simulation.json').read_text())
    config['defect_files'] = files
    source = tmp_path/'simulation.json'
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='defect_files'):
        load_config(source)


def test_missing_defect_file_reports_its_path(tmp_path):
    config = json.loads((ROOT/'config/simulation.json').read_text())
    source = tmp_path/'simulation.json'
    source.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match='defect_1.json'):
        load_config(source)


@pytest.mark.parametrize('field,value', [('initial_f', [0., 1.]),
                                        ('initial_f', 1.1), ('initial_f', float('nan')),
                                        ('defect_labels', ['one', 'two'])])
def test_per_species_settings_have_matching_lengths_and_valid_values(tmp_path, field, value):
    config = load_config(ROOT/'config/simulation.json')
    config['output' if field == 'defect_labels' else 'simulation'][field] = value
    source = tmp_path/'simulation.json'
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_config(source)


def test_defect_file_order_is_explicit_not_alphabetical(tmp_path):
    config = json.loads((ROOT/'config/simulation.json').read_text())
    config['defect_files'] = ['defect_10.json', 'defect_2.json']
    config['output']['defect_labels'] = ['ten', 'two']
    (tmp_path/'defect_10.json').write_text('{"Et": -4.1}')
    (tmp_path/'defect_2.json').write_text('{"Et": -4.2}')
    source = tmp_path/'simulation.json'
    source.write_text(json.dumps(config))
    assert [p['Et'] for p in load_config(source)['defects_parameters']] == [-4.1, -4.2]
