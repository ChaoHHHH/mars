"""只测试有效速率纯函数：声明的DFT归一化、空态、热平衡与下溢。"""

TARGET = "src.physics.solver.calculate_total_rates.calculate_total_rates"
import numpy as np
import pytest
from src.physics.solver.calculate_total_rates import calculate_total_rates

KB=8.6173332621e-5


def parameters(**changes):
    result=dict(log_k12_vbm=np.log(np.array([2.])),log_k12_cbm=np.log(np.array([11.])),
        transition_level_local=np.array([0.]),T=300.,psi_s=KB*300.*np.log(2.),
        cbm_local=0.,ef=0.,n0=2e20,p0=5e22,
        n_DFT_m3=1e27, Er12=np.array([1.]), Er21=np.array([1.]),
        log_k12_gate=np.array([-np.inf]),ef_gate=0.)
    result.update(changes);return result


def test_declared_normalization_and_cbm_empty_state():
    r=calculate_total_rates(**parameters())
    # p界面=p0/2，n界面=2*n0；CBM=EF时空态概率1/2；不乘缺陷浓度因子。
    expected_v=2*(5e22/2)/1e27;expected_c=11*(2*2e20)*.5/1e27
    for key in ["k12_vbm_effective","k21_vbm_effective"]:np.testing.assert_allclose(r[key],expected_v,rtol=1e-13)
    for key in ["k12_cbm_effective","k21_cbm_effective"]:np.testing.assert_allclose(r[key],expected_c,rtol=1e-13)
    np.testing.assert_allclose(r["rate_in"],expected_v+expected_c,rtol=1e-13)
    np.testing.assert_allclose(r["rate_out"],expected_v+expected_c,rtol=1e-13)


@pytest.mark.parametrize("offset",[-.2,0.,.2])
def test_both_reservoir_channels_share_fermi_equilibrium(offset):
    r=calculate_total_rates(**parameters(transition_level_local=np.array([offset])))
    expected=1/(1+np.exp(offset/(KB*300.)))
    np.testing.assert_allclose(r["rate_in"]/(r["rate_in"]+r["rate_out"]),expected,rtol=1e-13)
    np.testing.assert_allclose(r["log_rate_in"]-r["log_rate_out"],-offset/(KB*300.),atol=1e-13)


def test_zero_coupling():
    a=calculate_total_rates(**parameters(log_k12_cbm=np.array([-np.inf])))
    np.testing.assert_array_equal(a["k12_cbm_effective"],[0.])
    np.testing.assert_array_equal(a["k21_cbm_effective"],[0.])


def test_reverse_rate_survives_forward_underflow_without_mutating_inputs():
    p=parameters(log_k12_vbm=np.array([-900.]),log_k12_cbm=np.array([-np.inf]),
        p0=1e27,psi_s=0.,transition_level_local=np.array([-250*KB*300.]))
    before=p["log_k12_vbm"].copy();r=calculate_total_rates(**p)
    np.testing.assert_equal(r["rate_out"][0],0.)
    np.testing.assert_array_less(0.,r["rate_in"])
    np.testing.assert_allclose(r["log_rate_in"]-r["log_rate_out"],250.,atol=1e-12)
    np.testing.assert_array_equal(p["log_k12_vbm"],before)


def test_reference_density_scales_all_rates_without_changing_equilibrium():
    a=calculate_total_rates(**parameters())
    b=calculate_total_rates(**parameters(n_DFT_m3=2e27))
    for key in ('rate_in','rate_out','k12_vbm_effective','k21_vbm_effective',
                'k12_cbm_effective','k21_cbm_effective'):
        np.testing.assert_allclose(b[key], a[key]/2, rtol=1e-13)


@pytest.mark.parametrize('gate_ef',[-.1,0.,.1])
def test_gate_only_rates_follow_gate_fermi_level_without_density_factor(gate_ef):
    p=parameters(log_k12_vbm=np.array([-np.inf]),log_k12_cbm=np.array([-np.inf]),
                 log_k12_gate=np.log(np.array([3.])),ef_gate=gate_ef)
    r=calculate_total_rates(**p)
    expected_in=3*np.exp(gate_ef/(KB*300.))
    np.testing.assert_allclose(r['rate_out'],[3.],rtol=1e-13)
    np.testing.assert_allclose(r['rate_in'],[expected_in],rtol=1e-13)
    np.testing.assert_allclose(r['rate_in']/(r['rate_in']+r['rate_out']),
                               1/(1+np.exp(-gate_ef/(KB*300.))),rtol=1e-13)
    p['n_DFT_m3']*=10
    other=calculate_total_rates(**p)
    np.testing.assert_array_equal(other['rate_in'],r['rate_in'])
    np.testing.assert_array_equal(other['rate_out'],r['rate_out'])


def test_gate_and_channel_rates_add_with_separate_fermi_levels():
    p=parameters(log_k12_gate=np.log(np.array([3.])),ef_gate=.1)
    total=calculate_total_rates(**p)
    channel=calculate_total_rates(**parameters())
    np.testing.assert_allclose(total['rate_out'],channel['rate_out']+3.,rtol=1e-13)
    np.testing.assert_allclose(total['rate_in'],channel['rate_in']+3*np.exp(.1/(KB*300.)),rtol=1e-13)
    np.testing.assert_allclose(np.exp(total['log_rate_in']),total['rate_in'],rtol=1e-13)
    np.testing.assert_allclose(np.exp(total['log_rate_out']),total['rate_out'],rtol=1e-13)


def test_gate_reverse_rate_survives_emission_underflow():
    r=calculate_total_rates(**parameters(log_k12_vbm=np.array([-np.inf]),
        log_k12_cbm=np.array([-np.inf]),log_k12_gate=np.array([-900.]),
        ef_gate=250*KB*300.))
    np.testing.assert_array_equal(r['rate_out'],[0.])
    np.testing.assert_allclose(r['rate_in'],[np.exp(-650.)],rtol=1e-13)


@pytest.mark.parametrize('gate_ef',[-10.,0.,10.])
def test_zero_gate_rate_is_closed_at_any_finite_gate_fermi_level(gate_ef):
    reference=calculate_total_rates(**parameters())
    result=calculate_total_rates(**parameters(ef_gate=gate_ef))
    for key in ('k12_gate_effective','k21_gate_effective'):
        np.testing.assert_array_equal(result[key],[0.])
    for key in ('rate_in','rate_out','log_rate_in','log_rate_out'):
        np.testing.assert_array_equal(result[key],reference[key])


def test_all_channels_can_be_disabled_with_negative_infinite_log_rates():
    result=calculate_total_rates(**parameters(log_k12_vbm=np.array([-np.inf]),
                                             log_k12_cbm=np.array([-np.inf])))
    np.testing.assert_array_equal(result['rate_in'],[0.])
    np.testing.assert_array_equal(result['rate_out'],[0.])
    np.testing.assert_array_equal(result['log_rate_in'],[-np.inf])
    np.testing.assert_array_equal(result['log_rate_out'],[-np.inf])


@pytest.mark.parametrize("ratio", [0.25, 1.0, 4.0])
def test_curvature_entropy_changes_channel_and_gate_equilibrium(ratio):
    # Equal minimum energies: state weights are inverse square roots of curvature.
    p = parameters(psi_s=0., Er12=np.array([ratio]), Er21=np.array([1.]),
                   log_k12_gate=np.log(np.array([3.])), ef_gate=0.)
    result = calculate_total_rates(**p)
    expected_ratio = np.sqrt(ratio)
    for channel in ("vbm", "cbm", "gate"):
        np.testing.assert_allclose(result[f"k21_{channel}_effective"],
                                   result[f"k12_{channel}_effective"] * expected_ratio)
    expected_occupied = expected_ratio / (1. + expected_ratio)
    np.testing.assert_allclose(result["rate_in"] / (result["rate_in"] + result["rate_out"]),
                               expected_occupied)
