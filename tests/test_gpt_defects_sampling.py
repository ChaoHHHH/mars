"""只测试 Defects.sampling：离散测度、独立高斯分布和正重组能约束。"""

TARGET = "src.models.defect.Defects.sampling"
from math import erf, exp, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad


def gaussian_mass(lower, upper, mean, sigma):
    return .5 * (erf((upper-mean)/(sqrt(2)*sigma))
                 - erf((lower-mean)/(sqrt(2)*sigma)))


def test_delta_energy_distribution_and_uniform_position_measure(make_defect):
    defect = make_defect(dx=.7e-9)
    data = defect.sampling()
    np.testing.assert_equal(data is defect.data, True)
    np.testing.assert_array_equal(data["x"].shape, (4, 1, 1, 1))
    np.testing.assert_allclose(data["x"][:, 0, 0, 0], [0., 2e-9/3, 4e-9/3, 2e-9], rtol=1e-15, atol=0.)
    for key, value in [("Et", -4.5), ("Er12", .8), ("Er21", 1.4)]:
        np.testing.assert_array_equal(data[key], np.full((4, 1, 1, 1), value))
    np.testing.assert_array_equal(data["weights"], np.full((4, 1, 1, 1), .25))
    np.testing.assert_array_equal(data["f"], np.zeros((4, 1, 1, 1)))
    np.testing.assert_equal(data["f"].dtype.kind, "f")


@pytest.mark.parametrize("key,sigma", [("Et", .2), ("Er12", .1), ("Er21", .2)])
def test_gaussian_bin_weights_match_independent_numerical_integration(make_defect, key, sigma):
    defect = make_defect(**{key+"_sigma": sigma, "dEt": .09, "dEr": .09})
    data = defect.sampling()
    nodes = np.unique(data[key])
    step = nodes[1]-nodes[0]
    mean = getattr(defect, key)
    edges = np.concatenate(([0.], (nodes[:-1]+nodes[1:])/2, [np.inf])) if key != "Et" else np.concatenate((nodes-step/2, [nodes[-1]+step/2]))
    normalization = .5*(1+erf(mean/(sqrt(2)*sigma))) if key != "Et" else 1.
    expected = np.array([
        quad(lambda value: exp(-.5*((value-mean)/sigma)**2)/(sqrt(2*pi)*sigma),
             lower, upper, epsabs=1e-14)[0]/normalization
        for lower, upper in zip(edges[:-1], edges[1:])
    ])
    axis = {"Et": 1, "Er12": 2, "Er21": 3}[key]
    marginal = data["weights"].sum(axis=tuple(i for i in range(4) if i != axis))
    np.testing.assert_allclose(marginal, expected, rtol=2e-12, atol=2e-15)
    np.testing.assert_equal(np.all(data["weights"] >= 0.), True)
    np.testing.assert_array_less(0., marginal)
    if key == "Et":
        np.testing.assert_array_less(marginal.sum(), 1.)
    else:
        np.testing.assert_allclose(marginal.sum(), 1., atol=2e-15, rtol=0.)


@pytest.mark.parametrize("key,sigma", [("Et", .2), ("Er12", .1), ("Er21", .2)])
def test_fine_grid_gaussian_moments_and_probability_mass(make_defect, key, sigma):
    defect = make_defect(**{key+"_sigma": sigma, "dEt": sigma/20, "dEr": sigma/20})
    data = defect.sampling()
    nodes = np.unique(data[key]);step = nodes[1]-nodes[0]
    mu = getattr(defect, key)
    weights = data["weights"]
    observed_mean = np.sum(weights*data[key])/weights.sum()
    observed_variance = np.sum(weights*(data[key]-observed_mean)**2)/weights.sum()
    phi=lambda z: exp(-z*z/2)/sqrt(2*pi)
    if key == "Et":
        a=(nodes[0]-step/2-mu)/sigma;b=(nodes[-1]+step/2-mu)/sigma
        mass = gaussian_mass(nodes[0]-step/2, nodes[-1]+step/2, mu, sigma)
        z_mean=(phi(a)-phi(b))/mass
        expected_mean=mu+sigma*z_mean
        continuous_variance=sigma*sigma*(1+(a*phi(a)-b*phi(b))/mass-z_mean*z_mean)
    else:
        mass=1.
        a=-mu/sigma
        ratio=phi(a)/(.5*(1+erf(mu/(sqrt(2)*sigma))))
        expected_mean=mu+sigma*ratio
        continuous_variance=sigma*sigma*(1+a*ratio-ratio*ratio)
    np.testing.assert_allclose(weights.sum(), mass, rtol=5e-13, atol=2e-15)
    np.testing.assert_allclose(observed_mean, expected_mean, rtol=0., atol=2e-5)
    np.testing.assert_allclose(observed_variance, continuous_variance, rtol=5e-4, atol=0.)


def test_energy_refinement_converges_to_gaussian_variance(make_defect):
    errors=[]
    for step in [.2, .05, .01]:
        defect=make_defect(Et_sigma=.2,dEt=step,non_coverage=1e-8)
        data=defect.sampling();w=data["weights"]/data["weights"].sum()
        variance=np.sum(w*(data["Et"]-defect.Et)**2)
        errors.append(abs(variance-.2**2))
    np.testing.assert_array_less(errors[1:], errors[:-1])
    np.testing.assert_allclose(errors[-1], 0., atol=1e-5, rtol=0.)


def test_four_dimensional_measure_keeps_reorganization_axes_independent(make_defect):
    defect=make_defect(Et_sigma=.2,Er12_sigma=.1,Er21_sigma=.2,dEt=.15,dEr=.1)
    data=defect.sampling();w=data["weights"]/data["weights"].sum()
    nodes=[np.unique(data[key]) for key in ["x","Et","Er12","Er21"]]
    np.testing.assert_array_equal(data["weights"].shape,[len(axis) for axis in nodes])
    means={key:np.sum(w*data[key]) for key in ["Et","Er12","Er21"]}
    covariance=np.sum(w*(data["Er12"]-means["Er12"])*(data["Er21"]-means["Er21"]))
    np.testing.assert_allclose(covariance,0.,atol=2e-17,rtol=0.)
    spatial_mass=data["weights"].sum(axis=(1,2,3))
    np.testing.assert_allclose(spatial_mass,np.full(len(nodes[0]),data["weights"].sum()/len(nodes[0])),rtol=1e-15)
    np.testing.assert_array_less(0.,data["Er12"])
    np.testing.assert_array_less(0.,data["Er21"])


@pytest.mark.parametrize("key", ["Er12", "Er21"])
def test_positive_reorganization_distribution_is_conditional_gaussian(make_defect, key):
    defect=make_defect(**{key:.55,key+"_sigma":.70,"dEr":.25})
    data=defect.sampling()
    nodes=np.unique(data[key])
    edges=np.concatenate(([0.], (nodes[:-1]+nodes[1:])/2, [np.inf]))
    normalization=.5*(1+erf(.55/(sqrt(2)*.70)))
    expected=np.array([quad(
        lambda value: exp(-.5*((value-.55)/.70)**2)/(sqrt(2*pi)*.70),
        lo,hi,epsabs=1e-14)[0]/normalization
        for lo,hi in zip(edges[:-1],edges[1:])])
    axis={"Er12":2,"Er21":3}[key]
    marginal=data["weights"].sum(axis=tuple(i for i in range(4) if i!=axis))
    np.testing.assert_allclose(marginal,expected,rtol=2e-12,atol=2e-15)
    np.testing.assert_allclose(marginal.sum(),1.,atol=2e-15,rtol=0.)
    np.testing.assert_array_less(0.,nodes)


def test_both_wide_reorganization_distributions_preserve_total_population(make_defect):
    defect=make_defect(Er12=.55,Er21=.55,Er12_sigma=.70,Er21_sigma=.70,dEr=.25)
    data=defect.sampling()
    np.testing.assert_allclose(data["weights"].sum(),1.,atol=2e-15,rtol=0.)
    marginal12=data["weights"].sum(axis=(0,1,3))
    marginal21=data["weights"].sum(axis=(0,1,2))
    joint=data["weights"].sum(axis=(0,1))
    np.testing.assert_allclose(joint,np.outer(marginal12,marginal21),atol=2e-15)
    # donor 空态时总电荷应为 q * Nt * 厚度，避免浓度随 Gaussian 负尾改变。
    np.testing.assert_allclose(defect.get_charge_sheets().sum(),
                               1.602176634e-19*defect.Nt*(defect.x_range[1]-defect.x_range[0]),rtol=2e-15)


@pytest.mark.parametrize("key",["Er12","Er21"])
@pytest.mark.parametrize("mean",[0.,-1.])
def test_nonpositive_delta_reorganization_has_no_valid_positive_distribution(make_defect,key,mean):
    with np.testing.assert_raises(ValueError):
        make_defect(**{key:mean,key+"_sigma":0.})


@pytest.mark.parametrize("mean",[-2.,.55])
def test_truncated_reorganization_moments_converge(make_defect,mean):
    sigma=.70
    from scipy.special import log_ndtr
    a=-mean/sigma
    ratio=np.exp(-a*a/2-np.log(2*pi)/2-log_ndtr(-a))
    expected_mean=mean+sigma*ratio
    expected_var=sigma*sigma*(1+a*ratio-ratio*ratio)
    errors=[]
    for spacing in [.25,.05,.005]:
        defect=make_defect(Er12=mean,Er12_sigma=sigma,dEr=spacing,non_coverage=1e-8)
        w=defect.data["weights"];values=defect.data["Er12"]
        actual_mean=np.sum(w*values)
        actual_var=np.sum(w*(values-actual_mean)**2)
        errors.append(abs(actual_mean-expected_mean)+abs(actual_var-expected_var))
    np.testing.assert_array_less(errors[1:],errors[:-1])
    np.testing.assert_allclose(errors[-1],0.,atol=2e-5,rtol=0.)


def test_resampling_rebuilds_empty_electron_occupation(make_defect):
    defect=make_defect();defect.data["f"].fill(.9)
    data=defect.sampling()
    np.testing.assert_array_equal(data["f"],np.zeros_like(data["weights"]))


@pytest.mark.parametrize('non_coverage',[1e-2,1e-4])
def test_requested_gaussian_tail_coverage_and_grid_resolution(make_defect,non_coverage):
    defect=make_defect(Et_sigma=.2,dEt=.03,non_coverage=non_coverage)
    data=defect.sampling();nodes=np.unique(data['Et'])
    lower_probability=.5*(1+erf((nodes[0]-defect.Et)/(sqrt(2)*.2)))
    upper_probability=.5*(1-erf((nodes[-1]-defect.Et)/(sqrt(2)*.2)))
    np.testing.assert_allclose([lower_probability,upper_probability],non_coverage,atol=2e-16,rtol=1e-11)
    np.testing.assert_equal(np.all(np.diff(nodes)<=.03+1e-15),True)
