"""Focused checks for the independent, one-sample likelihood/BIC estimator.

Run: python experiments/test_ggmm_bic_estimator.py
These checks assess implementation contracts, not universal K recovery.
"""
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import digamma, gammaln
from scipy.stats import gennorm, norm
from threadpoolctl import threadpool_limits

import ggmm_bic_estimator as gg


def observed_sample():
    # Deterministic test fixture, with no hidden true parameters passed to fit.
    parts = []
    for n, mu, a, b in [(70, -4., .8, 1.5), (60, 0., 1., 2.), (70, 4., .9, 5.)]:
        p = (np.arange(n) + .5) / n
        parts.append(gennorm.ppf(p, b, loc=mu, scale=a))
    return np.concatenate(parts)


class GGMMBICEstimatorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = observed_sample()
        cls.config = gg.GGMMBICConfig(max_components=3, max_iter=250)
        with threadpool_limits(limits=1):
            cls.fit = gg.GGMMBICEstimator(cls.config).fit(cls.x)

    def test_ggd_normalization_and_scipy_logpdf(self):
        for b in (.7, 1., 2., 8., 30.):
            with self.subTest(b=b):
                model = gg.Mixture([1.], [.35], [1.25], [b])
                mass, _ = quad(lambda x: model.pdf([x])[0], -np.inf, np.inf,
                               epsabs=2e-8, epsrel=2e-8, limit=300,
                               points=None)
                self.assertAlmostEqual(mass, 1., delta=2e-7)
                grid = np.array([-1.1, 0., .35, .36, 1.3])
                np.testing.assert_allclose(model.logpdf(grid),
                    gennorm.logpdf(grid, b, loc=.35, scale=1.25), rtol=2e-13, atol=2e-13)

    def test_b2_gaussian_relation_and_derived_sigma(self):
        model = gg.Mixture([1.], [1.2], [2.4], [2.])
        sigma = 2.4 / np.sqrt(2.)
        self.assertAlmostEqual(model.sigmas[0], sigma, places=13)
        grid = np.linspace(-7., 9., 50)
        np.testing.assert_allclose(model.logpdf(grid),
            norm.logpdf(grid, loc=1.2, scale=sigma), rtol=2e-14, atol=2e-14)

    def test_exact_nll_derivatives_away_from_cusps(self):
        x = np.array([-3.17, -1.37, -.19, .47, 1.29, 2.91, 4.03])
        for weights, means, scales, shapes in [
            ([.35, .65], [-.7, 1.1], [1.3, 2.2], [.7, 8.]),
            ([.2, .3, .5], [-1.2, .3, 2.3], [1.1, .8, 1.4], [.7, 2., 8.]),
        ]:
            model = gg.Mixture(weights, means, scales, shapes)
            theta = gg.pack(model)
            value, analytic = gg.nll_gradient(theta, x, model.k)
            self.assertAlmostEqual(value, -model.logpdf(x).mean(), places=13)
            numeric = np.empty_like(theta)
            for j in range(len(theta)):
                step = np.zeros_like(theta)
                step[j] = 1e-6
                numeric[j] = (gg.nll_gradient(theta+step, x, model.k)[0]
                              - gg.nll_gradient(theta-step, x, model.k)[0]) / 2e-6
            np.testing.assert_allclose(analytic, numeric, rtol=3e-5, atol=2e-7)

    def test_exact_component_center_has_finite_convention(self):
        # A zero convention at b<1 is not a classical derivative claim.
        for b in (.7, 1., 2., 8.):
            model = gg.Mixture([1.], [0.], [1.], [b])
            value, gradient = gg.nll_gradient(gg.pack(model), np.array([-.3, 0., .3]), 1)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.isfinite(gradient).all())
            self.assertEqual(gradient[0], 0.)

    def test_negligible_overflow_tail_component_does_not_break_gradient(self):
        # A Gaussian still gives the point finite mixture density when the
        # very flat component has an unrepresentably tiny tail density.
        model = gg.Mixture([.5, .5], [0., 0.], [1., 1.], [2., 1e300])
        value, gradient = gg.nll_gradient(gg.pack(model), np.array([.2, 2.]), 2)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(gradient).all())

    def test_scale_floor_bounds_peak_without_shape_upper_bound(self):
        t_min = brentq(digamma, 1., 2.)
        a_min = .002
        peak_bound = 1. / (2*a_min*np.exp(gammaln(t_min)))
        for b in (.01, .1, .7, 2., 8., 30., 1e6, 1e100, 1e300):
            model = gg.Mixture([1.], [0.], [a_min], [b])
            peak = np.exp(model.component_logpdf([0.])[0, 0])
            self.assertTrue(np.isfinite(peak))
            self.assertLessEqual(peak, peak_bound*(1+1e-13))
        for model in self.fit.models_.values():
            self.assertTrue(np.all(model.scales >= self.fit.a_min_*(1-1e-12)))

    def test_bic_uses_actual_density_and_4k_minus_1_parameters(self):
        for row in self.fit.selection_:
            self.assertTrue(row['finite'])
            model = self.fit.models_[row['K']]
            ll = model.logpdf(self.x).sum()
            expected = -2*ll + (4*row['K']-1)*np.log(len(self.x))
            self.assertAlmostEqual(row['log_likelihood'], ll, delta=1e-7)
            self.assertAlmostEqual(row['BIC'], expected, delta=2e-7)
        winner = min(self.fit.selection_, key=lambda r: r['BIC'])
        self.assertEqual(self.fit.n_components_, winner['K'])
        self.assertEqual(sum(r['selected'] for r in self.fit.selection_), 1)
        self.assertEqual(winner['delta_BIC'], 0.)

    def test_fit_is_deterministic_and_does_not_consume_random_initialization(self):
        with patch.object(np.random, 'default_rng', side_effect=AssertionError('RNG used')), \
             patch.object(np.random, 'seed', side_effect=AssertionError('RNG used')), \
             patch.object(np.random, 'RandomState', side_effect=AssertionError('RNG used')), \
             threadpool_limits(limits=1):
            again = gg.GGMMBICEstimator(self.config).fit(self.x.copy())
        self.assertEqual(again.n_components_, self.fit.n_components_)
        for parameter in ('weights', 'means', 'scales', 'shapes'):
            np.testing.assert_array_equal(getattr(again.density_, parameter),
                                          getattr(self.fit.density_, parameter))
        source = inspect.getsource(gg)
        self.assertNotIn('from sklearn', source)
        self.assertNotIn('import sklearn', source)
        self.assertNotIn('np.random', source)

    def test_positive_affine_equivariance_including_floor_and_bic_shift(self):
        # Use exactly represented input coordinates to avoid changing the
        # nonconvex optimization through input rounding in this contract test.
        x = np.round(self.x*1024) / 1024
        cfg = gg.GGMMBICConfig(max_components=2, max_iter=100)
        with threadpool_limits(limits=1):
            left = gg.GGMMBICEstimator(cfg).fit(x)
            right = gg.GGMMBICEstimator(cfg).fit(8*x + 16)
        self.assertEqual(left.n_components_, right.n_components_)
        self.assertAlmostEqual(right.a_min_, 8*left.a_min_, places=13)
        np.testing.assert_allclose(right.density_.weights, left.density_.weights, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(right.density_.means, 8*left.density_.means+16, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(right.density_.scales, 8*left.density_.scales, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(right.density_.shapes, left.density_.shapes, rtol=1e-8, atol=1e-8)
        for lrow, rrow in zip(left.selection_, right.selection_):
            self.assertAlmostEqual(rrow['BIC']-lrow['BIC'], 2*len(x)*np.log(8.), delta=1e-6)

    def test_every_objective_receives_the_entire_single_sample(self):
        lengths = []
        original = gg.nll_gradient
        def record(theta, x, k):
            lengths.append(len(x))
            return original(theta, x, k)
        with patch.object(gg, 'nll_gradient', side_effect=record), threadpool_limits(limits=1):
            gg.GGMMBICEstimator(gg.GGMMBICConfig(max_components=2, max_iter=30)).fit(self.x)
        self.assertGreater(len(lengths), 1)
        self.assertEqual(set(lengths), {len(self.x)})

    def test_best_visited_trial_is_not_reported_as_converged_endpoint(self):
        def artificial_objective(theta, x, k):
            gradient = np.zeros_like(theta)
            gradient[0] = 2*(theta[0]-7.)
            return (theta[0]-7.)**2, gradient
        def artificial_optimizer(fun, theta, **kwargs):
            trial = theta.copy()
            trial[0] = 7.
            fun(trial)
            return SimpleNamespace(x=theta.copy(), success=True, nit=1, message='test endpoint')
        with patch.object(gg, 'nll_gradient', side_effect=artificial_objective), \
             patch.object(gg, 'minimize', side_effect=artificial_optimizer):
            model = gg.GGMMBICEstimator(gg.GGMMBICConfig(max_components=1)).fit(self.x)
        selected = model.selection_[0]
        self.assertFalse(selected['best_is_endpoint'])
        self.assertFalse(selected['converged'])
        self.assertAlmostEqual(model.density_.means[0], model.center_ + 7*model.unit_)

    def test_invalid_and_constant_inputs_are_explicit_failures(self):
        for x in ([], [1., 2., 3.], [1.]*10, [1., 2., 3., np.nan],
                  [1., 2., 3., np.inf], np.ones((10, 2))):
            with self.subTest(x=x), self.assertRaises(ValueError):
                gg.GGMMBICEstimator().fit(x)
        for cfg in (gg.GGMMBICConfig(max_components=0),
                    gg.GGMMBICConfig(scale_floor_relative=0),
                    gg.GGMMBICConfig(scale_floor_relative=np.nan),
                    gg.GGMMBICConfig(max_iter=0)):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                gg.GGMMBICEstimator(cfg).fit(self.x)

    def test_posterior_probabilities_sum_to_one(self):
        posterior = self.fit.predict_proba(self.x)
        self.assertEqual(posterior.shape, (len(self.x), self.fit.n_components_))
        self.assertTrue(np.isfinite(posterior).all())
        self.assertTrue((posterior >= 0).all())
        np.testing.assert_allclose(posterior.sum(axis=1), 1., atol=3e-15)


if __name__ == '__main__':
    unittest.main(verbosity=2)
