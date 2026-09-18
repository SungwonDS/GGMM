"""Result lifecycle and fresh-fit timing; use alongside estimator math checks."""
import contextlib
import io
import unittest
from unittest.mock import patch
from dataclasses import replace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ggmm_bic_experiment as ex
from ggmm_bic_estimator import GGMMBICConfig, GGMMBICEstimator, Mixture


class ResultLifecycleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = Mixture(**ex.TRUTH_PARAMETERS)
        cls.x = ex.sample_mixture(cls.truth, 120, 83)
        cls.config = GGMMBICConfig(max_components=2, max_iter=20)
        cls.baseline = ex.run_experiment(cls.x, config=cls.config, truth=cls.truth)

    def test_result_owns_sample_and_old_run_survives_new_fit(self):
        x = self.x.copy()
        first = ex.run_experiment(x, config=self.config, truth=self.truth)
        x[:] = 7
        np.testing.assert_array_equal(first['x'], self.x)
        second = ex.run_experiment(self.x+50, config=replace(self.config, max_components=3))
        self.assertIsNot(first['ggmm'], second['ggmm'])
        self.assertEqual(first['config']['max_components'], 2)
        self.assertEqual(second['config']['max_components'], 3)
        self.assertLess(first['ggmm'].density_.means.max(), second['ggmm'].density_.means.min())

    def test_failed_refit_cannot_expose_previous_selected_density(self):
        model = GGMMBICEstimator(self.config).fit(self.x)
        with self.assertRaises(ValueError):
            model.fit(np.ones(8))
        self.assertFalse(hasattr(model, 'density_'))
        with self.assertRaises(RuntimeError):
            ex.require_result(None)

    def test_fresh_timing_contains_all_candidates_and_starts(self):
        run = self.baseline
        fit = run['ggmm']
        self.assertGreater(run['fit_seconds'], 0)
        self.assertLessEqual(sum(r['seconds'] for r in fit.selection_), run['fit_seconds'])
        for rec in fit.selection_:
            starts = [s for s in fit.starts_ if s['K'] == rec['K']]
            self.assertEqual(rec['starts_count'], len(starts))
            self.assertLessEqual(rec['initialization_seconds']+sum(s['seconds'] for s in starts), rec['seconds'])
            for start in starts:
                self.assertGreater(start['nfev'], 0)
                self.assertGreaterEqual(start['seconds'], start['optimization_seconds'])

    def test_kmax_comparison_really_runs_fit_for_each_cap(self):
        seen = []
        original = GGMMBICEstimator.fit
        def observe(model, values):
            seen.append((model.config.max_components, np.array(values)))
            return original(model, values)
        with patch.object(GGMMBICEstimator, 'fit', observe), patch.object(ex, 'show_table'), \
             patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
            runs, table = ex.compare_kmax(self.baseline, [1, 3])
        self.assertEqual([cap for cap, _ in seen], [1, 3])
        for _, values in seen:
            np.testing.assert_array_equal(values, self.baseline['x'])
        self.assertEqual(table.K_max.tolist(), [1, 3])
        self.assertEqual(self.baseline['config']['max_components'], 2)
        self.assertEqual(len({r['run_id'] for r in runs}), 2)
        plt.close('all')

    def test_reports_accept_new_run_without_truth_or_other_report_state(self):
        shifted = ex.run_experiment(self.x+30, config=self.config)
        with patch.object(ex, 'show_table'), patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
            table = ex.report_all_k(shifted)
            same = ex.report_same_k(shifted)
            kde = ex.report_kde(shifted)
        self.assertTrue(table.IAE.isna().all())
        self.assertTrue(kde.ISE.isna().all())
        self.assertTrue((kde.Fit_n >= 3).all())
        plt.close('all')

    def test_Kmax_20_is_supported_without_cached_models(self):
        run = ex.run_experiment(self.x, config=replace(self.config, max_components=20, max_iter=3))
        self.assertEqual([r['K'] for r in run['ggmm'].selection_], list(range(1, 21)))
        self.assertEqual(run['ggmm'].diagnostics_['effective_Kmax'], 20)

    def test_scale_collapse_increases_likelihood_at_fixed_component_count(self):
        table = ex.scale_collapse_path(self.baseline)
        self.assertEqual(table.K.nunique(), 1)
        self.assertTrue((table.a > 0).all())
        # After the spike isolates one observation, LL grows at rate -log(a).
        tail = table.loc[table.Neg_log10_relative_a >= 16]
        np.testing.assert_allclose(np.diff(tail.LL),
            np.diff(tail.Neg_log10_relative_a)*np.log(10), rtol=1e-9)
        self.assertTrue((np.diff(tail.BIC) < 0).all())

    def test_boundary_alarm_with_descending_tail_or_winner_at_cap(self):
        rows = [dict(K=k, finite=True, BIC=b, converged=True)
                for k, b in enumerate([1, 6, 4, 2], 1)]
        status = ex.search_boundary(rows)
        self.assertFalse(status['selected_at_boundary'])
        self.assertTrue(status['review_larger_K'])
        rows[-1]['BIC'] = 0
        self.assertTrue(ex.search_boundary(rows)['selected_at_boundary'])
        rows[-1].update(finite=False, BIC=np.inf, converged=False)
        status = ex.search_boundary(rows)
        self.assertFalse(status['last_candidates_BIC_decreasing'])
        self.assertTrue(status['boundary_not_converged'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
