"""Shared evaluation, plotting and experiments for vggmm_merge_refit.

Importing this module does not generate samples, fit models, download data, or
write files. The notebook owns experimental settings; this module supplies
reusable operations. Density errors are defined exactly once.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import ssl
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp
from scipy.stats import gennorm, norm, laplace, t, skewnorm
from sklearn.mixture import BayesianGaussianMixture
from threadpoolctl import threadpool_limits

import vggmm_merge_refit as vg
from vggmm_merge_refit import Mixture, VGGMMConfig, VariationalGGMM

MODEL_FILE = Path(vg.__file__).resolve()
ABALONE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
ABALONE_SOURCE = "https://archive.ics.uci.edu/dataset/1/abalone"
METHODS = ("VGGMM-base", "VGGMM-merge", "VGM")


def sample_mixture(truth, n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice(truth.k, size=n, p=truth.weights)
    return np.asarray(gennorm.rvs(
        truth.shapes[labels],
        loc=truth.means[labels],
        scale=truth.scales[labels],
        random_state=rng,
    ), dtype=float)

def active_view(full, threshold):
    keep = full.weights > threshold
    retained = float(full.weights[keep].sum())
    excluded = float(full.weights[~keep].sum())

    density = (
        Mixture(
            full.weights[keep] / retained,
            full.means[keep],
            full.scales[keep],
            full.shapes[keep],
        )
        if keep.any() else None
    )
    return density, int(keep.sum()), excluded

def show_table(frame):
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(frame)
            return
    except ImportError:
        pass
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.7g}"))

def _density_points(density, tail):
    if hasattr(density, "evaluation_points"):
        return density.evaluation_points(tail)
    return _ggd_points(density, tail)


def _density_peak_upper(density):
    if hasattr(density, "peak_upper"):
        return float(density.peak_upper())
    return _ggd_peak_upper(density)


def _legendre_rule(order):
    # NumPy's Gauss-Legendre rule is exact for polynomials of degree 2*order-1.
    return np.polynomial.legendre.leggauss(order)

def _ggd_points(fitted, tail):
    probabilities = np.unique(np.array([
        tail, 1e-6, 1e-4, .001, .01, .1, .25, .5,
        .75, .9, .99, .999, 1-1e-4, 1-1e-6, 1-tail,
    ]))
    return gennorm.ppf(probabilities[:, None], np.asarray(fitted.shapes),
                       loc=np.asarray(fitted.means), scale=np.asarray(fitted.scales)).ravel()

def _ggd_peak_upper(fitted):
    # Each GGD attains its maximum at its location. The weighted sum of
    # component maxima bounds the entire mixture, even with distinct locations.
    weights = np.asarray(fitted.weights)
    log_peaks = (-np.log(2.)-np.log(np.asarray(fitted.scales))
                 -gammaln(1+1/np.asarray(fitted.shapes)))
    return float(np.sum(weights*np.exp(log_peaks)))

def _omitted_mass(model, left, right):
    values = np.asarray(model.cdf(np.array([left, right])), dtype=float).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise FloatingPointError("Density evaluation needs two finite CDF values.")
    if np.any(values < -1e-12) or np.any(values > 1+1e-12) or values[1] < values[0]-1e-12:
        raise ValueError("Density evaluation received invalid CDF values.")
    return float(np.clip(values[0]+(1-values[1]), 0., 1.))

_LEGENDRE = {order: _legendre_rule(order) for order in (32, 64)}

def density_errors(truth, fitted, tail=1e-9, atol=1e-7, rtol=1e-5, max_refinement=16):
    """Estimate IAE/ISE adaptively, and bound error omitted outside the interval.

    IAE = integral |fitted(x)-truth(x)| dx.
    ISE = integral (fitted(x)-truth(x))**2 dx.
    ``*_quad_error`` estimates numerical error using 32/64-point rule differences;
    it is not a rigorous quadrature bound. ``*_tail_upper`` bounds the omitted
    tails if truth.peak_upper() is a valid global density upper bound.
    Quadrature_OK checks rule agreement and refinement completion only.
    It does not assert model normalization or statistical goodness of fit.
    """
    if (not np.isfinite(tail) or not 0 < tail < .5
        or not np.isfinite(atol) or atol <= 0
        or not np.isfinite(rtol) or rtol < 0
        or not isinstance(max_refinement, (int, np.integer))
        or isinstance(max_refinement, (bool, np.bool_)) or max_refinement < 0):
        raise ValueError("Invalid density-error quadrature settings.")
    reference_points = np.asarray(_density_points(truth, tail), dtype=float).ravel()
    points = np.unique(np.r_[reference_points, _ggd_points(fitted, tail)])
    if len(points) < 2 or not np.isfinite(points).all():
        raise FloatingPointError("Evaluation quantiles cannot be represented as a finite interval.")
    truth_peak, fitted_peak = _density_peak_upper(truth), _ggd_peak_upper(fitted)
    if (not np.isfinite(truth_peak) or truth_peak < 0
        or not np.isfinite(fitted_peak) or fitted_peak < 0):
        raise FloatingPointError("Density tail bounds require finite nonnegative peak bounds.")

    left, right = points[:-1], points[1:]
    n_panels = len(left)
    depth = np.zeros(n_panels, dtype=int)
    total, error = np.zeros(2), np.zeros(2)
    unresolved = 0

    def panel(l, r, order):
        nodes, weights = _LEGENDRE[order]
        mid, half = l/2+r/2, (r-l)/2
        xx = mid[:, None]+half[:, None]*nodes
        fitted_pdf = np.asarray(fitted.pdf(xx.ravel()), dtype=float).reshape(xx.shape)
        true_pdf = np.asarray(truth.pdf(xx.ravel()), dtype=float).reshape(xx.shape)
        if not np.isfinite(fitted_pdf).all() or not np.isfinite(true_pdf).all():
            raise FloatingPointError("Nonfinite density value during quadrature.")
        delta = fitted_pdf-true_pdf
        return np.c_[np.abs(delta)@weights, delta**2@weights]*half[:, None]

    while len(left):
        low, high = panel(left, right, 32), panel(left, right, 64)
        difference = np.abs(high-low)
        good = np.all(difference <= atol/n_panels/(2.**depth[:, None])+rtol*np.abs(high), axis=1)
        stopped = depth >= max_refinement
        accept = good | stopped
        total += high[accept].sum(axis=0)
        error += difference[accept].sum(axis=0)
        unresolved += int(np.sum(stopped & ~good))
        ll, rr, dd = left[~accept], right[~accept], depth[~accept]
        mid = ll/2+rr/2
        left, right, depth = np.r_[ll, mid], np.r_[mid, rr], np.r_[dd+1, dd+1]

    lo, hi = float(points[0]), float(points[-1])
    true_tail, fitted_tail = _omitted_mass(truth, lo, hi), _omitted_mass(fitted, lo, hi)
    return dict(IAE=float(total[0]), ISE=float(total[1]),
        IAE_quad_error=float(error[0]), ISE_quad_error=float(error[1]),
        IAE_tail_upper=true_tail+fitted_tail,
        ISE_tail_upper=truth_peak*true_tail+fitted_peak*fitted_tail,
        Quadrature_OK=bool(unresolved == 0 and np.all(error <= atol+rtol*np.abs(total))),
        Evaluation_left=lo, Evaluation_right=hi, Unresolved_panels=unresolved)

def vgm_settings(config, cap=None, n_init=1, random_state=27):
    return dict(n_components=config.max_components if cap is None else int(cap),
        covariance_type="full", weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=config.concentration, init_params="kmeans",
        n_init=n_init, max_iter=1500, tol=1e-3, reg_covar=1e-6,
        random_state=random_state)


def fit_vgm(x, cap, cfg, *, n_init=1, random_state=27):
    estimator = BayesianGaussianMixture(**vgm_settings(cfg, cap, n_init, random_state))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        estimator.fit(np.array(x, dtype=float, copy=True).reshape(-1, 1))
        seconds = perf_counter()-started
    full = Mixture(estimator.weights_/estimator.weights_.sum(), estimator.means_[:, 0],
        np.sqrt(2*estimator.covariances_[:, 0, 0]), np.full(cap, 2.))
    return dict(full=full, model=estimator, fit_seconds=seconds,
        converged=bool(estimator.converged_), iterations=int(estimator.n_iter_),
        warnings=[str(w.message) for w in caught])


def evaluate_cap_density(
    method, cap, full, x, reference, cfg, run, fit_s,
    accepted_merges=0, merge_stop_reason="disabled", failed_starts=0,
):
    started = perf_counter()
    active, k, excluded = active_view(full, cfg.active_threshold)
    if active is None:
        raise RuntimeError("No active density at the reporting threshold.")

    full_errors = density_errors(reference, full)
    errors = density_errors(reference, active)

    row = dict(
        Method=method, Max_K=cap, Active_K=k, Full_slots=full.k,
        Excluded_mass=excluded,
        IAE=errors["IAE"], ISE=errors["ISE"],
        IAE_full=full_errors["IAE"], ISE_full=full_errors["ISE"],
        Quadrature_OK=bool(
            errors["Quadrature_OK"] and full_errors["Quadrature_OK"]
        ),
        ELBO_per_observation=run["elbo"]/len(x),
        LL_full_per_observation=float(full.logpdf(x).mean()),
        Converged=run["converged"],
        Inner_nonconverged=run["inner_nonconverged"],
        Accepted_merges=accepted_merges,
        Merge_stop_reason=merge_stop_reason,
        Fit_s=fit_s,
        Failed_initial_starts=failed_starts,
        IAE_quad_error=errors["IAE_quad_error"],
        ISE_quad_error=errors["ISE_quad_error"],
        IAE_tail_upper=errors["IAE_tail_upper"],
        ISE_tail_upper=errors["ISE_tail_upper"],
        Evaluation_s=perf_counter()-started,
    )
    return row, active

def _run_cap(x, cap, cfg, reference, output_dir, include_vgm=True):
    current = replace(cfg, max_components=cap)
    print(
        f"\nmax_K={cap}: baseline + merge ({cfg.initial_shape_mode})",
        flush=True,
    )
    with threadpool_limits(limits=1):
        model = VariationalGGMM(current).fit(x)

    rows, components, traces, starts, timing_rows, records = [], [], [], [], [], []
    stages = [
        ("VGGMM-base", model.baseline_run_, model.baseline_density_,
         model.baseline_seconds_, 0, "disabled"),
        ("VGGMM-merge", model.final_run_, model.full_density_,
         model.fit_seconds_, model.accepted_merges_, model.merge_stop_reason_),
    ]

    for method, run, full, seconds, accepted, stop_reason in stages:
        row, active = evaluate_cap_density(
            method, cap, full, x, reference, cfg, run, seconds,
            accepted_merges=accepted,
            merge_stop_reason=stop_reason,
            failed_starts=len(model.failed_starts_),
        )
        rows.append(row)
        records.append(dict(method=method, cap=cap, full=full, active=active))
        nk = run["state"]["r"].sum(axis=0)

        for k in range(full.k):
            components.append(dict(
                Method=method, Max_K=cap, Component=k,
                Weight=full.weights[k], Mean=full.means[k],
                Scale=full.scales[k], Shape=full.shapes[k],
                Sigma=full.sigmas[k], Nk=nk[k],
                Active=bool(full.weights[k] > cfg.active_threshold),
            ))

        print(
            f"  {method}: Active_K={row['Active_K']}, "
            f"ELBO/n={row['ELBO_per_observation']:.7f}, "
            f"IAE={row['IAE']:.7f}, merges={accepted}",
            flush=True,
        )

    timing_rows.append(dict(
        Method="VGGMM-base+merge", Max_K=cap,
        **model.timing_, **model.work_,
        Evaluation_s=sum(row["Evaluation_s"] for row in rows),
    ))

    for run in model.runs_:
        starts.append(dict(
            Max_K=cap, Start=run["name"],
            Selected=bool(run is model.baseline_run_),
            Converged=run["converged"],
            ELBO_per_observation=run["elbo"]/len(x),
            Active_K=model._active_k(run["state"]),
            Inner_nonconverged=run["inner_nonconverged"],
            Stop_reason=run["reason"], Fit_s=run["seconds"],
            Initialization_s=run["Initialization_s"],
            Optimize_wall_s=run["Optimize_wall_s"],
        ))

    checkpoints = [("baseline", model.baseline_run_)] + [
        (f"accepted-{i}", run)
        for i, run in enumerate(model.merge_checkpoints_[1:], start=1)
    ]
    for stage, run in checkpoints:
        for item in run["history"]:
            traces.append(dict(Max_K=cap, Stage=stage, **item))

    if include_vgm:
        rec = fit_vgm(x, cap, cfg)
        fake_run = dict(
            elbo=np.nan, converged=rec["converged"], inner_nonconverged=0
        )
        row, active = evaluate_cap_density(
            "VGM", cap, rec["full"], x, reference, cfg,
            fake_run, rec["fit_seconds"],
        )
        rows.append(row)
        records.append(dict(
            method="VGM", cap=cap, full=rec["full"], active=active
        ))
        timing_rows.append(dict(
            Method="VGM", Max_K=cap,
            Baseline_fit_s=rec["fit_seconds"], Merge_extra_s=0.,
            Finalization_s=0., Total_fit_s=rec["fit_seconds"],
            Evaluation_s=row["Evaluation_s"],
        ))

        full = rec["full"]
        for k in range(full.k):
            components.append(dict(
                Method="VGM", Max_K=cap, Component=k,
                Weight=full.weights[k], Mean=full.means[k],
                Scale=full.scales[k], Shape=full.shapes[k],
                Sigma=full.sigmas[k], Nk=np.nan,
                Active=bool(full.weights[k] > cfg.active_threshold),
            ))

    failed = pd.DataFrame(model.failed_starts_)
    if failed.empty and not len(failed.columns):
        failed = pd.DataFrame(columns=[
            "start", "error_type", "error",
            "Initialization_s", "Optimize_wall_s",
        ])

    result_tables = dict(
        metrics=pd.DataFrame(rows),
        components=pd.DataFrame(components),
        starts=pd.DataFrame(starts),
        traces=pd.DataFrame(traces),
        merge_log=pd.DataFrame(model.merge_log_),
        continuation_log=pd.DataFrame(model.continuation_log_),
        failed_starts=failed,
        timing=pd.DataFrame(timing_rows),
        seed_log=pd.DataFrame(model.seed_log_),
        initial_start_timing=pd.DataFrame(model.initial_start_timing_),
    )

    cap_dir = output_dir / f"cap_{cap}"
    cap_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result_tables.items():
        table.to_csv(cap_dir / f"{name}.csv", index=False)

    arrays = {}
    for name, run in (
        ("baseline", model.baseline_run_),
        ("final", model.final_run_),
    ):
        arrays.update({
            f"{name}_{key}": value
            for key, value in run["state"].items()
        })
    arrays.update(center=model.center_, unit=model.unit_)
    np.savez_compressed(cap_dir / "posterior_states.npz", **arrays)

    summary = dict(
        config=asdict(current),
        baseline_selection_fallback=model.baseline_selection_fallback_,
        accepted_merges=model.accepted_merges_,
        merge_stop_reason=model.merge_stop_reason_,
        baseline_seconds=model.baseline_seconds_,
        merge_seconds=model.merge_seconds_,
        total_fit_seconds=model.fit_seconds_,
        timing=model.timing_,
        work=model.work_,
        full_slots=model.full_k_,
        threshold=cfg.active_threshold,
    )
    (cap_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return model, result_tables, records

def combine_cap_tables(cap_tables):
    tables = {}
    for name in cap_tables[0][1]:
        frames = [
            table[name].assign(Max_K=cap)
            if "Max_K" not in table[name].columns else table[name]
            for cap, table in cap_tables if not table[name].empty
        ]
        tables[name] = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=cap_tables[0][1][name].columns)
        )

    rows = []
    for cap, group in tables["metrics"].groupby("Max_K", sort=True):
        by_method = group.set_index("Method")
        base = by_method.loc["VGGMM-base"]
        merge = by_method.loc["VGGMM-merge"]
        logs = tables["merge_log"]
        controls = tables["continuation_log"]
        accepted_gain, continuation_gain = 0., 0.

        if not logs.empty:
            chosen = logs[
                (logs["Max_K"] == cap)
                & logs["Accepted"].fillna(False).astype(bool)
            ]
            if not chosen.empty:
                accepted_gain = float(chosen["Gain_per_observation"].sum())

        if not controls.empty:
            used = controls[
                (controls["Max_K"] == cap)
                & controls["Used"].fillna(False).astype(bool)
            ]
            if not used.empty:
                continuation_gain = float(used["Gain_per_observation"].sum())

        merge_seconds = tables["timing"].loc[
            (tables["timing"]["Max_K"] == cap)
            & (tables["timing"]["Method"] == "VGGMM-base+merge"),
            "Merge_extra_s",
        ].iloc[0]

        rows.append(dict(
            Max_K=cap,
            Base_active_K=int(base["Active_K"]),
            Merge_active_K=int(merge["Active_K"]),
            Base_IAE=base["IAE"], Merge_IAE=merge["IAE"],
            Base_ISE=base["ISE"], Merge_ISE=merge["ISE"],
            Total_delta_ELBO_per_observation=(
                merge["ELBO_per_observation"]-base["ELBO_per_observation"]
            ),
            Accepted_merge_ELBO_gain_per_observation=accepted_gain,
            Continuation_ELBO_gain_per_observation=continuation_gain,
            Accepted_merges=int(merge["Accepted_merges"]),
            Merge_stop_reason=merge["Merge_stop_reason"],
            BaseFit_s=base["Fit_s"],
            MergeExtra_s=float(merge_seconds),
            TotalFit_s=merge["Fit_s"],
        ))

    tables["paired"] = pd.DataFrame(rows)
    return tables

class ReferenceMixture:
    """Evaluation/sampling object supporting different SciPy distribution families."""
    def __init__(self, weights, components, peak_bounds, description):
        self.weights = np.array(weights, dtype=float)
        self.components = tuple(components)
        self.peak_bounds = np.array(peak_bounds, dtype=float)
        self.description = description
        if (len(self.weights) != len(self.components) or self.peak_bounds.shape != self.weights.shape
            or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1)):
            raise ValueError("Invalid reference mixture.")

    def logpdf(self, x):
        x = np.asarray(x).reshape(-1)
        return logsumexp(np.column_stack([c.logpdf(x) for c in self.components])
                         + np.log(self.weights), axis=1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        return np.column_stack([c.cdf(x) for c in self.components]) @ self.weights

    def evaluation_points(self, tail):
        p = np.unique(np.array([tail, 1e-6, 1e-4, .001, .01, .1, .25, .5,
                                .75, .9, .99, .999, 1-1e-4, 1-1e-6, 1-tail]))
        return np.concatenate([c.ppf(p) for c in self.components])

    def peak_upper(self):
        return float(self.weights @ self.peak_bounds)

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        labels = rng.choice(len(self.weights), size=n, p=self.weights)
        x = np.empty(n)
        for k, component in enumerate(self.components):
            keep = labels == k
            x[keep] = component.rvs(size=int(keep.sum()), random_state=rng)
        return x

def gaussian_reference():
    # These scales are NORMAL standard deviations, rather than GGD scale a.
    weights, means, sigmas = [.2, .3, .5], [-5., 0., 6.], [1.2, 1.6, 2.]
    return ReferenceMixture(weights, [norm(loc=m, scale=s) for m, s in zip(means, sigmas)],
        [1/(np.sqrt(2*np.pi)*s) for s in sigmas],
        dict(weights=weights, means=means, normal_standard_deviations=sigmas,
             families=["Gaussian"]*3, generating_components=3))

def heterogeneous_reference():
    components = [norm(loc=-7., scale=.9), laplace(loc=-2., scale=.7),
                  t(df=5, loc=4., scale=1.), skewnorm(a=6., loc=9., scale=1.5)]
    return ReferenceMixture([.20, .25, .25, .30], components,
        [1/(.9*np.sqrt(2*np.pi)), 1/(2*.7), components[2].pdf(4.), 2/(1.5*np.sqrt(2*np.pi))],
        dict(weights=[.20, .25, .25, .30], generating_components=4,
             families=["Gaussian", "Laplace", "Student-t", "skew-normal"],
             parameters=[dict(loc=-7., scale=.9), dict(loc=-2., scale=.7),
                         dict(df=5, loc=4., scale=1.), dict(a=6., loc=9., scale=1.5)]))

def ggd_reference_3():
    return Mixture([.2, .3, .5], [-5., 0., 6.], [1.2, 1.6, 2.], [1.5, 4., 8.])


def ggd_reference_5():
    return Mixture([.10, .15, .20, .25, .30], [-10., -5., 0., 5., 10.],
        [1.2, 1.4, 1.6, 1.8, 2.0], [1.5, 2., 4., 6., 8.])


def synthetic_sample(reference, seed, *, n_train=5000, n_test=5000):
    return (reference.sample(n_train, seed), reference.sample(n_test, seed+1000), reference,
            dict(train_seed=seed, test_seed=seed+1000, reference=reference.description))


def load_abalone(path=None):
    """Use a local CSV when supplied; otherwise cache the official UCI file."""
    data_file = Path(path) if path is not None else MODEL_FILE.parent/"data"/"abalone.data"
    if not data_file.is_file():
        if path is not None:
            raise FileNotFoundError(data_file)
        context = ssl.create_default_context()
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(ABALONE_URL, context=context, timeout=45) as response:
            content = response.read()
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_bytes(content)
    columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight",
               "Viscera_weight", "Shell_weight", "Rings"]
    data = pd.read_csv(data_file, header=None, names=columns)
    x = data["Whole_weight"].to_numpy(dtype=float, copy=True)
    if len(data) != 4177 or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("UCI Abalone rows and Whole_weight need to be checked.")
    return x, dict(dataset="UCI Abalone", column="Whole_weight", rows=len(x),
        units="UCI stored scale (continuous values divided by 200 according to abalone.names)",
        source=ABALONE_SOURCE, raw_url=ABALONE_URL, license="CC BY 4.0",
        data_sha256=hashlib.sha256(data_file.read_bytes()).hexdigest())


def real_sample(x, metadata, seed):
    indices = np.random.default_rng(seed).permutation(len(x))
    cut = int(.8*len(x))
    return x[indices[:cut]], x[indices[cut:]], None, dict(**metadata, split_seed=seed,
        split="80/20 shuffled holdout", train_indices=indices[:cut].tolist(),
        test_indices=indices[cut:].tolist(), nonnegative_support=True)

def evaluate_holdout_density(density, x_train, x_test, truth, nonnegative, threshold=.005):
    begin = perf_counter()
    full = dict(Train_LL_per_obs=float(np.mean(density.logpdf(x_train))),
                Test_LL_per_obs=float(np.mean(density.logpdf(x_test))),
                Negative_mass=float(density.cdf([0.])[0]) if nonnegative else np.nan)
    if truth is not None:
        full.update(density_errors(truth, density))
    keep = density.weights > threshold
    retained = float(density.weights[keep].sum())
    full.update(Active_K=int(keep.sum()), Excluded_mass=float(density.weights[~keep].sum()))
    if keep.any():
        active = Mixture(density.weights[keep]/retained, density.means[keep],
                         density.scales[keep], density.shapes[keep])
        full["Active_Test_LL_per_obs"] = float(np.mean(active.logpdf(x_test)))
        if truth is not None:
            err = density_errors(truth, active)
            full.update({"Active_"+k: err[k] for k in ("IAE", "ISE", "Quadrature_OK")})
    full["Evaluation_s"] = perf_counter()-begin
    if not np.isfinite(full["Test_LL_per_obs"]):
        raise FloatingPointError("Test log density is nonfinite.")
    return full

def json_value(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)

def _fit_holdout_vgm(x, config, n_init=2):
    rec = fit_vgm(x, config.max_components, config, n_init=n_init)
    seconds, estimator = rec["fit_seconds"], rec["model"]
    return rec["full"], dict(Fit_s=seconds, Baseline_fit_s=seconds, Merge_extra_s=0.,
        Converged=rec["converged"], Iterations=rec["iterations"],
        Stop_reason="tolerance" if rec["converged"] else "iteration_limit",
        Inner_nonconverged=np.nan, Failed_starts=0, Accepted_merges=0,
        Chosen_start="kmeans", ELBO_gain_per_obs=np.nan,
        Warning_count=len(rec["warnings"]), Warnings=rec["warnings"])


def run_study(study_id, sample_factory, *, config=None,
              seeds=(20260912, 20260913, 20260914), output_root="results",
              vgm_n_init=2, show=True, formats=("png",)):
    """All fits use X_train only; truth and test data enter subsequent evaluation."""
    CONFIG = _shape2_config(config)
    MAX_K = CONFIG.max_components
    OUTPUT_ROOT = Path(output_root)
    VGM_SETTINGS = vgm_settings(CONFIG, n_init=vgm_n_init)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = OUTPUT_ROOT / study_id / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows, cases, arrays, log_rows, timing_rows = [], [], {}, [], []
    study_begin = perf_counter()
    actual_fit_s = 0.
    for seed in seeds:
        x_train, x_test, truth, metadata = sample_factory(seed)
        x_train = np.array(x_train, dtype=float, copy=True)
        x_test = np.array(x_test, dtype=float, copy=True)
        original = x_train.copy()
        prefix = f"seed_{seed}"
        arrays[prefix+"_train"], arrays[prefix+"_test"] = x_train, x_test
        case = dict(seed=int(seed), metadata=metadata, truth=truth, x_train=x_train,
                    densities={}, diagnostics={})
        fitted = {}
        print(f"{study_id}: seed={seed}, train={len(x_train)}, test={len(x_test)}, Max_K={MAX_K}", flush=True)
        begin = perf_counter()
        vggmm_time_recorded = False
        try:
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = VariationalGGMM(CONFIG).fit(x_train)
            actual_fit_s += model.fit_seconds_
            vggmm_time_recorded = True
            case["diagnostics"] = dict(timing=model.timing_, work=model.work_,
                initial_start_timing=model.initial_start_timing_, failed_starts=model.failed_starts_,
                warnings=[str(w.message) for w in caught],
                baseline_selection_fallback=model.baseline_selection_fallback_,
                merge_stop_reason=model.merge_stop_reason_)
            final_r = model.posterior_state_["r"]
            if final_r.shape != (len(x_train), MAX_K) or not np.allclose(final_r.sum(axis=1), 1., atol=1e-10):
                raise AssertionError("Responsibility rows or capacity are inconsistent.")
            if model.final_run_["elbo"] < model.baseline_run_["elbo"]-1e-7:
                raise AssertionError("Accepted search reduced the full ELBO.")
            if not np.isclose(model.timing_["Baseline_fit_s"]+model.timing_["Merge_extra_s"]
                              +model.timing_["Finalization_s"], model.fit_seconds_, atol=1e-9):
                raise AssertionError("Fit time accounting is inconsistent.")
            for name, state in (("baseline", model.baseline_run_["state"]),
                                ("final", model.posterior_state_)):
                for key, value in state.items():
                    arrays[prefix+"_"+name+"_"+key] = value
            for phase, entries in (("merge", model.merge_log_), ("continuation", model.continuation_log_),
                                   ("seed_screening", model.seed_log_)):
                for entry in entries:
                    log_rows.append(dict(Seed=seed, Phase=phase, **entry))
            timing_rows.append(dict(Seed=seed, **model.timing_, **model.work_))
            for method, density, run in (("VGGMM-base", model.baseline_density_, model.baseline_run_),
                                          ("VGGMM-merge", model.full_density_, model.final_run_)):
                merged = method == "VGGMM-merge"
                info = dict(Fit_s=model.fit_seconds_ if merged else model.baseline_seconds_,
                    Baseline_fit_s=model.baseline_seconds_, Merge_extra_s=model.merge_seconds_ if merged else 0.,
                    Converged=bool(run["converged"]), Iterations=len(run["history"])-1,
                    Stop_reason=run["reason"], Inner_nonconverged=run["inner_nonconverged"],
                    Failed_starts=len(model.failed_starts_), Warning_count=len(caught),
                    Chosen_start=model.baseline_run_["name"], Accepted_merges=model.accepted_merges_ if merged else 0,
                    ELBO_gain_per_obs=(run["elbo"]-model.baseline_run_["elbo"])/len(x_train))
                fitted[method] = (density, info)
        except Exception as exc:
            if not vggmm_time_recorded:
                actual_fit_s += perf_counter()-begin
            case["diagnostics"]["vggmm_error"] = f"{type(exc).__name__}: {exc}"
        begin = perf_counter()
        try:
            with threadpool_limits(limits=1):
                density, info = _fit_holdout_vgm(x_train, CONFIG, vgm_n_init)
            actual_fit_s += info["Fit_s"]
            fitted["VGM"] = (density, info)
        except Exception as exc:
            actual_fit_s += perf_counter()-begin
            case["diagnostics"]["vgm_error"] = f"{type(exc).__name__}: {exc}"
        if not np.array_equal(x_train, original):
            raise AssertionError("A fitter modified the input sample.")
        for method in METHODS:
            row = dict(Study=study_id, Seed=seed, Method=method, Max_K=MAX_K,
                       Train_N=len(x_train), Test_N=len(x_test), Status="fit_failed")
            if method in fitted:
                density, info = fitted[method]
                case["densities"][method] = density
                for key, value in density.payload().items():
                    arrays[prefix+"_"+method+"_"+key] = np.asarray(value)
                row.update(info, Status="ok")
                try:
                    row.update(evaluate_holdout_density(density, x_train, x_test, truth,
                                                metadata.get("nonnegative_support", False), CONFIG.active_threshold))
                except Exception as exc:
                    row.update(Status="evaluation_failed", Evaluation_error=f"{type(exc).__name__}: {exc}")
            else:
                row["Fit_error"] = case["diagnostics"].get("vgm_error" if method == "VGM" else "vggmm_error", "")
            rows.append(row)
        cases.append(case)
        print(" | ".join(f"{r['Method']}: K={r.get('Active_K', '?')}, "
            f"test LL={r.get('Test_LL_per_obs', np.nan):.6f}, {r.get('Fit_s', np.nan):.2f}s, "
            f"converged={r.get('Converged', False)}" for r in rows[-3:]), flush=True)
        # Save each completed seed, so interruption cannot discard completed fits.
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        np.savez_compressed(output / "samples_and_states.npz", **arrays)
    table = pd.DataFrame(rows)
    pd.DataFrame(log_rows).to_csv(output / "search_log.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "timing.csv", index=False)
    metadata_out = dict(study_id=study_id, seeds=list(seeds), config=asdict(CONFIG),
        vgm_settings=VGM_SETTINGS, density_kind="full posterior-parameter plug-in density",
        primary_metrics="heldout mean log density; synthetic IAE/ISE",
        versions=dict(python=platform.python_version(), **{p:version(p) for p in
            ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")}),
        model_sha256=hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        actual_fit_s=actual_fit_s, fit_evaluate_save_wall_s=perf_counter()-study_begin,
        cases=[dict(seed=c["seed"], metadata=c["metadata"], diagnostics=c["diagnostics"]) for c in cases])
    (output / "metadata.json").write_text(json.dumps(metadata_out, ensure_ascii=False,
        indent=2, default=json_value), encoding="utf-8")
    study = dict(table=table, cases=cases, output=output, metadata=metadata_out)
    study["paired"] = paired_comparisons(study)
    study["paired"].to_csv(output / "paired_differences.csv", index=False)
    summary = summarize_pairs(study["paired"])
    summary.to_csv(output / "paired_summary.csv", index=False)
    study["paired_summary"] = summary
    show_study(study, show=show, formats=formats)
    metadata_out["total_study_wall_s"] = perf_counter()-study_begin
    (output / "metadata.json").write_text(json.dumps(metadata_out, ensure_ascii=False,
        indent=2, default=json_value), encoding="utf-8")
    return study

def paired_comparisons(study):
    rows, table = [], study["table"]
    metrics = [m for m in ("IAE", "ISE", "Test_LL_per_obs", "Fit_s") if m in table]
    for seed in sorted(table["Seed"].unique()):
        group = table[table["Seed"] == seed].set_index("Method")
        for left, right in (("VGGMM-merge", "VGM"), ("VGGMM-merge", "VGGMM-base")):
            a, b = group.loc[left], group.loc[right]
            evaluated = a["Status"] == b["Status"] == "ok"
            converged = lambda row: pd.notna(row.get("Converged")) and bool(row.get("Converged"))
            row = dict(Study=a["Study"], Seed=seed, Comparison=left+" minus "+right,
                       Both_evaluated=evaluated,
                       Both_converged=bool(evaluated and converged(a) and converged(b)))
            for metric in metrics:
                row["Delta_"+metric] = a.get(metric, np.nan)-b.get(metric, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def summarize_pairs(paired):
    rows = []
    for comparison, group in paired.groupby("Comparison", sort=False):
        for metric in [c for c in paired if c.startswith("Delta_")]:
            values = group[metric].dropna().to_numpy()
            lower_is_better = metric != "Delta_Test_LL_per_obs"
            signs = -values if lower_is_better else values
            rows.append(dict(Comparison=comparison, Metric=metric, Valid_seeds=len(values),
                Total_seeds=len(group), Both_converged_seeds=int(group["Both_converged"].sum()),
                Mean_delta=float(values.mean()) if len(values) else np.nan,
                SD_delta=float(values.std(ddof=1)) if len(values)>1 else np.nan,
                Left_wins=int(np.sum(signs>1e-10)), Right_wins=int(np.sum(signs< -1e-10)),
                Ties=int(np.sum(np.abs(signs)<=1e-10))))
    return pd.DataFrame(rows)

def show_study(study, *, show=True, formats=("png",)):
    table = study["table"]
    threshold = study["metadata"]["config"]["active_threshold"]
    columns = [c for c in ("Seed", "Method", "Active_K", "IAE", "ISE", "Test_LL_per_obs",
        "Negative_mass", "Fit_s", "Merge_extra_s", "Converged", "Accepted_merges",
        "Inner_nonconverged", "Failed_starts", "Quadrature_OK", "Status") if c in table]
    try:
        from IPython.display import display
        display(table[columns], study["paired_summary"])
    except ImportError:
        print(table[columns].to_string(index=False))
        print(study["paired_summary"].to_string(index=False))
    fig, axes = plt.subplots(1, len(study["cases"]), figsize=(5*len(study["cases"]), 3.8),
                             squeeze=False, layout="constrained")
    styles = {"VGGMM-base":("#777777", "--"), "VGGMM-merge":("#c43c39", "-"), "VGM":("#2865b0", "-.")}
    for ax, case in zip(axes[0], study["cases"]):
        x = case["x_train"]
        truth = case["truth"]
        densities = case["densities"]
        lo, hi = np.quantile(x, [.001, .999])
        pad = .1*(hi-lo)
        grid = np.linspace(lo-pad, hi+pad, 2400)
        ax.hist(x, bins=65, density=True, color="#dce2e8", edgecolor="white", linewidth=.3)
        if truth is not None:
            ax.plot(grid, truth.pdf(grid), "k:", lw=2., label="Truth")
        for method, density in densities.items():
            color, style = styles[method]
            k = int(np.sum(density.weights > threshold))
            ax.plot(grid, density.pdf(grid), color=color, ls=style, lw=1.7,
                    label=f"{method}, active K={k}")
        xlabel = "Whole_weight (UCI stored scale)" if truth is None else "x"
        ax.set(title=f"seed={case['seed']}", xlabel=xlabel, ylabel="Density")
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    _export_figure(fig, study["output"]/"density", formats)
    if show:
        plt.show()
    plt.close(fig)
    print("주평가는 full density. Active_*는 문턱으로 자른 출력의 별도 진단입니다.")
    print("Fit_s: base는 재사용한 병합 전 적합 시간, merge는 base+병합 전체 시간. 행별 시간을 합치지 마세요.")
    print("저장:", study["output"], "| 실제 적합 시간 합:", round(study["metadata"]["actual_fit_s"], 2), "s")

def check_mathematics():
    xx, mean, sd = np.array([-3., -.1, 0., 2., 10.]), .4, .7
    np.testing.assert_allclose(
        np.exp(vg.normal_log_abs_moment(xx, mean, np.log(sd), 2.)),
        (xx-mean)**2 + sd**2,
        rtol=2e-13,
    )

    cfg = VGGMMConfig()
    r = np.array([.1, .7, .4, .8, .2])
    y = np.array([mean, np.log(sd), np.log(1.7)])
    profiled, a, lb, _ = vg.component_profile(y, xx, r, cfg)
    expected = vg.expected_log_components(
        xx, y[None, :], np.array([a]), np.array([lb])
    )[:, 0]
    expanded = r @ expected - vg.normal_kl(mean, np.log(sd), cfg.mean_prior_sd)
    expanded -= vg.gamma_kl(
        a, lb, cfg.precision_prior_shape, cfg.precision_prior_rate
    )
    np.testing.assert_allclose(
        profiled, expanded, rtol=2e-13, atol=1e-12
    )

    for nk in (np.array([100., 0., 40., 2.]), np.array([5.])):
        sa, sb = vg.stick_posterior(nk, cfg.concentration)
        w, elogw = vg.stick_statistics(sa, sb)
        np.testing.assert_allclose(w.sum(), 1., atol=1e-14)
        assert np.all(elogw <= np.log(w)+1e-10)

    print("Math checks passed: normal moment, Gamma profile/ELBO, sticks.")


def _shape2_config(config=None):
    config = VGGMMConfig() if config is None else config
    if config.initial_shape_mode != "shape2":
        raise ValueError("These experiments use initial_shape_mode='shape2'.")
    return config


def _new_output(root, experiment):
    path = Path(root)/experiment/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _export_figure(fig, stem, formats=("png",)):
    stem = Path(stem)
    with plt.rc_context({"svg.fonttype": "path"}):
        for fmt in formats:
            if fmt not in ("png", "svg", "pdf"):
                raise ValueError("Figure formats: png, svg, pdf.")
            fig.savefig(stem.with_suffix("."+fmt), dpi=180, bbox_inches="tight")


def run_cap_experiment(x, truth, *, caps=(10,), config=None, name="single_cap",
                       output_root="results", include_vgm=True):
    """One fresh fit per cap; truth is used only after each fit for evaluation."""
    config = _shape2_config(config)
    caps = tuple(caps)
    if not caps or len(set(caps)) != len(caps) or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1
        for k in caps
    ):
        raise ValueError("caps must contain distinct positive integers.")
    x = np.array(x, dtype=float, copy=True)
    original = x.copy()
    output = _new_output(output_root, name)
    models, tables_by_cap, records = {}, [], []
    for cap in caps:
        model, tables, items = _run_cap(x, cap, config, truth, output, include_vgm)
        models[cap] = model
        tables_by_cap.append((cap, tables))
        records.extend(items)
    tables = combine_cap_tables(tables_by_cap)
    true_k = getattr(truth, "k", None)
    if true_k is not None:
        metrics = tables["metrics"]
        metrics["True_K"] = int(true_k)
        metrics["Cap_below_true_K"] = metrics.Max_K < true_k
        metrics["At_cap"] = metrics.Active_K == metrics.Full_slots
    for key, table in tables.items():
        table.to_csv(output/(key+".csv"), index=False)
    np.savez_compressed(output/"sample.npz", X=x)
    metadata = dict(experiment=name, n=len(x), caps=caps, config=asdict(config),
        truth=truth.payload() if hasattr(truth, "payload") else truth.description,
        sample_sha256=hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest(),
        model_file=MODEL_FILE.name,
        model_sha256=hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization="initial components and merged-component seeds start at b=2; b stays free",
        fitting_uses_truth=False, density_kind="active parameter plug-in; full scores also reported",
        timing="merge total includes baseline; evaluation and plotting are separate",
        versions=dict(python=platform.python_version(),
            **{p:version(p) for p in ("numpy","scipy","pandas","matplotlib","scikit-learn","threadpoolctl")}))
    (output/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False,
        indent=2, default=json_value), encoding="utf-8")
    if not np.array_equal(x, original):
        raise AssertionError("Input sample was modified.")
    return dict(name=name, x=x, truth=truth, caps=caps, config=config, models=models,
                tables=tables, records=records, output=output, metadata=metadata)


def show_cap_results(result):
    metrics = result["tables"]["metrics"]
    cols = ["Method","Max_K","Active_K","Excluded_mass","IAE","ISE",
            "Converged","Inner_nonconverged","Accepted_merges","Fit_s","Evaluation_s","Quadrature_OK"]
    show_table(metrics[[c for c in cols if c in metrics]])
    show_table(result["tables"]["paired"])
    show_table(result["tables"]["timing"])
    print("Fit_s(merge) includes baseline; Evaluation_s is separate.")
    print("Density errors here use the active, renormalized density; *_full uses all slots.")
    print("Results:", result["output"])


def plot_cap_results(result, *, show=True, formats=("png",), components=False):
    x, truth, caps = result["x"], result["truth"], result["caps"]
    output, models = result["output"], result["models"]
    metrics, cfg = result["tables"]["metrics"], result["config"]
    styles = {"VGGMM-base":("#2865b0","--"),"VGGMM-merge":("#c43c39","-"),"VGM":("#2c8b66",":")}
    ncols, nrows = min(3,len(caps)), int(np.ceil(len(caps)/3))
    fig, axes = plt.subplots(nrows,ncols,figsize=(5.3*ncols,3.9*nrows),
        squeeze=False, sharex=True, sharey=True, layout="constrained")
    pts = _density_points(truth,1e-5)
    grid = np.linspace(min(float(np.min(pts)),x.min()),max(float(np.max(pts)),x.max()),2200)
    for ax,cap in zip(axes.flat,caps):
        ax.hist(x,bins=70,density=True,color="#dce2e8",edgecolor="white",linewidth=.3)
        ax.plot(grid,truth.pdf(grid),"k:",lw=1.8,label="True density")
        for rec in result["records"]:
            if rec["cap"] == cap:
                color,style = styles[rec["method"]]
                ax.plot(grid,rec["active"].pdf(grid),color=color,ls=style,lw=1.7,
                    label=f"{rec['method']}, active K={rec['active'].k}")
        ax.set(title=f"Max K={cap}; n={len(x)}; shape2 initialization",xlabel="x",ylabel="Density")
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(caps):]:
        ax.set_visible(False)
    _export_figure(fig,output/"density_comparison",formats)
    if show: plt.show()
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(10.5,7),layout="constrained")
    for method,part in metrics.groupby("Method",sort=False):
        color,style = styles[method]
        for ax,key in zip(axes.flat,("Active_K","IAE","ISE","ELBO_per_observation")):
            if method=="VGM" and key=="ELBO_per_observation": continue
            ax.plot(part.Max_K,part[key],marker="o",color=color,ls=style,label=method)
    axes[0,0].plot(caps,caps,":",color="gray",label="K=cap")
    for ax,key in zip(axes.flat,("Active K","IAE (lower is better)","ISE (lower is better)","ELBO per observation")):
        ax.set(xlabel="Maximum components",ylabel=key,xticks=caps)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    _export_figure(fig,output/"cap_comparison",formats)
    if show: plt.show()
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
    for cap,model in models.items():
        accepted = [model.baseline_run_]+model.merge_checkpoints_[1:]
        axes[0].plot(range(len(accepted)),[model._active_k(r["state"]) for r in accepted],"o-",label=f"Max K={cap}")
        axes[1].plot(range(len(accepted)),[r["elbo"]/len(x) for r in accepted],"o-",label=f"Max K={cap}")
    for ax,key in zip(axes,("Active K","ELBO per observation")):
        ax.set(xlabel="Accepted merge count",ylabel=key);ax.grid(alpha=.2);ax.legend(fontsize=8)
    _export_figure(fig,output/"accepted_moves",formats)
    if show: plt.show()
    plt.close(fig)
    if components and len(caps)==1:
        model=models[caps[0]]
        density=model.density_
        fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
        ax.hist(x,bins=70,density=True,color="#e5e9ed",edgecolor="white",linewidth=.3)
        ax.plot(grid,truth.pdf(grid),"k:",lw=2,label="True density")
        weighted=np.exp(density.component_logpdf(grid))*density.weights
        for k in range(density.k):
            ax.plot(grid,weighted[:,k],lw=1.2,alpha=.8,label=f"Component {k+1}, weight={density.weights[k]:.3f}")
        ax.plot(grid,density.pdf(grid),color="#c43c39",lw=2,label="Fitted active mixture")
        ax.set(title=f"Max K={caps[0]}, active K={density.k}, n={len(x)}",xlabel="x",ylabel="Density")
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
        _export_figure(fig,output/"components",formats)
        if show: plt.show()
        plt.close(fig)


def run_vgm_cap_experiment(*, n=5000, seed=20260912, caps=(3,10),
                           gaussian_control=True, config=None,
                           output_root="results", show=True, formats=("png",)):
    """Existing VGM-only capacity control, using the same shared IAE/ISE evaluator."""
    config=_shape2_config(config)
    ggd=ggd_reference_5()
    truths={"GGD_truth":ggd}
    if gaussian_control:
        truths["Gaussian_control"]=Mixture(ggd.weights,ggd.means,np.sqrt(2)*ggd.sigmas,np.full(ggd.k,2.))
    output=_new_output(output_root,"vgm_capacity_control")
    rows,samples,models,densities,arrays=[],{},{},{},{}
    for scenario,truth in truths.items():
        x=sample_mixture(truth,n,seed)
        xt=sample_mixture(truth,n,seed+1000)
        samples[scenario]=dict(train=x,test=xt)
        arrays[scenario+"_train"],arrays[scenario+"_test"]=x,xt
        for cap in caps:
            fit=fit_vgm(x,cap,config,n_init=2)
            fitted=fit["full"]
            weights=fitted.weights
            active=weights>config.active_threshold
            models[scenario,cap],densities[scenario,cap]=fit["model"],fitted
            row=dict(Scenario=scenario,Method="VGM",True_K=truth.k,Max_K=cap,
                Active_K=int(active.sum()),At_cap=bool(active.all()),
                Inactive_mass=float(weights[~active].sum()),**density_errors(truth,fitted),
                Test_LL_per_obs=float(fitted.logpdf(xt).mean()),Fit_s=fit["fit_seconds"],
                Converged=fit["converged"],Iterations=fit["iterations"],Fit_warnings=fit["warnings"])
            rows.append(row)
            print(f"{scenario} | Max_K={cap}: active K={active.sum()}, {fit['fit_seconds']:.2f}s",flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(output/"metrics.csv",index=False)
    np.savez_compressed(output/"samples.npz",**arrays)
    metadata=dict(n=n,seed=seed,caps=caps,config=asdict(config),
        vgm_n_init=2,density_kind="full parameter plug-in",fitting_uses_truth=False,
        model_sha256=hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        references={name:truth.payload() for name,truth in truths.items()})
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2,default=json_value),encoding="utf-8")
    show_table(table[["Scenario","True_K","Max_K","Active_K","At_cap","IAE","ISE",
                     "Test_LL_per_obs","Fit_s","Converged","Quadrature_OK"]])
    fig,axes=plt.subplots(len(truths),len(caps),figsize=(6.5*len(caps),3.8*len(truths)),
        squeeze=False,sharex="row",sharey="row",layout="constrained")
    for i,(scenario,truth) in enumerate(truths.items()):
        x=samples[scenario]["train"]
        grid=np.linspace(x.min()-1.,x.max()+1.,2400)
        for j,cap in enumerate(caps):
            ax=axes[i,j];fitted=densities[scenario,cap]
            ax.hist(x,bins=70,density=True,color="#dce2e8",edgecolor="white",linewidth=.3)
            ax.plot(grid,truth.pdf(grid),"k:",lw=2,label="True density: K=5")
            k=int(np.sum(fitted.weights>config.active_threshold))
            ax.plot(grid,fitted.pdf(grid),color="#c43c39",lw=2,label=f"VGM full density, active K={k}")
            ax.set(title=f"{scenario}; Max K={cap}; n={n}",xlabel="x",ylabel="Density")
            ax.legend(fontsize=8);ax.grid(alpha=.15)
    _export_figure(fig,output/"density_comparison",formats)
    if show: plt.show()
    plt.close(fig)
    return dict(table=table,samples=samples,models=models,densities=densities,
                truth=truths,output=output,metadata=metadata)
