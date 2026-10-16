"""Self-contained one-dimensional variational GGMM with merge/refit.

Only NumPy, SciPy and the standard library are required. All definitions used by
fit are in this file; no earlier notebook cell, project file or experiment is
imported. Importing this module never samples data, fits a model or writes files.

Inference: q(v)=Beta, q(z)=categorical, q(mu)=Normal, q(tau)=Gamma;
tau=a**(-b), b is a free positive point estimate. Default b starts at 2 in both initial fits and merge proposals; b remains free.
Merge proposals preserve Kmax, stick order and priors; all slots are refitted.
Exported densities use posterior parameter plug-ins, not posterior integration.
This is an independent truncated-DP implementation, not an exact reproduction
of IEEE 9002852's finite/Taylor algorithm.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import betaln, digamma, gammaln, hyp1f1, logsumexp, xlogy
from scipy.stats import gennorm

__all__ = ["Mixture", "VGGMMConfig", "VariationalGGMM"]

@dataclass(frozen=True)
class VGGMMConfig:
    """Priors and optimization settings.

    shape2 uses b=2 for both initial components and merged-component seeds.
    Subsequent optimization estimates b freely. moment remains available for
    older explicit configurations; the supplied experiments all use shape2.
    """
    max_components: int = 10
    concentration: float = .001
    active_threshold: float = .005
    max_iter: int = 300
    tol: float = 1e-5  # absolute ELBO change per observation, 3 consecutive times
    inner_max_iter: int = 30
    inner_ftol: float = 1e-8
    mean_prior_sd: float = 3.
    precision_prior_shape: float = 1.
    precision_prior_rate: float = .001
    starts: tuple[str, ...] = ("quantile", "lloyd")
    initial_shape_mode: str = "shape2"
    enable_merge: bool = True
    max_merge_rounds: int = 15
    max_candidates_per_round: int = 3
    merge_gain_per_observation: float = 1e-4  # 10 * default outer tol
    candidate_min_effective_count: float = 1e-6



@dataclass(init=False)
class Mixture:
    """Normalized GGD mixture; scales are GGD a, not standard deviations."""
    weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    shapes: np.ndarray

    def __init__(self, weights, means, scales, shapes):
        for name, values in zip(("weights", "means", "scales", "shapes"),
                                (weights, means, scales, shapes)):
            setattr(self, name, np.array(values, dtype=float, copy=True))
        if (self.weights.ndim != 1 or not self.weights.size
            or any(getattr(self, name).shape != self.weights.shape
                   for name in ("means", "scales", "shapes"))
            or not all(np.isfinite(getattr(self, name)).all()
                       for name in ("weights", "means", "scales", "shapes"))
            or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1.)
            or np.any(self.scales <= 0) or np.any(self.shapes <= 0)):
            raise ValueError("Invalid mixture parameters.")

    @property
    def k(self):
        return len(self.weights)

    @property
    def sigmas(self):
        return self.scales * np.exp(.5 * (gammaln(3/self.shapes) - gammaln(1/self.shapes)))

    def component_logpdf(self, x):
        return gennorm.logpdf(np.asarray(x).reshape(-1, 1), self.shapes,
                              loc=self.means, scale=self.scales)

    def logpdf(self, x):
        with np.errstate(divide="ignore"):
            return logsumexp(self.component_logpdf(x) + np.log(self.weights), axis=1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        return (gennorm.cdf(np.asarray(x).reshape(-1, 1), self.shapes,
                           loc=self.means, scale=self.scales) * self.weights).sum(axis=1)

    def payload(self):
        return {name: getattr(self, name).tolist()
                for name in ("weights", "means", "scales", "shapes")}



def normal_log_abs_moment(x, mean, log_sd, b):
    """log E|x-U|**b, U ~ Normal(mean, exp(2*log_sd)); exact identity."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        if not np.isfinite(b) or b <= 0 or not np.isfinite(log_sd):
            raise FloatingPointError("Invalid shape or posterior location SD.")
        sd = np.exp(log_sd)
        if sd == 0:
            raise FloatingPointError("Posterior location SD underflow.")
        z = -.5 * ((np.asarray(x) - mean)/sd)**2
        h = hyp1f1(-b/2, .5, z)
        if np.any(h <= 0) or not np.isfinite(h).all():
            raise FloatingPointError("Normal absolute moment cannot be represented.")
        answer = (b*log_sd + b*.5*np.log(2.) + gammaln((b+1)/2)
                  - .5*np.log(np.pi) + np.log(h))
    if not np.isfinite(answer).all():
        raise FloatingPointError("Nonfinite normal absolute moment.")
    return answer


def normal_kl(mean, log_sd, prior_sd):
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return .5*((np.exp(2*log_sd)+mean*mean)/prior_sd**2 - 1
                   + 2*np.log(prior_sd) - 2*log_sd)


def component_profile(y, x, r, cfg):
    """Optimize q(tau) analytically for fixed q(mu), b and responsibilities."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        m, ls, b = float(y[0]), float(y[1]), float(np.exp(y[2]))
        lm = normal_log_abs_moment(x, m, ls, b)
        nk = float(r.sum())
        log_r = np.full_like(r, -np.inf)
        np.log(r, out=log_r, where=r > 0)
        log_s = float(logsumexp(log_r + lm))
        a0, rate0 = cfg.precision_prior_shape, cfg.precision_prior_rate
        a = a0 + nk/b
        lb = float(np.logaddexp(np.log(rate0), log_s))
        value = (nk*(-np.log(2.) - gammaln(1+1/b)) + gammaln(a)-gammaln(a0)
                 + a0*np.log(rate0) - a*lb - normal_kl(m, ls, cfg.mean_prior_sd))
    if not np.isfinite(value) or not np.isfinite(a):
        raise FloatingPointError("Nonfinite profiled ELBO.")
    return float(value), float(a), lb, b


def stick_posterior(nk, alpha):
    nk = np.asarray(nk, dtype=float)
    return 1+nk[:-1], alpha+np.cumsum(nk[::-1])[::-1][1:]


def stick_statistics(a, b):
    if len(a) == 0:
        return np.ones(1), np.zeros(1)
    elogv = digamma(a)-digamma(a+b)
    elog1v = digamma(b)-digamma(a+b)
    elogw = np.r_[elogv, 0.] + np.r_[0., np.cumsum(elog1v)]
    log_residual = np.log(b)-np.log(a+b)
    logw = np.r_[np.log(a)-np.log(a+b), 0.] + np.r_[0., np.cumsum(log_residual)]
    w = np.exp(logw)
    return w/w.sum(), elogw


def stick_kl(a, b, alpha):
    return float(np.sum(betaln(1., alpha)-betaln(a, b)+(a-1)*digamma(a)
        +(b-alpha)*digamma(b)-(a+b-1-alpha)*digamma(a+b)))


def gamma_kl(a, log_rate, a0, rate0):
    return ((a-a0)*digamma(a)-gammaln(a)+gammaln(a0)
            +a0*(log_rate-np.log(rate0))-a+rate0*a*np.exp(-log_rate))


def expected_log_components(x, y, gamma_a, gamma_log_rate):
    out = np.empty((len(x), len(y)))
    for k, row in enumerate(y):
        b = float(np.exp(row[2]))
        lm = normal_log_abs_moment(x, row[0], row[1], b)
        with np.errstate(over="ignore", invalid="raise"):
            out[:, k] = (-np.log(2.)-gammaln(1+1/b)
                +(digamma(gamma_a[k])-gamma_log_rate[k])/b
                -np.exp(np.log(gamma_a[k])-gamma_log_rate[k]+lm))
    return out


def elbo_value(r, expected_logpdf, sa, sb, y, ga, glb, cfg):
    _, elogw = stick_statistics(sa, sb)
    term = np.zeros_like(r)
    np.multiply(r, expected_logpdf+elogw, out=term, where=r > 0)
    value = term.sum()-xlogy(r, r).sum()-stick_kl(sa, sb, cfg.concentration)
    value -= sum(normal_kl(row[0], row[1], cfg.mean_prior_sd) for row in y)
    value -= gamma_kl(ga, glb, cfg.precision_prior_shape, cfg.precision_prior_rate).sum()
    return float(value)


def moment_initial_shape(group):
    sd = float(np.std(group))
    ratio = float(np.mean(np.abs(group-group.mean()))/sd) if sd > 0 else np.nan
    if len(group) < 4 or not 0 < ratio < np.sqrt(3)/2:
        return 2.

    def f(eta):
        q = np.exp(-eta)
        return (.5*np.log(3)-np.log(2)+gammaln(1+2*q)
                -.5*(gammaln(1+q)+gammaln(1+3*q))-np.log(ratio))

    lo, hi = -1., 1.
    while f(lo) > 0 and lo > -16:
        lo -= 1
    while f(hi) < 0 and hi < 24:
        hi += 1
    return float(np.exp(brentq(f, lo, hi))) if f(lo)*f(hi) <= 0 else 2.


def partition(x, t, method):
    # x is sorted. Both arms use exactly the same deterministic partition.
    if method == "quantile":
        return np.minimum(np.arange(len(x))*t//len(x), t-1)
    if method != "lloyd":
        raise ValueError("Unknown deterministic initialization.")
    centers = np.quantile(x, (np.arange(t)+.5)/t)
    old = None
    for _ in range(100):
        distances = (x[:, None]-centers)**2
        labels = distances.argmin(axis=1)
        counts = np.bincount(labels, minlength=t)
        for j in np.flatnonzero(counts == 0):
            residual = distances[np.arange(len(x)), labels].copy()
            residual[counts[labels] <= 1] = -np.inf
            index = int(np.argmax(residual))
            counts[labels[index]] -= 1
            labels[index] = j
            counts[j] += 1
        if old is not None and np.array_equal(labels, old):
            break
        centers = np.array([x[labels == j].mean() for j in range(t)])
        old = labels.copy()
    order = np.argsort(centers, kind="stable")
    inverse = np.empty(t, int)
    inverse[order] = np.arange(t)
    return inverse[labels]


# Variational EM and merge/refit use the same posterior state and ELBO.
def copy_state(state):
    return {key: np.array(state[key], copy=True) for key in ("r", "y", "ga", "glb", "sa", "sb")}


class VariationalGGMM:
    """Fit X only; inspect density_, n_components_, timing_ and merge_log_.

    initial_shape_mode="shape2" sets starting b=2; later log(b) is optimized.
    VGM comparison, sample generation and truth-based metrics belong to examples.
    posterior_state_["r"] refers to sorted, standardized training observations.
    """
    def __init__(self, config=None):
        self.config = config or VGGMMConfig()

    def _empty_slot(self, state, k):
        # An unused b point estimate has no prior/likelihood contribution at Nk=0.
        # Canonical b=2 avoids exp(log(scale)) underflow from an old tiny shape.
        state["y"][k] = [0., np.log(self.config.mean_prior_sd), np.log(2.)]
        state["ga"][k] = self.config.precision_prior_shape
        state["glb"][k] = np.log(self.config.precision_prior_rate)

    def _profile_state(self, state, x):
        nk = state["r"].sum(axis=0)
        state["sa"], state["sb"] = stick_posterior(nk, self.config.concentration)
        for k in range(len(nk)):
            if nk[k] < 1e-12:
                self._empty_slot(state, k)
            else:
                _, state["ga"][k], state["glb"][k], _ = component_profile(
                    state["y"][k], x, state["r"][:, k], self.config)
        return state

    def _density(self, state):
        b = np.exp(state["y"][:, 2])
        loga = (state["glb"]-np.log(state["ga"]))/b + np.log(self.unit_)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            scales = np.exp(loga)
        weights, _ = stick_statistics(state["sa"], state["sb"])
        return Mixture(weights, state["y"][:, 0]*self.unit_+self.center_, scales, b)

    def _elbo(self, state, x):
        elogpdf = expected_log_components(x, state["y"], state["ga"], state["glb"])
        value = elbo_value(state["r"], elogpdf, state["sa"], state["sb"],
                           state["y"], state["ga"], state["glb"], self.config)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite full ELBO.")
        return value-len(x)*np.log(self.unit_), elogpdf

    def _component_update(self, x, r, initial_y):
        nk = float(r.sum())
        old_value, *_ = component_profile(initial_y, x, r, self.config)
        best = [old_value, initial_y.copy()]
        rejections = 0

        def objective(params):
            nonlocal rejections
            try:
                value, *_ = component_profile(params, x, r, self.config)
                if value > best[0]:
                    best[:] = [value, params.copy()]
                return -value/max(nk, 1.)
            except (FloatingPointError, ValueError, OverflowError):
                rejections += 1
                return 1e100

        opt = minimize(objective, initial_y, method="L-BFGS-B",
            options=dict(maxiter=self.config.inner_max_iter, ftol=self.config.inner_ftol, maxls=30))
        _, ga, glb, _ = component_profile(best[1], x, r, self.config)
        return best[1], ga, glb, int(not opt.success), rejections, int(opt.nfev)

    def _initial_state(self, x, t, name):
        labels = partition(x, t, name)
        r = np.eye(t)[labels]
        y = np.zeros((t, 3))
        for k in range(t):
            group = x[labels == k]
            sd = max(float(group.std()), np.sqrt(np.finfo(float).eps))
            b = 2. if self.config.initial_shape_mode == "shape2" else moment_initial_shape(group)
            y[k] = [group.mean(), np.log(sd/np.sqrt(len(group))), np.log(b)]
        state = dict(r=r, y=y, ga=np.empty(t), glb=np.empty(t), sa=np.empty(t-1), sb=np.empty(t-1))
        for k in range(t):
            try:
                _, state["ga"][k], state["glb"][k], _ = component_profile(y[k], x, r[:, k], self.config)
            except (FloatingPointError, ValueError):
                y[k, 2] = np.log(2.)
                _, state["ga"][k], state["glb"][k], _ = component_profile(y[k], x, r[:, k], self.config)
        state["sa"], state["sb"] = stick_posterior(r.sum(axis=0), self.config.concentration)
        return state

    def _optimize(self, x, initial_state, name):
        # Include state copying and the initial Gamma/stick reprofile in this refit.
        begin = perf_counter()
        cfg = self.config
        state = copy_state(initial_state)
        self._profile_state(state, x)
        history = []
        stable, previous, failures, rejections, nfev = 0, None, 0, 0, 0
        converged, reason = False, "outer_max_iter"
        for iteration in range(cfg.max_iter+1):
            value, elogpdf = self._elbo(state, x)
            weights, elogw = stick_statistics(state["sa"], state["sb"])
            history.append(dict(iteration=iteration, elbo=value,
                active_K=int(np.sum(weights > cfg.active_threshold)), seconds=perf_counter()-begin))
            if previous is not None:
                gain = (value-previous)/len(x)
                if gain < -1e-7:
                    reason = "ELBO_decreased_numerical_failure"
                    break
                stable = stable+1 if abs(gain) < cfg.tol else 0
                if stable >= 3:
                    converged, reason = True, "ELBO_gain_per_observation"
                    break
            if iteration == cfg.max_iter:
                break
            previous = value
            logjoint = elogpdf+elogw  # All Kmax slots, including newly emptied slots.
            lognorm = logsumexp(logjoint, axis=1, keepdims=True)
            if not np.isfinite(lognorm).all():
                raise FloatingPointError("No finite expected density at an observation.")
            state["r"] = np.exp(logjoint-lognorm)
            state["sa"], state["sb"] = stick_posterior(state["r"].sum(axis=0), cfg.concentration)
            for k in range(state["r"].shape[1]):
                rk = state["r"][:, k]
                if float(rk.sum()) < 1e-12:
                    self._empty_slot(state, k)
                    continue
                y, ga, glb, fail, reject, evaluations = self._component_update(x, rk, state["y"][k])
                state["y"][k], state["ga"][k], state["glb"][k] = y, ga, glb
                failures += fail
                rejections += reject
                nfev += evaluations
        # Do not require a plug-in density at every iteration; ELBO controls inference.
        # The exported final density must nevertheless have representable parameters.
        self._density(state)
        return dict(name=name, state=state, history=history, elbo=history[-1]["elbo"],
            converged=converged, reason=reason, seconds=perf_counter()-begin,
            inner_nonconverged=failures, numerical_rejections=rejections, nfev=nfev)

    def _active_k(self, state):
        weights, _ = stick_statistics(state["sa"], state["sb"])
        return int(np.sum(weights > self.config.active_threshold))


    # Merge proposals and full-refit acceptance (same class / same VS Code cell).
    def _pair_pool(self, state, x):
        r = state["r"]
        nk = r.sum(axis=0)
        live = np.flatnonzero(nk > self.config.candidate_min_effective_count)
        if len(live) < 2:
            return []
        # Observation centers/spreads, not the SD of the posterior over a mean.
        centers = np.array([np.dot(r[:, k], x)/nk[k] for k in live])
        order = live[np.argsort(centers, kind="stable")]
        pairs = []
        for left, right in zip(order[:-1], order[1:]):
            i, j = sorted((int(left), int(right)))  # Never reorder the sticks.
            overlap = float(np.sqrt(r[:, i]*r[:, j]).sum()/np.sqrt(nk[i]*nk[j]))
            pairs.append(dict(i=i, j=j, overlap=overlap))
        return pairs

    def _merge_seed(self, incumbent, x, i, j):
        state = copy_state(incumbent)
        parents = np.exp(state["y"][[i, j], 2])
        state["r"][:, i] += state["r"][:, j]
        state["r"][:, j] = 0.
        # The total responsibility remains one for every observation.
        pooled = state["r"][:, i]
        nk = float(pooled.sum())
        mean = float(np.dot(pooled, x)/nk)
        variance = float(np.dot(pooled, (x-mean)**2)/nk)
        posterior_sd = np.sqrt(max(variance, np.finfo(float).eps)/max(nk, 1.))
        # Shape seeds affect only proposal initialization. The fitted shape stays free.
        shapes = (np.array([2.]) if self.config.initial_shape_mode == "shape2"
                  else np.unique(np.r_[2., parents]))
        best, seed_failures = None, []
        for b in shapes:
            y0 = np.array([mean, np.log(posterior_sd), np.log(b)])
            try:
                y, ga, glb, fail, reject, nfev = self._component_update(x, pooled, y0)
                value, *_ = component_profile(y, x, pooled, self.config)
                if best is None or value > best[0]:
                    best = (value, y, ga, glb, fail, reject, nfev, float(b))
            except (FloatingPointError, ValueError, OverflowError) as exc:
                seed_failures.append(f"b={b:g}: {type(exc).__name__}: {exc}")
        if best is None:
            raise FloatingPointError(f"All merged-component seeds failed: {seed_failures}")
        _, state["y"][i], state["ga"][i], state["glb"][i], *info = best
        self._empty_slot(state, j)
        self._profile_state(state, x)
        seed_elbo, _ = self._elbo(state, x)
        return state, seed_elbo, best[-1], seed_failures

    def _merge_search(self, x, baseline):
        begin = perf_counter()
        cfg = self.config
        incumbent = baseline
        logs, controls, checkpoints, accepted = [], [], [baseline], 0
        seed_logs = []
        timing = dict(Continuation_s=0., Pair_pool_s=0., Seed_screening_s=0.,
                      Candidate_refit_s=0.)
        candidate_refits = 0
        stop_reason = "merge_round_budget_exhausted"
        for round_index in range(1, cfg.max_merge_rounds+1):
            # Give unmerged coordinate ascent another equal-budget continuation.
            # Compare proposals against the improved control, not an underoptimized baseline.
            control_entry = dict(Round=round_index, Used=False, Converged=False, Error="")
            tick = perf_counter()
            try:
                control = self._optimize(x, incumbent["state"], f"continuation-{round_index}")
            except (FloatingPointError, ValueError, OverflowError) as exc:
                control_entry["Error"] = f"{type(exc).__name__}: {exc}"
            finally:
                control_wall = perf_counter()-tick
                timing["Continuation_s"] += control_wall
                control_entry["Wall_s"] = control_wall
            if not control_entry["Error"]:
                control_gain = (control["elbo"]-incumbent["elbo"])/len(x)
                took_control = bool(control["converged"] and control_gain > 0.)
                control_entry.update(Before_ELBO=incumbent["elbo"],
                    After_ELBO=control["elbo"], Gain_per_observation=control_gain,
                    Converged=control["converged"], Used=took_control,
                    Inner_nonconverged=control["inner_nonconverged"])
                if took_control:
                    incumbent = control
            controls.append(control_entry)
            tick = perf_counter()
            try:
                pool = self._pair_pool(incumbent["state"], x)
            finally:
                timing["Pair_pool_s"] += perf_counter()-tick
            if not pool:
                stop_reason = "fewer_than_two_numerically_occupied_slots"
                break
            prepared = []
            for pair in pool:
                seed_entry = dict(Round=round_index, Pair_i=pair["i"], Pair_j=pair["j"],
                    Pair_overlap=pair["overlap"], Seed_ELBO=np.nan, Seed_b=np.nan,
                    Prepared=False, Error="")
                tick = perf_counter()
                try:
                    seed, value, seed_b, seed_errors = self._merge_seed(
                        incumbent["state"], x, pair["i"], pair["j"])
                except (FloatingPointError, ValueError, OverflowError) as exc:
                    seed_entry["Error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    seed_wall = perf_counter()-tick
                    timing["Seed_screening_s"] += seed_wall
                    seed_entry["Seed_wall_s"] = seed_wall
                if not seed_entry["Error"]:
                    prepared.append(dict(**pair, state=seed, seed_elbo=value, seed_b=seed_b,
                                         seed_errors=seed_errors, seed_wall_s=seed_wall))
                    seed_entry.update(Seed_ELBO=value, Seed_b=seed_b, Prepared=True,
                                      Seed_errors="; ".join(seed_errors))
                else:
                    logs.append(dict(Round=round_index, Pair_i=pair["i"], Pair_j=pair["j"],
                        Accepted=False, Status="seed_failed", Seed_wall_s=seed_wall,
                        Error=seed_entry["Error"]))
                seed_logs.append(seed_entry)
            # Full seed ELBO ranks proposals. A low seed ELBO is not itself a rejection.
            prepared.sort(key=lambda item: (-item["seed_elbo"], -item["overlap"], item["i"], item["j"]))
            if not prepared:
                stop_reason = "all_merge_seeds_failed"
                break
            any_accepted = False
            for rank, proposal in enumerate(prepared[:cfg.max_candidates_per_round], start=1):
                before = incumbent["elbo"]
                before_k = self._active_k(incumbent["state"])
                entry = dict(Round=round_index, Rank=rank, Pair_i=proposal["i"], Pair_j=proposal["j"],
                    Pair_overlap=proposal["overlap"], Seed_b=proposal["seed_b"],
                    Seed_errors="; ".join(proposal["seed_errors"]),
                    Seed_ELBO=proposal["seed_elbo"], Seed_wall_s=proposal["seed_wall_s"],
                    Before_ELBO=before,
                    Before_active_K=before_k, Pool_size=len(pool),
                    Prepared_candidates=len(prepared), Accepted=False, Error="")
                candidate_refits += 1
                tick = perf_counter()
                try:
                    candidate = self._optimize(x, proposal["state"],
                        f"merge-{round_index}-{proposal['i']}-{proposal['j']}")
                except (FloatingPointError, ValueError, OverflowError) as exc:
                    entry.update(Status="refit_failed", Error=f"{type(exc).__name__}: {exc}")
                finally:
                    refit_wall = perf_counter()-tick
                    timing["Candidate_refit_s"] += refit_wall
                    entry["Refit_wall_s"] = refit_wall
                if not entry["Error"]:
                    gain = (candidate["elbo"]-before)/len(x)
                    after_k = self._active_k(candidate["state"])
                    take = bool(candidate["converged"] and gain > cfg.merge_gain_per_observation)
                    entry.update(After_ELBO=candidate["elbo"], Gain_per_observation=gain,
                        After_active_K=after_k, Refit_converged=candidate["converged"],
                        Refit_stop_reason=candidate["reason"], Refit_iterations=len(candidate["history"])-1,
                        Inner_nonconverged=candidate["inner_nonconverged"],
                        Refit_s=candidate["seconds"], Accepted=take,
                        Status="accepted" if take else "rejected")
                    if take:
                        incumbent, any_accepted = candidate, True
                        accepted += 1
                        checkpoints.append(candidate)
                logs.append(entry)
                if entry["Accepted"]:
                    print(f"    merge ({proposal['i']}, {proposal['j']}): K {before_k}->{after_k}, "
                          f"ELBO/n +{gain:.6g}", flush=True)
                    break  # First improving proposal; rebuild the pool from the new state.
            if not any_accepted:
                stop_reason = ("candidate_budget_exhausted_without_improvement"
                    if len(prepared) > cfg.max_candidates_per_round
                    else "no_improvement_in_all_prepared_adjacent_pairs")
                break
        self.merge_log_, self.continuation_log_, self.merge_checkpoints_ = logs, controls, checkpoints
        self.seed_log_ = seed_logs
        self.accepted_merges_, self.merge_stop_reason_ = accepted, stop_reason
        self.merge_seconds_ = perf_counter()-begin
        # The measured categories above never nest; the residual covers ranking,
        # acceptance checks, logging and search management, including rejected moves.
        timing["Merge_management_s"] = self.merge_seconds_-sum(timing.values())
        self.merge_timing_ = timing
        self.work_.update(Continuation_refits=len(controls), Candidate_seeds=len(seed_logs),
                          Candidate_refits=candidate_refits, Accepted_merges=accepted)
        return incumbent

    def fit(self, X):
        """Estimate one finite nonconstant continuous column and return self."""
        cfg = self.config
        started = perf_counter()
        x = np.array(X, dtype=float, copy=True)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.std(x) <= 0:
            raise ValueError("X must be a finite nonconstant continuous column.")
        integer_fields = (cfg.max_components, cfg.max_iter, cfg.inner_max_iter,
                          cfg.max_merge_rounds, cfg.max_candidates_per_round)
        if (any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) or v < 1
                for v in integer_fields)
            or not cfg.starts or any(s not in ("quantile", "lloyd") for s in cfg.starts)
            or cfg.initial_shape_mode not in ("moment", "shape2")
            or not 0 <= cfg.active_threshold < 1
            or any(not np.isfinite(v) or v <= 0 for v in (cfg.concentration, cfg.tol,
                cfg.mean_prior_sd, cfg.precision_prior_shape, cfg.precision_prior_rate,
                cfg.inner_ftol, cfg.merge_gain_per_observation, cfg.candidate_min_effective_count))):
            raise ValueError("Invalid VGGMM config.")
        self.center_, self.unit_, self.raw_x_ = float(x.mean()), float(x.std()), x.copy()
        z = np.sort((x-self.center_)/self.unit_)
        self.standardized_x_ = z
        t = min(cfg.max_components, len(np.unique(x)), len(x))
        self.runs_, self.failed_starts_ = [], []
        self.initial_start_timing_ = []
        self.work_ = dict(Initial_starts_attempted=len(cfg.starts), Continuation_refits=0,
                          Candidate_seeds=0, Candidate_refits=0, Accepted_merges=0)
        for name in cfg.starts:
            start_begin = perf_counter()
            phase = "initialization"
            phase_begin = start_begin
            init_seconds, optimize_seconds = 0., 0.
            start_entry = dict(Start=name, Status="failed", Error="")
            try:
                state = self._initial_state(z, t, name)
                init_seconds = perf_counter()-phase_begin
                phase = "optimization"
                phase_begin = perf_counter()
                run = self._optimize(z, state, name)
                optimize_seconds = perf_counter()-phase_begin
                run["Initialization_s"] = init_seconds
                run["Optimize_wall_s"] = optimize_seconds
                self.runs_.append(run)
                start_entry["Status"] = "completed"
            except (FloatingPointError, ValueError, OverflowError) as exc:
                phase_seconds = perf_counter()-phase_begin
                if phase == "initialization":
                    init_seconds = phase_seconds
                else:
                    optimize_seconds = phase_seconds
                start_entry["Error"] = f"{type(exc).__name__}: {exc}"
                self.failed_starts_.append(dict(start=name, error_type=type(exc).__name__, error=str(exc),
                    Initialization_s=init_seconds, Optimize_wall_s=optimize_seconds))
            finally:
                start_entry.update(Initialization_s=init_seconds, Optimize_wall_s=optimize_seconds,
                                   Start_wall_s=perf_counter()-start_begin)
                self.initial_start_timing_.append(start_entry)
        if not self.runs_:
            raise RuntimeError(f"All starts failed: {self.failed_starts_}")
        # Select from converged starts if available; report a fallback explicitly.
        valid = [run for run in self.runs_ if run["converged"]]
        self.baseline_selection_fallback_ = not bool(valid)
        baseline = max(valid or self.runs_, key=lambda run: run["elbo"])
        self.baseline_run_ = baseline
        self.baseline_density_ = self._density(baseline["state"])
        self.baseline_seconds_ = perf_counter()-started
        self.merge_log_, self.continuation_log_, self.merge_checkpoints_ = [], [], [baseline]
        self.seed_log_ = []
        self.merge_timing_ = dict(Continuation_s=0., Pair_pool_s=0., Seed_screening_s=0.,
                                 Candidate_refit_s=0., Merge_management_s=0.)
        self.accepted_merges_, self.merge_seconds_, self.merge_stop_reason_ = 0, 0., "disabled"
        final = self._merge_search(z, baseline) if cfg.enable_merge else baseline
        self.final_run_, self.posterior_state_ = final, copy_state(final["state"])
        self.full_density_ = self._density(final["state"])
        self.weights_ = self.full_density_.weights.copy()
        keep = self.weights_ > cfg.active_threshold
        if not keep.any():
            raise RuntimeError("No active component at the reporting threshold.")
        self.density_ = Mixture(self.weights_[keep]/self.weights_[keep].sum(),
            self.full_density_.means[keep], self.full_density_.scales[keep], self.full_density_.shapes[keep])
        self.n_components_, self.full_k_ = int(keep.sum()), t
        self.converged_, self.termination_reason_ = final["converged"], final["reason"]
        self.history_ = final["history"]
        if not self.converged_:
            warnings.warn("Selected fit did not satisfy the outer convergence criterion.", RuntimeWarning)
        if self.failed_starts_:
            warnings.warn(f"{len(self.failed_starts_)} initial start(s) failed; see failed_starts_.", RuntimeWarning)
        self.fit_seconds_ = perf_counter()-started
        self.timing_ = dict(Baseline_fit_s=self.baseline_seconds_, Merge_extra_s=self.merge_seconds_,
            Finalization_s=self.fit_seconds_-self.baseline_seconds_-self.merge_seconds_,
            Total_fit_s=self.fit_seconds_, **self.merge_timing_)
        return self
