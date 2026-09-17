"""Independent, single-sample GGMM: bounded-scale likelihood fits + BIC.

No CDF objective, DKW selection, shape penalty, b upper bound, random
initialization, held-out sample, or fitted GMM input. This is numerical local
maximum likelihood, not a variational Bayesian GGMM or guaranteed global MLE.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import cast
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import gennorm


@dataclass(frozen=True)
class SimpleGGMMConfig:
    max_components: int = 10
    scale_floor_relative: float = 1e-3  # a_min / sample population SD
    max_iter: int = 1500
    ftol: float = 1e-10
    gtol: float = 1e-5


@dataclass
class Mixture:
    weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    shapes: np.ndarray

    def __post_init__(self):
        for name in ('weights', 'means', 'scales', 'shapes'):
            setattr(self, name, np.asarray(getattr(self, name), float))
        if (self.weights.ndim != 1 or len(self.weights) == 0
                or any(getattr(self, p).shape != self.weights.shape
                       for p in ('means', 'scales', 'shapes'))
                or not all(np.isfinite(getattr(self, p)).all()
                           for p in ('weights', 'means', 'scales', 'shapes'))
                or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1)
                or np.any(self.scales <= 0) or np.any(self.shapes <= 0)):
            raise ValueError('Invalid mixture parameters.')

    @property
    def k(self):
        return len(self.weights)

    @property
    def sigmas(self):
        q = 1 / self.shapes
        with np.errstate(over='ignore'):
            return np.exp(np.log(self.scales) + .5 * (
                gammaln(1 + 3*q) - gammaln(1 + q) - np.log(3.)))

    def component_logpdf(self, x):
        return component_terms(np.asarray(x).reshape(-1), self.means,
                               np.log(self.scales), np.log(self.shapes))[0]

    def logpdf(self, x):
        return cast(np.ndarray, logsumexp(self.component_logpdf(x) + np.log(self.weights), axis=1))

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        return (gennorm.cdf(np.asarray(x).reshape(-1, 1), self.shapes,
                            loc=self.means, scale=self.scales) * self.weights).sum(axis=1)

    def component_ppf(self, p, labels):
        labels = np.asarray(labels, int)
        return gennorm.ppf(p, self.shapes[labels], loc=self.means[labels],
                           scale=self.scales[labels])

    def payload(self):
        return {p: getattr(self, p).tolist() for p in ('weights','means','scales','shapes')}

    def formula(self, digits=6):
        """g(x;mu,a,b) parameterization; rounded for display only."""
        f = lambda v: format(float(v), f'.{digits}g')
        return ' + '.join(f'{f(w)} g(x; {f(m)}, {f(a)}, {f(b)})'
                         for w,m,a,b in zip(self.weights,self.means,self.scales,self.shapes))


def unpack(theta, k):
    logits = np.r_[theta[:k-1], 0.]
    lw = logits - logsumexp(logits)
    return lw, theta[k-1:2*k-1], theta[2*k-1:3*k-1], theta[3*k-1:]


def pack(model):
    return np.r_[np.log(model.weights[:-1]/model.weights[-1]), model.means,
                 np.log(model.scales), np.log(model.shapes)]


def component_terms(x, mu, la, eta):
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        b, q = np.exp(eta), np.exp(-eta)
        if np.any(b == 0) or np.any(q == 0):
            raise FloatingPointError('Shape cannot be represented.')
        norm = -np.log(2.) - la - gammaln(1 + q)
        dc = q * digamma(1 + q)
    if not np.isfinite(norm).all() or not np.isfinite(dc).all():
        raise FloatingPointError('Shape normalizer cannot be represented.')
    delta = x[:, None] - mu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'):
        lr = np.log(np.abs(delta)) - la
        lt = b * lr
        lp = norm - np.exp(lt)
    if np.isnan(lp).any():
        raise FloatingPointError('Invalid component density.')
    return lp, delta, lr, lt, dc


def nll_gradient(theta, x, k):
    """Mean NLL and exact derivatives where location derivatives exist.

    For b<=1, x=mu is nonsmooth; a zero location subgradient/convention is
    used at exact equality and this event is reported. No exponent clipping.
    """
    lw, mu, la, eta = unpack(theta, k)
    lp, delta, lr, lt, dc = component_terms(x, mu, la, eta)
    joint = lp + lw
    denom = cast(np.ndarray, logsumexp(joint, axis=1))
    if not np.isfinite(denom).all():
        raise FloatingPointError('Nonfinite mixture log density at an observation.')
    log_r = joint - denom[:, None]
    r = np.exp(log_r)
    with np.errstate(over='raise', invalid='raise', divide='ignore', under='ignore'):
        valid = np.isfinite(joint) & np.isfinite(lt)
        log_rbt = np.full_like(log_r, -np.inf)
        np.add(log_r + eta, lt, out=log_rbt, where=valid)
        rbt = np.exp(log_rbt)
        log_location = np.full_like(log_rbt, -np.inf)
        np.subtract(log_rbt, np.log(np.abs(delta)), out=log_location, where=delta != 0)
        location = np.sign(delta)*np.exp(log_location)
        shape_term = np.zeros_like(rbt)
        np.multiply(rbt, lr, out=shape_term, where=rbt != 0)
        grad = np.r_[np.exp(lw[:-1])-r.mean(axis=0)[:-1], -location.mean(axis=0),
                     (r-rbt).mean(axis=0), (shape_term-r*dc).mean(axis=0)]
    if not np.isfinite(grad).all():
        raise FloatingPointError('Nonfinite likelihood derivative.')
    return float(-denom.mean()), grad


def initial_shape(group):
    """Match mean absolute deviation / SD; fallback only if moment equation fails.

    The theoretical ratio increases from 0 to sqrt(3)/2. A finite sample's
    moment ratio can lie outside that interval. The b=2 fallback is a starting
    point, not a shape restriction or representative-shape assertion.
    """
    mu, sd = float(np.mean(group)), float(np.std(group))
    ratio = float(np.mean(np.abs(group-mu))/sd) if sd > 0 else np.nan
    if len(group) < 4 or not 0 < ratio < np.sqrt(3)/2:
        return 2., True
    def residual(eta):
        q = np.exp(-eta)
        return .5*np.log(3)-np.log(2)+gammaln(1+2*q)-.5*(
            gammaln(1+q)+gammaln(1+3*q))-np.log(ratio)
    lo, hi = -1., 1.
    while residual(lo) > 0 and lo > -16:
        lo -= 1
    while residual(hi) < 0 and hi < 24:
        hi += 1
    if residual(lo) * residual(hi) > 0:
        return 2., True
    return float(np.exp(cast(float, brentq(residual, lo, hi)))), False


def initializations(x, k, floor):
    """Two deterministic partitions, no density model fitted during initialization."""
    quantile = np.minimum(np.arange(len(x))*k//len(x), k-1)
    centers = np.quantile(x, (np.arange(k)+.5)/k)
    labels = None
    for _ in range(100):
        distance = (x[:, None]-centers)**2
        new = distance.argmin(axis=1)
        counts = np.bincount(new, minlength=k)
        for j in np.flatnonzero(counts == 0):
            residual = distance[np.arange(len(x)), new].copy()
            residual[counts[new] <= 1] = -np.inf
            i = int(np.argmax(residual))
            counts[new[i]] -= 1
            new[i] = j
            counts[j] += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centers = np.array([x[labels == j].mean() for j in range(k)])
    partitions = [('quantile', quantile)]
    if labels is not None and not np.array_equal(labels, quantile):
        partitions.append(('lloyd', labels))
    for name, groups in partitions:
        parts = [x[groups == j] for j in range(k)]
        bs = [initial_shape(p) for p in parts]
        b = np.array([t[0] for t in bs])
        sigma = np.array([max(p.std(), floor) for p in parts])
        la = np.log(sigma) + .5*(np.log(3)+gammaln(1+1/b)-gammaln(1+3/b))
        a = np.exp(np.maximum(la, np.log(floor)))
        yield name, Mixture(np.array([len(p) for p in parts])/len(x),
                            np.array([p.mean() for p in parts]), a, b), sum(t[1] for t in bs)


class SimpleGGMM:
    def __init__(self, config=None):
        self.config = config or SimpleGGMMConfig()

    def fit(self, X):
        cfg = self.config
        x = np.asarray(X, float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
            raise ValueError('Provide one finite continuous sample, with at least four observations.')
        if (not isinstance(cfg.max_components, int) or cfg.max_components < 1
                or not np.isfinite(cfg.scale_floor_relative) or cfg.scale_floor_relative <= 0
                or cfg.max_iter < 1 or not 0 < cfg.ftol < 1 or not 0 < cfg.gtol < 1):
            raise ValueError('Invalid configuration.')
        start = perf_counter()
        self.center_, self.unit_ = float(np.mean(x)), float(np.std(x))
        if not np.isfinite(self.unit_) or self.unit_ <= 0:
            raise ValueError('Constant or unrepresentable range: use a separate constant-column handler.')
        z = np.sort((x-self.center_)/self.unit_)
        if not np.isfinite(z).all():
            raise ValueError('Rescale the observations before fitting.')
        self.a_min_ = cfg.scale_floor_relative*self.unit_
        self.starts_, self.selection_, self.models_ = [], [], {}
        upper = min(cfg.max_components, len(np.unique(z)), len(z))
        previous_ll = -np.inf
        for k in range(1, upper+1):
            tick = perf_counter()
            bounds = [(None, None)]*(2*k-1) + [(np.log(cfg.scale_floor_relative), None)]*k + [(None, None)]*k
            candidates = []
            for name, initial, fallback_count in initializations(z, k, cfg.scale_floor_relative):
                best = [np.inf, None]
                rejected = [0]
                def objective(theta):
                    try:
                        value, grad = nll_gradient(theta, z, k)
                        if value < best[0]:
                            best[:] = value, theta.copy()
                        return value, grad
                    except (FloatingPointError, ValueError):
                        rejected[0] += 1
                        return np.inf, np.zeros_like(theta)
                theta0 = pack(initial)
                objective(theta0)
                opt = minimize(objective, theta0, jac=True, method='L-BFGS-B', bounds=bounds,
                               options=dict(maxiter=cfg.max_iter, ftol=cfg.ftol, gtol=cfg.gtol, maxls=50))
                theta = best[1]
                if theta is None:
                    self.starts_.append(dict(K=k, start=name, finite=False, converged=False,
                                             message='No finite likelihood', rejected_steps=rejected[0]))
                    continue
                value, grad = nll_gradient(theta, z, k)
                # Returning the best visited point avoids discarding a better finite trial.
                # If it differs from the optimizer endpoint, do not claim convergence.
                at_endpoint = bool(np.allclose(theta, opt.x, rtol=1e-10, atol=1e-10))
                for j in range(2*k-1, 3*k-1):
                    if theta[j] <= np.log(cfg.scale_floor_relative)+1e-8 and grad[j] > 0:
                        grad[j] = 0
                lw, mu, la, eta = unpack(theta, k)
                order = np.argsort(mu)
                try:
                    model = Mixture(np.exp(lw)[order], (self.center_+self.unit_*mu)[order],
                                    (self.unit_*np.exp(la))[order], np.exp(eta)[order])
                except ValueError as exc:
                    self.starts_.append(dict(K=k, start=name, finite=False, converged=False,
                        message=f'Unrepresentable final mixture: {exc}', rejected_steps=rejected[0]))
                    continue
                ll = float(-len(z)*(value+np.log(self.unit_)))
                row = dict(K=k, start=name, finite=True, log_likelihood=ll,
                           BIC=-2*ll+(4*k-1)*np.log(len(z)), converged=bool(opt.success and at_endpoint),
                           projected_gradient=float(np.max(np.abs(grad))), iterations=int(opt.nit),
                           message=str(opt.message), best_is_endpoint=at_endpoint,
                           rejected_steps=rejected[0], initial_shapes=initial.shapes.tolist(),
                           moment_fallback_components=int(fallback_count),
                           a_floor_hits=int(np.sum(model.scales <= self.a_min_*(1+1e-6))),
                           max_b=float(model.shapes.max()), min_b=float(model.shapes.min()),
                           nonsmooth_coincidences=int(np.sum((z[:, None] == mu) & (np.exp(eta) <= 1))))
                self.starts_.append(row)
                candidates.append((row, model))
            if not candidates:
                self.selection_.append(dict(K=k, finite=False, converged=False, BIC=np.inf))
                continue
            row, model = min(candidates, key=lambda item: item[0]['BIC'])
            row = row.copy()
            row.update(seconds=perf_counter()-tick,
                       likelihood_decreased=bool(row['log_likelihood'] < previous_ll-1e-5))
            previous_ll = max(previous_ll, row['log_likelihood'])
            self.models_[k] = model
            self.selection_.append(row)
        finite = [r for r in self.selection_ if r['finite']]
        if not finite:
            raise RuntimeError('All fits failed; inspect starts_.')
        winner = min(finite, key=lambda r: r['BIC'])
        self.n_components_ = winner['K']
        self.density_ = self.models_[self.n_components_]
        for r in self.selection_:
            r['delta_BIC'] = r['BIC']-winner['BIC']
            r['selected'] = r['K'] == self.n_components_
        self.diagnostics_ = dict(n=len(x), selected_K=self.n_components_, config=asdict(cfg),
            scale_floor_original_units=self.a_min_, selected_converged=winner['converged'],
            any_K_nonconverged=any(not r['converged'] for r in self.selection_),
            any_likelihood_decrease=any(r.get('likelihood_decreased',False) for r in self.selection_),
            selected_at_Kmax=self.n_components_ == upper,
            status='numerical_local_fit; no global optimum or true-K guarantee', seconds=perf_counter()-start)
        return self

    def predict_proba(self, X):
        joint = self.density_.component_logpdf(X) + np.log(self.density_.weights)
        denominator = cast(np.ndarray, logsumexp(joint, axis=1))
        if not np.isfinite(denominator).all():
            raise FloatingPointError('Posterior probabilities cannot be represented for these values.')
        return np.exp(joint-denominator[:, None])
