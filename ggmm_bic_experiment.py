"""Synthetic experiment and comparison for the BIC-selected GGMM estimator.

Truth is passed to generation and POST-FIT evaluation only. All reported
densities are normalized plug-in mixtures, including sklearn's VGM comparator.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, TypedDict, cast
import platform
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln
from scipy.stats import gennorm, gaussian_kde
from scipy.integrate import trapezoid, quad_vec
from IPython.display import display
from sklearn.mixture import BayesianGaussianMixture
from threadpoolctl import threadpool_limits
from ggmm_bic_estimator import Mixture, GGMMBICEstimator, GGMMBICConfig


TRUTH_PARAMETERS = dict(weights=[.2,.3,.5], means=[-5.,0.,6.],
                        scales=[1.2,1.6,2.], shapes=[1.5,4.,8.])
SAMPLE_SIZE = 5000
SAMPLE_SEED = 20260912
VGM_SEED = 27


class ExperimentResult(TypedDict):
    run_id: str
    x: np.ndarray
    truth: Mixture | None
    ggmm: GGMMBICEstimator
    config: dict[str, Any]
    fit_seconds: float
    selection: pd.DataFrame
    starts: pd.DataFrame
    parameters: pd.DataFrame
    metadata: dict[str, Any]


def sample_mixture(truth, n, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = rng.choice(truth.k, size=n, p=truth.weights)
    return cast(np.ndarray, gennorm.rvs(truth.shapes[labels], loc=truth.means[labels],
                                       scale=truth.scales[labels], random_state=rng))


def _points(model, tail):
    probabilities = [tail,1e-7,1e-5,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1-1e-5,1-1e-7,1-tail]
    p = np.tile(probabilities, model.k)
    labels = np.repeat(np.arange(model.k), len(probabilities))
    return np.unique(np.r_[model.component_ppf(p, labels), model.means,
                           model.means-model.scales, model.means+model.scales])


def density_errors(truth, fitted, tail=1e-9, atol=1e-7, rtol=1e-5):
    """Post-fit IAE/ISE: adaptive 32/64-node panels and omitted-tail bounds.

    Quadrature errors are estimates, not formal interval arithmetic bounds.
    The true density is never used for K selection or any fitting setting.
    """
    points = np.unique(np.r_[_points(truth,tail), _points(fitted,tail)])
    if not np.isfinite(points).all():
        raise FloatingPointError('Evaluation quantiles cannot be represented.')
    left,right = points[:-1],points[1:]
    n_panels = len(left)
    depth = np.zeros(n_panels,int)
    total,error = np.zeros(2),np.zeros(2)
    unresolved = 0
    def panel(l,r,order):
        nodes,weights = leggauss(order)
        mid,half = l/2+r/2,(r-l)/2
        xx = mid[:,None]+half[:,None]*nodes
        delta = (fitted.pdf(xx.ravel())-truth.pdf(xx.ravel())).reshape(xx.shape)
        return np.c_[np.abs(delta)@weights, delta**2@weights]*half[:,None]
    while len(left):
        low, high = panel(left,right,32), panel(left,right,64)
        err = np.abs(high-low)
        good = np.all(err <= atol/n_panels/(2.**depth[:,None])+rtol*np.abs(high),axis=1)
        stopped = depth >= 12
        accept = good|stopped
        total += high[accept].sum(axis=0)
        error += err[accept].sum(axis=0)
        unresolved += int(np.sum(stopped&~good))
        ll,rr,dd = left[~accept],right[~accept],depth[~accept]
        mid = ll/2+rr/2
        left,right,depth = np.r_[ll,mid],np.r_[mid,rr],np.r_[dd+1,dd+1]
    lo,hi = points[0],points[-1]
    def omitted(model):
        return float(np.clip(model.cdf([lo])[0]+1-model.cdf([hi])[0],0,1))
    def peak(model):
        return float(np.sum(model.weights*np.exp(-np.log(2)-np.log(model.scales)-gammaln(1+1/model.shapes))))
    tt,tf = omitted(truth),omitted(fitted)
    return dict(IAE=float(total[0]), ISE=float(total[1]),
        IAE_quadrature_error=float(error[0]), ISE_quadrature_error=float(error[1]),
        IAE_omitted_tail_upper=tt+tf, ISE_omitted_tail_upper=peak(truth)*tt+peak(fitted)*tf,
        quadrature_tolerance_met=bool(np.all(error <= atol+rtol*np.abs(total))),
        unresolved_panels=unresolved, integration_left=float(lo), integration_right=float(hi))


def fit_vgm(x, max_components=10):
    """RDT-style prior and active threshold; larger convergence budget disclosed."""
    tick = perf_counter()
    fitted = BayesianGaussianMixture(n_components=max_components, covariance_type='full',
        weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=.001,
        init_params='kmeans', n_init=1, max_iter=1500, tol=1e-3, reg_covar=1e-6, random_state=VGM_SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted.fit(x.reshape(-1,1))
    seconds = perf_counter()-tick
    w,mu,a = fitted.weights_,fitted.means_[:,0],np.sqrt(2*fitted.covariances_[:,0,0])
    full = Mixture(w/w.sum(),mu,a,np.full(len(w),2.))
    mask = w > .005
    active = Mixture(w[mask]/w[mask].sum(),mu[mask],a[mask],np.full(int(mask.sum()),2.))
    diagnostic = dict(converged=bool(fitted.converged_),iterations=int(fitted.n_iter_),
        seconds=seconds,active_K=int(mask.sum()),max_components=max_components,
        active_threshold=.005,inactive_mass=float(w[~mask].sum()),
        settings=fitted.get_params(), warnings=[str(v.message) for v in caught],
        comparison_note='Raw X; RDT-style DP prior/threshold; max_iter raised from 100 to 1500.')
    return full,active,diagnostic


def parameter_table(model):
    return pd.DataFrame(dict(component=np.arange(1,model.k+1),weight=model.weights,
                             mu=model.means,a=model.scales,b=model.shapes,
                             sigma_derived=model.sigmas))



def run_experiment(x, *, config=None, truth=None, sample_seed=None,
                   vgm_max_components=10, threads=1) -> ExperimentResult:
    """Always fit a new GGMM from one X. No cache, files or truth in fit().

    The returned result owns a copy of X and its configuration. Replacing the
    notebook's result does not mutate earlier results kept by the caller.
    """
    cfg = config or GGMMBICConfig()
    observed = np.array(x, dtype=float, copy=True)
    if observed.ndim == 2 and observed.shape[1] == 1:
        observed = observed[:, 0].copy()
    if observed.ndim != 1:
        raise ValueError('X must be one continuous column.')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer.')
    if not isinstance(vgm_max_components, int) or vgm_max_components < 1:
        raise ValueError('vgm_max_components must be a positive integer.')
    reference = None if truth is None else Mixture(**truth.payload())
    created = datetime.now(timezone.utc).isoformat(timespec='microseconds')
    fitted = GGMMBICEstimator(cfg)
    with threadpool_limits(limits=threads):
        # Time the complete fit() call, including initialization and K selection.
        # Input copying, generation, evaluation and plotting are outside it.
        tick = perf_counter()
        fitted.fit(observed)
        fit_seconds = perf_counter()-tick
    observed.setflags(write=False)
    return ExperimentResult(
        run_id=created, x=observed, truth=reference, ggmm=fitted,
        config=asdict(cfg), fit_seconds=fit_seconds,
        selection=pd.DataFrame(fitted.selection_),
        starts=pd.DataFrame(fitted.starts_),
        parameters=parameter_table(fitted.density_),
        metadata=dict(sample_seed=sample_seed, sample_size=len(observed), threads=threads,
            vgm_max_components=vgm_max_components, vgm_seed=VGM_SEED,
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__),
            timing='New GGMM fit call; includes all K and BIC; excludes reports and I/O.'))


def require_result(result):
    if not isinstance(result, dict) or not {'x', 'ggmm', 'config', 'fit_seconds'} <= result.keys():
        raise RuntimeError('먼저 GGMM 적합 셀을 성공적으로 실행하세요.')
    if result['ggmm'].fit_status_ != 'complete':
        raise RuntimeError('완료된 GGMM 적합 결과가 아닙니다.')
    return result


def show_table(table):
    with pd.option_context('display.max_columns', None, 'display.max_rows', 100,
                           'display.float_format', '{:.6f}'.format):
        display(table)


def report_context(result):
    require_result(result)
    print(f"Run: {result['run_id']} | n={len(result['x']):,} | "
          f"K_max={result['config']['max_components']} | "
          f"선택 K={result['ggmm'].n_components_}")
    if result['truth'] is None:
        print('참 밀도가 없는 자료: IAE·ISE와 참 밀도 오차 그림은 계산하지 않습니다.')
    return result['x'], result['truth'], result['ggmm']


def plot_grid(result, models, padding: float = 0):
    x = result['x']
    knots = np.unique(np.concatenate([x, *[_points(m, 1e-6) for m in models if m is not None]]))
    lo, hi = min(knots.min(), x.min()-padding), max(knots.max(), x.max()+padding)
    return np.unique(np.r_[np.linspace(lo, hi, 5000), knots])


def show_density_errors(result, grid, curves, title):
    x, truth = result['x'], result['truth']
    fig = plt.figure(figsize=(12, 7 if truth is not None else 4), layout='constrained')
    if truth is not None:
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0, :])
        errors = [fig.add_subplot(gs[1, j]) for j in range(2)]
        true_pdf = truth.pdf(grid)
        ax.plot(grid, true_pdf, 'k:', lw=1.8, label='True density')
    else:
        ax, errors, true_pdf = fig.add_subplot(), [], None
    ax.hist(x, bins=85, density=True, color='#dce2e8', edgecolor='white',
            linewidth=.3, label=f'Sample (n={len(x):,})')
    for label, pdf, color, style in curves:
        ax.plot(grid, pdf, color=color, ls=style, lw=2, label=label)
        if true_pdf is not None:
            delta = pdf-true_pdf
            errors[0].plot(grid, np.abs(delta), color=color, ls=style)
            errors[1].plot(grid, delta**2, color=color, ls=style)
    ax.set(title=title, ylabel='Density')
    ax.legend(ncol=2, fontsize=9, frameon=False)
    for panel, name in zip(errors, ['Absolute error', 'Squared error']):
        panel.set(title=name, ylabel=name)
    for panel in [ax, *errors]:
        panel.set(xlabel='x', xlim=(grid[0], grid[-1]), ylim=(0, None))
        panel.spines[['top', 'right']].set_visible(False)
    plt.show()


def mixture_row(result, name, model, seconds, converged) -> dict[str, Any]:
    tick = perf_counter()
    with threadpool_limits(limits=result['metadata']['threads']):
        ll = float(model.logpdf(result['x']).sum())
        errors = (density_errors(result['truth'], model) if result['truth'] is not None
                  else dict(IAE=np.nan, ISE=np.nan, quadrature_tolerance_met=None))
    return dict(Method=name, K=model.k, LL=ll, IAE=errors['IAE'], ISE=errors['ISE'],
        Fit_s=seconds, Evaluation_s=perf_counter()-tick, Converged=converged,
        Integral_OK=errors['quadrature_tolerance_met'])


def compare_vgm(result, cap):
    # Refit on this run's X. No global VGM is inherited from another report.
    with threadpool_limits(limits=result['metadata']['threads']):
        return fit_vgm(result['x'], max_components=min(cap, len(result['x'])))


def make_kde(result, subset=False) -> tuple[gaussian_kde, dict[str, Any]]:
    x = result['x']
    indices = (np.random.default_rng(20260914).choice(len(x), 3, replace=False)
               if subset else np.arange(len(x)))
    if np.std(x[indices]) <= 0:
        raise ValueError('선택된 KDE 관측값이 모두 같습니다. 다른 표본에서는 다시 확인해야 합니다.')
    with threadpool_limits(limits=result['metadata']['threads']):
        tick = perf_counter()
        model = gaussian_kde(x[indices], bw_method='scott')
        fit_s = perf_counter()-tick
        tick = perf_counter()
        ll = float(model.logpdf(x).sum())
        ll_s = perf_counter()-tick
    return model, dict(indices=indices, fit_s=fit_s, ll_s=ll_s, ll=ll,
                      h=float(np.sqrt(model.covariance[0, 0])))


def adaptive_kde_error(result, kde):
    x, truth = result['x'], result['truth']
    if truth is None:
        return dict(IAE=np.nan, ISE=np.nan, Integral_OK=None)
    h = float(np.sqrt(kde.covariance[0, 0]))
    points = np.unique(np.r_[_points(truth, 1e-9), np.quantile(x, np.linspace(0, 1, 21)),
                             x.min()-8*h, x.max()+8*h])
    def integrand(t):
        delta = float(kde([t])[0]-truth.pdf([t])[0])
        return np.array([abs(delta), delta**2])
    with threadpool_limits(limits=result['metadata']['threads']):
        values, error, info = quad_vec(integrand, points[0], points[-1],
            points=points[1:-1], epsabs=1e-8, epsrel=0, norm='max', full_output=True)
    return dict(IAE=float(values[0]), ISE=float(values[1]),
                Integral_OK=bool(info.success and error <= 1e-8))


def report_overview(result):
    x, truth, fit = report_context(result)
    selected = fit.density_
    cap = result['metadata']['vgm_max_components']
    _, active, info = compare_vgm(result, cap)
    print('g(x; mu,a,b) = b / [2a Gamma(1/b)] * exp(-|(x-mu)/a|^b)')
    if truth is not None:
        print('참 분포:', truth.formula())
    print('추정 GGMM:', selected.formula())
    show_table(parameter_table(selected)[['component', 'weight', 'mu', 'a', 'b']])
    table = pd.DataFrame([
        mixture_row(result, 'GGMM + BIC', selected, result['fit_seconds'],
                    fit.diagnostics_['selected_converged']),
        mixture_row(result, 'VGM active', active, info['seconds'], info['converged'])])
    show_table(table)
    print(f"VGM: 최대 {info['max_components']}성분, weight>0.005, 실제 유효 K={active.k}.")
    print('Fit_s: GGMM은 이 result를 만든 전체 탐색 시간, VGM은 이 셀의 새 적합 시간.')
    print('평가·그림 시간은 적합시간에 포함하지 않습니다. LL↑, IAE·ISE↓.')
    grid = plot_grid(result, [truth, selected, active])
    show_density_errors(result, grid, [
        (f'GGMM K={selected.k}', selected.pdf(grid), '#c43c39', '-'),
        (f'VGM active K={active.k}', active.pdf(grid), '#2865b0', '--')], 'Selected GGMM and VGM')
    return table


def report_initialization(result):
    _, _, fit = report_context(result)
    records = {r['K']: r for r in fit.selection_}
    rows = []
    for k, rec in sorted(records.items()):
        starts = {r['start']: r for r in fit.starts_ if r['K'] == k}
        row = dict(K=k, Chosen_start=rec.get('start', 'none'))
        for method in ['quantile', 'lloyd']:
            start = starts.get(method, {})
            row[method+'_LL'] = start.get('log_likelihood', np.nan)
            row[method+'_status'] = ('converged' if start.get('converged') else
                                     'not converged' if start else 'duplicate / omitted')
        rows.append(row)
    table = pd.DataFrame(rows)
    show_table(table)
    show_table(pd.DataFrame([dict(Start=r['start'],
        Initial_b=np.round(r.get('initial_shapes', []), 5).tolist(),
        Chosen=r['start'] == records[fit.n_components_].get('start'),
        Optimize_s=r['optimization_seconds'], Start_s=r['seconds'],
        Iterations=r.get('iterations'), Nfev=r['nfev'])
        for r in fit.starts_ if r['K'] == fit.n_components_]))
    print('Initial_b는 최종 추정값이 아닌 출발점입니다. 미수렴·중복 생략을 함께 표시합니다.')
    fig, ax = plt.subplots(figsize=(10, 3.5), layout='constrained')
    for method in ['quantile', 'lloyd']:
        ax.plot(table['K'], table[method+'_LL'], 'o-', label=method)
    ax.set(xlabel='K', ylabel='Log likelihood after fitting', xticks=table['K'])
    ax.legend(frameon=False)
    plt.show()
    return table


def report_selection(result):
    _, _, fit = report_context(result)
    scores = pd.DataFrame(fit.selection_).sort_values('K').copy()
    scores['AIC'] = -2*scores['log_likelihood']+2*(4*scores['K']-1)
    scores['LL_gain'] = scores['log_likelihood'].diff()
    scores['delta_AIC'] = scores['AIC']-scores['AIC'].min()
    show_table(scores.reindex(columns=['K', 'log_likelihood', 'LL_gain', 'AIC', 'BIC',
        'selected', 'converged', 'max_b', 'a_floor_hits', 'seconds']))
    print('계산된 유한 후보의 선택:', {'LL 최대': int(scores.loc[scores.log_likelihood.idxmax(), 'K']),
        'AIC 최소': int(scores.loc[scores.AIC.idxmin(), 'K']), 'BIC 최소': fit.n_components_})
    print('미수렴 후보를 숨기지 않습니다. 현재 선택은 전역 최적해나 참 K 복원을 보장하지 않습니다.')
    boundary = search_boundary(fit.selection_)
    print('탐색 상한 진단:', boundary)
    if boundary['review_larger_K']:
        print('상한을 넓혀 재적합할 필요가 있습니다. 현재 추정기는 자동으로 상한을 늘리지 않습니다.')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    for ax, metric in zip(axes, ['log_likelihood', 'delta_AIC', 'delta_BIC']):
        ax.plot(scores.K, scores[metric], 'o-', label='Computed fit')
        bad = scores.loc[~scores.converged]
        ax.scatter(bad.K, bad[metric], marker='x', color='red', label='Not converged')
        ax.set(xlabel='K', ylabel=metric, xticks=scores.K)
        ax.legend(fontsize=8, frameon=False)
    plt.show()
    return scores


def search_boundary(selection):
    """Diagnostic only, not a guarantee that no better larger K exists."""
    rows = sorted(selection, key=lambda r: r['K'])
    finite = [r for r in rows if r['finite']]
    best = min(finite, key=lambda r: r['BIC'])
    last = rows[-1]
    # Check adjacent actual candidates, without jumping over a failed fit.
    tail = rows[-3:]
    descending = len(tail) >= 2 and all(r['finite'] for r in tail) and all(
        b['BIC'] < a['BIC'] for a, b in zip(tail, tail[1:]))
    at_boundary = best['K'] == last['K']
    return dict(effective_Kmax=last['K'], selected_at_boundary=at_boundary,
                last_candidates_BIC_decreasing=descending,
                boundary_not_converged=not last['converged'],
                review_larger_K=bool(at_boundary or descending))


def report_timing(result):
    _, _, fit = report_context(result)
    table = pd.DataFrame(fit.selection_).reindex(columns=['K', 'seconds', 'initialization_seconds',
        'starts_count', 'converged']).rename(columns={'seconds': 'Candidate_s'})
    table['Cumulative_candidate_s'] = table.Candidate_s.cumsum()
    starts = pd.DataFrame(fit.starts_)
    show_table(table)
    show_table(starts.reindex(columns=['K', 'start', 'seconds', 'optimization_seconds',
                                       'iterations', 'nfev', 'converged']))
    print(f"전체 GGMM fit() 실측: {result['fit_seconds']:.6f}초")
    print('Candidate_s는 해당 K의 모든 시작점·초기화·후처리를 포함합니다.')
    print('초기화는 K별로 공동 측정합니다. starts.seconds는 초기화 이후 시도 전체입니다.')
    print('누적 후보 시간은 현재 실행의 부분합이며, 다른 K_max 독립 실행의 실측값은 아닙니다.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout='constrained')
    axes[0].bar(table.K, table.Candidate_s, label='All starts + initialization')
    axes[0].bar(table.K, table.initialization_seconds, label='Initialization')
    axes[0].set(xlabel='K', ylabel='Seconds', xticks=table.K, title='Time per candidate')
    axes[0].legend(fontsize=8)
    axes[1].plot(table.K, table.Cumulative_candidate_s, 'o-')
    axes[1].set(xlabel='K', ylabel='Seconds', xticks=table.K, title='Cumulative candidate time')
    plt.show()
    return table


def report_kde(result):
    x, truth, fit = report_context(result)
    selected = fit.density_
    full, full_info = make_kde(result)
    try:
        three, three_info = make_kde(result, subset=True)
    except ValueError as exc:
        print('3개 관측값 KDE를 계산할 수 없습니다:', exc)
        return pd.DataFrame()
    margin = 5*max(full_info['h'], three_info['h'])
    grid = np.linspace(x.min()-margin, x.max()+margin, 4000)
    with threadpool_limits(limits=result['metadata']['threads']):
        curves = [(f'GGMM K={selected.k}', selected.pdf(grid), '#c43c39', '-'),
                  (f'Gaussian KDE n={len(x):,}', full(grid), '#26836b', '--'),
                  ('Gaussian KDE n=3', three(grid), '#365ab0', '-.')]
    specs = [(len(x), selected.k, np.nan, np.nan, result['fit_seconds'], np.nan,
              float(selected.logpdf(x).sum())),
             (len(x), full.n, float(cast(float, full.factor)), full_info['h'], full_info['fit_s'], full_info['ll_s'], full_info['ll']),
             (3, 3, float(cast(float, three.factor)), three_info['h'], three_info['fit_s'], three_info['ll_s'], three_info['ll'])]
    rows = []
    for (name, pdf, _, _), (n, terms, factor, h, fit_s, ll_s, ll) in zip(curves, specs):
        delta = pdf-truth.pdf(grid) if truth is not None else np.full_like(grid, np.nan)
        rows.append(dict(Method=name, Fit_n=n, Terms=terms, Factor=factor, h=h,
            LL_on_all_X=ll, IAE=float(trapezoid(abs(delta), grid)),
            ISE=float(trapezoid(delta**2, grid)), Fit_s=fit_s, LL_s=ll_s))
    table = pd.DataFrame(rows)
    show_table(table)
    print('3-kernel 추출 시드: 20260914 | 인덱스:', three_info['indices'].tolist())
    print('추출 관측값:', np.round(x[three_info['indices']], 6).tolist())
    print('세 모형의 LL은 전체 X에서 평가합니다. 대역폭: Scott. IAE·ISE: 4,000점 사다리꼴 근사.')
    visible = (grid >= min(x.min()-3*full_info['h'], x[three_info['indices']].min()-3*three_info['h'])) & (
               grid <= max(x.max()+3*full_info['h'], x[three_info['indices']].max()+3*three_info['h']))
    show_density_errors(result, grid[visible], [(n, p[visible], c, s) for n, p, c, s in curves],
                        'GGMM and Gaussian KDE')
    return table


def report_same_k(result):
    _, truth, fit = report_context(result)
    records = {r['K']: r for r in fit.selection_}
    rows, matches = [], []
    # Match GGMM to the observed active VGM K, never force VGM to a desired K.
    for cap in dict.fromkeys([3, result['metadata']['vgm_max_components']]):
        _, active, info = compare_vgm(result, cap)
        k = active.k
        if k not in fit.models_:
            print(f'VGM 상한={cap}, 실제 K={k}: 해당 GGMM 후보가 없어 같은 K 비교를 생략합니다.')
            continue
        gg = fit.models_[k]
        for name, model, seconds, converged in [
            ('GGMM', gg, records[k]['seconds'], records[k]['converged']),
            ('VGM active', active, info['seconds'], info['converged'])]:
            row = mixture_row(result, name, model, seconds, converged)
            row.update(VGM_cap=cap, Parameters=(4*k-1 if name == 'GGMM' else 3*k-1))
            rows.append(row)
        matches.append((cap, k, gg, active))
    table = pd.DataFrame(rows)
    show_table(table)
    print('VGM의 실제 유효 K와 같은 GGMM 후보를 비교합니다. K=3·5를 강제로 만들지 않습니다.')
    print('GGMM은 해당 K의 시간, VGM은 최대 성분 수 전체의 시간으로 비용 범위가 다릅니다.')
    for cap, k, gg, vg in matches:
        grid = plot_grid(result, [truth, gg, vg])
        show_density_errors(result, grid, [(f'GGMM K={k}', gg.pdf(grid), '#c43c39', '-'),
            (f'VGM active K={k}', vg.pdf(grid), '#2865b0', '--')], f'{k} vs {k} components (VGM cap={cap})')
    return table


def report_all_k(result):
    _, truth, fit = report_context(result)
    selected = fit.density_
    records = {r['K']: r for r in fit.selection_}
    full, info = make_kde(result)
    rows = [mixture_row(result, f'GGMM K={k}', model, records[k]['seconds'], records[k]['converged'])
            for k, model in sorted(fit.models_.items())]
    tick = perf_counter()
    err = adaptive_kde_error(result, full)
    rows.append(dict(Method='KDE full sample', K=np.nan, LL=info['ll'], Fit_s=info['fit_s'],
                     Evaluation_s=info['ll_s']+perf_counter()-tick, Converged='Scott rule', **err))
    table = pd.DataFrame(rows)
    show_table(table)
    print('참 밀도를 아는 경우에만 IAE·ISE를 사후 계산합니다. K 선택에는 사용하지 않습니다.')
    print('적분은 넓은 유한 구간의 수치 근사입니다. 실패 후보는 선택·시간 표에서 확인하세요.')
    ks = list(dict.fromkeys([selected.k, *[k for k in (3, 4, 5, 9) if k in fit.models_]]))
    grid = plot_grid(result, [truth, *[fit.models_[k] for k in ks]], padding=4*info['h'])
    with threadpool_limits(limits=result['metadata']['threads']):
        kde_pdf = full(grid)
    for k in ks:
        label = f'GGMM K={k}' + (' [not converged]' if not records[k]['converged'] else '')
        show_density_errors(result, grid, [(label, fit.models_[k].pdf(grid), '#c43c39', '-'),
            ('KDE full sample', kde_pdf, '#26836b', '--')], f'Candidate K={k} and KDE')
    return table


def compare_kmax(result, maxima=(10, 20)):
    """Independent fresh fits; do not relabel a prefix sum as measured runtime."""
    report_context(result)
    runs, rows = [], []
    for cap in maxima:
        print(f'새 적합: K_max={cap}', flush=True)
        run = run_experiment(result['x'], truth=result['truth'],
            config=replace(result['ggmm'].config, max_components=int(cap)),
            sample_seed=result['metadata']['sample_seed'],
            vgm_max_components=result['metadata']['vgm_max_components'],
            threads=result['metadata']['threads'])
        runs.append(run)
        row = mixture_row(run, f'K_max={cap}', run['ggmm'].density_, run['fit_seconds'],
                          run['ggmm'].diagnostics_['selected_converged'])
        row.update(K_max=cap, Run=run['run_id'],
                   BIC=float(run['selection'].loc[run['selection'].selected, 'BIC'].iloc[0]))
        rows.append(row)
    table = pd.DataFrame(rows)
    show_table(table)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for ax, metric in zip(axes, ['Fit_s', 'K']):
        ax.plot(table.K_max, table[metric], 'o-')
        ax.set(xlabel='Requested K_max', ylabel=metric, xticks=table.K_max)
    plt.show()
    print('각 점은 같은 X의 독립 재적합입니다. 단일 시간 측정으로 반복 변동까지 평가하지는 않습니다.')
    return runs, table


def compare_scale_floors(result, floors=(1e-4, 1e-3, 1e-2)):
    report_context(result)
    runs, rows = [], []
    for floor in floors:
        print(f'새 적합: scale_floor_relative={floor:g}', flush=True)
        run = run_experiment(result['x'], truth=result['truth'],
            config=replace(result['ggmm'].config, scale_floor_relative=float(floor)),
            sample_seed=result['metadata']['sample_seed'],
            vgm_max_components=result['metadata']['vgm_max_components'],
            threads=result['metadata']['threads'])
        runs.append(run)
        rows.append(dict(scale_floor_relative=floor, Run=run['run_id'],
            **mixture_row(run, 'GGMM', run['ggmm'].density_, run['fit_seconds'],
                          run['ggmm'].diagnostics_['selected_converged'])))
    table = pd.DataFrame(rows)
    show_table(table)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout='constrained')
    for ax, metric in zip(axes, ['K', 'IAE', 'Fit_s']):
        ax.semilogx(table.scale_floor_relative, table[metric], 'o-')
        ax.set(xlabel='Relative scale floor', ylabel=metric)
    plt.show()
    print('양의 하한을 바꾼 새 적합이며 하한을 완전히 제거한 실험은 아닙니다.')
    return runs, table


def report_restricted_and_high_k(result):
    """Reuse this run's actual candidates; no invented separate fit timing."""
    _, truth, fit = report_context(result)
    records = {r['K']: r for r in fit.selection_ if r['finite']}
    restricted = [r for r in records.values() if r['K'] <= 2]
    if not restricted:
        raise RuntimeError('K<=2의 유한 적합 후보가 없습니다.')
    limited_k = min(restricted, key=lambda r: r['BIC'])['K']
    highest_ll_k = max(records.values(), key=lambda r: r['log_likelihood'])['K']
    ks = list(dict.fromkeys([limited_k, fit.n_components_, highest_ll_k]))
    rows = []
    for k in ks:
        rec, model = records[k], fit.models_[k]
        row = mixture_row(result, f'GGMM K={k}', model, rec['seconds'], rec['converged'])
        row.update(BIC=rec['BIC'], Max_b=rec['max_b'], Floor_hits=rec['a_floor_hits'],
                   Iterations=rec['iterations'], Stop_reason=rec['message'])
        rows.append(row)
    table = pd.DataFrame(rows)
    show_table(table)
    print(f'후보를 K<=2로 제한한 BIC 선택: K={limited_k}')
    print(f'전체 후보의 BIC 선택 K={fit.n_components_}; 훈련 로그우도 최대 K={highest_ll_k}.')
    print('같은 실행의 후보를 제한한 진단입니다. K_max=2의 별도 실행시간을 측정한 것은 아닙니다.')
    print('참 성분 수 3을 알아서 선택한 것이 아닙니다. 참 밀도는 오차 평가에만 사용합니다.')
    print('LL 최대 후보의 파라미터:')
    show_table(parameter_table(fit.models_[highest_ll_k]))
    show_table(pd.DataFrame([r for r in fit.starts_ if r['K'] == highest_ll_k]).reindex(
        columns=['K', 'start', 'log_likelihood', 'converged', 'iterations',
                 'projected_gradient', 'max_b', 'a_floor_hits', 'message']))
    grid = plot_grid(result, [truth, *[fit.models_[k] for k in ks]])
    colors = ['#a16b1b', '#c43c39', '#365ab0']
    for k, color in zip(ks, colors):
        label = f'GGMM K={k}' + (' [not converged]' if not records[k]['converged'] else '')
        show_density_errors(result, grid, [(label, fit.models_[k].pdf(grid), color, '-')],
                            f'Candidate K={k}: fitted density and error')
    print('큰 K 자체를 실패 원인으로 단정하지 않습니다. 수렴·형상·오차를 함께 확인하세요.')
    return table


def scale_collapse_path(result):
    """Unconstrained mixture counterexample, not optimizer outputs or a fit.

    Add a b=2 GGD of fixed weight 1/n at X[0], and let its strictly positive
    scale tend to zero. Other components remain fixed. The component count
    is fixed throughout the path, so BIC's parameter-count penalty is fixed.
    """
    require_result(result)
    x, base = result['x'], result['ggmm'].density_
    base_log = base.logpdf(x)
    alpha = 1/len(x)
    unit = float(np.std(x))
    rows = []
    for exponent in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        log_a = np.log(unit)-exponent*np.log(10.)
        # b=2: density at its centre is 1/(a*sqrt(pi)).
        with np.errstate(divide='ignore', over='ignore'):
            power = np.exp(2*(np.log(np.abs(x-x[0]))-log_a))
        spike_log = -log_a-.5*np.log(np.pi)-power
        ll = float(np.logaddexp(np.log1p(-alpha)+base_log, np.log(alpha)+spike_log).sum())
        rows.append(dict(Neg_log10_relative_a=exponent, a=float(np.exp(log_a)),
            LL=ll, LL_gain_vs_base=ll-float(base_log.sum()),
            Log_weighted_peak=float(np.log(alpha)-log_a-.5*np.log(np.pi)),
            K=base.k+1, BIC=-2*ll+(4*(base.k+1)-1)*np.log(len(x))))
    return pd.DataFrame(rows)


def report_scale_collapse(result):
    report_context(result)
    table = scale_collapse_path(result)
    show_table(table)
    print('하한 없는 모형의 반례: 기존 밀도에 X[0] 중심, b=2, weight=1/n 성분 하나를 추가합니다.')
    print('모든 a는 양수입니다. a=0은 올바른 연속 밀도가 아닙니다.')
    print('최적화 결과가 아니라, 허용 파라미터 경로에서 계산한 실제 로그우도입니다.')
    print('K는 경로 내내 고정됩니다. a→0에서 한 관측점의 밀도가 무한대로 가므로 LL도 발산합니다.')
    print('나머지 점은 기존 성분이 맡습니다. 따라서 BIC의 고정된 파라미터 벌점도 이를 막지 못합니다.')
    print('현재 fit은 양의 하한을 요구합니다. 작은 양수로 바꾼 실험과 하한 제거는 다릅니다.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained')
    for ax, metric in zip(axes, ['LL_gain_vs_base', 'BIC']):
        ax.plot(table.Neg_log10_relative_a, table[metric], 'o-')
        ax.set(xlabel='-log10(a / sample SD)', ylabel=metric)
    plt.show()
    return table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fresh GGMM fit, no disk cache or automatic exports.')
    parser.add_argument('--kmax', type=int, default=10)
    parser.add_argument('--n', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--seed', type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    reference = Mixture(**TRUTH_PARAMETERS)
    sample = sample_mixture(reference, args.n, args.seed)
    result = run_experiment(sample, truth=reference, sample_seed=args.seed,
                            config=GGMMBICConfig(max_components=args.kmax))
    print(result['selection'].to_string(index=False))
    print(result['ggmm'].density_.formula())
    print(f"Total fit seconds: {result['fit_seconds']:.6f}")
