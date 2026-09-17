"""Synthetic experiment and comparison for the BIC-selected GGMM estimator.

Truth is passed to generation and POST-FIT evaluation only. All reported
densities are normalized plug-in mixtures, including sklearn's VGM comparator.
"""
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import json
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
from scipy.stats import gennorm
from sklearn.mixture import BayesianGaussianMixture
from threadpoolctl import threadpool_limits
from ggmm_bic_estimator import Mixture, GGMMBICEstimator, GGMMBICConfig


TRUTH_PARAMETERS = dict(weights=[.2,.3,.5], means=[-5.,0.,6.],
                        scales=[1.2,1.6,2.], shapes=[1.5,4.,8.])
SAMPLE_SIZE = 5000
SAMPLE_SEED = 20260912
VGM_SEED = 27


def sample_mixture(truth, n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice(truth.k, size=n, p=truth.weights)
    return gennorm.rvs(truth.shapes[labels], loc=truth.means[labels],
                       scale=truth.scales[labels], random_state=rng)


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
    active = Mixture(w[mask]/w[mask].sum(),mu[mask],a[mask],np.full(mask.sum(),2.))
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


def make_figures(x, truth, fit, vgm, selection, out):
    grid = np.linspace(min(x.min(),_points(truth,1e-6).min()),
                       max(x.max(),_points(truth,1e-6).max()),3000)
    fig,axes = plt.subplots(1,2,figsize=(13,4.6),sharex=True,sharey=True)
    for ax,model,color,title in [(axes[0],fit,'#c43c39',f'GGMM: selected K = {fit.k}'),
                                  (axes[1],vgm,'#2865b0',f'VGM: active K = {vgm.k} (weight > 0.005)')]:
        ax.hist(x,bins=85,density=True,color='#dce2e8',edgecolor='white',linewidth=.3,label=f'Observed sample (n={len(x):,})')
        ax.plot(grid,truth.pdf(grid),color='#1c2530',lw=2,ls='--',label='True density (evaluation only)')
        ax.plot(grid,model.pdf(grid),color=color,lw=2.3,label='Fitted mixture density')
        for j in range(model.k):
            ax.plot(grid,model.weights[j]*np.exp(model.component_logpdf(grid)[:,j]),color=color,lw=.8,alpha=.4)
        ax.set(xlabel='x',ylabel='Density',title=title)
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    path=out/'density_comparison.png';fig.savefig(path,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].plot(selection.K,selection.BIC,'o-',color='#c43c39')
    selected=selection[selection.selected].iloc[0]
    axes[0].scatter([selected.K],[selected.BIC],s=130,color='#1c2530',zorder=5,
                    label=f'Selected K = {int(selected.K)}')
    axes[0].set(xlabel='Candidate K',ylabel='BIC (lower is better)',title='K selected from the same observed sample',xticks=selection.K)
    axes[0].legend()
    axes[1].plot(selection.K,selection.log_likelihood,'o-',color='#2865b0')
    axes[1].set(xlabel='Candidate K',ylabel='Log likelihood (higher is better)',
                title='Extra components must earn their BIC cost',xticks=selection.K)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    path2=out/'k_selection.png';fig.savefig(path2,dpi=170);plt.close(fig)
    return [path,path2]


def run_experiment(output_dir, run_sensitivity=True):
    out = Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    truth = Mixture(**TRUTH_PARAMETERS)
    x = sample_mixture(truth,SAMPLE_SIZE,SAMPLE_SEED)
    np.save(out/'observed_sample.npy',x)
    cfg=GGMMBICConfig()
    print('Fitting GGMM candidates K=1,...,10 using X only...',flush=True)
    with threadpool_limits(limits=1):
        ggmm=GGMMBICEstimator(cfg).fit(x)
        print(f'GGMM selected K={ggmm.n_components_}; {ggmm.diagnostics_["seconds"]:.2f} seconds',flush=True)
        full,active,vgm_diag=fit_vgm(x,cfg.max_components)
        rows=[]
        for name,model,seconds,converged in [
            ('GGMM + BIC',ggmm.density_,ggmm.diagnostics_['seconds'],ggmm.diagnostics_['selected_converged']),
            ('VGM active mixture',active,vgm_diag['seconds'],vgm_diag['converged']),
            ('VGM full mixture',full,vgm_diag['seconds'],vgm_diag['converged'])]:
            rows.append(dict(model=name,K=model.k,**density_errors(truth,model),fit_seconds=seconds,
                             converged=converged,sample_log_likelihood=float(model.logpdf(x).sum())))
        comparison=pd.DataFrame(rows)
        selection=pd.DataFrame(ggmm.selection_)
        parameters=parameter_table(ggmm.density_)
        comparison.to_csv(out/'comparison.csv',index=False)
        selection.to_csv(out/'k_selection.csv',index=False)
        parameters.to_csv(out/'estimated_parameters.csv',index=False)
        print(comparison[['model','K','IAE','ISE','fit_seconds','converged']].to_string(index=False),flush=True)
        figures=make_figures(x,truth,ggmm.density_,active,selection,out)
        failed=selection.loc[~selection.converged,'K'].tolist()
        if failed:
            print(f'Unconverged candidate K values: {failed}. BIC selection remains provisional.',flush=True)
        sensitivity=[]
        if run_sensitivity:
            for floor in [1e-4,1e-3,1e-2]:
                print(f'Same-sample scale-floor sensitivity: {floor:g}',flush=True)
                fitted=ggmm if floor == cfg.scale_floor_relative else GGMMBICEstimator(replace(cfg,scale_floor_relative=floor)).fit(x)
                sr=pd.DataFrame(fitted.selection_)
                sr.to_csv(out/f'k_selection_floor_{floor:g}.csv',index=False)
                sensitivity.append(dict(scale_floor_relative=floor,selected_K=fitted.n_components_,
                    selected_BIC=float(sr.loc[sr.selected,'BIC'].iloc[0]),
                    nearest_competitor_delta_BIC=float(sr.loc[~sr.selected,'delta_BIC'].min()),
                    a_floor_hits=int(sr.loc[sr.selected,'a_floor_hits'].iloc[0]),
                    selected_converged=fitted.diagnostics_['selected_converged'],
                    **density_errors(truth,fitted.density_),fit_seconds=fitted.diagnostics_['seconds']))
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(out/'scale_floor_sensitivity.csv',index=False)
    formula='g(x; mu,a,b) = b/(2*a*Gamma(1/b)) * exp(-abs((x-mu)/a)**b)\n\n'
    formula+='True density:\n'+truth.formula(9)+'\n\nEstimated GGMM:\n'+ggmm.density_.formula(9)+'\n'
    (out/'estimated_formula.txt').write_text(formula,encoding='utf-8')
    models=dict(truth=truth.payload(),ggmm=ggmm.density_.payload(),vgm_full=full.payload(),vgm_active=active.payload())
    (out/'models.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),sample_size=SAMPLE_SIZE,
        sample_seed=SAMPLE_SEED,vgm_seed=VGM_SEED,truth=TRUTH_PARAMETERS,
        sample_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ggmm_config=asdict(cfg),
        ggmm_diagnostics=ggmm.diagnostics_,vgm_diagnostics=vgm_diag,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      sklearn=sklearn.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        fitting_contract='One X; no truth, labels, split, or validation input. No fitting RNG in GGMM.',
        code_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ['ggmm_bic_estimator.py','ggmm_bic_experiment.py']},
        limitations=['One predeclared example, not a distribution-wide performance claim.',
            'BIC uses local numerical fits in a nonregular constrained mixture family.',
            'GGMM+BIC and VGM differ in both distribution and estimation/selection.',
            'No CTGAN training or generated-table quality evaluation.'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    (out/'initialization_log.json').write_text(json.dumps(ggmm.starts_,indent=2),encoding='utf-8')
    return dict(ggmm=ggmm,truth=truth,x=x,comparison=comparison,selection=selection,
                parameters=parameters,sensitivity=sensitivity,figures=figures,metadata=metadata)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='results/ggmm_bic_experiment')
    parser.add_argument('--skip-sensitivity',action='store_true')
    args=parser.parse_args()
    result=run_experiment(args.output_dir,not args.skip_sensitivity)
    print(result['selection'][['K','BIC','delta_BIC','converged','a_floor_hits','max_b']].to_string(index=False))
    print(result['ggmm'].density_.formula())
