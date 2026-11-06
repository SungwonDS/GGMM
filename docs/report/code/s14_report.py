"""14단계: 교수님 보고용 BGM vs VGGMM(병합/삭제/재정렬 이동) 종합 비교.

산출물 (results/report/)
- synth_runs.csv, real_runs.csv, kmax_runs.csv : 실행 기록
- fig_synth_overlay.png : 인공 분포 8종, 참 밀도 / BGM / VGGMM 겹친 그림 (n=5000)
- fig_synth_kl.png      : 표본 수에 따른 KL (참 분포와의 거리)
- fig_real_overlay.png  : 실데이터 열별 히스토그램 / BGM / VGGMM
- fig_kmax.png          : K_max에 따른 유효 성분 수, 평가 로그우도, 학습 시간
실행:  python experiments\\s14_report.py            (전체, 약 1시간)
       python experiments\\s14_report.py --quick    (n=5000, 시드 1개, K_max 최대 30)
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.datasets import load_breast_cancer
from sklearn.mixture import BayesianGaussianMixture

from ctgan.variational_ggmm import VariationalGGMM
from make_data import alpha_from_sigma

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
SIZES = [5000] if QUICK else [1000, 5000, 20000]
SEEDS = [0] if QUICK else [0, 1, 2]
KMAX_LIST = [5, 10, 20, 30] if QUICK else [5, 10, 20, 30, 50]


# ---------------- 참 분포 정의 ----------------
class Mixture:
    def __init__(self, weights, comps):
        self.w, self.comps = np.asarray(weights, float), comps

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        lab = rng.choice(len(self.w), size=n, p=self.w)
        x = np.empty(n)
        for k, c in enumerate(self.comps):
            idx = lab == k
            x[idx] = c.rvs(size=idx.sum(), random_state=rng)
        return x

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.w, self.comps))


def ggd(mu, sd, b):
    return stats.gennorm(b, loc=mu, scale=alpha_from_sigma(sd, b))


SCENARIOS = {
    "Gaussian mix (b=2)":      Mixture([.35, .40, .25], [ggd(-4, 1, 2), ggd(0, .8, 2), ggd(5, 1.2, 2)]),
    "Heavy tails (b=0.7-1.2)": Mixture([.40, .35, .25], [ggd(-6, 1.2, .7), ggd(0, 1, 1), ggd(6, 1, 1.2)]),
    "Flat tops (b=4-8)":       Mixture([.30, .40, .30], [ggd(-5, 1, 4), ggd(0, 1, 6), ggd(5, .8, 8)]),
    "Mixed 5 comps":           Mixture([.30, .25, .20, .15, .10],
                                       [ggd(-10, 1.5, .8), ggd(-4, 1, 1), ggd(0, 1, 2),
                                        ggd(5, .8, 4), ggd(10, .6, 8)]),
    "Overlapping":             Mixture([.30, .40, .30], [ggd(-2, 1, 1), ggd(0, .8, 2), ggd(2.5, 1, 4)]),
    "Student-t mix (df=3)*":   Mixture([.5, .5], [stats.t(3, loc=-4, scale=1), stats.t(3, loc=4, scale=1)]),
    "Log-normal (skewed)*":    Mixture([1.0], [stats.lognorm(0.6, scale=np.e)]),
    "Uniform blocks*":         Mixture([.4, .6], [stats.uniform(-3, 2), stats.uniform(0, 4)]),
}  # * = GGD 혼합 밖의 분포 (두 모형 모두 틀린 모형)


# ---------------- 모형 ----------------
def fit(method, x, seed, K=10):
    start = time.perf_counter()
    if method == "BGM":  # rdt ClusterBasedNormalizer와 같은 설정
        m = BayesianGaussianMixture(n_components=K, weight_concentration_prior_type="dirichlet_process",
                                    weight_concentration_prior=0.001, random_state=seed)
    else:
        m = VariationalGGMM(n_components=K, random_state=seed)
    m.fit(x.reshape(-1, 1))
    return m, time.perf_counter() - start


def logpdf(m, x):
    if hasattr(m, "plugin_logpdf"):
        return m.plugin_logpdf(x)
    sd = np.sqrt(m.covariances_.reshape(-1))
    comp = stats.norm.logpdf(np.asarray(x).reshape(-1, 1), m.means_.reshape(-1), sd)
    return logsumexp(comp + np.log(m.weights_), axis=1)


def active(m):
    return int((m.weights_ > 0.005).sum())


# ---------------- 1. 인공 분포 ----------------
def run_synthetic():
    rows, fitted = [], {}
    for name, truth in SCENARIOS.items():
        x_test = truth.sample(20_000, seed=999)
        true_ll = np.log(truth.pdf(x_test)).mean()
        lo, hi = np.quantile(x_test, [0.0005, 0.9995])
        grid = np.linspace(lo - 1, hi + 1, 4001)
        p_true = truth.pdf(grid)
        for n in SIZES:
            for seed in SEEDS:
                x = truth.sample(n, seed=seed)
                for method in ("BGM", "VGGMM"):
                    m, sec = fit(method, x, seed)
                    rows.append({"scenario": name, "n": n, "seed": seed, "method": method,
                                 "KL": true_ll - logpdf(m, x_test).mean(),
                                 "IAE": np.trapezoid(np.abs(np.exp(logpdf(m, grid)) - p_true), grid),
                                 "K": active(m), "fit_s": sec})
                    if n == 5000 and seed == 0:
                        fitted[(name, method)] = m
                print(f"[synth] {name:24s} n={n:5d} seed={seed}  KL BGM={rows[-2]['KL']:.4f} "
                      f"VGGMM={rows[-1]['KL']:.4f}  K {rows[-2]['K']}/{rows[-1]['K']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synth_runs.csv", index=False)

    # 겹친 그림 (n=5000, seed 0)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (name, truth) in zip(axes.ravel(), SCENARIOS.items()):
        x = truth.sample(5000, seed=0)
        lo, hi = np.quantile(x, [0.001, 0.999])
        grid = np.linspace(lo - 0.5, hi + 0.5, 1500)
        ax.hist(x, bins=120, range=(lo - .5, hi + .5), density=True, color="0.85")
        ax.plot(grid, truth.pdf(grid), "k", lw=2, label="true")
        for method, style in (("BGM", dict(color="tab:blue", ls="--")), ("VGGMM", dict(color="tab:red"))):
            m = fitted[(name, method)]
            kl = df[(df.scenario == name) & (df.n == 5000) & (df.seed == 0) & (df.method == method)].KL.iloc[0]
            ax.plot(grid, np.exp(logpdf(m, grid)), lw=1.4, **style,
                    label=f"{method} (K={active(m)}, KL={kl:.4f})")
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Synthetic distributions, n=5000: true density vs BGM vs VGGMM", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_synth_overlay.png", dpi=130)

    # KL vs n
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    for ax, name in zip(axes.ravel(), SCENARIOS):
        for method, color in (("BGM", "tab:blue"), ("VGGMM", "tab:red")):
            g = df[(df.scenario == name) & (df.method == method)].groupby("n").KL
            ax.errorbar(g.mean().index, g.mean().clip(lower=1e-5), yerr=g.std().fillna(0),
                        marker="o", color=color, label=method, capsize=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(name, fontsize=11)
        ax.set_xlabel("n"); ax.set_ylabel("KL (lower is better)"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_synth_kl.png", dpi=130)
    return df


# ---------------- 2. 실데이터 ----------------
def real_columns():
    cols = {}
    adult = pd.read_csv(ROOT / "CTGAN-main" / "examples" / "csv" / "adult.csv")
    for c in ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]:
        cols[f"adult:{c}"] = adult[c].to_numpy(float)
    br = pd.read_csv(ROOT / "CTGAN-main" / "examples" / "tsv" / "br2000.dat", sep="\t", header=None)
    for i in (2, 4, 10, 12):  # br2000.meta에서 C(연속)로 표시된 열
        cols[f"br2000:col{i}"] = br[i].to_numpy(float)
    bc = load_breast_cancer(as_frame=True).frame
    for c in ["mean radius", "mean area", "mean concavity", "worst area"]:
        cols[f"breast_cancer:{c}"] = bc[c].to_numpy(float)
    return cols


def run_real():
    rows, fitted = [], {}
    cols = real_columns()
    for name, x_all in cols.items():
        integer = bool(np.all(x_all == np.round(x_all)))
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(x_all))
            half = len(x_all) // 2
            x_tr, x_te = x_all[perm[:half]].copy(), x_all[perm[half:]].copy()
            if integer:  # 정수 열: 두 방법에 같은 디퀀타이즈
                x_tr += rng.uniform(-.5, .5, len(x_tr)); x_te += rng.uniform(-.5, .5, len(x_te))
            for method in ("BGM", "VGGMM"):
                m, sec = fit(method, x_tr, seed)
                rows.append({"column": name, "n_train": half, "integer": integer, "seed": seed,
                             "method": method, "test_ll": logpdf(m, x_te).mean(),
                             "K": active(m), "fit_s": sec})
                if seed == 0:
                    fitted[(name, method)] = (m, x_tr)
            print(f"[real] {name:30s} seed={seed}  ll BGM={rows[-2]['test_ll']:.4f} "
                  f"VGGMM={rows[-1]['test_ll']:.4f}  K {rows[-2]['K']}/{rows[-1]['K']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_runs.csv", index=False)

    names = list(cols)
    fig, axes = plt.subplots(3, 5, figsize=(22, 11))
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        x = fitted[(name, "BGM")][1]
        lo, hi = np.quantile(x, [0.002, 0.998])
        grid = np.linspace(lo, hi, 1500)
        ax.hist(x, bins=100, range=(lo, hi), density=True, color="0.85")
        for method, style in (("BGM", dict(color="tab:blue", ls="--")), ("VGGMM", dict(color="tab:red"))):
            m = fitted[(name, method)][0]
            ll = df[(df.column == name) & (df.seed == 0) & (df.method == method)].test_ll.iloc[0]
            ax.plot(grid, np.exp(logpdf(m, grid)), lw=1.3, **style,
                    label=f"{method} (K={active(m)}, ll={ll:.3f})")
        if "capital" in name:  # 0이 90% 넘게 몰린 열: 로그 눈금으로 꼬리를 보이게
            ax.set_yscale("log"); ax.set_ylim(bottom=1e-7)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7.5)
    fig.suptitle("Real columns (train half, seed 0): histogram vs BGM vs VGGMM", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_overlay.png", dpi=120)
    return df


# ---------------- 3. K_max ----------------
def run_kmax():
    cols = real_columns()
    data = {"Mixed 5 comps (synthetic)": SCENARIOS["Mixed 5 comps"].sample(10_000, seed=0),
            "Gaussian mix (synthetic)": SCENARIOS["Gaussian mix (b=2)"].sample(10_000, seed=0),
            "adult:age": cols["adult:age"], "adult:hours-per-week": cols["adult:hours-per-week"]}
    synth = {"Mixed 5 comps (synthetic)": "Mixed 5 comps", "Gaussian mix (synthetic)": "Gaussian mix (b=2)"}
    rows = []
    for name, x_all in data.items():
        rng = np.random.default_rng(0)
        if name in synth:  # 인공: 같은 참 분포에서 뽑은 평가 데이터
            x_tr, x_te = x_all, SCENARIOS[synth[name]].sample(20_000, seed=999)
        else:              # adult: 50/50 분할 + 디퀀타이즈
            perm = rng.permutation(len(x_all)); half = len(x_all) // 2
            x_tr = x_all[perm[:half]] + rng.uniform(-.5, .5, half)
            x_te = x_all[perm[half:]] + rng.uniform(-.5, .5, len(x_all) - half)
        for K in KMAX_LIST:
            for method in ("BGM", "VGGMM"):
                m, sec = fit(method, x_tr, 0, K=K)
                rows.append({"data": name, "K_max": K, "method": method, "K": active(m),
                             "test_ll": logpdf(m, x_te).mean(), "fit_s": sec})
                print(f"[kmax] {name:28s} K_max={K:2d} {method:5s} K={rows[-1]['K']:2d} "
                      f"ll={rows[-1]['test_ll']:.4f} ({sec:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kmax_runs.csv", index=False)

    fig, axes = plt.subplots(3, len(data), figsize=(4.6 * len(data), 11))
    for c, name in enumerate(data):
        for method, color in (("BGM", "tab:blue"), ("VGGMM", "tab:red")):
            d = df[(df.data == name) & (df.method == method)]
            axes[0, c].plot(d.K_max, d.K, "o-", color=color, label=method)
            axes[1, c].plot(d.K_max, d.test_ll, "o-", color=color, label=method)
            axes[2, c].plot(d.K_max, d.fit_s, "o-", color=color, label=method)
        axes[0, c].plot(KMAX_LIST, KMAX_LIST, ":", color="0.5", label="K = K_max")
        axes[0, c].set_title(name, fontsize=11)
        axes[0, c].set_ylabel("active K"); axes[1, c].set_ylabel("test log-lik (higher better)")
        axes[2, c].set_ylabel("fit time (s)"); axes[2, c].set_yscale("log")
        for r in range(3):
            axes[r, c].set_xlabel("K_max"); axes[r, c].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kmax.png", dpi=120)
    return df


if __name__ == "__main__":
    parts = [a for a in sys.argv[1:] if not a.startswith("--")] or ["synth", "real", "kmax"]
    if "synth" in parts:
        run_synthetic()
    if "real" in parts:
        run_real()
    if "kmax" in parts:
        run_kmax()
    print("saved to", OUT)
