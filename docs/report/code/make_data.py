"""2단계: 모수를 알고 있는 GGD 5개 혼합에서 인공 데이터를 만든다."""
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import gennorm

# 참 모수 (나중에 이 값을 맞히는 것이 목표)
TRUE = {
    "weights": np.array([0.30, 0.25, 0.20, 0.15, 0.10]),
    "means":   np.array([-10.0, -4.0, 0.0, 5.0, 10.0]),
    "sigmas":  np.array([1.5, 1.0, 1.0, 0.8, 0.6]),   # 표준편차
    "betas":   np.array([0.8, 1.0, 2.0, 4.0, 8.0]),   # 형상 (작을수록 꼬리가 두꺼움)
}

RESULTS = Path(__file__).resolve().parent.parent / "results"


def alpha_from_sigma(sigma, beta):
    """표준편차 sigma를 GGD 스케일 alpha로 바꾼다: sigma = alpha * sqrt(G(3/b)/G(1/b))."""
    return sigma / np.exp(0.5 * (gammaln(3 / beta) - gammaln(1 / beta)))


def excess_kurtosis(beta):
    """GGD의 초과 첨도. Gaussian(beta=2)이면 0, 클수록 꼬리가 두꺼움."""
    return np.exp(gammaln(5 / beta) + gammaln(1 / beta) - 2 * gammaln(3 / beta)) - 3


def sample_mixture(n, seed, true=TRUE):
    """혼합분포에서 n개를 뽑는다. (값 x, 각 값이 나온 성분 번호 label)을 돌려준다."""
    rng = np.random.default_rng(seed)
    label = rng.choice(len(true["weights"]), size=n, p=true["weights"])
    alphas = alpha_from_sigma(true["sigmas"], true["betas"])
    x = np.empty(n)
    for k in range(len(true["weights"])):
        idx = label == k
        x[idx] = gennorm.rvs(true["betas"][k], loc=true["means"][k],
                             scale=alphas[k], size=idx.sum(), random_state=rng)
    return x, label


def true_pdf(x, true=TRUE):
    """참 혼합밀도 p(x) = sum_k w_k * GGD(x; mu_k, alpha_k, beta_k)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    alphas = alpha_from_sigma(true["sigmas"], true["betas"])
    comp = gennorm.pdf(x, true["betas"], loc=true["means"], scale=alphas)
    return comp @ true["weights"]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
    import matplotlib.pyplot as plt

    RESULTS.mkdir(exist_ok=True)
    x_train, y_train = sample_mixture(10_000, seed=0)
    x_test, y_test = sample_mixture(10_000, seed=1)
    np.savez(RESULTS / "synthetic.npz", x_train=x_train, y_train=y_train,
             x_test=x_test, y_test=y_test)

    print(f"{'k':>2} {'w':>5} {'mu':>6} {'sigma':>6} {'beta':>5} {'alpha':>6} "
          f"{'ex_kurt':>8} {'count':>6}")
    alphas = alpha_from_sigma(TRUE["sigmas"], TRUE["betas"])
    for k in range(5):
        print(f"{k:>2} {TRUE['weights'][k]:5.2f} {TRUE['means'][k]:6.1f} "
              f"{TRUE['sigmas'][k]:6.2f} {TRUE['betas'][k]:5.1f} {alphas[k]:6.3f} "
              f"{excess_kurtosis(TRUE['betas'][k]):8.2f} {(y_train == k).sum():6d}")
    print(f"train n={len(x_train)}: min={x_train.min():.2f}, max={x_train.max():.2f}")

    grid = np.linspace(x_train.min() - 1, x_train.max() + 1, 2000)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(x_train, bins=300, density=True, alpha=0.35, color="gray", label="sample (n=10,000)")
    ax.plot(grid, true_pdf(grid), "k", lw=1.5, label="true density")
    for k in range(5):
        one = {key: val[[k]] for key, val in TRUE.items()}
        one["weights"] = np.array([1.0])
        ax.plot(grid, TRUE["weights"][k] * true_pdf(grid, one), "--", lw=1,
                label=f"k={k}: beta={TRUE['betas'][k]:g}")
    ax.set_xlim(-22, 14)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "synthetic.png", dpi=130)
    print("saved:", RESULTS / "synthetic.npz", "/", RESULTS / "synthetic.png")
