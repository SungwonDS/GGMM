"""Variational GGMM: sklearn BayesianGaussianMixture의 틀에 GGD 성분을 넣은 모델.

- 가중치: stick-breaking(디리클레 과정) -> sklearn 코드 그대로 사용
- 폭: tau = alpha**(-beta), q(tau) = Gamma(a, b)  -> 켤레, 공식으로 갱신
- 중심 mu, 형상 beta: 점추정 (1차원 탐색)
- 입력은 1차원. 내부에서 z = (x - center) / unit 으로 표준화해서 계산한다.
"""
from numbers import Real

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, digamma, gammaln, gammaincinv, logsumexp
from scipy.stats import gennorm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval
from threadpoolctl import threadpool_limits

LOG2 = np.log(2.0)
ACTIVE = 0.005  # 유효 성분 문턱 (rdt weight_threshold와 같음)


def _ggd_norm(beta):
    """GGD 로그밀도의 상수 부분: ln(beta) - ln 2 - lnGamma(1/beta)."""
    return np.log(beta) - LOG2 - gammaln(1.0 / beta)


def _sd_ratio(beta):
    """sigma / alpha = sqrt(Gamma(3/beta) / Gamma(1/beta))."""
    return np.exp(0.5 * (gammaln(3.0 / beta) - gammaln(1.0 / beta)))


def _weighted_median(z, w):
    order = np.argsort(z)
    cw = np.cumsum(w[order])
    return z[order][np.searchsorted(cw, 0.5 * cw[-1])]


class VariationalGGMM(BayesianGaussianMixture):
    """1차원 변분 GGMM. 학습 후 weights_, means_, covariances_, shapes_, scales_ 제공."""

    _parameter_constraints: dict = {
        **BayesianGaussianMixture._parameter_constraints,
        "shape_init": [Interval(Real, 0.0, None, closed="neither")],
        "estimate_shape": ["boolean"],
        "shape_bounds": [tuple],
        "scale_prior_shape": [Interval(Real, 0.0, None, closed="neither")],
        "scale_prior_rate": [None, Interval(Real, 0.0, None, closed="neither")],
        "scale_prior_guess": [Interval(Real, 0.0, None, closed="neither")],
        "min_count_for_shape": [Interval(Real, 0.0, None, closed="left")],
        "moves": ["boolean"],
        "move_iter": [Interval(Real, 1, None, closed="left")],
        "max_move_rounds": [Interval(Real, 0, None, closed="left")],
        "max_candidates": [Interval(Real, 1, None, closed="left")],
        "move_gain": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(self, *, n_components=10, tol=1e-3, max_iter=100, n_init=1,
                 init_params="kmeans", weight_concentration_prior=1e-3,
                 shape_init=2.0, estimate_shape=True, shape_bounds=(0.3, 10.0),
                 scale_prior_shape=0.5, scale_prior_rate=None, scale_prior_guess=0.1,
                 min_count_for_shape=5.0, moves=True, move_iter=20, max_move_rounds=50,
                 max_candidates=3, move_gain=1e-4,
                 random_state=None, warm_start=False, verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, covariance_type="full", tol=tol, reg_covar=0.0,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=weight_concentration_prior,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.shape_init = shape_init
        self.estimate_shape = estimate_shape
        self.shape_bounds = shape_bounds
        self.scale_prior_shape = scale_prior_shape
        self.scale_prior_rate = scale_prior_rate
        self.scale_prior_guess = scale_prior_guess
        self.min_count_for_shape = min_count_for_shape
        self.moves = moves
        self.move_iter = move_iter
        self.max_move_rounds = max_move_rounds
        self.max_candidates = max_candidates
        self.move_gain = move_gain

    # ---------- 준비: 입력 확인, 표준화, 사전값 ----------
    def _check_parameters(self, X, xp=None):
        if X.shape[1] != 1:
            raise ValueError("VariationalGGMM은 1차원 데이터(열 1개)만 받습니다.")
        self._check_weights_parameters()  # sklearn: weight_concentration_prior_ 설정
        x = X[:, 0]
        self.center_ = float(x.mean())
        self.unit_ = float(x.std()) or 1.0
        self.a0_ = float(self.scale_prior_shape)

    def _b0(self, beta):
        """tau의 사전 rate. 기본: a0 * c**beta (alpha의 사전 추측값 c, 표준화 단위).
        scale_prior_rate를 주면 beta와 무관한 상수 (sklearn 비교용)."""
        if self.scale_prior_rate is not None:
            return self.scale_prior_rate
        return self.a0_ * self.scale_prior_guess ** beta

    def _z(self, X):
        return (X[:, 0] - self.center_) / self.unit_

    # ---------- 성분 하나의 목적함수 F_k와 갱신 ----------
    def _F(self, n, s, beta):
        """F_k = N[ln b - ln2 - lnG(1/b)] + a0 ln b0 - lnG(a0) + lnG(a) - a ln(b0 + S)."""
        a, b0 = self.a0_ + n / beta, self._b0(beta)
        return (n * _ggd_norm(beta) + self.a0_ * np.log(b0) - gammaln(self.a0_)
                + gammaln(a) - a * np.log(b0 + s))

    @staticmethod
    def _S(z, r, mu, beta):
        return float(np.dot(r, np.abs(z - mu) ** beta))

    def _update_mu(self, z, r, n, mu, beta):
        """beta 고정에서 S(mu) = sum r|z-mu|^beta 를 줄이는 mu. 줄어들 때만 바꾼다."""
        if beta == 2.0:
            return float(np.dot(r, z) / n)  # 가중평균 (sklearn과 같음)
        best, best_s = mu, self._S(z, r, mu, beta)
        mean = float(np.dot(r, z) / n)
        cands = (mean, float(_weighted_median(z, r))) if beta < 1.5 else (mean,)
        for cand in cands:  # 중앙값(정렬 필요)은 꼬리가 두꺼울 때만 후보로
            s = self._S(z, r, cand, beta)
            if s < best_s:
                best, best_s = cand, s
        spread = np.sqrt(max(float(np.dot(r, (z - mean) ** 2) / n), 1e-12))
        res = minimize_scalar(lambda m: self._S(z, r, m, beta), method="bounded",
                              bounds=(best - spread, best + spread),
                              options={"xatol": 1e-4 * spread})
        return float(res.x) if res.fun < best_s else best

    def _update_beta(self, z, r, n, mu, beta):
        """mu 고정에서 F_k를 키우는 beta (ln beta 공간에서 bounded Brent).
        속도: ln|z-mu|를 한 번만 계산해 |z-mu|^b = exp(b ln|z-mu|)로 쓰고,
        탐색 구간은 현재 beta의 1/2배 ~ 2배로 좁힌다 (매 반복 조금씩 움직임)."""
        d = np.abs(z - mu)
        pos = d > 0
        logd, rp = np.log(d[pos]), r[pos]
        S = lambda b: float(np.dot(rp, np.exp(b * logd)))
        lo = max(np.log(self.shape_bounds[0]), np.log(beta) - LOG2)
        hi = min(np.log(self.shape_bounds[1]), np.log(beta) + LOG2)
        f = lambda lb: -self._F(n, S(np.exp(lb)), np.exp(lb))
        res = minimize_scalar(f, method="bounded", bounds=(lo, hi), options={"xatol": 1e-3})
        old = -f(np.log(beta))
        return float(np.exp(res.x)) if -res.fun > old else beta

    def _update_components(self, X, resp, update_shape):
        z = self._z(X)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # sklearn과 같은 보정
        self._estimate_weights(nk)  # sklearn: Beta(1+N_k, gamma+sum_{j>k} N_j)
        K = self.n_components
        mu, beta = self.mu_z_.copy(), self.shapes_.copy()
        s, F = np.empty(K), np.empty(K)
        for k in range(K):
            keep = resp[:, k] > 1e-12  # 책임도가 사실상 0인 점은 계산에서 뺌 (속도)
            if not keep.any():
                s[k], F[k] = 0.0, self._F(nk[k], 0.0, beta[k])
                continue
            zk, r = z[keep], resp[keep, k]
            if update_shape and nk[k] >= self.min_count_for_shape:
                mu[k] = self._update_mu(zk, r, nk[k], mu[k], beta[k])  # mu -> beta -> mu
                beta[k] = self._update_beta(zk, r, nk[k], mu[k], beta[k])
            mu[k] = self._update_mu(zk, r, nk[k], mu[k], beta[k])
            s[k] = self._S(zk, r, mu[k], beta[k])
            F[k] = self._F(nk[k], s[k], beta[k])
        self.mu_z_, self.shapes_ = mu, beta
        self.gamma_a_ = self.a0_ + nk / beta      # q(tau) = Gamma(a, b)
        self.gamma_b_ = self._b0(beta) + s
        self._F_k = F
        self._set_parameters(self._get_parameters())

    def _initialize(self, X, resp):
        self.mu_z_ = np.zeros(self.n_components)
        self.shapes_ = np.full(self.n_components, float(self.shape_init))
        self._update_components(X, resp, update_shape=False)

    def _m_step(self, X, log_resp, xp=None):
        self._update_components(X, np.exp(log_resp), update_shape=self.estimate_shape)

    # ---------- E-step 점수와 ELBO ----------
    def _estimate_log_prob(self, X, xp=None):
        """E_q[ln p(x | k)] (원래 단위). 가중치 항 E[ln pi]는 sklearn이 더해 준다."""
        z = self._z(X)[:, None]
        a, b, beta = self.gamma_a_, self.gamma_b_, self.shapes_
        return (_ggd_norm(beta) + (digamma(a) - np.log(b)) / beta
                - (a / b) * np.abs(z - self.mu_z_) ** beta - np.log(self.unit_))

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """M-step 직후의 ELBO = sum F_k - sum r ln r + sum[lnB(g1,g2) - lnB(1,gamma)] - N ln(unit)."""
        g1, g2 = self.weight_concentration_
        n = log_resp.shape[0]
        entropy = np.sum(np.exp(log_resp) * log_resp)
        sticks = np.sum(betaln(g1, g2)) - len(g1) * betaln(1.0, self.weight_concentration_prior_)
        return float(np.sum(self._F_k) - entropy + sticks - n * np.log(self.unit_))

    # ---------- 모수 저장/복원 (n_init 중 최고 선택에 사용) ----------
    def _get_parameters(self):
        return (self.weight_concentration_, self.mu_z_.copy(), self.shapes_.copy(),
                self.gamma_a_.copy(), self.gamma_b_.copy(), self._F_k.copy())

    def _set_parameters(self, params, xp=None):
        (self.weight_concentration_, self.mu_z_, self.shapes_,
         self.gamma_a_, self.gamma_b_, self._F_k) = params
        g1, g2 = self.weight_concentration_
        w = g1 / (g1 + g2) * np.hstack((1.0, np.cumprod((g2 / (g1 + g2))[:-1])))
        self.weights_ = w / w.sum()  # sklearn과 같은 방식
        # 대표값: 정밀도 사후평균 대입 alpha = (b/a)^(1/beta)  (sklearn covariances_ 규칙)
        alpha_z = (self.gamma_b_ / self.gamma_a_) ** (1.0 / self.shapes_)
        self.scales_ = alpha_z * self.unit_
        self.means_ = (self.center_ + self.unit_ * self.mu_z_).reshape(-1, 1)
        sigma = self.scales_ * _sd_ratio(self.shapes_)
        self.covariances_ = (sigma ** 2).reshape(-1, 1, 1)
        self.precisions_ = 1.0 / self.covariances_
        self.precisions_cholesky_ = np.sqrt(self.precisions_)

    # ---------- 전처리와 진단에 쓰는 함수 ----------
    def half_widths(self, eps=1e-4):
        """성분별로 (1-eps) 확률이 들어가는 반폭 h_k (원래 단위). beta=2면 약 3.89 sigma."""
        b = self.shapes_
        return self.scales_ * gammaincinv(1.0 / b, 1.0 - eps) ** (1.0 / b)

    def plugin_logpdf(self, X):
        """대입 밀도 ln sum_k w_k GGD(x; mu_k, alpha_k, beta_k)."""
        x = np.asarray(X, dtype=float).reshape(-1, 1)
        comp = gennorm.logpdf(x, self.shapes_, loc=self.means_[:, 0], scale=self.scales_)
        with np.errstate(divide="ignore"):
            return np.logaddexp.reduce(comp + np.log(self.weights_), axis=1)

    def sample(self, n_samples=1):
        rng = np.random.default_rng(self.random_state)
        label = rng.choice(self.n_components, size=n_samples, p=self.weights_)
        x = gennorm.rvs(self.shapes_[label], loc=self.means_[label, 0],
                        scale=self.scales_[label], random_state=rng)
        return x.reshape(-1, 1), label

    # ---------- 6단계: 학습 중 성분 정리 이동 (ELBO가 오를 때만 받아들임) ----------
    def fit_predict(self, X, y=None):
        # 작은 배열 연산이 많아서 BLAS 스레드 1개가 더 빠르다 (속도 개선)
        with threadpool_limits(limits=1):
            return self._fit_predict(X, y)

    def _fit_predict(self, X, y=None):
        labels = super().fit_predict(X, y)  # 1) sklearn 방식 기본 학습
        self.move_log_ = []
        if not self.moves:
            return labels
        X = check_array(X, dtype=np.float64)
        base_bounds = list(self.lower_bounds_)
        self._move_search(X)                # 2) 병합/삭제/재정렬 이동
        lb, trace, conv = self._em(X, self.max_iter)  # 3) 수렴까지 이어서 학습
        self.lower_bound_, self.converged_ = lb, conv
        self.lower_bounds_ = base_bounds + trace
        self.final_trace_ = trace
        self.n_iter_ += len(trace)
        _, log_resp = self._e_step(X)
        return log_resp.argmax(axis=1)

    def _em(self, X, n_iter):
        """sklearn과 같은 E -> M -> ELBO 반복. (마지막 ELBO, 기록, 수렴 여부)를 돌려준다."""
        trace = []
        for _ in range(int(n_iter)):
            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            trace.append(self._compute_lower_bound(log_resp, log_prob_norm))
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < self.tol:
                return trace[-1], trace, True
        return trace[-1], trace, False

    def _candidates(self, resp):
        """이번 라운드에 시도할 이동 목록.
        앞쪽: 재정렬, 겹침이 큰 이웃 병합 상위 max_candidates개, 작은 성분 삭제 상위 max_candidates개.
        뒤쪽: 나머지 이웃 병합과 삭제 전부 (앞쪽이 모두 실패했을 때만 시도된다)."""
        nk = resp.sum(axis=0)
        out = []
        if np.any(np.diff(nk) > 1e-9 * len(resp)):     # 큰 순서로 안 놓여 있으면
            out.append(("reorder", None, None))
        active = np.flatnonzero(self.weights_ > ACTIVE)
        by_mean = active[np.argsort(self.mu_z_[active])]
        pairs = []
        for a, b in zip(by_mean[:-1], by_mean[1:]):
            overlap = np.sqrt(resp[:, a] * resp[:, b]).sum() / np.sqrt(nk[a] * nk[b])
            keep, drop = (a, b) if nk[a] >= nk[b] else (b, a)
            pairs.append((overlap, keep, drop))
        pairs.sort(reverse=True)
        merges = [("merge", i, j) for _, i, j in pairs]
        small = active[np.argsort(self.weights_[active])]
        deletes = [("delete", None, j) for j in small] if len(active) > 1 else []
        c = int(self.max_candidates)
        return out + merges[:c] + deletes[:c] + merges[c:] + deletes[c:]

    def _apply(self, X, resp, kind, i, j):
        """이동 하나를 적용하고 move_iter번 학습한 뒤의 ELBO를 돌려준다."""
        if kind == "reorder":
            order = np.argsort(-resp.sum(axis=0), kind="stable")
            resp = resp[:, order]
            self.mu_z_, self.shapes_ = self.mu_z_[order], self.shapes_[order]
        elif kind == "merge":
            resp = resp.copy()
            resp[:, i] += resp[:, j]
            resp[:, j] = 0.0
            self.shapes_ = self.shapes_.copy()
            self.shapes_[[i, j]] = self.shape_init
        else:  # delete: j를 빼고 나머지 성분 점수로 책임도를 다시 나눔
            logw = self._estimate_weighted_log_prob(X)
            logw[:, j] = -np.inf
            resp = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
            self.shapes_ = self.shapes_.copy()
            self.shapes_[j] = self.shape_init
        if kind != "reorder":  # 병합/삭제 뒤에는 큰 성분이 앞 막대에 오도록 재정렬
            order = np.argsort(-resp.sum(axis=0), kind="stable")
            resp = resp[:, order]
            self.mu_z_, self.shapes_ = self.mu_z_[order], self.shapes_[order]
        self._update_components(X, resp, update_shape=self.estimate_shape)
        return self._em(X, self.move_iter)[0]

    def _move_search(self, X):
        n = len(X)
        for rnd in range(int(self.max_move_rounds)):
            base = self._get_parameters()
            control_lb = self._em(X, self.move_iter)[0]   # 대조군: 이동 없이 같은 횟수 학습
            control = self._get_parameters()
            self._set_parameters(base)
            resp = np.exp(self._e_step(X)[1])
            accepted = False
            for kind, i, j in self._candidates(resp):
                self._set_parameters(base)
                lb = self._apply(X, resp, kind, i, j)
                ok = lb > control_lb + self.move_gain * n
                self.move_log_.append(dict(
                    round=rnd, kind=kind, keep=i, drop=j, control_elbo=control_lb,
                    candidate_elbo=lb, accepted=ok,
                    active_K=int((self.weights_ > ACTIVE).sum())))
                if ok:
                    accepted = True
                    break
            if not accepted:
                self._set_parameters(control)
                return
