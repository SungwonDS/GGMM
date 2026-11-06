# VariationalGGMM 설계 문서

- 작성일: 2026-11-04
- 상태: **설계안(코드 없음)**. 이 문서는 구현 전에 방향과 검증 기준을 정하기 위한 것이다.
- 목표: sklearn `BayesianGaussianMixture`(이하 **BGM**)의 구조를 그대로 따르면서 성분 분포만 Gaussian에서 **Generalized Gaussian(GGD)** 으로 바꾼 `VariationalGGMM`을 만든다. 성분 수 상한 `K_max`(예: 10~30)를 크게 줘도, 학습 중에 쓸모없는 성분의 가중치가 0 근처로 줄어 **유효 성분 수가 스스로 줄어들어야** 한다.

---

## 0. 이 문서를 읽는 법

각 절은 같은 순서로 쓰여 있다.

1. **쉬운 말**: 무엇을 왜 하는지 한두 문장으로 설명한다.
2. **식 또는 표**: 정확한 내용을 적는다.
3. **확인 방법**: 구현했을 때 맞았는지 확인하는 방법을 적는다.

식이 어렵게 느껴지면 "쉬운 말" 부분만 먼저 읽어도 전체 흐름은 이해할 수 있다.

### 용어집

| 용어 | 뜻 (쉬운 말) |
|---|---|
| 혼합 모형(mixture) | 데이터가 여러 개의 "작은 분포(성분)"를 섞어서 만들어졌다고 보는 모형. 성분마다 가중치가 있다. |
| 성분(component) | 혼합 모형을 이루는 작은 분포 하나. 여기서는 GGD 하나. |
| GGD | Gaussian을 일반화한 분포. 형상 β로 꼬리 두께를 조절한다. β=2이면 Gaussian, β=1이면 Laplace(뾰족하고 꼬리가 두꺼움), β가 크면 평평한 상자 모양에 가까워진다. |
| μ (중심) | 성분의 위치(가운데). |
| α (스케일) | 성분의 폭. 표준편차와 비례하지만 같지는 않다. σ = α·√(Γ(3/β)/Γ(1/β)). |
| β (형상) | 꼬리 두께. 작을수록 꼬리가 두껍다. |
| τ (정밀도 비슷한 값) | τ = α^(−β). β=2이면 τ = 1/α² = 1/(2σ²). τ로 쓰면 Gamma 분포와 "켤레"가 되어 계산이 쉬워진다. |
| 켤레(conjugate) | 사전분포와 사후분포가 같은 종류가 되는 성질. 덕분에 갱신식이 닫힌꼴(공식 한 줄)이 된다. |
| 책임도 r_nk | "데이터 n이 성분 k에서 나왔을 확률"의 추정값. 각 행의 합은 1. |
| N_k | 성분 k가 맡은 데이터의 개수(책임도의 합). 정수가 아니어도 된다. |
| stick-breaking | 길이 1인 막대를 차례로 쪼개서 가중치를 만드는 방식. 디리클레 과정(DP)의 표현법. 앞에서 많이 쪼개 가져가면 뒤쪽 성분 가중치는 자연히 작아진다. |
| γ (집중 모수) | stick-breaking 사전의 모수. 작을수록 "적은 수의 성분만 쓰자"는 쪽으로 강하게 민다. RDT 기본값은 0.001. |
| 변분 추론(VI) | 진짜 사후분포 대신 계산하기 쉬운 분포 q로 근사하는 방법. |
| ELBO | 변분 추론이 최대화하는 값. "모형이 데이터를 얼마나 잘 설명하는가 − 복잡도 벌점"의 하한. 매 반복마다 **절대 내려가면 안 된다**. 내려가면 식이나 코드가 틀린 것이다. |
| 점추정 | 분포 대신 숫자 하나로 추정하는 것. 이 설계에서는 μ와 β를 점추정한다. |
| E-step / M-step | E-step은 현재 모수로 책임도 r을 계산하는 단계, M-step은 r을 고정하고 모수를 갱신하는 단계. 둘을 번갈아 반복한다. |
| 유효 성분(active) | 가중치가 문턱값(RDT 기본 0.005)보다 큰 성분. |

---

## 1. 조사 범위: 읽은 것, 못 읽은 것, 가정

### 1.1 실제로 읽은 코드

| 대상 | 버전 | 확인한 내용 |
|---|---|---|
| `sklearn/mixture/_base.py` | 1.5.2 | `fit_predict`, `_e_step`, `_estimate_log_prob_resp`, 초기화 |
| `sklearn/mixture/_bayesian_mixture.py` | 1.5.2 | `_initialize`, `_m_step`, `_estimate_weights`, `_estimate_log_weights`, `_estimate_log_prob`, `_compute_lower_bound`, `_set_parameters` |
| `rdt/transformers/numerical.py` (`ClusterBasedNormalizer`) | 2.1.0 | `_fit`, `_transform`, `_reverse_transform_helper` |
| `ctgan/data_transformer.py` | 0.13.0 (PyPI 원본) | `_fit_continuous`, `_transform_continuous`, `_inverse_transform_continuous` |
| GitHub `SungwonDS/GGMM`의 `vggmm_merge_refit.py`, `vggmm_experiments.py`, `results/` | main (b8cc3f9) | 기존 변분 GGMM 구현과 실험 결과 |

### 1.2 읽지 못한 코드 (중요)

다음 5개 파일은 사용자 PC에만 있어서 이 문서를 쓰는 동안 열어볼 수 없었다.

- `ctgan/ggmm_mle.py`
- `ctgan/ggd_utils.py`
- `ctgan/ggmm_normalizer.py`
- `ctgan/ggmm_data_transformer.py`
- `ctgan/synthesizers/ggmm_ctgan.py`

그래서 이 파일들과 관련된 부분은 **아래 가정을 바탕으로** 썼다. 구현 0단계(§9)에서 실제 파일을 읽고 가정이 맞는지 먼저 확인해야 한다.

| 파일 | 가정한 역할 | 이 설계에서 쓰는 방식 |
|---|---|---|
| `ggd_utils.py` | GGD의 logpdf, cdf, 표본 추출, α↔σ 변환 | 그대로 재사용. 없는 함수(분위수 반폭 `h`)만 추가 |
| `ggmm_mle.py` | EM으로 GGMM을 최대우도 추정(MLE) | 비교 기준선, 그리고 초기값 후보 |
| `ggmm_normalizer.py` | `ClusterBasedNormalizer`를 상속해 GGMM을 쓰는 정규화기 | 모델 생성 부분만 `VariationalGGMM`으로 바꾸고, 4σ를 `h_k`로 바꾼다 |
| `ggmm_data_transformer.py` | ctgan `DataTransformer`의 GGMM 버전 | 어떤 정규화기를 쓸지 고르는 옵션만 추가 |
| `synthesizers/ggmm_ctgan.py` | 위 변환기를 쓰는 CTGAN | 변경 최소화(옵션 전달만) |

### 1.3 기존 실험에서 얻은 중요한 증거

`SungwonDS/GGMM/results/cap_sweep/.../metrics.csv`에 있는 결과다. 실제 성분 수 3개(True_K=3)인 데이터에 K_max=10을 줬다.

| 방법 | 유효 성분 수 | 학습 시간 |
|---|---|---|
| `VGGMM-base` (기존 변분 GGMM, 병합 없음) | **10** (하나도 안 줄었음) | 33~136초 |
| `VGGMM-merge` (ELBO로 검증한 병합 사용) | 3 | 42~170초 |
| sklearn BGM (`VGM`) | 5 | 0.3~0.7초 |

> **쉬운 말**: 변분 갱신(E-step/M-step 반복)만으로는 성분이 기대만큼 줄지 않았다. Gaussian용 sklearn도 3이 아니라 5에서 멈췄다. 그러니 이 설계의 핵심 목표("스스로 줄어든다")는 **식만 바꿔서는 달성되지 않을 가능성이 높다**. §5.5와 §11에서 이 문제를 따로 다룬다. 또 기존 구현은 sklearn보다 수십~수백 배 느렸다.

---

## 2. sklearn BGM은 어떻게 돌아가나

### 2.1 전체 흐름 (`BaseMixture.fit_predict`)

```
fit_predict(X)
 ├─ 파라미터 검사 (_check_parameters)
 ├─ n_init번 반복:
 │   ├─ 초기화 (_initialize_parameters)
 │   │    kmeans로 각 점에 성분 번호를 붙여 책임도 resp(0/1)를 만들고
 │   │    → _initialize(X, resp): 사실상 M-step을 1번 실행
 │   └─ max_iter번 반복:
 │        ├─ _e_step(X)        → log_resp (책임도) 계산
 │        ├─ _m_step(X, log_resp) → 모수 갱신
 │        ├─ _compute_lower_bound(log_resp, log_prob_norm) → ELBO
 │        └─ |ELBO 변화| < tol 이면 멈춤
 ├─ n_init 중 ELBO가 가장 큰 결과를 채택 (_set_parameters)
 └─ 마지막으로 E-step을 한 번 더 해서 라벨 반환
```

> **쉬운 말**: "책임도 계산 → 모수 갱신 → 점수(ELBO) 계산"을 점수가 더 이상 안 오를 때까지 반복한다. 이 바깥 틀은 성분이 Gaussian이든 GGD든 똑같다. 그래서 **바깥 틀은 상속해서 그대로 쓰고, 안쪽 계산만 바꾸면 된다.**

### 2.2 BGM 안쪽 계산에서 알아둘 점

1. **ELBO 계산식은 "M-step 직후에만 맞는 축약식"이다.**
   `_compute_lower_bound(log_resp, log_prob_norm)`는 X를 받지 않는다. M-step 직후에는 여러 항이 서로 지워져서, 남은 항(책임도 엔트로피, Wishart 정규화 상수, `betaln`, `log κ`)만으로 ELBO를 계산할 수 있기 때문이다. 상수항도 빠져 있다. 따라서 **sklearn ELBO의 절대값을 우리 ELBO와 직접 비교하면 안 된다.**
2. **`covariances_`는 `1 / E[정밀도]`** 이다. 즉 sklearn이 내놓는 표준편차는 "정밀도의 사후평균을 대입한 값"이다(§8에서 사용).
3. **μ에 대한 사후분포가 있다.** BGM은 μ와 정밀도를 함께 다루는 Normal-Wishart를 쓴다. 그래서 E-step 점수에 `−1/(2κ_k)` 항이, ELBO에 `−½ Σ ln κ_k` 항이 있다(κ_k = κ0 + N_k). 이 항은 "성분이 많으면 벌점"을 주는 **오컴 벌점** 역할을 한다.
4. **stick 개수는 K개다.** `_estimate_weights`는 마지막 성분에도 막대 v_K ~ Beta(1+N_K, γ)를 둔다. 그래서 가중치의 합이 정확히 1이 아니고, `_set_parameters`에서 정규화한다. 기존 repo의 `stick_posterior`는 K−1개(마지막 v_K = 1)를 쓴다. 1단계에서 sklearn과 맞추려면 **sklearn 방식(K개)** 을 써야 한다.
5. `tol`은 **표본당이 아니라 전체 ELBO의 절대 변화량**이다(기본 1e-3). n이 크면 매우 엄격해진다.
6. `reg_covar=1e-6`이 분산에 더해진다. sklearn과 정확히 비교할 때는 `reg_covar=0`으로 둔다.

---

## 3. RDT와 CTGAN은 모델에 무엇을 요구하나

### 3.1 `rdt.ClusterBasedNormalizer` (rdt 2.1.0)

```python
# _fit
self._bgm_transformer = BayesianGaussianMixture(
    n_components=self.max_clusters,
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.001,
    random_state=...)
self._bgm_transformer.fit(data.reshape(-1, 1))
self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold  # 0.005

# _transform
means = model.means_.reshape((1, max_clusters))[:, valid]
stds  = np.sqrt(model.covariances_).reshape((1, max_clusters))[:, valid]
normalized = (x - means) / (4 * stds)                    # STD_MULTIPLIER = 4
probs = model.predict_proba(x)[:, valid]
selected = 각 행마다 probs(+1e-6)에 따라 무작위로 성분 하나 선택   # argmax가 아님
normalized = clip(normalized[selected], -0.99, 0.99)

# _reverse_transform_helper
x = clip(normalized, -1, 1) * 4 * std[selected] + mean[selected]
```

**모델이 반드시 가져야 하는 것**

| 속성/메서드 | 모양 | 비고 |
|---|---|---|
| `fit(X)` | X: (n, 1) | |
| `weights_` | (K,) | 유효 성분 판단에 사용 |
| `means_` | 원소 K개 | `reshape((1, K))`가 되어야 함 |
| `covariances_` | 원소 K개 | `sqrt` 후 `reshape((1, K))`. 이 값으로 σ를 만든다 |
| `predict_proba(X)` | (n, K) | |

GGD 버전 정규화기는 여기에 더해 `shapes_`(β, K개)와 `scales_`(α, K개)가 필요하다. 반폭 `h_k`를 계산하기 위해서다(§7).

### 3.2 ctgan `DataTransformer` (0.13.0)

- `_fit_continuous`: `ClusterBasedNormalizer(max_clusters=min(len(data), 10), weight_threshold=0.005)`를 만들고, `valid_component_indicator.sum()`으로 출력 모양을 `[SpanInfo(1,'tanh'), SpanInfo(K_유효,'softmax')]`로 정한다.
- 정규화된 값은 생성기의 **tanh 출력(−1~1)** 으로 만들어지므로 반드시 (−1, 1) 안에 있어야 한다. 그래서 클리핑이 있다.
- `_inverse_transform_continuous`: `sigmas`가 주어지면 정규화 공간에서 `N(값, sigma)` 잡음을 더한 다음 역변환한다. 정규화 폭을 넓히면(예: 두꺼운 꼬리에서 h_k가 커지면) **같은 잡음이 원래 단위에서는 더 커진다**는 점을 기억해 둔다.

---

## 4. (a) BGM에서 그대로 쓰는 부분 vs 바꾸는 부분

**권장 방식**: `BayesianGaussianMixture`를 직접 상속하고, 아래 표의 "바꿈" 메서드만 덮어쓴다. 입력은 1차원으로 제한하고 `covariance_type`은 내부적으로 `'full'`로 고정한다(1차원에서는 full과 spherical이 같다).

> **쉬운 말**: 학습 루프, 초기화, 책임도 정규화, 가중치(stick-breaking) 계산은 sklearn 것을 그대로 빌려 쓴다. "한 성분이 데이터 한 점을 얼마나 잘 설명하는가"를 계산하는 부분과 그 성분의 모수를 갱신하는 부분만 GGD용으로 새로 쓴다.

| sklearn 메서드 | 처리 | 이유 / 내용 |
|---|---|---|
| `fit`, `fit_predict` (n_init, warm_start, max_iter, tol, verbose, random_state) | **그대로** | 바깥 반복 틀은 분포 종류와 무관 |
| `_initialize_parameters` (kmeans 등으로 resp 생성) | **그대로** | 같은 random_state면 sklearn과 같은 초기 resp를 얻음 → 1단계 비교가 쉬워짐 |
| `_e_step`, `_estimate_log_prob_resp`, `_estimate_weighted_log_prob` | **그대로** | "성분 점수 + 가중치 점수 → logsumexp"는 공통 |
| `predict`, `predict_proba`, `score_samples`, `score` | **그대로** | 단, sklearn과 마찬가지로 점수는 "기대 로그밀도" 기준(대입 밀도 아님) |
| `_estimate_weights` (Beta 사후: 1+N_k, γ+Σ_{j>k}N_j) | **그대로** | stick-breaking(DP) 유지 |
| `_estimate_log_weights` (E[ln π_k]) | **그대로** | 같은 이유 |
| `__init__`, `_parameter_constraints` | **바꿈** | `shape_init=2.0`, `estimate_shape`, `shape_bounds=(0.3,10)`, `scale_prior_shape(a0)`, `scale_prior_rate(b0)`, `standardize=True`, `sort_components`, `moves` 등 추가. 공분산 관련 인자 제거 |
| `_check_parameters` | **바꿈** | 1차원 확인, 사전 모수 설정, 표준화 기준값 저장 |
| `_initialize(X, resp)` | **바꿈** | β=shape_init, μ=가중평균, q(τ)=Gamma 갱신 |
| `_m_step(X, log_resp)` | **바꿈** | Beta(그대로 호출) + Gamma(τ) + μ + β 갱신 |
| `_estimate_log_prob(X)` | **바꿈** | GGD 기대 로그밀도(§6.1) |
| `_compute_lower_bound` | **바꿈** | GGD용 축약 ELBO(§6.3). 검증용으로 X를 받는 **전체 ELBO 함수**를 따로 둔다 |
| `_get_parameters`, `_set_parameters` | **바꿈** | Gamma 모수(a_k, b_k), `shapes_`, `scales_` 저장. 호환용 `weights_`, `means_`, `covariances_`(=σ_k²), `precisions_` 생성 |
| `sample` | **바꿈** | Gaussian 대신 GGD(`scipy.stats.gennorm`)로 표본 생성 |
| `_estimate_means`, `_estimate_precisions`, `_estimate_wishart_*` | 쓰지 않음 | 호출되지 않도록 `_m_step`/`_initialize`에서 우회 |
| (신규) `_sort_components` | **추가** | 성분을 N_k 큰 순서로 재정렬(§5.5) |
| (신규, 옵션) 병합/삭제 이동 | **추가** | ELBO가 오를 때만 받아들임(§5.5) |
| (신규) `half_widths(eps)`, `posterior_mean_std()` | **추가** | 정규화(§7)와 진단(§8)용 |

**상속 시 주의점**

- `sklearn.mixture._base`, `_bayesian_mixture`의 밑줄(`_`) 메서드는 **비공개 API**라서 버전마다 바뀔 수 있다. → `requirements.txt`에 sklearn 버전을 고정하고(예: `scikit-learn==1.5.*`), 1단계 테스트를 CI 역할로 쓴다.
- sklearn의 `get_params`/`clone`은 우리 클래스 `__init__`의 인자 목록을 읽는다. 새 인자는 모두 `__init__`에 명시하고 `self.이름 = 이름`으로 그대로 저장해야 한다.
- 대안으로 `BaseMixture`를 상속하고 가중치 관련 함수 30줄 정도를 복사하는 방법도 있다. 더 깔끔하지만, "BGM을 상속했다"는 설명이 약해지고 sklearn 갱신을 직접 따라가야 한다. 처음에는 BGM 상속을 권한다.

---

## 5. (b) 설계 방향

### 5.1 모형

> **쉬운 말**: 가중치는 sklearn과 같은 stick-breaking으로 만든다. 각 성분은 GGD이고, 폭은 τ로 표현해서 Gamma 사전과 켤레가 되게 한다. 중심 μ와 형상 β는 분포 없이 숫자 하나로 추정한다.

데이터는 내부에서 표준화한다: z = (x − m)/s (m은 평균, s는 표준편차). 아래 식은 모두 z 기준이다.

$$
v_k \sim \mathrm{Beta}(1,\gamma),\qquad \pi_k = v_k\prod_{j<k}(1-v_j)\qquad(\gamma = 0.001)
$$

$$
z_n \mid \pi \sim \mathrm{Cat}(\pi),\qquad \tau_k \sim \mathrm{Gamma}(a_0,\ b_0)\ \ (\text{shape},\ \text{rate})
$$

$$
p(x_n \mid z_n=k) = \frac{\beta_k\,\tau_k^{1/\beta_k}}{2\,\Gamma(1/\beta_k)}\exp\!\left(-\tau_k\,|x_n-\mu_k|^{\beta_k}\right),\qquad \alpha_k = \tau_k^{-1/\beta_k}
$$

(GGD의 원래 식 `β/(2αΓ(1/β))·exp(−(|x−μ|/α)^β)`에 α = τ^(−1/β)를 넣은 것과 같다.)

**근사 분포(변분 분포)**

$$
q = \prod_k q(v_k)\;\prod_k q(\tau_k)\;\prod_n q(z_n),\qquad q(v_k)=\mathrm{Beta},\ \ q(\tau_k)=\mathrm{Gamma}(a_k,b_k),\ \ q(z_n)=\mathrm{Cat}(r_n)
$$

μ_k와 β_k는 점추정한다. 기본은 사전 없는 최대우도(ML)이고, 옵션으로 약한 사전(예: ln β ~ N(ln 2, 1²))을 붙인 MAP를 쓸 수 있다.

**사전 기본값**

| 모수 | 기본값 | 이유 |
|---|---|---|
| γ | 0.001 | RDT `ClusterBasedNormalizer`와 동일 |
| a0 | 0.5 | sklearn의 `degrees_of_freedom_prior = n_features = 1`은 τ의 Gamma shape 1/2에 해당 |
| b0 | 표준화 데이터의 표본분산(ddof=1, ≈1) | sklearn의 `covariance_prior = var(X)`에 해당. β=2일 때 **정확히** sklearn과 같은 사전 |
| β 범위 | [0.3, 10] | 경계로 도망가는 것을 막는 안전장치 |
| β 초기값 | 2 | 기존 repo `initial_shape_mode="shape2"`와 같음. 1단계는 이 값에 고정 |

**왜 표준화하나**: τ의 단위는 (데이터 단위)^(−β)라서, β가 바뀌면 같은 b0라도 뜻이 달라진다. 데이터를 표준편차 1로 맞춰 두면 |x−μ|가 1 근처라 |x−μ|^β도 β와 상관없이 1 근처가 되어, b0 하나로 β 전 범위에서 비슷한 의미를 유지할 수 있다. 결과를 내보낼 때는 원래 단위로 되돌린다(μ → m + sμ, α → sα, ELBO에 −N ln s).

### 5.2 μ, β를 점추정해도 "변분 베이지안"이라고 부를 수 있나

> **쉬운 말**: 엄밀히 말하면 **"부분적으로" 베이지안**이다. 가중치(v), 소속(z), 폭(τ)은 분포로 다루지만, 중심(μ)과 형상(β)은 숫자 하나로만 다룬다. 정확한 이름은 **VB-EM**(변분 베이즈 EM) 또는 "점추정된 위치·형상을 갖는 변분 GGMM"이다.

- **ELBO가 무엇의 하한인가**: 완전 베이지안이면 ELBO ≤ ln p(X)이다. 이 설계에서는 **ELBO ≤ ln p(X | μ, β)** 이다(MAP라면 ln p(X, μ, β)). μ와 β를 데이터에 맞춰 고른 다음의 하한이므로, 성분 수를 비교하는 근거로서는 그만큼 약하다.
- **한계 1. 불확실성이 없다**: μ, β가 얼마나 확실한지 알 수 없다. N_k가 작은 성분도 확신에 찬 μ, β를 갖는다.
- **한계 2. 오컴 벌점이 약해진다**: sklearn ELBO에 있는 `−½ ln κ_k ≈ −½ ln N_k`(성분 하나당 BIC와 비슷한 벌점)가 사라진다. β에도 벌점이 없다. 그래서 **성분을 줄이려는 힘이 sklearn보다 약할 수 있다.** 이 설계의 핵심 목표와 부딪히는 부분이다.
- **한계 3. β가 경계로 도망간다**: N_k가 작거나 책임도가 퍼져 있으면 β가 범위 끝(0.3 또는 10)에 붙는다. 실제 검산에서 무작위 책임도로 β를 추정했더니 β*=10(상한)이 나왔다(부록 A). → β 범위 제한, (옵션) ln β 사전, N_k가 작으면 β를 갱신하지 않는 규칙이 필요하다.
- **한계 4. β<1이면 μ 갱신이 비볼록이다**: 목적함수가 여러 개의 골짜기를 가질 수 있어서 단순 최적화로는 전역 최적을 보장하지 못한다.
- **그래도 지켜지는 것**: μ, β 갱신이 "같은 ELBO를 올리는 방향으로만" 움직이면 **ELBO 단조 증가는 보장된다**(좌표 상승법). 따라서 수렴 판정과 디버깅은 sklearn처럼 할 수 있다.

### 5.3 대안 비교

| 안 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **A (제안)** | q(v), q(τ)=Gamma, q(z) + μ, β 점추정 | 단순, 빠름, β=2에서 sklearn과 거의 일치(§6.4), 디버깅 쉬움 | 부분 베이지안. 오컴 벌점 약화 |
| B | q(μ)=Normal 추가 + β 점추정 (기존 `vggmm_merge_refit.py` 방식) | μ도 베이지안, μ 오컴 벌점 있음 | E\|x−U\|^β에 `hyp1f1`이 필요해 느리고 수치적으로 깨지기 쉬움. 기존 결과에서 병합 없이는 성분이 줄지 않았음 |
| C | β=2에서만 Normal-Gamma 결합 (= sklearn) | 완전 켤레, 정확 | β≠2에서는 쓸 수 없음 |
| **D** | β를 격자 {0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8} 위의 이산 분포 q(β)로 둠. 각 격자점마다 q(τ\|β)=Gamma | β까지 베이지안, 모든 갱신이 여전히 닫힌꼴, ELBO 단조 유지 | 계산량이 격자 수(J≈9)배. μ 갱신이 β 격자 전체에 걸친 가중합이 됨 |
| E | MCMC(Gibbs 등) | 완전 베이지안 | 느림, 수렴 판정이 어려움, RDT에 넣기 무거움 |

**권고**: A로 시작한다. 교수님이 "β 점추정은 안 된다"고 하면 D로 바꾼다. D는 A의 코드 대부분(Gamma 갱신, E-step 형태)을 재사용할 수 있다.

### 5.4 성분 축소는 어떻게 일어나나 (원리)

> **쉬운 말**: 데이터를 거의 맡지 않은 성분은 E[ln π_k]가 매우 작아진다(digamma 때문). 그러면 E-step에서 그 성분을 고르는 점수가 낮아지고, 맡는 데이터가 더 줄고, 다시 가중치가 줄어드는 되먹임이 생긴다. γ가 작을수록 이 효과가 강하다.

반대로 성분 두 개가 같은 덩어리를 **반씩 나눠 맡고 있으면** 둘 다 N_k가 커서 이 되먹임이 안 생긴다. 한쪽으로 몰아주면 ELBO가 오르더라도, E-step/M-step을 한 번씩 하는 작은 움직임으로는 그 상태에 도달하지 못한다(**국소 최적**). 기존 실험에서 K_max=10일 때 10개가 그대로 남은 이유가 이것이다.

### 5.5 학습 중 성분 축소를 돕는 장치 (모두 "같은 ELBO를 올릴 때만" 적용)

1. **크기순 재정렬** (기본 켬): 매 반복 후 성분을 N_k 큰 순서로 다시 줄 세운다. stick-breaking은 순서에 따라 ELBO가 달라지고, 큰 성분이 앞에 오는 쪽이 보통 ELBO가 높다(Kurihara 등 2007의 가속 변분 DP에서 쓰는 방법). 재정렬 전후 ELBO를 비교해서 오를 때만 받아들인다.
2. **삭제 이동** (옵션): N_k가 작은 성분을 비우고 그 책임도를 나머지에 다시 나눈 뒤 몇 번 갱신한다. ELBO가 오르면 받아들인다.
3. **병합 이동** (옵션): μ 순서상 이웃한 두 성분을 합친다. 기존 `vggmm_merge_refit._merge_search`의 방식(합친 뒤 전체 재적합, ELBO 증가량이 표본당 1e-4보다 클 때만 수락)을 재사용한다. Hughes & Sudderth(2013)의 memoized VB에 있는 birth/merge/delete 이동과 같은 생각이다.

**"후처리인가, 추론의 일부인가"에 대한 이 문서의 입장**

- **추론의 일부**: 이동이 **같은 목적함수(ELBO)** 를 올릴 때만 받아들여지고, 받아들인 뒤에도 같은 E/M 반복을 계속한다. 그러면 이동은 "더 좋은 국소 최적으로 건너가는 최적화 단계"이다.
- **후처리**: 학습이 끝난 결과를 ELBO와 상관없는 기준(겹침 정도, 거리 문턱 등)으로 고치는 것. 예를 들어 "가까운 성분은 무조건 합친다"는 후처리다.
- 이 판단은 연구 설계의 문제이므로 **교수님께 확인해야 한다**(§11 질문 1). 확인 전까지는 이동을 옵션(기본 끔)으로 두고, 켰을 때와 껐을 때 결과를 모두 보고한다.

---

## 6. (c) 식: E-step, M-step, ELBO

기호: r_nk는 책임도, N_k = Σ_n r_nk, ψ는 digamma, Γ는 감마함수.

### 6.1 E-step 점수 (성분 k가 점 n을 얼마나 잘 설명하나)

$$
\ln\rho_{nk} = \mathbb{E}[\ln\pi_k] + \ln\beta_k - \ln 2 - \ln\Gamma(1/\beta_k) + \frac{\psi(a_k)-\ln b_k}{\beta_k} - \frac{a_k}{b_k}\,|x_n-\mu_k|^{\beta_k}
$$

$$
r_{nk} = \frac{\rho_{nk}}{\sum_j \rho_{nj}}\quad(\text{로그 공간에서 logsumexp로 계산})
$$

$$
\mathbb{E}[\ln\pi_k] = \psi(\gamma_{1k}) - \psi(\gamma_{1k}+\gamma_{2k}) + \sum_{j<k}\big[\psi(\gamma_{2j})-\psi(\gamma_{1j}+\gamma_{2j})\big]\quad(\text{sklearn 코드 그대로})
$$

> **쉬운 말**: 뒤의 네 항이 GGD 로그밀도의 "평균값"이다. τ가 분포(Gamma)이기 때문에 ln τ 자리에는 E[ln τ] = ψ(a) − ln b가, τ 자리에는 E[τ] = a/b가 들어간다. 코드에서는 `_estimate_log_prob`가 첫 항을 뺀 나머지를 (n, K) 배열로 돌려주면, sklearn이 첫 항을 더하고 정규화까지 해 준다.

### 6.2 M-step 갱신식

**(1) 가중치 (그대로)**

$$
\gamma_{1k} = 1 + N_k,\qquad \gamma_{2k} = \gamma + \sum_{j>k} N_j
$$

**(2) 폭 τ (Gamma 켤레, 닫힌꼴)**: 현재 μ_k, β_k에서

$$
S_k(\mu,\beta) = \sum_n r_{nk}\,|x_n-\mu|^{\beta},\qquad a_k = a_0 + \frac{N_k}{\beta_k},\qquad b_k = b_0 + S_k(\mu_k,\beta_k)
$$

> **쉬운 말**: "사전에서 온 값 + 데이터에서 온 값"이다. β=2이면 S_k는 가중 제곱합이고, sklearn의 분산 갱신과 같은 식이 된다.

**(3) 성분별 축약 목적함수**: q(τ)를 최적으로 놓고 τ를 적분하면, 성분 k의 ELBO 기여분이 μ, β만의 함수로 정리된다.

$$
F_k(\mu,\beta) = N_k\big[\ln\beta - \ln 2 - \ln\Gamma(1/\beta)\big] + a_0\ln b_0 - \ln\Gamma(a_0) + \ln\Gamma(a_k) - a_k\ln b_k\ \ \big(+\ln p(\mu)+\ln p(\beta)\ \text{MAP일 때}\big)
$$

(a_k, b_k는 위 식대로 μ, β에 따라 바뀐다.)

**(4) 중심 μ**: β를 고정하면 a_k는 μ와 무관하므로, F_k 최대화는 S_k 최소화와 같다.

$$
\mu_k = \arg\min_{\mu}\ \sum_n r_{nk}\,|x_n-\mu|^{\beta_k}
$$

- β=2: 가중평균 (sklearn과 같음)
- β=1: 가중중앙값
- β≥1: 볼록함수 → 구간 [min x, max x]에서 1차원 Brent 탐색이면 충분
- β<1: 비볼록 → 가중분위수 몇 개를 후보로 넣고, 가장 좋은 후보 주변에서 국소 탐색

**(5) 형상 β**: μ를 고정하고

$$
\beta_k = \arg\max_{\beta\in[0.3,\,10]} F_k(\mu_k,\beta)\quad(\ln\beta\ \text{공간에서 bounded Brent})
$$

**(6) 교대와 단조성**: (4)와 (5)를 2~3번 번갈아 하고, F_k가 **오를 때만** 새 값을 받아들인다. 마지막에 (2)로 a_k, b_k를 다시 계산한다. 각 단계가 같은 ELBO를 올리거나 유지하므로 ELBO 단조 증가가 보장된다.

**(7) 빈 성분 처리**: N_k < 1e-12이면 μ, β는 기본값(β=2)으로 되돌리고 q(τ)는 사전으로 둔다. 기존 repo `VariationalGGMM._empty_slot`과 같은 방식이다.

**(8) 비용 줄이기 규칙**: N_k가 작은 성분(예: N_k < 5)은 β를 갱신하지 않는다. β는 매 반복이 아니라 몇 반복마다 갱신할 수도 있다. 이전 값에서 탐색을 시작한다(warm start).

### 6.3 ELBO

**전체 ELBO** (언제 계산해도 맞는 식, 검증용, X 필요):

$$
\mathcal{L} = \sum_{n,k} r_{nk}\big(\mathbb{E}[\ln\pi_k] + \mathbb{E}[\ln p(x_n\mid k)]\big) - \sum_{n,k} r_{nk}\ln r_{nk} - \sum_k \mathrm{KL}\big(q(v_k)\,\|\,\mathrm{Beta}(1,\gamma)\big) - \sum_k \mathrm{KL}\big(\mathrm{Gamma}(a_k,b_k)\,\|\,\mathrm{Gamma}(a_0,b_0)\big) + \sum_k\big[\ln p(\mu_k)+\ln p(\beta_k)\big] - N\ln s
$$

(E[ln p(x_n|k)]는 §6.1의 뒤 네 항이다. 마지막 −N ln s는 표준화를 되돌리는 보정이고, MAP 사전항은 옵션이다.)

**축약 ELBO** (M-step 직후에만 맞는 식, `_compute_lower_bound`에서 사용, X 불필요):

$$
\mathcal{L} = \sum_k F_k(\mu_k,\beta_k) - \sum_{n,k} r_{nk}\ln r_{nk} + \sum_{k}\Big[\ln B(\gamma_{1k},\gamma_{2k}) - \ln B(1,\gamma)\Big] - N\ln s
$$

(B는 베타함수. sklearn의 `−log_norm_weight = +Σ betaln(γ1, γ2)`와 같은 항이다.)

> **쉬운 말**: M-step 직후에는 많은 항이 서로 지워져서 짧은 식만 남는다. sklearn도 같은 요령을 쓴다. 다만 짧은 식은 "M-step 직후"라는 조건에서만 맞으므로, 테스트에서는 두 식이 같은 값을 내는지 반드시 확인한다. F_k는 M-step에서 이미 계산하므로 저장해 두었다가 `_compute_lower_bound`에서 꺼내 쓰면 된다.

**검산 결과**: 무작위 책임도와 서로 다른 β(0.8, 1.5, 2, 3.5)에서 전체 ELBO와 축약 ELBO의 차이는 2.7e-12였다. sklearn 방식(stick K개)과 repo 방식(stick K−1개) 모두에서 일치했다(부록 A).

### 6.4 식이 바뀌면 ELBO도 바뀌나? → 바뀐다. 어디가 바뀌는지 정리

| ELBO 구성 요소 | sklearn BGM | VariationalGGMM | 바뀜? |
|---|---|---|---|
| 우도 기대값 | Gaussian: −½ln2π + ½E[ln λ] − ½E[λ](x−μ)² − 1/(2κ) | GGD: ln β − ln 2 − lnΓ(1/β) + E[ln τ]/β − E[τ]\|x−μ\|^β | **바뀜** |
| 정밀도/폭 KL | Wishart (`_log_wishart_norm`) | Gamma KL | **바뀜** |
| μ 관련 항 | −½ Σ ln κ_k (q(μ) 엔트로피 + 사전) | 없음 (점추정) | **사라짐** |
| μ, β 사전 | 없음 | ln p(μ) + ln p(β) (MAP 옵션일 때) | **추가** |
| 가중치 항 | +Σ betaln(γ1, γ2) | 같음 | 그대로 |
| 책임도 엔트로피 | −Σ r ln r | 같음 | 그대로 |

**β=2에서 sklearn과 같아지는지 확인 (유도)**

β=2이면 ln 2 − ln 2 − lnΓ(½) = −½ ln π이다. τ = λ/2(λ는 Gaussian 정밀도)이고, sklearn의 ν = 2a, 그리고 b = W0⁻¹ + S라 두면 E[ln λ] = ψ(a) − ln b + ln 2, E[λ] = 2a/b이다. 대입하면

$$
\text{sklearn 점수} = -\tfrac12\ln\pi + \tfrac12\big(\psi(a)-\ln b\big) - \tfrac{a}{b}(x-\mu)^2 - \tfrac{1}{2\kappa_k} = \text{우리 점수} - \tfrac{1}{2\kappa_k}
$$

**즉 두 모형의 차이는 q(μ)의 분산에서 오는 −1/(2κ_k) 한 항뿐이다.** κ_k = κ0 + N_k이므로 성분이 데이터를 많이 맡을수록 차이가 작아진다. sklearn에서 `mean_precision_prior`(κ0)를 아주 작게(1e-10) 두면 M-step 갱신값도 우리와 같아진다. 이 계산을 실제 sklearn 1.5.2로 확인했다(오차 1e-14, 부록 A).

---

## 7. (d) 정규화: 4σ 대신 무엇을 쓸까

### 7.1 문제

RDT는 `(x − μ)/(4σ)`로 정규화하고 ±0.99로 자른다. 주석에는 "99.99% 확률로 [−1, 1] 안에 들어온다"고 되어 있다. 이 말은 Gaussian(β=2)에서만 맞다. GGD에서는 β에 따라 크게 달라진다.

| β | 4σ 밖으로 나갈 확률 (잘리는 비율) | 99.99% 반폭 h / σ | 4σ / h |
|---|---|---|---|
| 0.5 | **1.0e-2** (100개 중 1개 잘림) | 12.62 | 0.32 |
| 1 (Laplace) | **3.5e-3** | 6.51 | 0.61 |
| 1.5 | 6.5e-4 | 4.72 | 0.85 |
| 2 (Gaussian) | 6.3e-5 | 3.89 | 1.03 |
| 4 | 4.3e-15 | 2.74 | 1.46 |
| 8 | ≈0 | 2.21 | **1.81** |

(scipy로 직접 계산한 값, 부록 A)

> **쉬운 말**: β가 작으면(꼬리가 두꺼우면) 4σ 밖의 값이 많아서 너무 많이 잘린다. 잘린 값은 역변환 때 ±4σ 위치로 뭉개지므로 꼬리 정보가 사라진다. β가 크면(상자 모양) 실제 데이터는 ±2.2σ 안에 다 있는데 ±4σ를 쓰니, tanh 범위의 절반 가까이를 버리게 된다(해상도 손실).

### 7.2 제안: 분위수 기반 반폭 h_k

성분마다 "이 성분 데이터의 99.99%가 들어가는 반폭"을 쓴다.

$$
h_k = \alpha_k\cdot\Big[\mathrm{gammaincinv}\big(1/\beta_k,\ 1-\varepsilon\big)\Big]^{1/\beta_k},\qquad \varepsilon = 10^{-4}
$$

여기서 `gammaincinv`는 scipy의 정규화 하위 불완전감마함수의 역함수다. |X−μ|/α를 β제곱하면 Gamma(1/β, 1)을 따르기 때문에 이 식이 나온다. `scipy.stats.gennorm.ppf(1 − ε/2, β) * α`와 같은 값이다(검산 완료).

- 정규화: `u = (x − μ_k)/h_k`, 그다음 `clip(u, −0.99, 0.99)` (RDT와 같음)
- 역변환: `x = clip(u, −1, 1)·h_k + μ_k`
- **β=2이면 h = 3.89σ** → RDT의 4σ와 같은 뜻(99.99%)을 β 전 범위로 넓힌 규칙이다.
- 기대 클리핑 비율 ≈ ε. 0.99에서 자르므로 실제로는 조금 더 크다. 정확히 맞추려면 h_k/0.99를 쓴다.
- α_k는 §8의 대입값 (b_k/a_k)^(1/β_k)을 쓴다.
- ε은 인자로 노출한다(기본 1e-4).

**주의**: RDT는 성분을 argmax가 아니라 확률에 따라 **무작위로** 고른다(+1e-6 보정). 그래서 가끔 멀리 있는 성분이 골라져 클리핑되는 경우가 있다. 클리핑 비율을 측정할 때 "꼬리 때문에 잘림"과 "엉뚱한 성분이 골라져서 잘림"을 나눠서 본다(§10).

### 7.3 대안 (실험 옵션): 확률적분변환(PIT)

$$
u = 2F_k(x) - 1\quad(F_k\text{는 성분 k의 GGD 누적분포})
$$

- 장점: 어떤 β에서도 클리핑이 거의 없고, (−1, 1)을 고르게 쓴다.
- 단점: 비선형이라 생성기 출력이 ±1 근처에서 조금만 틀려도 원래 단위에서는 아주 먼 꼬리 값이 된다(오차 증폭). CTGAN의 `sigmas` 잡음도 꼬리에서 크게 증폭된다.
- 권고: 기본은 §7.2(선형, RDT와 같은 의미), PIT는 비교 실험용 옵션.

---

## 8. (e) 대표 표준편차는 무엇으로 할까

τ는 분포(Gamma(a_k, b_k))이므로 σ도 하나로 정해지지 않는다. 숫자 하나를 고르는 규칙이 필요하다.

**기본 규칙: "정밀도 사후평균을 대입한 값"**

$$
\hat\alpha_k = \big(\mathbb{E}[\tau_k]\big)^{-1/\beta_k} = \Big(\frac{b_k}{a_k}\Big)^{1/\beta_k},\qquad \hat\sigma_k = \hat\alpha_k\sqrt{\frac{\Gamma(3/\beta_k)}{\Gamma(1/\beta_k)}}
$$

호환 속성은 `covariances_ = σ̂_k²` (모양 (K,1,1)), `scales_ = α̂_k`, `shapes_ = β_k`로 둔다.

이유:

1. sklearn `covariances_ = 1/E[λ]`와 같은 규칙이라 **β=2에서 sklearn과 정확히 같다**(1단계 검증 가능, 오차 1e-14 확인).
2. 기존 repo `VariationalGGMM._density`(`loga = (glb − ln ga)/b`)와도 같다.
3. 계산이 간단하고 항상 정의된다.

**진단용으로 함께 보고**: σ의 사후평균

$$
\mathbb{E}[\sigma_k] = \sqrt{\frac{\Gamma(3/\beta)}{\Gamma(1/\beta)}}\; b_k^{1/\beta}\,\frac{\Gamma(a_k-1/\beta)}{\Gamma(a_k)}\quad(a_k>1/\beta\text{일 때 정의})
$$

검산(β=1.5)에서 두 값의 상대 차이는 N_k=5일 때 17.6%, 50일 때 1.7%, 500일 때 0.17%였다. 유효 성분(N_k가 큰 성분)에서는 거의 같고, 작은 성분에서만 다르다.

**주의**: 정규화에는 σ가 아니라 §7의 h_k를 쓴다. σ는 sklearn/RDT와의 호환과 보고용이다.

---

## 9. (f) 단계별 구현 순서와 통과 기준

> **쉬운 말**: 한 번에 다 만들지 않는다. 각 단계는 "앞 단계가 통과해야 시작"한다. 실패하면 그 단계 안에서만 원인을 찾으면 된다.

### 0단계: 준비

- 사용자 로컬 5개 파일(§1.2)을 읽고 가정 표를 확정한다. 다르면 이 문서를 먼저 고친다.
- sklearn, rdt, ctgan 버전을 고정한다.
- 기준선을 저장한다: 같은 데이터에서 sklearn BGM과 `ggmm_mle`의 결과(유효 K, 로그우도, 시간).
- **통과**: 기준선 표가 저장되고, 테스트 데이터 생성 함수(시드 고정)가 있다.

### 1단계: β=2 고정, sklearn과 일치

설정: `estimate_shape=False`, `shape_init=2`. 비교 대상은 `BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=1e-3, mean_precision_prior=1e-10, degrees_of_freedom_prior=1, reg_covar=0)`.

| 검사 | 방법 | 통과 기준 |
|---|---|---|
| 1a. M-step 1회 | 같은 resp로 양쪽 `_initialize` | weights, means, σ 최대 오차 < 1e-8 (검산 1e-14) |
| 1b. E-step 점수 | 같은 모수에서 `_estimate_log_prob` 비교 | (우리 − sklearn) − 1/(2κ_k)의 최대 오차 < 1e-8 (검산 2.6e-14) |
| 1c. 전체 학습 | 같은 random_state(같은 kmeans 초기값), 데이터 3종 × 시드 5개 | 유효 K 같음, 정렬한 가중치 차 < 0.01, 평균 차 < 0.05σ, 표본당 로그우도 차 < 1e-3 |
| 1d. ELBO | 매 반복 기록 | 감소 0회, 전체 ELBO = 축약 ELBO (< 1e-8) |

참고: sklearn 기본값(κ0=1)과의 차이는 통과/실패가 아니라 보고만 한다.

### 2단계: β 추정

| 검사 | 통과 기준 |
|---|---|
| K=1, β ∈ {0.5, 1, 2, 4, 8}, n=10,000, 시드 5개 | β 상대오차 ≤ 10% (β≤4), ≤ 20% (β=8); μ 오차 ≤ 0.05α |
| 혼합(β=1, 2, 4), 잘 떨어진 경우 | 매칭 후 β 상대오차 ≤ 15%, μ 오차 ≤ 0.1σ, 가중치 오차 ≤ 0.02 |
| ELBO | 감소 0회, 전체 = 축약 |
| β가 경계(0.3 또는 10)에 붙은 **유효** 성분 수 | 0 (붙으면 원인 기록) |

### 3단계: 성분 축소 검증

- K_max = 10, 20, 30 × {정렬만, 정렬+삭제, 정렬+삭제+병합} × 시드 10개.
- **통과 기준**
  - 유효하지 않은 슬롯의 가중치 합 < 0.01 (가중치가 실제로 0 쪽으로 모였는가)
  - 이동을 켰을 때 최종 ELBO ≥ 껐을 때 ELBO (같은 시드 기준)
  - 홀드아웃 표본당 로그우도가 sklearn BGM보다 나쁘지 않음 (Gaussian 데이터: 허용 오차 0.01/표본; GGD 데이터: 더 좋아야 함)
- **보고만 하는 진단 지표 (통과/실패 아님)**: 유효 K, K_max에 따른 유효 K의 변화, 제외된 질량(excluded mass), 반복 횟수.

### 4단계: RDT 연결

- `GGMMNormalizer(ClusterBasedNormalizer)`: `_fit`에서 `VariationalGGMM` 생성, `_transform`과 `_reverse_transform_helper`에서 `4*std`를 `h_k`로 바꾼다. 출력 열 이름과 구조(`.normalized`, `.component`, 결측 열)는 그대로 둔다.
- **통과 기준**: 출력 열 구조가 원본과 같음; 잘리지 않은 행의 왕복 오차 < 1e-8 (상대); 정규화 값이 [−0.99, 0.99] 안에 있음; 정상 데이터에서 꼬리 클리핑 비율 ≤ 3ε; `valid_component_indicator.sum() ≥ 1`; pickle 저장/불러오기 가능; 결측값 처리(`missing_value_generation='from_column'`) 동작.

### 5단계: CTGAN 연결

- `ggmm_data_transformer.py`에서 정규화기 선택 옵션 추가 → `ggmm_ctgan.py`에서 전달.
- **통과 기준**: 1 epoch 스모크 테스트 통과; `output_info`의 차원 합 = 변환 데이터 열 수; `sample()`이 원래 열 타입과 범위로 동작.
- **보고**: 기준 CTGAN 대비 열별 KS 통계량/TV 거리, 학습 시간.

---

## 10. (g) 테스트 계획

| 분류 | 내용 | 판정 |
|---|---|---|
| ELBO 단조성 | 매 반복 ΔL ≥ −1e-8·\|L\|. 전체 ELBO와 축약 ELBO 일치(< 1e-8). 이동(정렬/삭제/병합) 전후도 확인 | 통과/실패 |
| 식 검산 | E[ln p(x\|k)]를 q(τ)에서 τ를 10만 번 뽑는 몬테카를로로 확인(3 표준오차 이내); GGD pdf 적분 = 1 (β 0.3~10) | 통과/실패 |
| sklearn 일치 (β=2) | §9 1단계 | 통과/실패 |
| 모수 복원 | 3성분: β=(1, 2, 4), μ=(−4, 0, 4), σ=1, w=(0.3, 0.4, 0.3). 성분은 μ 기준 헝가리안 매칭. 지표: β 상대오차, μ 오차/σ, 가중치 오차, 참 밀도 대비 IAE | 통과/실패 (§9 기준) |
| 겹침이 큰 경우 | 위 혼합에서 μ 간격을 2σ로 줄임 | 보고 |
| 표본 수 변화 | n = 500, 2,000, 10,000, 50,000. 오차가 대략 1/√n 속도로 줄어드는지(로그-로그 기울기 ≈ −0.5) | 보고 + 추세 확인 |
| 시드 변화 | 시드 0~9. 지표의 평균 ± 표준편차, n_init 최고 ELBO 선택 효과 | 보고 |
| K_max 변화 | K_max = 10/20/30에서 유효 K 분포(최소/중앙/최대), 제외 질량, 시간 | **보고만 (진단 지표)** |
| 클리핑 비율 | β ∈ {0.5, 1, 2, 4, 8} 단일 성분과 혼합에서 4σ 방식 vs h_k 방식. "꼬리 때문에 잘림"과 "무작위 성분 선택 때문에 잘림"을 분리 | h_k 방식 ≤ 3ε (꼬리 원인) |
| 실행 시간 | n × K_max 격자에서 fit 시간, 반복당 시간, sklearn 대비 배수, 단계별(E/M/β 탐색/이동) 시간. 목표 예: n=10k, K_max=30에서 30초 이내 | 목표 대비 보고 |
| 경계 사례 | 상수 열, n < K_max, 중복값이 많은 열, 극단값 1개, 정수형 열, 결측값 | 오류 없이 동작하거나 명확한 오류 메시지 |
| sklearn 규약 | `clone`, `get_params`/`set_params`, `fit` 후 `predict_proba` 모양, `random_state` 재현성 | 통과/실패 |

테스트 데이터는 모두 시드를 고정한 함수로 생성하고, 결과는 기존 repo처럼 CSV + 메타데이터(JSON: 버전, 시드, 설정)로 저장한다.

---

## 11. (h) 위험 요소와 교수님께 드릴 질문

### 11.1 위험 요소

| 위험 | 왜 문제인가 | 대응 |
|---|---|---|
| **변분 갱신만으로는 성분이 안 줄 수 있음** | 기존 결과: K_max=10에서 Active_K=10 (참 K=3). 성분이 덩어리를 나눠 가지는 국소 최적 | 크기순 정렬, ELBO 검증 삭제/병합 이동(§5.5), K_max별 진단 보고 |
| 점추정 → 오컴 벌점 약화 | μ, β에 벌점이 없어 성분이 남기 쉬움 | 대안 D(이산 q(β)), μ에 약한 사전, 결과 보고 |
| β 경계 도망 | 작은 N_k나 퍼진 책임도에서 β가 0.3 또는 10에 붙음(검산에서 실제로 β*=10) | 범위 제한, N_k 작으면 β 고정, ln β 사전(옵션) |
| β<1일 때 μ 비볼록 | 국소 최적에 빠짐 | 가중분위수 후보 + 국소 탐색 |
| τ 사전의 의미가 β에 따라 바뀜 | 단위가 (데이터 단위)^(−β) | 내부 표준화(§5.1) |
| sklearn 비공개 API 의존 | 버전이 바뀌면 깨질 수 있음 | 버전 고정, 1단계 테스트를 회귀 테스트로 사용 |
| 수치 문제 | β가 작으면 lnΓ(1/β)가 큼; β가 크면 \|x−μ\|^β가 넘침 | 로그 공간 계산, 표준화, `gammaln`/`logsumexp` 사용 |
| 실행 시간 | β 탐색이 반복마다 성분 수 × n 계산. 기존 VGGMM은 sklearn보다 수십~수백 배 느렸음 | 점추정(hyp1f1 불필요), warm start, β 갱신 주기 조절, 작은 성분 생략 |
| rdt/ctgan 버전 차이 | 속성 이름이나 흐름이 다를 수 있음 | 0단계에서 버전 고정, 4단계 테스트 |
| 로컬 파일 미확인 | 이 문서의 §1.2는 가정 | 0단계에서 확인 후 문서 수정 |
| 정규화 폭과 CTGAN 잡음 | h_k가 크면 `sigmas` 잡음이 원래 단위에서 커짐 | 5단계에서 열별 분포 비교로 확인 |
| sklearn의 stick 개수(K개)와 repo(K−1개) 차이 | 결과가 약간 다름 | 1단계는 sklearn 방식, 차이를 문서화 |

### 11.2 교수님께 드릴 질문

1. **학습 중 인접 성분 병합(그리고 삭제)이 후처리인가?**
   "ELBO가 오를 때만 받아들이고, 받아들인 뒤 같은 E/M 반복을 계속하는 이동"을 추론 알고리즘의 일부로 봐도 되는지 여쭌다. 성분을 크기순으로 다시 정렬하는 것은 허용되는지도 함께 여쭌다. (이 문서의 입장은 §5.5.)
2. **μ, β 점추정을 허용하는가?**
   허용하지 않으면 (B) q(μ)=Normal과 (D) 이산 q(β) 중 무엇을 원하시는지 여쭌다. 논문이나 보고서에서 "변분 베이지안 GGMM"이라고 불러도 되는지, 아니면 "VB-EM GGMM"으로 불러야 하는지도 여쭌다.
3. **사전 값**: γ=0.001(RDT와 같음), a0=0.5, b0≈1(표준화 데이터), β 범위 [0.3, 10]을 그대로 써도 되는가?
4. **성공 기준**: "유효 K = 참 K"인가, "밀도 품질(홀드아웃 로그우도, IAE)"인가, 아니면 CTGAN 합성 데이터 품질인가?
5. **비교 기준선**: sklearn BGM, `ggmm_mle`+BIC 외에 꼭 넣어야 할 비교 대상이 있는가?
6. **정규화**: 분위수 반폭 h_k(선형)로 충분한가, PIT 같은 비선형 방식도 실험해야 하는가? ε=1e-4가 적절한가?
7. **대표 σ 규칙**: "정밀도 사후평균 대입"(sklearn과 같음)으로 정해도 되는가?
8. 기존 코드 주석에 나온 **IEEE 9002852 논문의 알고리즘을 재현해야 하는가**, 아니면 이 독립 설계로 충분한가?

---

## 부록 A. 수치 검산 결과

sklearn 1.5.2, scipy 1.18.1, numpy 2.5.3으로 직접 계산했다. 데이터는 N(−3, 1) 400개 + N(2, 0.5) 600개, K=4, 무작위 책임도.

| 검사 | 결과 |
|---|---|
| M-step 1회, β=2, sklearn(κ0=1e-10, reg_covar=0) 대비 means 최대 오차 | 2.2e-14 |
| 같은 조건 σ² 최대 오차 | 1.2e-14 |
| sklearn `degrees_of_freedom_/2`와 a_k | 완전히 같음 (0.0) |
| E-step 점수: (우리 − sklearn) − 1/(2κ_k) | 2.6e-14 |
| E[ln π]: 우리 식 vs sklearn `_estimate_log_weights` | 0.0 |
| 전체 ELBO − 축약 ELBO (β = 0.8, 1.5, 2, 3.5; stick K개) | −2.7e-12 |
| 같은 검사 (stick K−1개, repo 방식) | −1.8e-12 |
| β 한 번 갱신 시 F_k 변화 (무작위 책임도) | −511.09 → −424.63 (증가), 단 β* = 10(상한)에 붙음 → §5.2 한계 3의 예 |
| h/σ (ε = 1e-4) | β = 0.5: 12.62, 1: 6.51, 2: 3.89, 4: 2.74, 8: 2.21 |
| `gammaincinv` 식 vs `gennorm.ppf` | 모든 β에서 같음 |
| 대입 σ vs E[σ] (β = 1.5) | N_k = 5: 17.6%, 50: 1.7%, 500: 0.17% 차이 |

## 부록 B. 기존 코드에서 재사용할 것 (`SungwonDS/GGMM/vggmm_merge_refit.py`)

| 함수/클래스 | 용도 | 비고 |
|---|---|---|
| `stick_kl`, `gamma_kl` | 전체 ELBO(검증용) | 그대로 |
| `stick_posterior`, `stick_statistics` | 비교용 | sklearn 방식(K개)과 다르므로 1단계에서는 sklearn 쪽을 사용 |
| `Mixture` (gennorm 기반 pdf/cdf/logpdf) | 밀도 평가, 표본 생성, 실험 지표 | 그대로 |
| `partition` (quantile/lloyd 결정적 초기화) | 초기화 옵션 | sklearn kmeans 초기화 외 추가 선택지 |
| `VariationalGGMM._empty_slot` | 빈 성분 처리 | 같은 규칙 |
| `VariationalGGMM._merge_search`, `_merge_seed`, `_pair_pool` | 병합 이동 | q(μ) 부분을 점추정에 맞게 단순화 |
| `vggmm_experiments.py`의 `density_errors`, `run_cap_experiment` | 3단계 지표(IAE/ISE), K_max 스윕 | 그대로 또는 약간 수정 |

## 참고문헌

- Bishop, *Pattern Recognition and Machine Learning*, 10장 (변분 Gaussian 혼합). sklearn BGM의 바탕.
- Blei & Jordan (2006), Variational inference for Dirichlet process mixtures. stick-breaking 변분 추론.
- Kurihara, Welling & Vlassis (2007), Accelerated variational Dirichlet process mixtures. 성분 크기순 정렬.
- Hughes & Sudderth (2013), Memoized online variational inference for Dirichlet process mixture models. ELBO 검증 birth/merge/delete 이동.
- scikit-learn 1.5.2 `sklearn/mixture/_base.py`, `_bayesian_mixture.py`; rdt 2.1.0 `ClusterBasedNormalizer`; ctgan 0.13.0 `DataTransformer`.
