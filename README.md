# GGMM with BIC selection

연구 결과를 확인하고 실행하려면 **`GGMM_BIC.ipynb` 하나만 필요합니다.**
Google Colab 또는 Jupyter에서 노트북을 열고 위에서 아래로 실행하면 됩니다.
표본과 기존 GGMM 적합 결과가 노트북 안에 들어 있으므로 별도의 `.py` 파일이나
`ggmm_result.pkl`을 같은 폴더에 둘 필요가 없습니다.

실행 환경에는 NumPy, SciPy, pandas, Matplotlib, scikit-learn,
threadpoolctl, IPython이 필요합니다. Google Colab에는 일반적으로 포함되어 있습니다.

연구 구현을 별도 Python 모듈로 살펴볼 때의 역할은 다음과 같습니다.

- `ggmm_bic_estimator.py`: 후보 K별 GGMM 수치 적합과 BIC 선택을 수행하는 추정기
- `ggmm_bic_experiment.py`: 합성자료 생성, GGMM·VGM 비교, IAE·ISE 평가를 수행하는 실험 실행기

위 두 Python 파일은 구현 재현과 개발을 위한 자료이며, `GGMM_BIC.ipynb` 실행에는 필요하지 않습니다.
