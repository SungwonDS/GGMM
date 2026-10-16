import numpy as np
import matplotlib
# 독립적인 GUI 창을 띄우기 위한 백엔드 설정
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import gamma

# GGD 확률밀도함수 정의
def ggd_pdf(x, mu=0, alpha=1.0, beta=2.0):
    coeff = beta / (2 * alpha * gamma(1.0 / beta))
    return coeff * np.exp(- (np.abs(x - mu) / alpha) ** beta)

x = np.linspace(-6, 6, 1000)
init_alpha, init_beta = 1.0, 2.0

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

y = ggd_pdf(x, 0, init_alpha, init_beta)
line, = ax.plot(x, y, lw=2.5, color='#1f77b4')
fill = ax.fill_between(x, 0, y, alpha=0.2, color='#1f77b4')

ax.set_ylim(0, 1.5)
ax.set_xlim(-6, 6)
ax.set_title(f'Generalized Gaussian Distribution\nScale (α) = {init_alpha:.2f}, Shape (β) = {init_beta:.2f}', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

# 슬라이더 위치 설정
ax_alpha = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_beta = plt.axes([0.15, 0.05, 0.65, 0.03])

# 객체 참조 유지 (이 변수들이 사라지면 슬라이더가 멈춥니다)
slider_alpha = Slider(ax_alpha, 'Scale (a)', 0.1, 3.0, valinit=init_alpha, valstep=0.1)
slider_beta = Slider(ax_beta, 'Shape (b)', 0.2, 10.0, valinit=init_beta, valstep=0.1)

def update(val):
    current_alpha = slider_alpha.val
    current_beta = slider_beta.val
    
    new_y = ggd_pdf(x, 0, current_alpha, current_beta)
    line.set_ydata(new_y)
    
    ax.collections.clear()
    ax.fill_between(x, 0, new_y, alpha=0.2, color='#1f77b4')
    
    ax.set_title(f'Generalized Gaussian Distribution\nScale (α) = {current_alpha:.2f}, Shape (β) = {current_beta:.2f}', fontsize=14)
    fig.canvas.draw_idle()

slider_alpha.on_changed(update)
slider_beta.on_changed(update)

# 대화형 GUI 창 실행
plt.show()
