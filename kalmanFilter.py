import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial estimation error covariance
        self.x = x  # Initial state

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.4) 

t = np.arange(0, 1, 0.001)  
true_signal = 10 + 5 * np.sin(2 * np.pi * 1 * t) 
noise_std = 2.0 
noisy_signal = true_signal + np.random.normal(0, noise_std, len(t))

F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[1]])
R = np.array([[10]])
P = np.array([[1]])
x = np.array([[0]])

kf = KalmanFilter(F, H, Q, R, P, x)

kalman_estimates = []
for measurement in noisy_signal:
    kf.predict()
    estimate = kf.update(measurement)
    kalman_estimates.append(estimate[0][0])

ax = plt.gca()
line_noisy, = plt.plot(t, noisy_signal, 'orange', alpha=0.6, label='Зашумлений сигнал')
line_true, = plt.plot(t, true_signal, 'b--', label='Справжній сигнал')
line_kalman, = plt.plot(t, kalman_estimates, 'g', label='Оцінка фільтра Калмана')
plt.grid(True)
plt.legend()
plt.title('Фільтр Калмана - Інтерактивна візуалізація')
plt.xlabel('Час (с)')
plt.ylabel('Значення')

axQ = plt.axes([0.1, 0.25, 0.65, 0.03])
axR = plt.axes([0.1, 0.20, 0.65, 0.03])
axP = plt.axes([0.1, 0.15, 0.65, 0.03])
axNoise = plt.axes([0.1, 0.10, 0.65, 0.03])

sQ = Slider(axQ, 'Q', 0.1, 50.0, valinit=1.0, valstep=0.1)
sR = Slider(axR, 'R', 0.1, 50.0, valinit=10.0, valstep=0.1)
sP = Slider(axP, 'P', 0.1, 50.0, valinit=1.0, valstep=0.1)
sNoise = Slider(axNoise, 'Шум', 0.1, 10.0, valinit=2.0, valstep=0.1)

def update(val):
    kf.Q = np.array([[sQ.val]])
    kf.R = np.array([[sR.val]])
    kf.P = np.array([[sP.val]])
    
    noisy_signal = true_signal + np.random.normal(0, sNoise.val, len(t))
    
    kf.x = np.array([[0]])
    
    kalman_estimates = []
    for measurement in noisy_signal:
        kf.predict()
        estimate = kf.update(measurement)
        kalman_estimates.append(estimate[0][0])
    
    line_noisy.set_ydata(noisy_signal)
    line_kalman.set_ydata(kalman_estimates)
    
    noise_var_before = np.var(noisy_signal - true_signal)
    noise_var_after = np.var(kalman_estimates - true_signal)
    plt.title(f'Дисперсія шуму: {noise_var_before:.2f} → {noise_var_after:.2f}')
    
    plt.draw()

reset_ax = plt.axes([0.8, 0.15, 0.1, 0.04])
reset_button = Button(reset_ax, 'Скинути')

def reset(event):
    sQ.reset()
    sR.reset()
    sP.reset()
    sNoise.reset()
    update(None)

reset_button.on_clicked(reset)

sQ.on_changed(update)
sR.on_changed(update)
sP.on_changed(update)
sNoise.on_changed(update)

plt.show()
