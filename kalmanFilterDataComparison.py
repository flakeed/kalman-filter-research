import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from datetime import datetime

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  
        self.H = H  
        self.Q = Q 
        self.R = R  
        self.P = P 
        self.x = x 

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

class ExperimentRunner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"kalman_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.frequency = 1
        self.amplitude = 5
        self.sampling_interval = 0.001
        self.total_time = 1
        
        self.parameters = {
            'Q': [0.1, 10.0],    
            'R': [1.0, 20.0],    
            'P': [0.1, 10.0],    
            'x0': [-2.0, 2.0],  
            'offset': [0.0, 5.0]  
        }

    def run_single_experiment(self, Q, R, P, x0, offset):
        time_steps = np.arange(0, self.total_time, self.sampling_interval)
        
        true_signal = offset + self.amplitude * np.sin(2 * np.pi * self.frequency * time_steps)
        
        noise_std_dev = 2.0
        noisy_signal = true_signal + np.random.normal(0, noise_std_dev, len(true_signal))
        
        F = np.array([[1]])
        H = np.array([[1]])
        x = np.array([[x0]])
        
        kf = KalmanFilter(F, H, np.array([[Q]]), np.array([[R]]), np.array([[P]]), x)
        
        kalman_estimates = []
        for measurement in noisy_signal:
            kf.predict()
            estimate = kf.update(measurement)
            kalman_estimates.append(estimate[0][0])
        
        error_before = noisy_signal - true_signal
        error_after = np.array(kalman_estimates) - true_signal
        
        return {
            'time_steps': time_steps,
            'true_signal': true_signal,
            'noisy_signal': noisy_signal,
            'kalman_estimates': kalman_estimates,
            'variance_before': np.var(error_before),
            'variance_after': np.var(error_after),
            'mse_before': np.mean(error_before**2),
            'mse_after': np.mean(error_after**2),
            'convergence_time': self.calculate_convergence_time(error_after)
        }

    def calculate_convergence_time(self, error, threshold=0.1):
        abs_error = np.abs(error)
        converged_indices = np.where(abs_error < threshold)[0]
        if len(converged_indices) > 0:
            return converged_indices[0] * self.sampling_interval
        return self.total_time

    def plot_and_save_results(self, results, params):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['time_steps'], results['noisy_signal'], 
                label='Зашумлений сигнал', color='orange', alpha=0.6)
        plt.plot(results['time_steps'], results['true_signal'], 
                label='Істинний сигнал', linestyle='--', color='blue')
        plt.plot(results['time_steps'], results['kalman_estimates'], 
                label='Оцінка фільтра Калмана', color='green')
        
        plt.xlabel('Час (с)')
        plt.ylabel('Значення')
        plt.title(f'Результати фільтрації Калмана\n' + 
                 f'Q={params["Q"]}, R={params["R"]}, P={params["P"]}, ' +
                 f'x0={params["x0"]}, offset={params["offset"]}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        error = np.array(results['kalman_estimates']) - results['true_signal']
        plt.plot(results['time_steps'], error, label='Помилка оцінки', color='red')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Час (с)')
        plt.ylabel('Помилка')
        plt.title('Помилка оцінки')
        plt.legend()
        plt.grid(True)
        
        filename = (f'kalman_Q{params["Q"]}_R{params["R"]}_P{params["P"]}_' +
                   f'x0{params["x0"]}_offset{params["offset"]}.png')
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
        
        return filename

    def run_all_experiments(self):
        results_file = os.path.join(self.results_dir, 'results.txt')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Результати експериментів з фільтром Калмана\n")
            f.write("=" * 50 + "\n\n")
            
            param_combinations = product(
                self.parameters['Q'],
                self.parameters['R'],
                self.parameters['P'],
                self.parameters['x0'],
                self.parameters['offset']
            )
            
            for Q, R, P, x0, offset in param_combinations:
                params = {
                    'Q': Q,
                    'R': R,
                    'P': P,
                    'x0': x0,
                    'offset': offset
                }
                
                results = self.run_single_experiment(Q, R, P, x0, offset)
                filename = self.plot_and_save_results(results, params)
                
                f.write(f"Параметри експерименту:\n")
                f.write(f"  Q (коваріація шуму процесу): {Q}\n")
                f.write(f"  R (коваріація шуму вимірювань): {R}\n")
                f.write(f"  P (початкова коваріація помилки): {P}\n")
                f.write(f"  x0 (початкова оцінка стану): {x0}\n")
                f.write(f"  offset (зміщення сигналу): {offset}\n")
                f.write(f"\nРезультати:\n")
                f.write(f"  Дисперсія до фільтрації: {results['variance_before']:.2f}\n")
                f.write(f"  Дисперсія після фільтрації: {results['variance_after']:.2f}\n")
                f.write(f"  Зменшення дисперсії: {((results['variance_before'] - results['variance_after']) / results['variance_before'] * 100):.2f}%\n")
                f.write(f"  MSE до фільтрації: {results['mse_before']:.2f}\n")
                f.write(f"  MSE після фільтрації: {results['mse_after']:.2f}\n")
                f.write(f"  Час збіжності: {results['convergence_time']:.3f} с\n")
                f.write(f"  Графік збережено як: {filename}\n")
                f.write("=" * 50 + "\n\n")

runner = ExperimentRunner()
runner.run_all_experiments()
print(f"Результати збережено в директорії: {runner.results_dir}")
