import numpy as np
import matplotlib.pyplot as plt

#Генерация сигнала
def generate_signal(time, num_exp, num_cos, num_log, amp_exp, amp_cos, amp_log):
    total_signal = np.zeros_like(time)

    # Константы
    exp_const = 1 
    freq_base = 2 * np.pi
    phase_shift = 0  
    log_const = 1 
    k_base = 1  

    # Добавляем экспоненциальные члены
    if num_exp > 0:
        for i in range(num_exp):
            exp_amp = amp_exp[i] if i < len(amp_exp) else 0
            total_signal += exp_amp * np.exp(-time / exp_const)

    # Добавляем косинусные члены
    if num_cos > 0:
        for j in range(num_cos):
            cos_amp = amp_cos[j] if j < len(amp_cos) else 0
            frequency = freq_base * (j + 1)
            total_signal -= cos_amp * np.cos(frequency * time + phase_shift)

    # Добавляем логарифмические члены
    if num_log > 0:
        for k in range(num_log):
            log_amp = amp_log[k] if k < len(amp_log) else 0
            k_value = k_base * (k + 1)
            log_term = np.log10(np.maximum(log_const * k_value * time, 1e-10))
            total_signal += log_amp * log_term

    return total_signal

t = np.linspace(0.1, 10, 1000)
n = 3 
m = 1  
l = 0  
A_vals = [0.73, 0.15, 0.5]
B_vals = [1]
C_vals = [0]

signal = generate_signal(t, n, m, l, A_vals, B_vals, C_vals)

plt.plot(t, signal)
plt.title('Сгенерированный сигнал')
plt.xlabel('Время (t)')
plt.ylabel('Сигнал (s)')
plt.show()