import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import requests


# Функция для генерации сигнала
def generate_signal_single_point(time_point, num_exp, num_cos, num_log, amp_exp, amp_cos, amp_log):
    total_signal = 0
    exp_const = 1
    freq_base = 2 * np.pi
    phase_shift = 0
    log_const = 1
    k_base = 1

    # Экспоненциальные компоненты
    if num_exp > 0:
        for i in range(num_exp):
            exp_amp = amp_exp[i] if i < len(amp_exp) else 0
            total_signal += exp_amp * np.exp(-time_point / exp_const)

    # Косинусные компоненты
    if num_cos > 0:
        for j in range(num_cos):
            cos_amp = amp_cos[j] if j < len(amp_cos) else 0
            frequency = freq_base * (j + 1)
            total_signal -= cos_amp * np.cos(frequency * time_point + phase_shift)

    # Логарифмические компоненты
    if num_log > 0:
        for k in range(num_log):
            log_amp = amp_log[k] if k < len(amp_log) else 0
            k_value = k_base * (k + 1)
            log_input = max(log_const * k_value * time_point, 1e-10)
            total_signal += log_amp * np.log10(log_input)

    return total_signal


# Класс для графического интерфейса
class SignalApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Генерация и обработка сигнала")

        # Параметры для генерации сигнала
        self.num_exp = 3
        self.num_cos = 1
        self.num_log = 0
        self.amp_exp = [0.73, 0.15, 0.5]
        self.amp_cos = [1]
        self.amp_log = [0]

        # Параметры для расчета EMA
        self.N = 5
        self.alpha = 2 / (self.N + 1)
        self.EMA_prev = None

        # Параметры скользящего среднего
        self.window_size = 3
        self.moving_avg_window = []

        # Массивы для хранения данных
        self.time_values = []
        self.signal_values = []
        self.ema_values = []
        self.moving_avg_values = []

        # Параметры времени
        self.sampling_interval = 0.05
        self.total_duration = 90
        self.start_time = None
        self.current_time = 0
        self.running = False

        # Создаем элементы интерфейса
        self.create_widgets()

        # Инициализация источника данных
        self.data_source = "mathematical_model" 

    def create_widgets(self):
        # Кнопки управления
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.start_button = ttk.Button(control_frame, text="Старт", command=self.start_signal)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Стоп", command=self.stop_signal, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Поля для отображения текущих значений
        value_frame = ttk.Frame(self.master)
        value_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(value_frame, text="Текущее время:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.time_label = ttk.Label(value_frame, text="0.00 с")
        self.time_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(value_frame, text="Текущий сигнал:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.signal_label = ttk.Label(value_frame, text="0.0000")
        self.signal_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(value_frame, text="Текущий EMA:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.ema_label = ttk.Label(value_frame, text="0.0000")
        self.ema_label.grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Label(value_frame, text="Скользящее среднее:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.moving_avg_label = ttk.Label(value_frame, text="0.0000")
        self.moving_avg_label.grid(row=3, column=1, sticky=tk.W, padx=5)

        # График
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.line1, = self.ax.plot([], [], label='Сигнал')
        self.line2, = self.ax.plot([], [], label='EMA', linestyle='--')
        self.line3, = self.ax.plot([], [], label='Скользящее среднее', linestyle=':')
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Значение')
        self.ax.set_title('Сигнал, EMA и Скользящее Среднее')
        self.ax.legend()
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def start_signal(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.start_time = time.time()
            self.EMA_prev = None
            self.time_values.clear()
            self.signal_values.clear()
            self.ema_values.clear()
            self.moving_avg_values.clear()
            self.ax.clear()
            self.line1, = self.ax.plot([], [], label='Сигнал')
            self.line2, = self.ax.plot([], [], label='EMA', linestyle='--')
            self.line3, = self.ax.plot([], [], label='Скользящее среднее', linestyle=':')
            self.ax.set_xlabel('Время (с)')
            self.ax.set_ylabel('Значение')
            self.ax.set_title('Сигнал, EMA и Скользящее Среднее')
            self.ax.legend()
            self.ax.grid(True)

            # Запуск потока для генерации сигнала
            self.signal_thread = threading.Thread(target=self.generate_signal)
            self.signal_thread.start()

    def stop_signal(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def get_signal_from_source(self):
        if self.data_source == "mathematical_model":
            return generate_signal_single_point(self.current_time, self.num_exp, self.num_cos, self.num_log, self.amp_exp, self.amp_cos, self.amp_log)
        elif self.data_source == "analog_sensor":
            # Имитация данных от аналогового датчика
            return np.random.rand()
        elif self.data_source == "digital_sensor":
            # Имитация данных от цифрового датчика
            return np.random.randint(0, 256)
        elif self.data_source == "api":
            try:
                response = requests.get('https:127.0.0.1:5000/signal')
                return response.json().get('signal', 0)
            except Exception as e:
                print(f"Ошибка при получении данных из API: {e}")
                return 0

    def generate_signal(self):
        while self.running and self.current_time < self.total_duration:
            self.current_time = time.time() - self.start_time
            signal_value = self.get_signal_from_source()

            # Расчет EMA
            if self.EMA_prev is None:
                self.EMA_prev = signal_value
            else:
                self.EMA_prev = (self.alpha * signal_value) + ((1 - self.alpha) * self.EMA_prev)

            # Расчет скользящего среднего
            self.moving_avg_window.append(signal_value)
            if len(self.moving_avg_window) > self.window_size:
                self.moving_avg_window.pop(0)
            moving_avg = sum(self.moving_avg_window) / len(self.moving_avg_window)

            # Сохраняем значения для построения графиков
            self.time_values.append(self.current_time)
            self.signal_values.append(signal_value)
            self.ema_values.append(self.EMA_prev)
            self.moving_avg_values.append(moving_avg)

            # Обновляем линии на графике
            self.line1.set_data(self.time_values, self.signal_values)
            self.line2.set_data(self.time_values, self.ema_values)
            self.line3.set_data(self.time_values, self.moving_avg_values)

            # Устанавливаем пределы осей
            self.ax.set_xlim(0, max(self.time_values) if self.time_values else 1)

            # Избегаем одинаковых значений границ по оси Y
            min_value = min(min(self.signal_values), min(self.ema_values), min(self.moving_avg_values))
            max_value = max(max(self.signal_values), max(self.ema_values), max(self.moving_avg_values))

            if min_value == max_value:
                min_value -= 0.1
                max_value += 0.1

            self.ax.set_ylim(min_value, max_value)

            # Обновляем интерфейс
            self.canvas.draw()

            # Обновляем текстовые метки
            self.time_label.config(text=f"{self.current_time:.2f} с")
            self.signal_label.config(text=f"{signal_value:.4f}")
            self.ema_label.config(text=f"{self.EMA_prev:.4f}")
            self.moving_avg_label.config(text=f"{moving_avg:.4f}")

            time.sleep(self.sampling_interval)


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalApp(master=root)
    root.mainloop()