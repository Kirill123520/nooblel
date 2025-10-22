import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# Устанавливаем Matplotlib для работы с Tkinter
import matplotlib

matplotlib.use("TkAgg")


# --- 1. euler method solver  ---
def f(t, x):
    """the right side of the ODE: dx/dt = t - x^2"""
    return t - x ** 2


def explicit_euler(T, h, x0):
    """solves the ODE from 0 to T with step h and initial condition x0."""
    N = int(T / h)
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0

    for k in range(N):
        x[k + 1] = x[k] + h * f(t[k], x[k])

        if abs(x[k + 1]) > 1e10:
            return t[:k + 2], x[:k + 2], True  # added 'True' for explosion

    return t, x, False  # added 'False' for stability


# --- 2. main logic---
def run_and_plot(t_entry, h_entry, x0_entry, plot_frame, toolbar_frame):
    # 2a. reading parameters (must use float())
    try:
        T = float(t_entry.get())
        h = float(h_entry.get())
        x0 = float(x0_entry.get())
    except ValueError:
        tk.messagebox.showerror("input error", "t, h, and x0 must be !SIMPLE! numbers,ты че еблан?")
        return

    # 2b. clear old plot
    for widget in plot_frame.winfo_children():
        widget.destroy()
    for widget in toolbar_frame.winfo_children():
        widget.destroy()

    # 2c. run the solver
    t, x, exploded = explicit_euler(T, h, x0)

    # 2d. plotting setup
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    # 2e. plot the results
    ax.plot(t, x, label=f'h={h}, x0={x0}')

    # 2f. nice labels and title
    ax.set_xlabel('time $t$')
    ax.set_ylabel('solution  $x(t)$')
    title_status = "exploded pizda... " if exploded else "stable "
    ax.set_title(f'explicit euler result : {title_status} (T={T})')
    ax.grid(True)
    ax.legend()

    # 2g. embed matplotlib into tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 2h. add the toolbar (zoom, pan, save)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)


# --- 3. gui setup ---
root = tk.Tk()
root.title("euler method solver:)")

# 3a. input controls frame
input_frame = ttk.Frame(root, padding="10")
input_frame.pack(fill='x')

# T input
ttk.Label(input_frame, text="T (end time):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
t_entry = ttk.Entry(input_frame, width=10)
t_entry.insert(0, "9")  # default to problem 1a short time
t_entry.grid(row=0, column=1, padx=5, pady=5)

# h input
ttk.Label(input_frame, text="h (time step):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
h_entry = ttk.Entry(input_frame, width=10)
h_entry.insert(0, "0.05")  # default step
h_entry.grid(row=0, column=3, padx=5, pady=5)

# x0 input
ttk.Label(input_frame, text="x0 (start value):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
x0_entry = ttk.Entry(input_frame, width=10)
x0_entry.insert(0, "1.0")  # default start
x0_entry.grid(row=1, column=1, padx=5, pady=5)

# run button
solve_button = ttk.Button(
    input_frame,
    text="CLICK HERE",
    command=lambda: run_and_plot(t_entry, h_entry, x0_entry, plot_frame, toolbar_frame)
)
solve_button.grid(row=1, column=3, padx=5, pady=5)

# 3b. plot area frames
toolbar_frame = ttk.Frame(root)
toolbar_frame.pack(fill='x')
plot_frame = ttk.Frame(root)
plot_frame.pack(fill='both', expand=True)

# 3c. run the app
root.mainloop()