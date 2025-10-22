import numpy as np
import matplotlib.pyplot as plt


# --- 1. function f(t, x) ---
def f(t, x):
    """the right side of the ODE: dx/dt = t - x^2"""
    return t - x ** 2


# --- 2. euler method solver ---
def explicit_euler(T, h, x0):
    """solves the ODE from 0 to T with step h and initial condition x0."""
    N = int(T / h)
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0

    for k in range(N):
        x[k + 1] = x[k] + h * f(t[k], x[k])

        # simple stability check: if it blows up, we stop.
        if abs(x[k + 1]) > 1e10:
            print(f"  oops: solution exploded around t={t[k + 1]:.2f}. aborting run.")
            return t[:k + 2], x[:k + 2]

    return t, x


# --- 3. plotting function for problem 1a ---
def solve_and_plot_1a(T, h, x0_list):
    """solves for multiple x0 and plots them on one figure."""
    plt.figure(figsize=(10, 6))
    plt.suptitle(f'problem 1a: effect of initial condition (T={T}, h={h})', fontsize=14)
    print(f"\n--- starting calculations for problem 1a: T={T}, h={h} ---")

    for x0 in x0_list:
        t, x = explicit_euler(T, h, x0)
        plt.plot(t, x, label=f'$x_0 = {x0}$')
        print(f"  done with $x_0 = {x0}$. final value x({T})={x[-1]:.4f}")

    plt.xlabel('time $t$')
    plt.ylabel('solution $x(t)$')
    plt.grid(True)
    plt.legend()
    plt.show()


# --- 4. plotting function for problems 1b and 1c ---
def solve_and_plot_1bc(T, x0, h_b, h_c):
    """compares stability for two different step sizes (h_b and h_c)."""

    # 1b calculation (big step)
    t_b, x_b = explicit_euler(T, h_b, x0)
    print(f"\n--- starting calculations for problem 1b: T={T}, h={h_b} (expecting trouble) ---")

    # 1c calculation (small step)
    t_c, x_c = explicit_euler(T, h_c, x0)
    print(f"\n--- starting calculations for problem 1c: T={T}, h={h_c} (this should be okay) ---")

    # Plotting setup
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'problem 1b and 1c: explicit euler stability check (T={T}, x0={x0})', fontsize=14)

    # Plot 1b
    plt.subplot(1, 2, 1)
    plt.plot(t_b, x_b, label=f'$h = {h_b}$')
    plt.title(f'1b: big step $h={h_b}$')
    plt.xlabel('time $t$')
    plt.ylabel('solution $x(t)$')
    plt.grid(True)
    plt.legend()
    plt.ylim(min(x_b) if x_b.size > 0 else -1, max(x_b) if x_b.size > 0 else 1)

    # Plot 1c
    plt.subplot(1, 2, 2)
    plt.plot(t_c, x_c, label=f'$h = {h_c}$')
    plt.title(f'1c: smaller step $h={h_c}$')
    plt.xlabel('time $t$')
    plt.ylabel('solution $x(t)$')
    plt.grid(True)