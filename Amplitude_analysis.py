#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def sine_fit_func(t, A, phi, offset, f):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

def analyze_file(file_path, freq):
    try:
        # Read comma-separated data; files have no header.
        data = np.loadtxt(file_path, delimiter=",")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    # Assume columns: 0 = time, 1 = V_acc, 2 = V_in
    t = data[:, 0]
    V_acc = data[:, 1]
    V_in = data[:, 2]
    
    # --- Fit V_acc (blue curve) ---
    p0_acc = [(np.max(V_acc)-np.min(V_acc))/2, 0, np.mean(V_acc)]   # initial guess for [amplitude, phase, dc offset]
    try:
        popt_acc, _ = curve_fit(
            lambda t, A, phi, offset: sine_fit_func(t, A, phi, offset, freq),
            t, V_acc, p0=p0_acc,
            bounds=([0, -np.pi, -np.inf], [np.inf, np.pi, np.inf])  # Force amplitude A >= 0
        )
    except Exception as e:
        print(f"Fit failed for V_acc in {file_path}: {e}")
        return None
    A_acc, phi_acc, offset_acc = popt_acc

    # --- Fit V_in (red curve) ---
    p0_in = [(np.max(V_in)-np.min(V_in))/2, 0, np.mean(V_in)]   # initial guess for [amplitude, phase, dc offset]
    try:
        popt_in, _ = curve_fit(
            lambda t, A, phi, offset: sine_fit_func(t, A, phi, offset, freq),
            t, V_in, p0=p0_in,
            bounds=([0, -np.pi, -np.inf], [np.inf, np.pi, np.inf])  # Force amplitude A >= 0
        )
    except Exception as e:
        print(f"Fit failed for V_in in {file_path}: {e}")
        return None
    A_in, phi_in, offset_in = popt_in

    # Compute constant phase difference 
    print("Phi_acc: ", phi_acc)
    print("Phi_in: ", phi_in)
    dphi = phi_acc - phi_in
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # Normalize to [-π, π]
    
    # Generate fitted curves for plotting
    t_fit = np.linspace(np.min(t), np.max(t), 1000)
    V_acc_fit = sine_fit_func(t_fit, A_acc, phi_acc, offset_acc, freq)
    V_in_fit = sine_fit_func(t_fit, A_in, phi_in, offset_in, freq)

    # Print the fit parameters
    print("Fit parameters for V_acc:")
    print(f"  Amplitude: {A_acc:.4f}, Phase: {phi_acc:.4f} rad, Offset: {offset_acc:.4f}")
    print("Fit parameters for V_in:")
    print(f"  Amplitude: {A_in:.4f}, Phase: {phi_in:.4f} rad, Offset: {offset_in:.4f}")
    print(f"Constant Phase Difference (V_acc - V_in): {dphi:.4f} rad")

    # Plot the time series data with fits
    plt.figure(figsize=(10,6))
    plt.plot(t, V_acc, 'bo', label='V_acc data')
    plt.plot(t, V_in, 'ro', label='V_in data')
    plt.plot(t_fit, V_acc_fit, 'b-', label='V_acc fit')
    plt.plot(t_fit, V_in_fit, 'r-', label='V_in fit')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Time Series Analysis at {freq} Hz\nFile: {os.path.basename(file_path)}")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the specified directory
    save_dir = '/Users/test1/Desktop/Python/Lab 6/Plots'
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = f"Amplitude_{int(freq)}Hz.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()
    
    return (A_acc, phi_acc, offset_acc, A_in, phi_in, offset_in, dphi)

def main():
    # Directory where files are stored.
    data_dir = '/Users/test1/Desktop/Python/Lab 6/Data/Max Damping'
    
    # Let the user specify which frequency file to analyze (default is 12 Hz).
    try:
        freq_input = input("Enter frequency (Hz) to analyze (default 12): ")
        freq = 12.0 if freq_input.strip() == "" else float(freq_input)
    except Exception as e:
        print("Invalid input; defaulting to 12 Hz.")
        freq = 12.0

    # Construct file name. Expected format: "Lab_6Vacc_Vin_xhz.lvm" where x is the frequency.
    file_name = f"Lab_6Vacc_Vin_{freq:.1f}hz.lvm"
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    analyze_file(file_path, freq)

if __name__ == "__main__":
    main()
