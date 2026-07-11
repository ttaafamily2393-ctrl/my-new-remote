import numpy as np
import scipy.signal as signal

print("[*] Initializing Gyrophone Micro-vibration Signal Analysis Engine...")

# 1. Simulate a high-frequency mobile sensor stream (sampling at 400Hz)
fs = 400  # Sampling frequency (Hz)
t = np.arange(0, 3, 1/fs)  # 3-second sample timeline track

# Sub-10Hz kinetic movement (tilt/hand shake)
hand_shake = 0.5 * np.sin(2 * np.pi * 1.5 * t)
# 100Hz acoustic leak (simulated conversation vibration caught by MEMS sensor)
acoustic_leak = 0.02 * np.sin(2 * np.pi * 100 * t)
raw_gyro_z = hand_shake + acoustic_leak + np.random.normal(0, 0.005, len(t))

print(f"[+] Ingested {len(raw_gyro_z)} angular velocity data frames across Z-axis.")

# 2. Apply a Butterworth Highpass Filter to remove low-frequency physical tilts (>20Hz)
cutoff = 20.0
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
filtered_gyro = signal.filtfilt(b, a, raw_gyro_z)

# 3. Perform a Fast Fourier Transform (FFT) to parse the hidden frequency peaks
fft_values = np.abs(np.fft.rfft(filtered_gyro))
frequencies = np.fft.rfftfreq(len(filtered_gyro), d=1/fs)
dominant_freq = frequencies[np.argmax(fft_values)]

print("\n================ GYROPHONE SIGNAL FORENSICS ================")
print(f"Kinetic Noise Floor (Raw Mean): {np.mean(np.abs(hand_shake)):.4f} rad/s")
print(f"Isolated Acoustic Frequency:    {dominant_freq:.2f} Hz")
print("[+] Target signal vector cleanly parsed into frequency matrices.")
