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

def regularize_gyro_timestamps(raw_timestamps, raw_data, target_fs=400):
    import scipy.interpolate as interpolate
    print("[*] Technique 1: Regularizing erratic sensor timestamps...")
    t_fixed = np.arange(raw_timestamps[0], raw_timestamps[-1], 1/target_fs)
    interp_func = interpolate.interp1d(raw_timestamps, raw_data, kind='linear')
    return t_fixed, interp_func(t_fixed)

def isolate_acoustic_band(data, fs=400):
    print("[*] Technique 2: Applying sharp Chebyshev Highpass isolation...")
    # Cut off all human kinetic hand motion below 30 Hz
    b, a = signal.cheby1(4, rp=0.5, Wn=30.0/(0.5*fs), btype='high')
    return signal.filtfilt(b, a, data)

def match_acoustic_signature(signal_a, template_b):
    print("[*] Technique 3: Running signature cross-correlation match...")
    correlation = signal.correlate(signal_a, template_b, mode='same')
    return np.max(np.abs(correlation))

def generate_spectrogram_matrix(data, fs=400):
    print("[*] Technique 4: Compiling time-frequency STFT spectrogram...")
    f, t_spec, Zxx = signal.stft(data, fs=fs, nperseg=64)
    print(f"[+] Matrix constructed. Frequency bins: {len(f)}, Time segments: {len(t_spec)}")
    return f, t_spec, np.abs(Zxx)
