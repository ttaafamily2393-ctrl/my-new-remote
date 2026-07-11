import numpy as np
import scipy.signal as signal

print("[*] Initializing EMG Digital Signal Processing (DSP) Pipeline...")

# 1. Generate an explicit dummy EMG dataset (simulating 1000Hz hardware sampling)
fs = 1000  # Sampling Frequency (Hz)
t = np.arange(0, 5, 1/fs)
raw_emg = np.random.normal(0, 0.2, size=len(t)) 
raw_emg[2000:3500] *= 4  # Inject a contraction spike in the middle of the timeline

print(f"[+] Data points ingested: {len(raw_emg)} samples across {t[-1]:.1f} seconds.")

# 2. Apply a 4th-order Butterworth Bandpass Filter (20Hz to 450Hz baseline limits)
lowcut = 20.0
highcut = 450.0
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b_band, a_band = signal.butter(4, [low, high], btype='band')
filtered_emg = signal.filtfilt(b_band, a_band, raw_emg)

# 3. Apply a Notch Filter to completely choke out 60Hz AC electrical line noise
notch_freq = 60.0  
q_factor = 30.0    
b_notch, a_notch = signal.iirnotch(notch_freq / nyq, q_factor)
cleaned_emg = signal.filtfilt(b_notch, a_notch, filtered_emg)

# 4. Calculate the Root Mean Square (RMS) Envelope to track exact muscle activation
window_size = 100  # 100ms moving window
rms_envelope = np.sqrt(np.convolve(cleaned_emg**2, np.ones(window_size)/window_size, mode='same'))

print("\n================= EMG FORENSIC DATA METRICS =================")
print(f"Baseline Noise Floor (RMS):   {np.mean(rms_envelope[:1000]):.4f}")
print(f"Maximum Contraction Power:    {np.max(rms_envelope):.4f}")
print("[+] Processing complete. Matrix arrays ready for feature extraction.")
