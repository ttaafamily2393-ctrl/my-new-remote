import socket
import numpy as np
import subprocess
import datetime
import time

ALERT_EMAIL = "ttaafamily2393@gmail.com"
MAGNITUDE_THRESHOLD = 0.85
COOLDOWN_SECONDS = 300  # 5-minute strict cooldown threshold restriction

last_alert_time = 0

def send_emergency_email(peak_val):
    global last_alert_time
    current_time = time.time()
    
    # Enforce strict calculation boundary limits
    if current_time - last_alert_time < COOLDOWN_SECONDS:
        print(f"[i] Cooldown active. Suppressing email transmission. Peak: {peak_val:.4f}")
        return
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = "🚨 ALERT: Sustained RF Anomaly Logged on Workstation-7"
    body = (
        f"Time Vector: {timestamp}\n"
        f"Hardware Sensor: Nooelec NESDR v5 SMArt\n"
        f"Intercepted Amplitude Magnitude Peak: {peak_val:.4f}\n\n"
        f"[!] WARNING: A high-energy continuous wave or burst signal was captured. "
        f"The system has locked outbound emails for 5 minutes to prevent rate limiting."
    )
    
    email_message = f"To: {ALERT_EMAIL}\nSubject: {subject}\n\n{body}"
    
    try:
        process = subprocess.Popen(['msmtp', ALERT_EMAIL], stdin=subprocess.PIPE, text=True)
        process.communicate(input=email_message)
        last_alert_time = current_time
        print(f"[+] Emergency Alert successfully dispatched to {ALERT_EMAIL}")
    except Exception as e:
        print(f"[-] Failed to execute mail transmission: {e}")

print("[*] Fusing Rate-Limited Radio Interceptor Core with Outbound SMTP Delivery...")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    server_socket.connect(('127.0.0.1', 1234))
    print("[+] RF Alert Matrix fully active. Passively sniffing local airspace fields...")
    
    while True:
        raw_data = server_socket.recv(8192)
        if not raw_data:
            break
            
        data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
        data = (data - 127.5) / 127.5
        
        I_samples = data[0::2]
        Q_samples = data[1::2]
        magnitude = np.sqrt(I_samples**2 + Q_samples**2)
        peak = np.max(magnitude)
        
        if peak >= MAGNITUDE_THRESHOLD:
            send_emergency_email(peak)
            
except ConnectionRefusedError:
    print("[-] CRITICAL: Connection refused. Ensure rtl_tcp is actively running on port 1234 first!")
except KeyboardInterrupt:
    print("\n[*] Deactivating radio alert matrix pipelines safely.")
finally:
    server_socket.close()
