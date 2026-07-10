import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("[*] Loading CSTR Anomaly Dataset...")
# Load datasets directly from the extracted matrices
train_attack = scipy.io.loadmat('CSTR_train_attack.mat')['CSTR_train_attack']
train_noise = scipy.io.loadmat('CSTR_train_noise.mat')['CSTR_train_noise']
train_normal = scipy.io.loadmat('CSTR_train_normal.mat')['CSTR_train_normal']

test_attack = scipy.io.loadmat('CSTR_test_attack.mat')['CSTR_test_attack']
test_noise = scipy.io.loadmat('CSTR_test_noise.mat')['CSTR_test_noise']
test_normal = scipy.io.loadmat('CSTR_test_normal.mat')['CSTR_test_normal']

# Strip trailing labels and build feature spaces (200 dimensions per sample)
X_train = np.vstack((train_attack[:, :200], train_noise[:, :200], train_normal[:, :200]))
X_test = np.vstack((test_attack[:, :200], test_noise[:, :200], test_normal[:, :200]))

# Create clean integer target label streams (0 = Attack, 1 = Noise, 2 = Normal)
y_train = np.hstack((np.zeros(len(train_attack)), np.ones(len(train_noise)), np.ones(len(train_normal)) * 2))
y_test = np.hstack((np.zeros(len(test_attack)), np.ones(len(test_noise)), np.ones(len(test_normal)) * 2))

print(f"[+] Data successfully unified.")
print(f"    -> Training Vector Space: {X_train.shape}")
print(f"    -> Testing Vector Space:  {X_test.shape}")

print("[*] Training Hardware-Safe Random Forest Classifier...")
# Spawn an optimized Celeron-compatible model mapping all available CPU cores
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("[*] Evaluating Performance Metrics...")
y_pred = model.predict(X_test)

print("\n================ CYBER ATTACK DETECTION REPORT ================")
print(f"Classification Performance Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nStatistical Analysis Breakdown:")
print(classification_report(y_test, y_pred, target_names=['Attack Anomaly', 'Background Noise', 'Normal Baseline']))
