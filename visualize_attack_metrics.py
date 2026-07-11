import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("[*] Processing matrix arrays for visualization panels...")
# Load datasets directly from the extracted matrices
train_attack = scipy.io.loadmat('CSTR_train_attack.mat')['CSTR_train_attack']
train_noise = scipy.io.loadmat('CSTR_train_noise.mat')['CSTR_train_noise']
train_normal = scipy.io.loadmat('CSTR_train_normal.mat')['CSTR_train_normal']

test_attack = scipy.io.loadmat('CSTR_test_attack.mat')['CSTR_test_attack']
test_noise = scipy.io.loadmat('CSTR_test_noise.mat')['CSTR_test_noise']
test_normal = scipy.io.loadmat('CSTR_test_normal.mat')['CSTR_test_normal']

X_train = np.vstack((train_attack[:, :200], train_noise[:, :200], train_normal[:, :200]))
X_test = np.vstack((test_attack[:, :200], test_noise[:, :200], test_normal[:, :200]))

y_train = np.hstack((np.zeros(len(train_attack)), np.ones(len(train_noise)), np.ones(len(train_normal)) * 2))
y_test = np.hstack((np.zeros(len(test_attack)), np.ones(len(test_noise)), np.ones(len(test_normal)) * 2))

# Train the Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Setup a clean 1x2 plotting grid configuration layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

print("[*] Compiling Panel 1: Confusion Matrix...")
# 1. Renders the Confusion Matrix Layout
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Attack', 'Noise', 'Normal'])
disp.plot(ax=ax1, cmap='Purples', values_format='d')
ax1.set_title("CSTR Attack Confusion Matrix")

print("[*] Compiling Panel 2: Sensor Feature Importances...")
# 2. Extract and Plot top 10 most critical reactor sensor columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

ax2.bar(range(10), importances[indices], color='#7B4F3AD1', align='center')
ax2.set_xticks(range(10))
ax2.set_xticklabels([f"Sensor {i}" for i in indices], rotation=45)
ax2.set_title("Top 10 Most Critical Reactor Sensors")
ax2.set_ylabel("Importance Score Metric")

plt.tight_layout()
print("[+] Saving data charts cleanly to disk path: cstr_detection_charts.png")
plt.savefig('cstr_detection_charts.png', dpi=300)
print("[*] Execution complete.")
