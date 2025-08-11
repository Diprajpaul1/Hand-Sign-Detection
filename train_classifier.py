# train_model.py

import os, pickle, cv2, numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

# MediaPipe Setup
mp_hands = mp.solutions.hands

# Load and process dataset
data, labels = [], []
DATA_DIR = './data'

for label_folder in sorted(os.listdir(DATA_DIR), key=int):
    folder = os.path.join(DATA_DIR, label_folder)
    for img_file in tqdm(os.listdir(folder), desc=f"Processing {label_folder}"):
        img = cv2.imread(os.path.join(folder, img_file))
        if img is None:
            continue
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0]
        xs = [pt.x for pt in lm.landmark[:21]]
        ys = [pt.y for pt in lm.landmark[:21]]
        sample = []
        for pt in lm.landmark[:21]:
            sample += [pt.x - min(xs), pt.y - min(ys)]
        if len(sample) == 42:
            data.append(sample)
            labels.append(int(label_folder))

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# ✅ Accuracy
train_accuracy = clf.score(x_train, y_train)
test_accuracy = clf.score(x_test, y_test)
print(f"✅ Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# Save trained model
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z
pickle.dump({'model': clf, 'labels_dict': labels_dict}, open('model.p', 'wb'))
print("✅ Model trained and saved as 'model.p'")

# Predict
y_pred = clf.predict(x_test)

# Used labels (for classes that actually appear in y_test)
used_labels = sorted(list(unique_labels(y_test, y_pred)))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=used_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[labels_dict[i] for i in used_labels],
            yticklabels=[labels_dict[i] for i in used_labels])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ✅ Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=used_labels,
    target_names=[labels_dict[i] for i in used_labels]
))

# ✅ Precision, Recall, F1 Score
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nPrecision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ✅ Accuracy vs Training Size (example plot)
train_sizes = [100, 200, 300, 400, 500][:len(x_train)//100]  # Adjusted if dataset is smaller
accuracies = []

for size in train_sizes:
    clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_temp.fit(x_train[:size], y_train[:size])
    acc = clf_temp.score(x_test, y_test)
    accuracies.append(acc)

plt.figure()
plt.plot(train_sizes, accuracies, marker='o', linestyle='-', color='green')
plt.title('Accuracy vs. Training Size')
plt.xlabel('Training Sample Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
