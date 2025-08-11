import os
import cv2
import string

# Directory to store data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 26 alphabets: A-Z → labeled as 0–25
number_of_classes = 26
dataset_size = 10  # Images per class

# Start webcam (change index 2 to 0 if your default camera is different)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera not found or cannot be opened!")

# Map index to letter (0 → A, ..., 25 → Z)
index_to_char = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for label {j} (letter: {index_to_char[j]})')

    # Wait for user to press 'Q' to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Get ready for: {index_to_char[j]} (label {j}) - Press "Q" to start!',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture dataset_size images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

print("✅ Data collection complete.")
cap.release()
cv2.destroyAllWindows()