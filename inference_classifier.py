import pickle, cv2, numpy as np, mediapipe as mp, time

# Load model
md = pickle.load(open('model.p','rb'))
model, labels_dict = md['model'], md['labels_dict']
print("Model expects:", model.n_features_in_, "features")

# MediaPipe and webcam
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit("Camera not working")

sentence = ""
prev_char = ""
last_update_time = 0
delay = 1.5  # seconds between accepting new letters

while True:
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        xs = [pt.x for pt in lm.landmark[:21]]
        ys = [pt.y for pt in lm.landmark[:21]]
        sample = []
        for pt in lm.landmark[:21]:
            sample += [pt.x - min(xs), pt.y - min(ys)]

        if len(sample) == model.n_features_in_:
            pred = model.predict([sample])[0]
            char = labels_dict.get(pred, '?')
            x1, y1 = int(min(xs)*W)-20, int(min(ys)*H)-20
            cv2.rectangle(frame, (x1,y1), (int(max(xs)*W)+20, int(max(ys)*H)+20), (0,255,0), 2)
            cv2.putText(frame, char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            # Add to sentence if enough time has passed
            current_time = time.time()
            if char != prev_char and (current_time - last_update_time) > delay:
                sentence += char
                prev_char = char
                last_update_time = current_time

    # Display current sentence
    cv2.rectangle(frame, (10, 10), (W - 10, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("Hand Sign Detection - Press Q to Quit, C to Clear", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = sentence[:-1]

        prev_char = ""

cap.release()
cv2.destroyAllWindows()