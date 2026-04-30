from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque
import pyttsx3
import threading
from queue import Queue

# ================= SPEECH =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# ================= APP =================
app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier(
    "D:/converted_keras/keras_model.h5",
    "D:/converted_keras/labels.txt"
)

labels = open("D:/converted_keras/labels.txt").read().splitlines()

imgSize = 300
offset = 20

buffer = deque(maxlen=5)

latest_prediction = {"word": "", "confidence": 0}
last_spoken = ""

# ================= CAMERA =================
def generate_frames():
    global latest_prediction, last_spoken

    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    new_w = int(w * k)
                    imgResize = cv2.resize(imgCrop, (new_w, imgSize))
                    gap = (imgSize - new_w) // 2
                    imgWhite[:, gap:gap + new_w] = imgResize
                else:
                    k = imgSize / w
                    new_h = int(h * k)
                    imgResize = cv2.resize(imgCrop, (imgSize, new_h))
                    gap = (imgSize - new_h) // 2
                    imgWhite[gap:gap + new_h, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                confidence = float(max(prediction))

                # ===== BUFFER FIX =====
                if len(buffer) == 0 or index != buffer[-1]:
                    buffer.clear()

                buffer.append(index)

                if len(buffer) >= 4:
                    stable_idx = max(set(buffer), key=buffer.count)
                    detected_word = labels[stable_idx]

                    latest_prediction["word"] = detected_word
                    latest_prediction["confidence"] = round(confidence * 100, 2)

                    # ===== SPEAK =====
                    if detected_word != last_spoken:
                        speech_queue.put(detected_word)
                        last_spoken = detected_word

                    cv2.putText(imgOutput, detected_word,
                                (x, y - 20),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.5, (0, 0, 0), 2)

                    cv2.rectangle(imgOutput, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 3)

        _, buffer_img = cv2.imencode('.jpg', imgOutput)
        frame = buffer_img.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify(latest_prediction)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)