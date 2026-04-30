# SignVox — AI Sign Language Detector

> Real-time sign language recognition system powered by AI.
> Detect hand gestures through camera and convert them into text and voice instantly.

## Features

* **Real-Time Camera Detection**
*  **AI-Based Gesture Recognition**
*  **Text-to-Speech Output**
*  **Live Confidence Score**
*  **Detection History Tracking**
*  **Fast & Responsive UI**
*  **Modern Web Interface**

## Tech Stack

| Technology         | Usage           |
| ------------------ | --------------- |
| Python             | Backend logic   |
| Flask              | Web framework   |
| OpenCV             | Camera handling |
| CVZone / MediaPipe | Hand tracking   |
| TensorFlow / Keras | AI model        |
| HTML/CSS/JS        | Frontend UI     |


##  Installation & Setup

### Clone the repository

git clone https://github.com/Deepika-sharma2008/signvox.git
cd signvox

### Install dependencies

pip install flask opencv-python numpy cvzone tensorflow pyttsx3

###  Run the application

python app.py

### 4️⃣ Open in browser

link eg:-http://127.0.0.1:5000

## How It Works

1. Start the camera from UI
2. Show your hand gesture
3. AI model detects the sign
4. Output displayed as text
5. Voice speaks detected word

## API Endpoints

| Endpoint          | Method | Description        |
| ----------------- | ------ | ------------------ |
| `/`               | GET    | Load UI            |
| `/video_feed`     | GET    | Live camera stream |
| `/get_prediction` | GET    | Get detected sign  |

##  Example Response

json
{
  "word": "A",
  "confidence": 92
}

## Requirements

* Python 3.8+
* Webcam enabled device
* Modern browser (Chrome / Edge recommended)

## Future Improvements

* Full A-Z sign detection
* Multi-language support
* Cloud deployment
* Mobile app version
* Sentence formation

## Author

**DEEPIKA SHARMA**


---
