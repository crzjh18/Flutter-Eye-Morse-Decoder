# 👁️ Hybrid Eye-Blink Morse Code Converter (Thesis Project)

A real-time mobile application designed to assist individuals with motor impairments (ALS, LIS) by converting eye blinks into speech. 

This project utilizes a **Hybrid Ensemble Strategy**, combining a custom Convolutional Neural Network (CNN) with Google's ML Kit Face Detection to achieve high-accuracy blink detection, even in varying lighting conditions.

## 🚀 Key Features

* **🧠 Hybrid AI Engine:** Uses a voting system between a TFLite CNN model (trained on open/closed eyes) and ML Kit's geometric landmarks for 99% reliability.
* **🔒 Focus Lock:** Intelligent head-pose detection prevents accidental typing when the user looks away (yaw > 30°).
* **🗣️ Morse-to-Speech:** automatically converts dot/dash sequences into spoken words using Text-to-Speech (TTS).
* **⚡ Smart Shortcuts:** Users can map specific blink patterns (e.g., `....`) to full phrases (e.g., "WATER") for rapid communication.
* **🔊 Audio Feedback:** Distinct beep tones for Dots vs. Dashes to guide the user without looking at the screen.
* **📚 Integrated Learning:** Built-in Morse Code cheatsheet for quick reference.

## 🛠️ Tech Stack

* **Framework:** Flutter (Dart)
* **Machine Learning:** TensorFlow Lite (Custom CNN), Google ML Kit (Face Mesh)
* **State Management:** `setState` (Optimized for high-frequency camera stream updates)
* **Audio:** `flutter_tts` (Speech), `audioplayers` (Feedback tones)

## 📦 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    ```

2.  **Install dependencies**
    ```bash
    flutter pub get
    ```

3.  **Add Assets**
    Ensure the following files are in your `assets/` folder (not included in repo for privacy/size reasons):
    * `eye_state_cnn.tflite` (The trained model)
    * `beep.mp3` (Audio feedback file)

4.  **Run on Device**
    (Note: This requires a physical device with a camera; it will not work on iOS/Android Simulators).
    ```bash
    flutter run --release
    ```

## 📐 System Architecture

The app processes camera frames at 30FPS:
1.  **Face Detection:** Locates the face and checks Head Euler Y (Yaw).
2.  **ROI Cropping:** Extracts the eye region (64x64 grayscale).
3.  **Inference:** Runs the CNN model to get raw logits (`Closed Score` vs `Open Score`).
4.  **Ensemble Vote:** Compares CNN result with ML Kit's `eyeOpenProbability`.
5.  **Logic Engine:** Converts time-series blink data into Dots (`.`) or Dashes (`-`).

## 🎓 Thesis Context

This software was developed as part of a Computer Science thesis to explore "Low-Latency Eye Tracking on Mobile Devices using Hybrid Neural Networks."

---
*Developed by Me
