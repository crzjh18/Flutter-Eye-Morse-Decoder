# Decoding Morse Code via Eye Blinks using CNN and LSTM Machine Learning Algorithms

**A Thesis Project Presented to the Faculty of the College of Computing Studies, EARIST Manila**

This project details a **modular deep learning pipeline** designed to translate intentional eye blinks into Morse code and subsequently into textual output. Anchored on Sustainable Development Goals (SDG) 9.5 and 10.2, this system aims to provide an accessible, low-cost communication tool for individuals with severe motor impairments (e.g., ALS, Quadriplegia).

The system leverages a **Convolutional Neural Network (CNN)** for robust spatial eye-state classification and a **Long Short-Term Memory (LSTM)** network for temporal blink-event modeling, ensuring reliable decoding even without rigid thresholding.

## Current Development Status (Mobile)

> **Developer Note:**
> The mobile application implementation is currently a **Work in Progress**.
> * **LSTM Integration:** I am currently working on the LSTM module for this mobile app. I am encountering challenges converting the trained LSTM model from **ONNX to TensorFlow Lite (TFLite)**, which is required for the mobile inference engine.
> * **CNN Updates:** I will also be adding the latest, more accurate CNN model and changing the parameters according to it in a future update.

## Key Features

* **Hybrid CNN-LSTM Architecture:** Combines spatial feature extraction with temporal sequence modeling to distinguish between voluntary "Dots," "Dashes," and involuntary blinks.
* **Robust Eye Tracking:** Utilizes **MediaPipe Face Mesh** for precise, real-time eye-region localization and cropping, adaptable to varied lighting and head poses.
* **Adaptive Temporal Learning:** The LSTM module autonomously learned a decision boundary of **0.49 seconds** for distinguishing dots and dashes, eliminating the need for manual threshold tuning.
* **Ultra-Low Latency:** Achieves an average inference speed of **0.88 ms per frame** on benchmark tests, ensuring seamless real-time performance well below the 33ms limit for 30FPS video.
* **Text-to-Speech (TTS):** Converts decoded text into auditory feedback to assist users in communicating effectively.

## Tech Stack

* **Mobile Framework:** Flutter (Dart)
* **Deep Learning Models:**
    * **CNN Backbone:** MobileNetV3-Small (Modified for Grayscale Input)
    * **Sequence Model:** Bidirectional LSTM
* **Computer Vision:** OpenCV, MediaPipe Face Mesh
* **Training Environment:** PyTorch, Google Colab (GPU)
* **Audio:** `flutter_tts`

## System Architecture

The system follows a modular deep-learning pipeline:

1.  **Face & Eye Localization:** The system uses **MediaPipe Face Mesh** to extract eye landmarks and crop the eye region from the video frame.
2.  **Spatial Classification (CNN):** A customized **MobileNetV3-Small** CNN processes the cropped frame (64x64 grayscale) to classify the eye state as "Open" or "Closed".
    * *Performance:* 90.71% Accuracy, 99.63% Precision for Closed Eyes.
3.  **Temporal Modeling (LSTM):** The sequence of eye states is buffered and fed into the LSTM. The LSTM detects the full blink transition (`open -> closed -> open`) and classifies the duration as a "Dot" or "Dash".
4.  **Decoding:** A rule-based decoder maps the sequence of dots and dashes to alphanumeric characters based on International Morse Code standards.

## How to Run (Mobile)

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    ```

2.  **Install dependencies**
    ```bash
    flutter pub get
    ```

3.  **Add Assets**
    Ensure the following model files are in your `assets/` folder:
    * `cnn_model.tflite` (Current CNN implementation)
    * *(Note: LSTM model integration is pending conversion fix)*

4.  **Run on Device**
    (Requires a physical device with a camera).
    ```bash
    flutter run --release
    ```

## Thesis Results

* **CNN Precision (Closed Class):** 99.63% — Highly conservative to prevent "phantom" inputs.
* **Symbol Error Rate (SER):** 0.0000 on validation dataset.
* **Decision Boundary:** The model learned to split Dots and Dashes at **0.49s** (Theoretical optimal: 0.50s).

## Authors

* **Cruz, Josh Harold H.**
* **Narciso, Victor Prince**
* **Palagam, Erika Van Imeren A.**
* **Rosales, Aries P.**

**Thesis Adviser:** Dr. Jesus S. Paguigan

---
*Developed by Group 2, BSCS 4th Year, EARIST Manila*
