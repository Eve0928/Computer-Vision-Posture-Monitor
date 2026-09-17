# 🖥️  Computer Vision Posture Monitor 

An end-to-end computer vision application that tracks desk ergonomics and focus time in real-time. Built with a YOLOv8 pose estimation model and OpenCV, this tool analyzes webcam feeds to detect 'lazy posture' and track continuous focus. Runs strictly locally to ensure complete data privacy.

## ✨ Key Features

*   **Real-Time Pose Tracking:** Uses YOLOv8 to track specific facial and shoulder keypoints at high framerate, also to calibrate the baseline posture.
*   **Scale-Invariant Math:** Normalizes the vertical drop between your ears and shoulders against your shoulder width; this ensures tracking works whether you sit close to or far from the monitor.
*   **Smart Alert System:** Triggers a visual warning and a Mac-native audio alarm (chime and voice) after 3 seconds of sustained slouching to prevent false positives.
*   **Privacy First:** Processes all video frames locally on your machine with no cloud storage or external API calls. Detects your posture through your computer/laptop webcam.

## Installation

You will need Python installed on your machine. Open your terminal and run the following command to install the required dependencies:

    pip3 install ultralytics opencv-python numpy

## How to Use

1.  Clone this repository to your laptop/computer and navigate to the project folder.
2.  Change 'Evelyn' to 'Your Name' in the main.py file
3.  Run the main script in your terminal: `python3 main.py`
4.  When the webcam window opens, sit upright and press **c** on your keyboard to calibrate your baseline posture.
4.  Press **q** at any time to close the video window and exit the application.

## Project Structure

*   `config.py`: Acts as the control panel; it holds all hardcoded settings, model paths, and YOLO keypoint mappings.
*   `tracker.py`: The logic engine; it handles raw data extraction and calculates the normalized posture metric.
*   `main.py`: The main application loop; it manages the webcam hardware, AI inference, and visual user interface.