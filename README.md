
# Real-Time Classroom Concentration Analysis System

A Python-based desktop application developed as my Final Year Project to explore real-time computer vision and eye movement analysis.

## Overview

The application uses a webcam to detect faces, identify eye landmarks, monitor blinking and eye closure, and display an estimated concentration indicator.

The concentration indicator is based on custom eye-movement heuristics. It is not a scientifically validated measurement of attention.

## Technologies Used

- Python
- OpenCV
- dlib
- NumPy
- SciPy
- Tkinter
- Pillow
- imutils

## Key Features

- Real-time webcam video processing
- Face detection using Haar Cascade
- Facial landmark detection using dlib
- Eye Aspect Ratio (EAR) calculation
- Blink and eye-closure detection
- Estimated concentration indicator
- Desktop interface with Start and Stop controls

## How It Works

1. Capture video frames from a webcam.
2. Detect faces and facial landmarks.
3. Calculate Eye Aspect Ratio values.
4. Detect blinking and eye-closure events.
5. Update a heuristic concentration indicator.
6. Display the live results in a Tkinter interface.

## Requirements

Python and the required libraries must be installed.

The application also requires:

- `haarcascade_frontalface_default.xml`
- `haarcascade_eye.xml`
- `shape_predictor_68_face_landmarks.dat`

These model files must be available at the paths expected by the source code.

## Running the Project

Install the required Python dependencies and model files, then run the main Python script.

A working webcam is required.

Note: This is an academic prototype. Compatibility and setup may depend on the operating system and Python environment.

## What I Learned

This project gave me hands-on experience combining computer vision, real-time processing, application logic, and desktop UI development into a working prototype.

It also strengthened my debugging and problem-solving skills.
