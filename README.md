# AI-Enabled Surveillance Platform for Public Parks with Behaviour-Based Safety Monitoring

## Overview

The **AI-Enabled Surveillance Platform for Public Parks** is an AI/ML-based video surveillance system developed as part of **Infosys Springboard Virtual Internship 6.0**.

The project aims to automate human activity recognition and behaviour-based safety monitoring in public park environments using Computer Vision and Deep Learning techniques.

The system processes video streams, extracts human body landmarks, analyses movement patterns, and classifies activities using an LSTM-based sequence learning model. A Flask-based web application provides an interface for video processing, activity prediction, and monitoring.

## Problem Statement

Traditional surveillance systems primarily depend on manual monitoring of CCTV footage. Continuously observing large amounts of video data is time-consuming and makes it difficult to identify potentially unsafe activities promptly.

This project explores an AI-powered approach to video surveillance that can automatically recognize human activities and support more efficient monitoring of public spaces.

## Objectives

- Automate human activity recognition from video footage.
- Extract human body posture and movement information using pose estimation.
- Analyse temporal movement patterns using deep learning.
- Classify activities into predefined categories.
- Provide a web-based interface for video analysis and monitoring.
- Maintain user authentication and activity history.

## Key Features

- **Video Processing:** Extracts frames from uploaded or supported video streams.
- **Human Pose Estimation:** Identifies body landmarks using MediaPipe.
- **Feature Extraction:** Extracts skeletal coordinates and joint-angle information.
- **Activity Recognition:** Uses an LSTM model to learn movement patterns across consecutive frames.
- **Web Interface:** Flask-based application for video processing and prediction.
- **User Authentication:** Provides login functionality.
- **Activity History:** Stores monitoring and prediction records using SQLite.

## System Workflow

1. **Video Input:** Video footage is provided to the system.
2. **Frame Extraction:** OpenCV reads the video and extracts individual frames.
3. **Pose Estimation:** MediaPipe detects human body landmarks in each frame.
4. **Feature Extraction:** Body coordinates and joint angles are extracted to represent human posture.
5. **Sequence Generation:** Features from consecutive frames are grouped into sequences to capture movement over time.
6. **LSTM Classification:** The trained LSTM model analyses the sequences and predicts the corresponding activity class.
7. **Result Display:** The predicted activity is presented through the Flask web application.
8. **Database Storage:** User information and activity history are maintained using SQLite.

## AI/ML Methodology

### 1. Video Frame Extraction – OpenCV

OpenCV is used to read video files and extract frames for further processing.

Each frame is passed to the pose estimation module to identify human body landmarks.

### 2. Human Pose Estimation – MediaPipe

MediaPipe Pose is used to detect human body landmarks, including:

- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

These landmarks provide information about human posture and body positioning.

### 3. Feature Extraction

The detected landmarks are converted into numerical features.

The extracted features include:

- Normalized body landmark coordinates.
- Joint positions.
- Joint angles.
- Movement-related features across consecutive frames.

These features represent the person's posture and movement and are used as input to the activity recognition model.

### 4. Activity Recognition – LSTM

A Long Short-Term Memory (LSTM) network is used for sequential activity recognition.

Unlike models that independently classify individual frames, LSTM can learn temporal relationships between consecutive frames.

**Working:**

1. Extracted features from consecutive frames are grouped into sequences.
2. These sequences are provided as input to the LSTM network.
3. LSTM learns movement patterns and temporal dependencies.
4. The final classification layer predicts the corresponding activity category.

This helps the system analyse how a person's posture changes over time rather than relying only on a single frame.

### 5. Activity Classification

The trained model classifies video sequences into predefined activity categories.

The predictions are displayed through the web application to support automated surveillance and activity monitoring.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe |
| Deep Learning | LSTM, TensorFlow/Keras |
| Feature Processing | NumPy |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Development | VS Code, Jupyter Notebook |

## Application Architecture

The application consists of three major components:

**1. AI/ML Processing Layer**
- Video frame extraction.
- Human pose estimation.
- Feature extraction.
- Sequential activity classification.

**2. Web Application Layer**
- Flask backend.
- Video processing interface.
- Activity prediction display.
- User authentication.

**3. Database Layer**
- SQLite database.
- User information.
- Activity and prediction history.

## Project Outcomes

- Developed an AI-based approach for automated human activity recognition.
- Integrated pose estimation and sequential movement analysis.
- Combined computer vision and deep learning with a web-based surveillance interface.
- Reduced the need for continuous manual observation of video footage.


## Future Enhancements

- Integration with live CCTV camera streams.
- Real-time detection and alert notifications.
- Improved activity classification using larger and more diverse datasets.
- Multi-person tracking and activity recognition.
- Deployment on edge devices for low-latency surveillance.
- Integration with cloud-based monitoring systems.



This project was developed for educational and internship purposes. The system is intended as a prototype for AI-assisted surveillance and is not a replacement for human supervision or emergency response systems.
