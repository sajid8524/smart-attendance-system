# smart-attendance-system

# Smart Attendance System using Face Recognition

An automated attendance system that uses face recognition to identify students from classroom video sessions and generate attendance reports automatically.

## Features

- Automated attendance marking using face recognition
- Face detection and recognition using ArcFace
- Blur filtering and duplicate-face removal
- Attendance report generation in Excel format
- Automatic email reporting to faculty
- Supports webcam, IP camera, and video file input

## Tech Stack

- Python
- OpenCV
- DeepFace (ArcFace)
- NumPy
- Pandas
- smtplib

## Project Structure

```bash
Smart-Attendance-System/
│
├── data/
├── build_db.py
├── run_recognition.py
├── send_email.py
├── face_db.pkl
├── requirements.txt
└── README.md
Workflow
Build face embedding database from student videos
Detect and recognise faces from classroom session
Match identities using cosine similarity
Generate attendance report in Excel format
Automatically send report through email
How to Run
Install Dependencies
pip install -r requirements.txt
Build Face Database
python build_db.py
Run Recognition System
python run_recognition.py
Send Attendance Report
python send_email.py
