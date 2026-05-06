import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace
from collections import deque

# ---------------- PATHS ----------------
DB_PATH = "db/face_db.pkl"
# ---------------- DYNAMIC OUTPUT FILE ----------------
now = datetime.now()
filename = now.strftime("attendance_%Y-%m-%d_%H-%M-%S.xlsx")
OUTPUT_FILE = os.path.join("output", filename)

os.makedirs("output", exist_ok=True)

# ---------------- LOAD DB ----------------
with open(DB_PATH, "rb") as f:
    database = pickle.load(f)

# ---------------- SETTINGS ----------------
THRESHOLD = 0.6
FRAME_SKIP = 2

# stability buffer (prevents flicker)
recent_predictions = deque(maxlen=5)

# ---------------- ATTENDANCE ----------------
def mark_attendance(name):
    time_now = datetime.now().strftime("%H:%M:%S")
    date_today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_excel(OUTPUT_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if ((df["Name"] == name) & (df["Date"] == date_today)).any():
        return

    new_entry = pd.DataFrame([[name, date_today, time_now]],
                             columns=["Name", "Date", "Time"])

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(OUTPUT_FILE, index=False)

# ---------------- SIMILARITY ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- INPUT ----------------
print("\nSelect Input Source:")
print("1: Webcam")
print("2: Video File")
print("3: Image")
print("4: Mobile Camera (IP Webcam)")

choice = input("Enter choice: ")

if choice == "1":
    cap = cv2.VideoCapture(0)

elif choice == "2":
    path = input("Enter video path: ")
    cap = cv2.VideoCapture(path)

elif choice == "3":
    path = input("Enter image path: ")
    frame = cv2.imread(path)
    cap = None

elif choice == "4":
    url = input("Enter mobile stream URL (e.g. http://192.168.x.x:8080/video): ")
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

else:
    print("Invalid choice")
    exit()

frame_count = 0

# ---------------- MAIN LOOP ----------------
while True:
    if cap:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    # resize for speed
    #frame = cv2.resize(frame, (640, 480))
    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    for (x, y, w, h) in faces:
        pad = 10
        face = frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

        if face.size == 0:
            continue

        try:
            rep = DeepFace.represent(
                img_path=face,
                model_name="ArcFace",
                enforce_detection=False
            )

            emb = np.array(rep[0]["embedding"])
            emb = emb / np.linalg.norm(emb)

            best_match = "Unknown"
            best_score = -1

            for person, data in database.items():
                scores = [
                cosine_similarity(emb, db_emb)
                for db_emb in data["embeddings"]
                ]

                score = max(scores) 

                if score > best_score:
                    best_score = score
                    best_match = person

            if best_score < THRESHOLD:
                best_match = "Unknown"

        except Exception as e:
            print("Embedding error:", e)
            best_match = "Error"
            best_score = 0

        # ---------------- STABILIZATION ----------------
        recent_predictions.append(best_match)
        final_name = max(set(recent_predictions), key=recent_predictions.count)

        label = f"{final_name} ({best_score:.2f})"

        if final_name != "Unknown" and final_name != "Error":
            mark_attendance(final_name)

        cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(display, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Recognition", display)

    if choice == "3":
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()

cv2.destroyAllWindows()