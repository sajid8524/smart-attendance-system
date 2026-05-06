import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace

# ---------------- PATHS ----------------
VIDEO_FOLDER = "data/videos"
PREVIEW_FOLDER = "data/preview_faces"
DB_PATH = "db/face_db.pkl"

os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs("db", exist_ok=True)

# ---------------- SETTINGS ----------------
FRAME_SKIP = 5
MAX_PREVIEW = 30
MAX_EMBEDDINGS_PER_PERSON = 30   # 👈 controlled size
SIMILARITY_THRESHOLD = 0.85      # 👈 filter duplicates

# ---------------- LOAD EXISTING DB ----------------
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        database = pickle.load(f)
    print("Loaded existing DB (append mode)")
else:
    database = {}

# ---------------- UTILS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_diverse(new_emb, emb_list):
    for emb in emb_list:
        if cosine_similarity(new_emb, emb) > SIMILARITY_THRESHOLD:
            return False
    return True

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- MAIN ----------------
for video_file in os.listdir(VIDEO_FOLDER):
    if not video_file.endswith((".mp4", ".avi", ".mov")):
        continue

    person = os.path.splitext(video_file)[0]
    print(f"\nProcessing {person}...")

    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_file))

    if person in database:
        embeddings = database[person]["embeddings"]
    else:
        embeddings = []

    samples = []
    frame_count = 0
    preview_count = 0

    person_preview_path = os.path.join(PREVIEW_FOLDER, person)
    os.makedirs(person_preview_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))

            # -------- PREVIEW --------
            if preview_count < MAX_PREVIEW:
                img_name = f"{preview_count}.jpg"
                cv2.imwrite(os.path.join(person_preview_path, img_name), face)
                preview_count += 1

            # -------- EMBEDDING --------
            try:
                rep = DeepFace.represent(
                    img_path=face,
                    model_name="ArcFace",
                    enforce_detection=False
                )

                emb = np.array(rep[0]["embedding"])
                emb = emb / np.linalg.norm(emb)

                # -------- SMART FILTER --------
                if is_diverse(emb, embeddings):
                    embeddings.append(emb)

                # keep size bounded
                if len(embeddings) > MAX_EMBEDDINGS_PER_PERSON:
                    embeddings.pop(0)

            except:
                continue

        frame_count += 1

    cap.release()

    if len(embeddings) > 0:
        database[person] = {
            "embeddings": embeddings
        }
        print(f"{person}: stored {len(embeddings)} diverse embeddings")

    else:
        print(f"{person}: no valid faces")

# ---------------- SAVE DB ----------------
with open(DB_PATH, "wb") as f:
    pickle.dump(database, f)

print("\nDatabase saved (optimized embeddings)")