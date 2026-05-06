import os
import cv2
import pickle
import numpy as np

DB_PATH = "db/face_db.pkl"
PREVIEW_FOLDER = "data/preview_faces"

# ---------------- LOAD DB ----------------
with open(DB_PATH, "rb") as f:
    db = pickle.load(f)

people = list(db.keys())
person_index = 0
sample_index = 0

print("Controls:")
print("n → next sample")
print("p → previous sample")
print("c → next person")
print("q → quit")

while True:
    person = people[person_index]
    samples = db[person]["samples"]

    if len(samples) == 0:
        print(f"No samples for {person}")
        break

    sample = samples[sample_index]

    img_name = sample["image"]
    aug_type = sample["type"]
    embedding = sample["embedding"]

    img_path = os.path.join(PREVIEW_FOLDER, person, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print(f"Image not found: {img_path}")
        continue

    # Resize for display
    img = cv2.resize(img, (300, 300))

    # Display text info
    text1 = f"Person: {person}"
    text2 = f"Sample: {sample_index+1}/{len(samples)}"
    text3 = f"Type: {aug_type}"
    text4 = f"Emb norm: {np.linalg.norm(embedding):.2f}"

    cv2.putText(img, text1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img, text2, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img, text3, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img, text4, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Show image
    cv2.imshow("Embedding Viewer", img)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        sample_index = (sample_index + 1) % len(samples)

    elif key == ord('p'):
        sample_index = (sample_index - 1) % len(samples)

    elif key == ord('c'):
        person_index = (person_index + 1) % len(people)
        sample_index = 0

    elif key == ord('q'):
        break

cv2.destroyAllWindows()