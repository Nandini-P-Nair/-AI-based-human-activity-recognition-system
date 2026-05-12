import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter

# ---------------- LOAD MODEL ----------------
model = joblib.load("activity_model.pkl")

# ---------------- MEDIAPIPE SETUP ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ---------------- ANGLE FUNCTION ----------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# ---------------- LOAD VIDEO ----------------
video_path = "C:/Users/User/Desktop/p/infosys db/new2/test_video.mp4"
cap = cv2.VideoCapture(video_path)

all_predictions = []

# =================================================
# 1️⃣ FIRST PASS → ANALYZE VIDEO (NO DISPLAY)
# =================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        left_wrist = [landmarks[15].x, landmarks[15].y]

        right_shoulder = [landmarks[12].x, landmarks[12].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]

        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]

        right_hip = [landmarks[24].x, landmarks[24].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]

        features = np.array([[ 
            calculate_angle(left_shoulder, left_elbow, left_wrist),
            calculate_angle(right_shoulder, right_elbow, right_wrist),
            calculate_angle(left_hip, left_knee, left_ankle),
            calculate_angle(right_hip, right_knee, right_ankle),
            calculate_angle(left_elbow, left_shoulder, left_hip),
            calculate_angle(right_elbow, right_shoulder, right_hip),
            calculate_angle(left_shoulder, left_hip, left_knee),
            calculate_angle(right_shoulder, right_hip, right_knee)
        ]])

        prediction = model.predict(features)[0]
        all_predictions.append(prediction)

cap.release()

# =================================================
# 2️⃣ FINAL DECISION
# =================================================
if len(all_predictions) == 0:
    print("No activity detected.")
    exit()

final_activity = Counter(all_predictions).most_common(1)[0][0]

print("\n====================================")
print("FINAL ACTIVITY DETECTED:", final_activity)
print("====================================")

# =================================================
# 3️⃣ SECOND PASS → SHOW VIDEO WITH GREEN BOX + FINAL RESULT
# =================================================
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # ---- Compute Bounding Box ----
        x_list = [int(lm.x * w) for lm in landmarks]
        y_list = [int(lm.y * h) for lm in landmarks]

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # ---- Draw Green Box ----
        cv2.rectangle(frame,
                      (x_min, y_min),
                      (x_max, y_max),
                      (0, 255, 0),
                      2)

        # ---- Show Final Activity ----
        cv2.putText(frame,
                    f'SAFE Activity: {final_activity}',
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

    cv2.imshow("Video Result", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()