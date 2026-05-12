import cv2
import os
import csv
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# --------- Angle Calculation Function ----------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# --------- Activities ----------
activities = ["walking", "jogging", "sitting", "yoga"]

# --------- CSV File ----------
csv_file = open("dataset_angles.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Header
csv_writer.writerow([
    "video_name",
    "frame",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "label"
])

with mp_pose.Pose() as pose:

    for activity in activities:
        folder_path = activity

        for video_name in os.listdir(folder_path):

            if video_name.endswith(".mp4"):

                video_path = os.path.join(folder_path, video_name)
                cap = cv2.VideoCapture(video_path)

                frame_number = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_number += 1

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    if results.pose_landmarks:

                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        def get_point(id):
                            return [landmarks[id].x, landmarks[id].y]

                        # Left side
                        left_shoulder = get_point(11)
                        left_elbow = get_point(13)
                        left_wrist = get_point(15)
                        left_hip = get_point(23)
                        left_knee = get_point(25)
                        left_ankle = get_point(27)

                        # Right side
                        right_shoulder = get_point(12)
                        right_elbow = get_point(14)
                        right_wrist = get_point(16)
                        right_hip = get_point(24)
                        right_knee = get_point(26)
                        right_ankle = get_point(28)

                        # Calculate angles
                        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

                        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                        # Write row
                        csv_writer.writerow([
                            video_name,
                            frame_number,
                            left_elbow_angle,
                            right_elbow_angle,
                            left_knee_angle,
                            right_knee_angle,
                            left_shoulder_angle,
                            right_shoulder_angle,
                            left_hip_angle,
                            right_hip_angle,
                            activity
                        ])

                cap.release()

csv_file.close()

print("Dataset generation complete!")