import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("dataset_angles.csv")

# -------------------------------
# 2. Split by VIDEO (Very Important)
#    This prevents data leakage
# -------------------------------
unique_videos = data["video_name"].unique()

train_videos, test_videos = train_test_split(
    unique_videos,
    test_size=0.2,
    random_state=42
)

train_data = data[data["video_name"].isin(train_videos)]
test_data = data[data["video_name"].isin(test_videos)]

# -------------------------------
# 3. Features and Labels
# -------------------------------
feature_columns = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip"
]

X_train = train_data[feature_columns]
y_train = train_data["label"]

X_test = test_data[feature_columns]
y_test = test_data["label"]

# -------------------------------
# 4. Random Forest Model
#    Reduced complexity to avoid overfitting
# -------------------------------
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,          # Increased trees
        max_depth=20,              # Allow deeper learning
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # VERY IMPORTANT
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
)

# -------------------------------
# 5. Train Model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\nTrain Videos:", len(train_videos))
print("Test Videos:", len(test_videos))

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
joblib.dump(model, "activity_model.pkl")
print("\nRandom Forest model saved as activity_model.pkl")