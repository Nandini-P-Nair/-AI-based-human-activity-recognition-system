from flask import Flask, render_template, request, jsonify, session
import subprocess
import sqlite3
import os
from database import init_db

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# Initialize database
init_db()

# Ensure static folder exists
os.makedirs("static", exist_ok=True)


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["POST"])
def signup():

    data = request.get_json()

    username = data["username"]
    email = data["email"]
    password = data["password"]
    confirm = data["confirm"]

    if password != confirm:
        return jsonify({"status": "fail", "message": "Passwords do not match"})

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (username, email, password)
        )
        conn.commit()

        return jsonify({"status": "success"})

    except sqlite3.IntegrityError:
        return jsonify({"status": "fail", "message": "User already exists"})

    finally:
        conn.close()


# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    username = data["username"]
    password = data["password"]

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()

    if user:
        session["user"] = username
        return jsonify({"status": "success", "user": username})

    return jsonify({"status": "fail"})


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return jsonify({"status": "logged_out"})


# ---------------- ANALYZE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():

    if "user" not in session:
        return jsonify({"result": "Please login first"})

    file = request.files["video"]
    file.save("test_video.mp4")

    # Run ML script — writes annotated frames to static/output_result.mp4
    result = subprocess.run(
        ["python", "combine.py"],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()

    # Extract final line
    if "UNSAFE -" in output:
        final_line = [line for line in output.split("\n") if "UNSAFE" in line][-1]
    elif "FINAL ACTIVITY DETECTED" in output:
        final_line = [line for line in output.split("\n") if "FINAL ACTIVITY DETECTED" in line][-1]
    else:
        final_line = "No result detected"

    # -------------------------------------------------------------------
    # Re-encode with ffmpeg → H.264 + faststart so browsers can play it
    # -------------------------------------------------------------------
    raw_path = "static/output_result.mp4"
    web_path = "static/output_web.mp4"

    # Remove stale web file first
    if os.path.exists(web_path):
        os.remove(web_path)

    ffmpeg_ok = False
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", raw_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",                     # drop audio (there is none)
                    "-movflags", "+faststart",
                    web_path
                ],
                capture_output=True,
                timeout=300
            )
            if proc.returncode == 0 and os.path.exists(web_path) and os.path.getsize(web_path) > 0:
                ffmpeg_ok = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Fallback: serve the raw file directly if ffmpeg unavailable/failed
    if not ffmpeg_ok:
        import shutil
        if os.path.exists(raw_path):
            shutil.copy(raw_path, web_path)

    # -------------------------------------------------------------------

    # Save history
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history(username,video_name,result) VALUES(?,?,?)",
        (session["user"], file.filename, final_line)
    )

    conn.commit()
    conn.close()

    video_url = "/static/output_web.mp4" if os.path.exists(web_path) else None
    return jsonify({"result": final_line, "video_url": video_url})


# ---------------- HISTORY ----------------
@app.route("/history")
def history():

    if "user" not in session:
        return jsonify({"history": []})

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT video_name,result FROM history WHERE username=?",
        (session["user"],)
    )

    rows = cursor.fetchall()
    conn.close()

    return jsonify({"history": rows})


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)