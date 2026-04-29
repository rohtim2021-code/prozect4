# ================= IMPORTS =================
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient, ReturnDocument
from werkzeug.utils import secure_filename
import os, time, requests
import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
import urllib.request

# ================= APP =================
app = Flask(__name__, static_folder="frontend")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DATABASE =================
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Mongo URI
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")

try:
    client = MongoClient(MONGO_URI)
    db = client["lost_finder"]

    # Collections
    users_col   = db["users"]
    missing_col = db["missing"]
    found_col   = db["found"]
    family_col  = db["family"]
    counters    = db["counters"]

    print("MongoDB Connected Successfully")

except Exception as e:
    print("MongoDB Connection Error:", e)
except Exception as e:
    print("❌ MongoDB Connection Error:", e)

#======= Cloud CREDENTIALS =================

    cloudinary.config(
    cloud_name="YOUR_CLOUD_NAME",
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET"
)
    

# ================= ADMIN CREDENTIALS =================
ADMIN_EMAIL    = "admin@gmail.com"
ADMIN_PASSWORD = "admin123"

# ================= SMS CONFIG (Fast2SMS) =================
# Sign up free at https://www.fast2sms.com  → API → Dev API
# Paste your API key below
FAST2SMS_API_KEY = "YOUR_FAST2SMS_API_KEY_HERE"
SMS_ENABLED      = True   # Set False to disable SMS


def send_sms(phone: str, message: str) -> bool:
    """
    Send SMS via Fast2SMS (India).
    phone  : 10-digit Indian mobile number
    message: text to send
    Returns True on success, False on failure.
    """
    if not SMS_ENABLED:
        print(f"[SMS DISABLED] To: {phone} | {message[:60]}")
        return False

    phone = str(phone).strip().replace("+91", "").replace(" ", "")
    if len(phone) != 10 or not phone.isdigit():
        print(f"[SMS] Invalid phone: {phone}")
        return False

    try:
        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = {
            "authorization": FAST2SMS_API_KEY,
            "message":       message,
            "language":      "english",
            "route":         "q",          # Quick Transactional
            "numbers":       phone,
        }
        resp = requests.post(url, data=payload, timeout=10)
        result = resp.json()
        if result.get("return"):
            print(f"[SMS ✅] Sent to {phone}")
            return True
        else:
            print(f"[SMS ❌] Failed: {result}")
            return False
    except Exception as e:
        print(f"[SMS ❌] Exception: {e}")
        return False


def build_match_sms(missing_person: dict, found_person: dict, score: int) -> str:
    """Build SMS text for match alert."""
    msg  = f"🚨 LOST & FINDER ALERT\n"
    msg += f"Match Found: {score}%\n\n"
    msg += f"Your missing person:\n"
    msg += f"Name: {missing_person.get('name','Unknown')}\n"
    msg += f"Age:  {missing_person.get('age','?')} | {missing_person.get('gender','?')}\n\n"
    msg += f"Possible match found at:\n"
    msg += f"Location: {found_person.get('location','?')}\n"
    city  = found_person.get("found_city") or found_person.get("city","")
    state = found_person.get("found_state") or found_person.get("state","")
    if city:  msg += f"City: {city}\n"
    if state: msg += f"State: {state}\n"
    msg += f"\nPlease login to Lost & Finder to verify.\n"
    msg += f"Helpline: +91-8707624604"
    return msg


def build_family_sms(family_member: dict, found_person: dict, score: int) -> str:
    """Build SMS for family member match alert."""
    msg  = f"🚨 LOST & FINDER ALERT\n"
    msg += f"Family Match Found: {score}%\n\n"
    msg += f"Your family member:\n"
    msg += f"Name: {family_member.get('name','Unknown')} ({family_member.get('relation','')})\n\n"
    msg += f"Possible match found at:\n"
    msg += f"Location: {found_person.get('location','?')}\n"
    city  = found_person.get("found_city") or found_person.get("city","")
    state = found_person.get("found_state") or found_person.get("state","")
    if city:  msg += f"City: {city}\n"
    if state: msg += f"State: {state}\n"
    msg += f"\nPlease login to Lost & Finder to verify.\n"
    msg += f"Helpline: +91-8707624604"
    return msg


# ================= OPENCV FACE SETUP =================
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def read_image_from_url(url):
    try:
        resp = urllib.request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        return cv2.imdecode(img, cv2.IMREAD_COLOR)
    except:
        return None

def extract_face(image_path):
    if image_path.startswith("http"):
             img = read_image_from_url(image_path)
    else:
             img = cv2.imread(image_path)
    if img is None:
        return None
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return cv2.resize(img, (100, 100))
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = 10
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return cv2.resize(img[y1:y2, x1:x2], (100, 100))


def compare_histograms(img1, img2):
    h1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)],
                      [0, 1], None, [50, 60], [0, 180, 0, 256])
    h2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)],
                      [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return max(0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) * 100)


def compare_orb(img1, img2):
    orb = cv2.ORB_create(nfeatures=500)
    g1  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return 0
    good = [m for pair in matches if len(pair)==2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    mx = min(len(kp1), len(kp2))
    return min((len(good) / mx) * 100, 100) if mx > 0 else 0


def face_score(path1, path2):
    try:
        f1 = extract_face(path1)
        f2 = extract_face(path2)
        if f1 is None or f2 is None:
            return 0
        return round(compare_histograms(f1, f2) * 0.6 + compare_orb(f1, f2) * 0.4, 2)
    except Exception as e:
        print(f"face_score error: {e}")
        return 0


# ================= HELPERS =================
def next_id(name):
    doc = counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return int(doc["seq"])


def save_file(file):
    if not file or file.filename == "":
        return ""
    try:
        res = cloudinary.uploader.upload(file)
        return res.get("secure_url")   # URL store hoga
    except Exception as e:
        print("Cloudinary Upload Error:", e)
        return ""

def norm(s):
    return (s or "").strip().lower()


# ================= TEXT MATCH SCORE =================
def calc_score(a, b):
    score = 0

    if norm(a.get("name")) and norm(a.get("name")) == norm(b.get("name")):
        score += 25

    loc_a = norm(" ".join(filter(None, [
        a.get("location",""), a.get("city",""), a.get("state",""),
        a.get("found_city",""), a.get("found_state","")
    ])))
    loc_b = norm(" ".join(filter(None, [
        b.get("location",""), b.get("city",""), b.get("state",""),
        b.get("found_city",""), b.get("found_state","")
    ])))
    if loc_a and loc_b:
        common = set(loc_a.split()) & set(loc_b.split())
        score += min(20, len(common) * 5)

    if norm(a.get("gender")) and norm(a.get("gender")) == norm(b.get("gender")):
        score += 20

    try:
        if abs(int(a.get("age", 0)) - int(b.get("age", 0))) <= 3:
            score += 20
    except Exception:
        pass

    if a.get("photo") and b.get("photo"):
        p1 = a["photo"]
        p2 = b["photo"]
        if os.path.exists(p1) and os.path.exists(p2):
            score += int(face_score(p1, p2) * 0.15)

    return min(score, 100)


# ================= AUTO MATCH + AUTO SMS =================
def auto_match():
    """
    Full three-way matching + auto SMS on 80%+ match.
    """
    missing_list = list(missing_col.find({"status": "approved"}))
    found_list   = list(found_col.find())
    family_list  = list(family_col.find())

    # track which numbers already got SMS this run (avoid duplicate)
    sms_sent = set()

    # ---- RESET ----
    missing_col.update_many({}, {"$set": {"match": 0, "match_id": "-"}})
    found_col.update_many({}, {"$set": {
        "match": 0, "match_id": "-",
        "family_match": 0, "family_match_id": "-"
    }})
    family_col.update_many({}, {"$set": {"match": 0, "match_id": "-"}})

    # ---- FOUND <-> MISSING ----
    for f in found_list:
        best_score = 0
        best_id    = "-"
        best_miss  = None

        for m in missing_list:
            s = calc_score(f, m)
            if s > best_score:
                best_score = s
                best_id    = m["id"]
                best_miss  = m

        found_col.update_one({"id": f["id"]}, {"$set": {
            "match": best_score, "match_id": best_id
        }})

        # AUTO SMS: missing person's phone when match >= 80
        if best_score >= 80 and best_miss:
            phone = best_miss.get("phone") or best_miss.get("mobile")
            key   = f"miss_{best_miss['id']}_found_{f['id']}"
            if phone and key not in sms_sent:
                msg = build_match_sms(best_miss, f, best_score)
                send_sms(phone, msg)
                sms_sent.add(key)

    # ---- FOUND <-> FAMILY ----
    for f in found_list:
        best_score = 0
        best_id    = "-"
        best_fam   = None

        for fam in family_list:
            score = 0

            if f.get("photo") and fam.get("photo"):
                p1 = os.path.join(UPLOAD_FOLDER, f["photo"])
                p2 = os.path.join(UPLOAD_FOLDER, fam["photo"])
                if os.path.exists(p1) and os.path.exists(p2):
                    score += face_score(p1, p2) * 0.5

            if norm(f.get("gender")) and norm(f.get("gender")) == norm(fam.get("gender")):
                score += 25

            loc_f   = norm(" ".join(filter(None, [f.get("location",""),   f.get("found_city",""),   f.get("found_state","")])))
            loc_fam = norm(" ".join(filter(None, [fam.get("location",""), fam.get("city",""),        fam.get("state","")])))
            if loc_f and loc_fam and (set(loc_f.split()) & set(loc_fam.split())):
                score += 25

            score = min(int(score), 100)
            if score > best_score:
                best_score = score
                best_id    = fam["id"]
                best_fam   = fam

        found_col.update_one({"id": f["id"]}, {"$set": {
            "family_match": best_score, "family_match_id": best_id
        }})

        # AUTO SMS: family member's phone when match >= 80
        if best_score >= 80 and best_fam:
            phone = best_fam.get("phone") or best_fam.get("mobile")
            key   = f"fam_{best_fam['id']}_found_{f['id']}"
            if phone and key not in sms_sent:
                msg = build_family_sms(best_fam, f, best_score)
                send_sms(phone, msg)
                sms_sent.add(key)

    # ---- MISSING <- reverse from FOUND ----
    updated_found = list(found_col.find())
    for m in missing_list:
        best_score = 0; best_id = "-"
        for f in updated_found:
            if str(f.get("match_id")) == str(m["id"]) and f.get("match", 0) > best_score:
                best_score = f["match"]; best_id = f["id"]
        missing_col.update_one({"id": m["id"]}, {"$set": {
            "match": best_score, "match_id": best_id
        }})

    # ---- FAMILY <- reverse from FOUND ----
    for fam in family_list:
        best_score = 0; best_id = "-"
        for f in updated_found:
            if str(f.get("family_match_id")) == str(fam["id"]) and f.get("family_match", 0) > best_score:
                best_score = f["family_match"]; best_id = f["id"]
        family_col.update_one({"id": fam["id"]}, {"$set": {
            "match": best_score, "match_id": best_id
        }})

    print("✅ auto_match done")


# ================= STATIC FILES =================
@app.route("/")
def home():
    return send_from_directory(os.path.join(os.getcwd(), "frontend"), "login.html")

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/<path:path>")
def static_files(path):
    frontend_path = os.path.join(os.getcwd(), "frontend")
    if path.startswith("api"):
        return "Not Found", 404
    return send_from_directory(frontend_path, path)

# ================= AUTH =================
@app.route("/api/signup", methods=["POST"])
def signup():
    data  = request.get_json()
    email = norm(data.get("email"))
    if users_col.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400
    users_col.insert_one({
        "name":     data.get("name"),
        "email":    email,
        "password": data.get("password"),
        "mobile":   data.get("mobile"),
        "role":     "user"
    })
    return jsonify({"msg": "Signup successful"})


@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = norm(data.get("email"))
    password = data.get("password")
    role     = data.get("role")

    if role == "admin":
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            return jsonify({"redirect": "/admin.html"})
        return jsonify({"error": "Invalid admin credentials"}), 401

    user = users_col.find_one({"email": email, "password": password})
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"redirect": "/index.html"})


# ================= REPORT MISSING =================
@app.route("/api/report", methods=["POST"])
def add_missing():
    filename = save_file(request.files.get("photo"))
    doc = {
        "id":       next_id("missing"),
        "name":     request.form.get("name"),
        "age":      request.form.get("age"),
        "gender":   request.form.get("gender"),
        "height":   request.form.get("height"),
        "aadhar":   request.form.get("aadhar"),
        "phone":    request.form.get("phone"),
        "email":    request.form.get("email"),
        "location": request.form.get("location"),
        "village":  request.form.get("village"),
        "city":     request.form.get("city"),
        "state":    request.form.get("state"),
        "pincode":  request.form.get("pincode"),
        "nearby":   request.form.get("nearby"),
        "date":     request.form.get("date"),
        "details":  request.form.get("details"),
        "photo":    filename,
        "status":   "pending",
        "match":    0,
        "match_id": "-"
    }
    missing_col.insert_one(doc)
    return jsonify({"msg": "Missing report submitted"})


# ================= REPORT FOUND =================
@app.route("/api/found", methods=["POST"])
def add_found():
    filename = save_file(request.files.get("photo"))
    doc = {
        "id":            next_id("found"),
        "name":          request.form.get("name"),
        "age":           request.form.get("age"),
        "gender":        request.form.get("gender"),
        "height":        request.form.get("height"),
        "aadhar":        request.form.get("aadhar"),
        "phone":         request.form.get("phone"),
        "location":      request.form.get("location"),
        "found_village": request.form.get("found_village"),
        "found_city":    request.form.get("found_city"),
        "found_state":   request.form.get("found_state"),
        "found_pincode": request.form.get("found_pincode"),
        "found_nearby":  request.form.get("found_nearby"),
        "home_address":  request.form.get("home_address"),
        "home_village":  request.form.get("home_village"),
        "home_city":     request.form.get("home_city"),
        "home_state":    request.form.get("home_state"),
        "home_pincode":  request.form.get("home_pincode"),
        "home_nearby":   request.form.get("home_nearby"),
        "date":          request.form.get("date"),
        "details":       request.form.get("details"),
        "photo":         filename,
        "match":           0,
        "match_id":        "-",
        "family_match":    0,
        "family_match_id": "-"
    }
    found_col.insert_one(doc)
    auto_match()
    return jsonify({"msg": "Found report submitted"})


# ================= ADD FAMILY MEMBER =================
@app.route("/api/family/add", methods=["POST"])
def add_family():
    filename = save_file(request.files.get("photo"))
    doc = {
        "id":       next_id("family"),
        "name":     request.form.get("name"),
        "age":      request.form.get("age"),
        "gender":   request.form.get("gender"),
        "height":   request.form.get("height"),
        "relation": request.form.get("relation"),
        "aadhar":   request.form.get("aadhar"),
        "phone":    request.form.get("phone"),
        "email":    request.form.get("email"),
        "location": request.form.get("location"),
        "village":  request.form.get("village"),
        "city":     request.form.get("city"),
        "state":    request.form.get("state"),
        "pincode":  request.form.get("pincode"),
        "nearby":   request.form.get("nearby"),
        "photo":    filename,
        "match":    0,
        "match_id": "-"
    }
    family_col.insert_one(doc)
    auto_match()
    return jsonify({"msg": "Family member added"})


# ================= GET ALL RECORDS =================
@app.route("/api/admin/missing")
def get_missing():
    return jsonify(list(missing_col.find({}, {"_id": 0})))

@app.route("/api/found/all")
def get_found():
    return jsonify(list(found_col.find({}, {"_id": 0})))

@app.route("/api/family/all")
def get_family():
    return jsonify(list(family_col.find({}, {"_id": 0})))


# ================= HOME CASES (latest 4) =================
@app.route("/api/home_cases")
def home_cases():
    missing = list(missing_col.find({}, {"_id": 0}))
    found   = list(found_col.find({}, {"_id": 0}))

    for m in missing: m["type"] = "missing"
    for f in found:   f["type"] = "found"

    all_cases = missing + found
    # sort by id descending, take latest 4
    all_cases.sort(key=lambda x: x.get("id", 0), reverse=True)
    return jsonify({"cases": all_cases[:4]})


# ================= APPROVE =================
@app.route("/api/approve", methods=["POST"])
def approve():
    mid = int(request.json["id"])
    missing_col.update_one({"id": mid}, {"$set": {"status": "approved"}})
    found_col.update_one(  {"id": mid}, {"$set": {"status": "approved"}})
    family_col.update_one( {"id": mid}, {"$set": {"status": "approved"}})
    auto_match()
    return jsonify({"msg": "Approved"})


# ================= DELETE =================
@app.route("/api/delete", methods=["POST"])
def delete():
    mid = int(request.json["id"])
    missing_col.delete_one({"id": mid})
    found_col.delete_one(  {"id": mid})
    family_col.delete_one( {"id": mid})
    auto_match()
    return jsonify({"msg": "Deleted"})


# ================= MANUAL MATCH =================
@app.route("/api/match")
def match_now():
    auto_match()
    return jsonify({"msg": "Matching complete"})


# ================= SEARCH BY AADHAAR =================
@app.route("/api/search")
def search_aadhar():
    aadhar = request.args.get("aadhar", "").strip()
    if not aadhar:
        return jsonify({"error": "Aadhaar required"}), 400

    rec = (missing_col.find_one({"aadhar": aadhar}, {"_id": 0}) or
           found_col.find_one(  {"aadhar": aadhar}, {"_id": 0}) or
           family_col.find_one( {"aadhar": aadhar}, {"_id": 0}))

    if not rec:
        return jsonify({"error": "No record found"}), 404
    return jsonify(rec)


# ================= FACE SEARCH =================
@app.route("/api/search-face", methods=["POST"])
def search_face():
    try:
        file = request.files.get("photo")
        if not file:
            return jsonify({"match": False, "error": "No file uploaded"}), 400

        filename = str(int(time.time())) + "_query_" + secure_filename(file.filename)
        upload = cloudinary.uploader.upload(file)
        path = upload.get("secure_url")

        best_score  = 0
        best_person = None

        for m in missing_col.find({}):
            if m.get("photo"):
                p2 = m["photo"] 
                if os.path.exists(p2):
                    s = face_score(path, p2)
                    if s > best_score:
                        best_score  = s
                        best_person = m

        try:
            os.remove(path)
        except Exception:
            pass

        if best_score >= 60 and best_person:
            best_person.pop("_id", None)
            return jsonify({
                "match":   True,
                "percent": best_score,
                "person":  best_person
            })

        return jsonify({"match": False, "percent": best_score})

    except Exception as e:
        print(f"search_face ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ================= STATS =================
@app.route("/api/stats")
def stats():
    found_list  = list(found_col.find({}, {"match": 1, "family_match": 1}))
    alert_count = sum(
        1 for f in found_list
        if f.get("match", 0) >= 80 or f.get("family_match", 0) >= 80
    )
    return jsonify({
        "total_missing": missing_col.count_documents({}),
        "approved":      missing_col.count_documents({"status": "approved"}),
        "pending":       missing_col.count_documents({"status": "pending"}),
        "total_found":   found_col.count_documents({}),
        "alerts":        alert_count
    })


# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))