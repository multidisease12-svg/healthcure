import sqlite3
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_dance.contrib.google import make_google_blueprint, google
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import requests
import markdown
import io
from flask import Flask
from flask_mail import Mail
from flask_dance.contrib.google import make_google_blueprint
from dotenv import load_dotenv
import os


os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Allow OAuth over HTTP (for local development only)

load_dotenv()   # Make sure this runs before using os.getenv()


# Groq API credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")


app = Flask(__name__)
app.secret_key = "FLASK_SECRET_KEY"  # Replace with your own secret key
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    redirect_to="google_login"
)

app.register_blueprint(google_bp, url_prefix="/login")


app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD")
)

mail = Mail(app)



# Serializer for password reset token
s = URLSafeTimedSerializer(app.secret_key)

# --- NEW: safe users.db path and ensure instance dir exists ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
USERS_DB_PATH = os.path.join(INSTANCE_DIR, "users.db")

# Ensure users table (with email column) exists
try:
    conn_init = sqlite3.connect(USERS_DB_PATH)
    cur_init = conn_init.cursor()
    cur_init.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        email TEXT
    )
    """)
    conn_init.commit()
    cur_init.close()
    conn_init.close()
except Exception:
    # If creation fails, we continue — the routes will handle errors and flash messages.
    pass
# ------------------------------------------------------------------

# Configure SQLite database for SQLAlchemy predictions.db (unchanged)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # remove ForeignKey
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    disease = db.Column(db.String(50), nullable=False)
    image = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


    def __repr__(self):
        return f"<Prediction {self.disease} for {self.firstname} {self.lastname}>"

# Create the predictions database and tables
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash("Please enter both username and password")
            return render_template('index.html')  # Stay on login page

        # Connect to SQLite
        conn = sqlite3.connect(USERS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            db_password = user[2]
            if password == db_password:
                session['username'] = user[1]
                session['user_id'] = user[0]
                flash(f"Welcome, {user[1]}!")
                return redirect(url_for('main'))

        # Login failed
        flash("Invalid username or password")
        return redirect(url_for('login'))  # ✅ redirect instead of render_template

    # GET request
    return render_template('index.html')


@app.route('/main')
def main():
    if 'username' in session:
        return render_template('main.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET','POST'])
def signup():
    # Clear old flash messages
    session.pop('_flashes', None)

    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')

    if not username or not password or not email:
        flash("Please enter username, password, and email")
        return redirect(url_for('login'))


    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()

    # Check if username already exists
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        db_password = existing_user[0]
        if db_password == password:
            # Both username & password are same → tell user to log in
            flash("User already exists. Please log in.")
        else:
            # Username exists with different password → ask for valid password
            flash("Username exists. Please enter the correct password.")
    else:
        # Username does not exist → insert new user
        cursor.execute("INSERT INTO users (username, password,email) VALUES (?, ?,?)", (username, password,email))
        conn.commit()
        flash("Signup successful! Please log in.")

    cursor.close()
    conn.close()
    return redirect(url_for('login'))





@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Show only that user's predictions
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    return render_template('results.html', predictions=predictions)

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))

    if session['username'] != 'admin':  # 👈 adjust if you want a different admin name
        flash("Access denied! Admins only.")
        return redirect(url_for('results'))

    all_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template('admin.html', predictions=all_predictions)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction_text = None
    
    # Personal details
    firstname = ''
    lastname = ''
    email = ''
    phone_no = ''
    gender = ''
    age = ''
    filename = None

    # Heart form fields (all persistent)
    form = {
        "trestbps": "",
        "chol": "",
        "fbs": "",
        "cp": "",
        "restecg": "",
        "thalach": "",
        "exang": "",
        "oldpeak": "",
        "slope": "",
        "ca": "",
        "thal": ""
    }

    if request.method == 'POST':
        try:
            data = pd.read_csv("dataset/heart.csv")
        except FileNotFoundError:
            flash("Error: 'heart.csv' missing in dataset folder.")
            return render_template(
                'heart.html',
                prediction=prediction_text,
                firstname=firstname, lastname=lastname,
                email=email, phone_no=phone_no,
                age=age, gender=gender,
                filename=filename,
                form=form
            )

        X = data.drop("target", axis=1)
        y = data["target"]

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # Personal info
        firstname = request.form.get('firstname', '')
        lastname = request.form.get('lastname', '')
        email = request.form.get('email', '')
        phone_no = request.form.get('phone_no', '')
        gender = request.form.get('gender', '')
        age = request.form.get('age', '')

        # Fill form dictionary (PERSISTENCE)
        for key in form.keys():
            form[key] = request.form.get(key, "")

        try:
            # Convert
            sex = 1 if gender.lower() == 'male' else 0
            trestbps = float(form["trestbps"])
            chol = float(form["chol"])
            fbs = int(form["fbs"])
            cp = int(form["cp"])
            restecg = int(form["restecg"])
            thalach = float(form["thalach"])
            exang = int(form["exang"])
            oldpeak = float(form["oldpeak"])
            slope = int(form["slope"])
            ca = int(form["ca"])
            thal = int(form["thal"])

        except:
            flash("Please enter valid numeric values.")
            return render_template(
                'heart.html',
                prediction=prediction_text,
                firstname=firstname, lastname=lastname,
                email=email, phone_no=phone_no,
                age=age, gender=gender,
                filename=filename,
                form=form
            )

        # Model prediction
        user_data = pd.DataFrame({
            "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
            "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
            "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
        })

        pred = model.predict(user_data)
        prediction_text = "Positive" if pred[0] == 1 else "Negative"

        # Save in DB
        if 'user_id' in session:
            new_prediction = Prediction(
                user_id=session['user_id'],
                firstname=firstname,
                lastname=lastname,
                age=int(age),
                gender=gender,
                prediction=prediction_text,
                disease="Heart Disease",
                image=filename
            )
            db.session.add(new_prediction)
            db.session.commit()

    return render_template(
        'heart.html',
        prediction=prediction_text,
        firstname=firstname,
        lastname=lastname,
        email=email,
        phone_no=phone_no,
        age=age,
        gender=gender,
        filename=filename,
        form=form
    )



@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction_text = None

    # Initialize all fields
    firstname = ''
    lastname = ''
    email = ''
    phone_no = ''
    gender = ''
    age = ''
    hypertension = ''
    heart_disease = ''
    bmi = ''
    hbA1c_level = ''
    glucose_level = ''
    filename = None

    if request.method == 'POST':
        try:
            # Load dataset and train model (optional, replace with your pre-trained model)
            data = pd.read_csv("dataset/diabetes.csv")
            X = data.drop("diabetes", axis=1)
            y = data["diabetes"]
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            # --- Personal info ---
            firstname = request.form.get('firstname', '')
            lastname = request.form.get('lastname', '')
            email = request.form.get('email', '')
            phone_no = request.form.get('phone_no', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')

            # --- Diabetes-specific fields ---
            hypertension = request.form.get('hypertension', '')
            heart_disease = request.form.get('heart_disease', '')
            bmi = request.form.get('bmi', '')
            hbA1c_level = request.form.get('hbA1c_level', '')
            glucose_level = request.form.get('glucose_level', '')

            # --- File upload ---
            file = request.files.get('image')
            if file and file.filename != '':
                filename = file.filename
                file.save(os.path.join("upload/", filename))

            # Convert gender and numeric fields for model
            gender_val = 1 if gender.lower() == 'male' else 0
            user_data = pd.DataFrame({
                "gender": [gender_val],
                "age": [float(age) if age else 0],
                "hypertension": [float(hypertension) if hypertension else 0],
                "heart_disease": [float(heart_disease) if heart_disease else 0],
                "bmi": [float(bmi) if bmi else 0],
                "HbA1c_level": [float(hbA1c_level) if hbA1c_level else 0],
                "blood_glucose_level": [float(glucose_level) if glucose_level else 0]
            })

            prediction = model.predict(user_data)
            prediction_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

            # Save to database
            if 'user_id' in session:
                new_prediction = Prediction(
                    user_id=session['user_id'],
                    firstname=firstname,
                    lastname=lastname,
                    age=int(age) if age else None,
                    gender=gender,
                    prediction=prediction_text,
                    disease="Diabetes",
                    image=filename
                )
                db.session.add(new_prediction)
                db.session.commit()

        except Exception as e:
            flash(f"Error: {str(e)}")

    return render_template('diabetes.html',
                           prediction=prediction_text,
                           firstname=firstname,
                           lastname=lastname,
                           email=email,
                           phone_no=phone_no,
                           age=age,
                           gender=gender,
                           hypertension=hypertension,
                           heart_disease=heart_disease,
                           bmi=bmi,
                           hbA1c_level=hbA1c_level,
                           glucose_level=glucose_level,
                           filename=filename)



@app.route('/covid19', methods=['GET', 'POST'])
def covid19():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from flask import request, render_template, flash
    import os

    # Feature columns exactly as in CSV
    feature_cols = [
        "Breathing Problem","Fever","Dry Cough","Sore throat","Running Nose",
        "Asthma","Chronic Lung Disease","Headache","Heart Disease","Diabetes",
        "Hyper Tension","Fatigue","Gastrointestinal","Abroad travel",
        "Contact with COVID Patient","Attended Large Gathering",
        "Visited Public Exposed Places","Family working in Public Exposed Places",
        "Wearing Masks","Sanitization from Market"
    ]

    # ---- GET REQUEST → RESET EVERYTHING ----
    if request.method == "GET":
        empty_features = {col: "" for col in feature_cols}  # empty “Select”
        return render_template(
            'covid19.html',
            prediction=None,
            firstname="",
            lastname="",
            age="",
            gender="",
            filename=None,
            user_features=empty_features
        )

    # ---- POST REQUEST ----
    prediction_text = None
    firstname = lastname = gender = age = ''
    phone_no = request.form.get("phone_no", "")
    email = request.form.get("email", "")
    filename = None
    user_features = {}

    try:
        data = pd.read_csv("dataset/covid.csv")
        data.columns = [c.strip() for c in data.columns]
    except FileNotFoundError:
        flash("Dataset not found!")
        return render_template('covid19.html')

    # Convert Yes/No → 1/0
    for col in feature_cols + ["COVID-19"]:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    X = data[feature_cols]
    y = data["COVID-19"]

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Personal info
    firstname = request.form.get('firstname', '')
    lastname = request.form.get('lastname', '')
    age = request.form.get('age', '')
    gender = request.form.get('gender', '')

    # Symptoms + risk factors
    for col in feature_cols:
        value = request.form.get(col, "")
        if value == "":
            user_features[col] = ""
        else:
            user_features[col] = int(value)

    # Prediction
    filled = {k: (0 if v == "" else v) for k, v in user_features.items()}
    covid_prediction = model.predict(pd.DataFrame([filled]))[0]
    prediction_text = "Positive" if covid_prediction == 1 else "Negative"

    # File Upload
    file = request.files.get('image')
    if file and file.filename != "":
        filename = file.filename
        os.makedirs("upload", exist_ok=True)
        file.save(os.path.join("upload", filename))
    
    if "user_id" in session:
        new_prediction = Prediction(
            user_id=session["user_id"],
            firstname=firstname,
            lastname=lastname,
            age=int(age) if age else None,
            gender=gender,
            prediction=prediction_text,
            disease="COVID-19",
            image=filename
        )
        db.session.add(new_prediction)
        db.session.commit()

    # Show result WITH submitted values
    return render_template(
        'covid19.html',
        prediction=prediction_text,
        firstname=firstname,
        lastname=lastname,
        age=age,
        gender=gender,
        phone_no=phone_no,
        email=email,
        filename=filename,
        user_features=user_features
    )

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response



@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor():
    prediction_text = None
    firstname = lastname = gender = age = phone_no = email = ''
    filename = None
    image_base64 = None

    if request.method == 'POST':
        try:
            file = request.files.get('image')
            if not file or file.filename == '':
                flash("Please select a file")
                return render_template('brain_tumor.html',
                                       prediction=prediction_text,
                                       firstname=firstname,
                                       lastname=lastname,
                                       age=age,
                                       gender=gender,
                                       phone_no=phone_no,
                                       email=email,
                                       filename=filename)

            # Save uploaded file and keep filename
            filename = file.filename
            img_path = os.path.join("upload/", filename)
            file.save(img_path)

            # Convert image to base64
            with open(img_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Brain tumor prediction
            categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
            model = load_model("models/brainTumor.keras")

            def predict(img_path, categories, model):
                img_size = 224
                label_dict = {i: category for i, category in enumerate(categories)}
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized_img = cv2.resize(img_rgb, (img_size, img_size))
                normalized_img = resized_img / 255.0
                input_img = normalized_img.reshape(-1, img_size, img_size, 3)
                prediction = model.predict(input_img)
                predicted_class_index = np.argmax(prediction)
                return label_dict[predicted_class_index], input_img

            prediction_text, _ = predict(img_path, categories, model)

            # Get form data
            firstname = request.form.get('firstname', '')
            lastname = request.form.get('lastname', '')
            phone_no = request.form.get('phone_no', '')
            email = request.form.get('email', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')

            # Save to database
            new_prediction = Prediction(
                user_id=session.get('user_id'),
                firstname=firstname,
                lastname=lastname,
                age=int(age) if age else None,
                gender=gender,
                prediction=prediction_text,
                disease="Brain Tumor",
                image=image_base64
            )
            db.session.add(new_prediction)
            db.session.commit()

        except Exception as e:
            flash(f"Error processing image: {str(e)}")

    return render_template('brain_tumor.html',
                           prediction=prediction_text,
                           firstname=firstname,
                           lastname=lastname,
                           age=age,
                           gender=gender,
                           phone_no=phone_no,
                           email=email,
                           filename=filename)


@app.route('/alzheimer', methods=['GET', 'POST'])
def alzheimer():
    prediction_text = None
    firstname = lastname = email = phone_no = gender = age = ''
    filename = None
    image_base64 = None  # ✅ Initialize image_base64

    if request.method == 'POST':
        try:
            # Get form values
            firstname = request.form.get('firstname', '')
            lastname = request.form.get('lastname', '')
            email = request.form.get('email', '')
            phone_no = request.form.get('phone_no', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')

            # File handling
            file = request.files.get('image')
            if not file or file.filename == '':
                flash("Please select a file")
                return render_template('alzheimer.html', 
                                       prediction=prediction_text,
                                       firstname=firstname, lastname=lastname,
                                       email=email, phone_no=phone_no,
                                       age=age, gender=gender)

            # Save uploaded file
            filename = file.filename
            upload_path = os.path.join("upload/", filename)
            file.save(upload_path)

            # Convert image to base64
            with open(upload_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Prediction logic
            categories = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
            model = load_model("models/alzheimer.keras")

            def predict(img_path, categories, model):
                img_size = 224
                label_dict = {i: category for i, category in enumerate(categories)}
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized_img = cv2.resize(img_rgb, (img_size, img_size))
                normalized_img = resized_img / 255.0
                input_img = normalized_img.reshape(-1, img_size, img_size, 3)
                prediction = model.predict(input_img)
                predicted_class_index = np.argmax(prediction)
                return label_dict[predicted_class_index]

            prediction_text = predict(upload_path, categories, model)

            # Save to database
            if 'user_id' in session:
                new_prediction = Prediction(
                    user_id=session['user_id'],
                    firstname=firstname,
                    lastname=lastname,
                    age=int(age) if age else None,
                    gender=gender,
                    prediction=prediction_text,
                    disease="Alzheimer",
                    image=image_base64  # ✅ now defined
                )
                db.session.add(new_prediction)
                db.session.commit()

        except Exception as e:
            flash(f"Error: {str(e)}")

    return render_template('alzheimer.html', 
                           prediction=prediction_text,
                           firstname=firstname,
                           lastname=lastname,
                           email=email,
                           phone_no=phone_no,
                           age=age,
                           gender=gender,
                           filename=filename,
                           )


@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    prediction_text = None
    firstname = lastname = gender = age = phone_no = email = ''
    filename = None
    image_base64 = None

    if request.method == 'POST':
        try:
            file = request.files.get('image')
            if not file or file.filename == '':
                flash("Please select a file")
                return render_template('pneumonia.html',
                                       prediction=prediction_text,
                                       firstname=firstname,
                                       lastname=lastname,
                                       age=age,
                                       gender=gender,
                                       phone_no=phone_no,
                                       email=email,
                                       filename=filename)

            # Save uploaded file and retain filename
            filename = file.filename
            img_path = os.path.join("upload/", filename)
            file.save(img_path)

            # Convert image to base64
            with open(img_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Pneumonia prediction
            categories = ['Normal', 'Pneumonia']
            model = load_model("models/pneumonia.keras")

            def predict(img_path, categories, model):
                img_size = 224
                label_dict = {i: category for i, category in enumerate(categories)}
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized_img = cv2.resize(img_rgb, (img_size, img_size))
                normalized_img = resized_img / 255.0
                input_img = normalized_img.reshape(-1, img_size, img_size, 3)
                prediction = model.predict(input_img)
                predicted_class_index = np.argmax(prediction)
                return label_dict[predicted_class_index], input_img

            prediction_text, _ = predict(img_path, categories, model)

            # Get form data
            firstname = request.form.get('firstname', '')
            lastname = request.form.get('lastname', '')
            phone_no = request.form.get('phone_no', '')
            email = request.form.get('email', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')

            # Save to database
            new_prediction = Prediction(
                user_id=session.get('user_id'),
                firstname=firstname,
                lastname=lastname,
                age=int(age) if age else None,
                gender=gender,
                prediction=prediction_text,
                disease="Pneumonia",
                image=image_base64
            )
            db.session.add(new_prediction)
            db.session.commit()

        except Exception as e:
            flash(f"Error processing image: {str(e)}")

    return render_template('pneumonia.html',
                           prediction=prediction_text,
                           firstname=firstname,
                           lastname=lastname,
                           age=age,
                           gender=gender,
                           phone_no=phone_no,
                           email=email,
                           filename=filename)


@app.route('/breast', methods=['GET', 'POST'])
def breast():
    prediction_text = None
    firstname = lastname = gender = age = phone_no = email = ''
    filename = None
    image_base64 = None

    if request.method == 'POST':
        try:
            file = request.files.get('image')
            if not file or file.filename == '':
                flash("Please select a file")
                return render_template('breast.html',
                                       prediction=prediction_text,
                                       firstname=firstname,
                                       lastname=lastname,
                                       age=age,
                                       gender=gender,
                                       phone_no=phone_no,
                                       email=email,
                                       filename=filename)

            # Save uploaded file and retain filename
            filename = file.filename
            img_path = os.path.join("upload/", filename)
            file.save(img_path)

            # Convert image to base64
            with open(img_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            # Breast cancer prediction
            categories = ['Benign', 'Malignant', 'Normal']

            model = load_model("models/breastCancer.keras")

            def predict(img_path, categories, model):
                img_size = 224
                label_dict = {i: category for i, category in enumerate(categories)}
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized_img = cv2.resize(img_rgb, (img_size, img_size))
                normalized_img = resized_img / 255.0
                input_img = normalized_img.reshape(-1, img_size, img_size, 3)
                prediction = model.predict(input_img)
                predicted_class_index = np.argmax(prediction)
                return label_dict[predicted_class_index], input_img

            prediction_text, _ = predict(img_path, categories, model)

            # Get form data
            firstname = request.form.get('firstname', '')
            lastname = request.form.get('lastname', '')
            phone_no = request.form.get('phone_no', '')
            email = request.form.get('email', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')

            # Save to database
            new_prediction = Prediction(
                user_id=session.get('user_id'),
                firstname=firstname,
                lastname=lastname,
                age=int(age) if age else None,
                gender=gender,
                prediction=prediction_text,
                disease="Breast Cancer",
                image=image_base64
            )
            db.session.add(new_prediction)
            db.session.commit()

        except Exception as e:
            flash(f"Error processing image: {str(e)}")

    return render_template('breast.html',
                           prediction=prediction_text,
                           firstname=firstname,
                           lastname=lastname,
                           age=age,
                           gender=gender,
                           phone_no=phone_no,
                           email=email,
                           filename=filename)


@app.route("/google_login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if resp.ok:
        user_info = resp.json()
        username = user_info.get("name")
        email = user_info.get("email")

        # Save user in SQLite if not exists (use safe path)
        conn = sqlite3.connect(USERS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        if not user:
            # insert username, oauth marker as password, and email
            cursor.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, "google_oauth", email))
            conn.commit()
            user_id = cursor.lastrowid
        else:
            user_id = user[0]

        cursor.close()
        conn.close()

        session['username'] = username
        session['user_id'] = user_id
        flash(f"Welcome, {username}! (Google login)")
        return redirect(url_for("main"))

    flash("Failed to fetch user info from Google.")
    return redirect(url_for("login"))



def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

@app.route("/medical_chatbot")
def chatbot_page():
    return render_template("medical_chatbot.html", chat_history=session.get("chat_history", []))

@app.route('/download_report')
def download_report():
    if 'user_id' not in session:
        flash("Please login to download report")
        return redirect(url_for('login'))

    # Fetch predictions for logged-in user
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setTitle("HealthCure - Full Prediction Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 50, "HealthCure - Full Prediction Report")

    # Table headers
    pdf.setFont("Helvetica-Bold", 12)
    headers = ["First Name", "Last Name", "Age", "Gender", "Disease", "Prediction", "Image", "Timestamp"]
    x_positions = [30, 100, 170, 210, 260, 350, 470, 540]  # 8 columns
    y = height - 80
    for i, header in enumerate(headers):
        pdf.drawString(x_positions[i], y, header)

    pdf.setFont("Helvetica", 12)
    y -= 20
    row_height = 80

    # Add all predictions
    for pred in predictions:
        pdf.drawString(x_positions[0], y, str(pred.firstname))
        pdf.drawString(x_positions[1], y, str(pred.lastname))
        pdf.drawString(x_positions[2], y, str(pred.age))
        pdf.drawString(x_positions[3], y, str(pred.gender))
        pdf.drawString(x_positions[4], y, str(pred.disease))
        pdf.drawString(x_positions[5], y, str(pred.prediction))

        # Draw image if exists
        if pred.image:
            img_data = base64.b64decode(pred.image)
            img = ImageReader(io.BytesIO(img_data))
            pdf.drawImage(img, x_positions[6], y - 45, width=60, height=50, preserveAspectRatio=True)

        # Draw timestamp
        # Draw timestamp as date only
        pdf.drawString(x_positions[7], y, pred.timestamp.strftime('%Y-%m-%d'))


        y -= row_height

        # Create new page if y too low
        if y < 50:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 12)

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="HealthCure_Full_Report.pdf", mimetype='application/pdf')



@app.route("/ask", methods=["POST"])
def ask_bot():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return redirect(url_for("chatbot_page"))

    # Initialize chat history in session
    if "chat_history" not in session:
        session["chat_history"] = []

    # Add user message
    session["chat_history"].append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": session["chat_history"],  # full chat history
        "temperature": 1,
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()
        answer = output["choices"][0]["message"]["content"]
        html_answer = markdown.markdown(answer)  # convert to HTML safely
        # Add assistant message
        session["chat_history"].append({"role": "assistant", "content": html_answer})
        session.modified = True
    except Exception as e:
        html_answer = f"<strong>Error:</strong> {e}<br><br><code>{response.text if 'response' in locals() else ''}</code>"
        session["chat_history"].append({"role": "assistant", "content": html_answer})
        session.modified = True

    return render_template("medical_chatbot.html", chat_history=session["chat_history"])


@app.route("/chatbot_reset", methods=["POST"])
def chatbot_reset():
    session.pop("chat_history", None)
    session.modified = True
    return {"success": True}




@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form.get('email')
    if not email:
        flash("Please enter your email")
        return redirect(url_for('login'))

    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        flash("Email not found")
        return redirect(url_for('login'))

    token = s.dumps(email, salt='password-reset-salt')
    reset_url = url_for('reset_password', token=token, _external=True)

    msg = Message("Password Reset Request", sender="youremail@gmail.com", recipients=[email])
    msg.body = f"Click this link to reset your password: {reset_url}"
    mail.send(msg)

    flash("Password reset link sent to your email")
    return redirect(url_for('login'))

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)  # 1-hour expiry
    except Exception:
        flash("The reset link is invalid or expired")
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        if not new_password:
            flash("Enter a new password")
            return redirect(request.url)

        conn = sqlite3.connect(USERS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password=? WHERE email=?", (new_password, email))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Password updated successfully! Please login.")
        return redirect(url_for('login'))

    return render_template('reset_password.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

