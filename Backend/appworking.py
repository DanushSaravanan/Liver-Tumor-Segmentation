from flask import Flask, render_template, redirect, url_for, session, flash, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
from flask_mysqldb import MySQL
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mydatabase'
app.secret_key = 'your_secret_key_here'

mysql = MySQL(app)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained CNN model
DATADIR = "dataset"
CATEGORIES = os.listdir(DATADIR)
model = load_model('CNN.model')

# Function to preprocess the selected image
def preprocess_image(file_path):
    # Read the image using OpenCV
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing steps similar to what you did in creating training data
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    median = cv2.medianBlur(img_array.astype('uint8'), 5)
    median = 255 - median
    ret, thresh = cv2.threshold(median.astype('uint8'), 165, 255, cv2.THRESH_BINARY_INV)
    resized_img = cv2.resize(thresh, (100, 100))
    resized_img = resized_img.reshape(-1, 100, 100, 1)
    resized_img = resized_img / 255.0  # Normalize pixel values
    
    return resized_img

# Function to predict category
def predict_category(file_path):
    # Preprocess the selected image
    processed_img = preprocess_image(file_path)
        
    # Predict the category using the loaded model
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    predicted_category = CATEGORIES[predicted_class]
        
    return predicted_category

class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        # store data into database 
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        mysql.connection.commit()
        cursor.close()
        flash("Registration successful. You can now login.")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        cursor.close()
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close()

    prediction = None
    filename = None

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without filename
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        # Check if the file extension is allowed
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict category
            predicted_category = predict_category(file_path)
            prediction = predicted_category

        else:
            flash("File type not allowed")
            return redirect(request.url)

    if user:
        return render_template('dashboard.html', user=user, prediction=prediction, filename=filename)

    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)