from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATADIR = "dataset"
CATEGORIES = os.listdir(DATADIR)
# Load the trained CNN model
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

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without filename
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        # Check if the file extension is allowed
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict category
            predicted_category = predict_category(file_path)
            
            return render_template('index.html', prediction=predicted_category, filename=filename)
        else:
            return render_template('index.html', error="File type not allowed")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

