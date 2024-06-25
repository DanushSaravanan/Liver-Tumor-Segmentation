import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

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

# Function to handle button click event and predict category
def predict_category():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Preprocess the selected image
        processed_img = preprocess_image(file_path)
        
        # Predict the category using the loaded model
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        predicted_category = CATEGORIES[predicted_class]
        
        # Display the predicted category
        prediction_label.config(text=f'Predicted Category: {predicted_category}')

# Create Tkinter window
window = tk.Tk()
window.title("Image Category Predictor")

# Button to select image file
select_button = tk.Button(window, text="Select Image", command=predict_category)
select_button.pack(pady=20)

# Label to display predicted category
prediction_label = tk.Label(window, text="")
prediction_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
