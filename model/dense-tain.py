import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

DATADIR = "dataset"
CATEGORIES = os.listdir(DATADIR)

import cv2
a=input("Enter image path")
img_array = cv2.imread(a,0)
img_array = cv2.resize(img_array, (250, 250))
print("ORIGINAL IMAGE")
cv2.imshow("",img_array)
cv2.waitKey(0)

#img_array = cv2.Canny(img_array, threshold1=50, threshold2=10)

clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
img_array = clahe.apply(img_array)
median  = cv2.medianBlur(img_array.astype('uint8'), 5)
median = 255-median
ret,thresh = cv2.threshold(median.astype('uint8'),165,255,cv2.THRESH_BINARY_INV)
img_array=cv2.fastNlMeansDenoising(img_array)
print(img_array)
print("PRE-PROCESSED IMAGE")
cv2.imshow("",img_array)
cv2.waitKey(0)

# Limiting the number of images per category to 500
LIMIT_PER_CATEGORY = 5000

# Image size for the neural network
IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        count = 0  # Counter to keep track of the number of images processed
        for img in os.listdir(path):
            try:
                if count >= LIMIT_PER_CATEGORY:
                    break  # Stop processing images for this category if limit reached
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
                img_array = clahe.apply(img_array)
                median  = cv2.medianBlur(img_array.astype('uint8'), 5)
                median = 255-median
                ret, thresh = cv2.threshold(median.astype('uint8'), 165, 255, cv2.THRESH_BINARY_INV)
                new_array = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                count += 1
            except Exception as e:
                pass

# Call the function to create training data
create_training_data()

# Shuffle the training data
random.shuffle(training_data)

# Prepare features (X) and labels (y)
X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert lists to numpy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0  # Normalize pixel values to be between 0 and 1
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Define the CNN model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(len(CATEGORIES)))  # Output layer with the number of categories
model.add(Activation("softmax"))

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

# Evaluate the model on the training set
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f'Training accuracy: {train_acc:.4f}')

# Save the model
model.save('CNN.h5')

# Plotting accuracy and loss graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate and print accuracy
test_accuracy = metrics.accuracy_score(y_test, y_pred_classes)
print(f'Test accuracy: {test_accuracy:.4f}')

# Generate confusion matrix and classification report
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
print('\nConfusion Matrix:\n', confusion_mtx)
print('\nClassification Report:\n', classification_report(y_test, y_pred_classes))

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
