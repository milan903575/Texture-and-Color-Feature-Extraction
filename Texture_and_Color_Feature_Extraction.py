import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load images and labels
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize for consistency
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Update the paths to the 'train' folder and use 'seg_pred' as the file name prefix.
forest_images_path = 'train/seg_pred/forest'  # Path to forest images in the train folder
mountain_images_path = 'train/seg_pred/mountain'  # Path to mountain images in the train folder

# Load forest and mountain images
forest_images, forest_labels = load_images(forest_images_path, 0)
mountain_images, mountain_labels = load_images(mountain_images_path, 1)

# Combine and shuffle
X = np.concatenate([forest_images, mountain_images])
y = np.concatenate([forest_labels, mountain_labels])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction (GLCM, LBP, and color histogram)
def glcm_features(image):
    gray = rgb2gray(image)
    glcm = greycomatrix((gray * 255).astype(np.uint8), distances=[5], angles=[0], levels=256, 
                        symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    return [contrast, homogeneity, energy, correlation]

def lbp_features(image):
    gray = rgb2gray(image)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract features for the dataset
def extract_features(images):
    features = []
    for img in images:
        glcm = glcm_features(img)
        lbp = lbp_features(img)
        color_hist = color_histogram(img)
        combined = np.concatenate([glcm, lbp, color_hist])
        features.append(combined)
    return np.array(features)

# Extract features for training and testing sets
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, 
                    validation_data=(X_test_scaled, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predict on test data
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting accuracy and loss graphs
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Forest', 'Mountain'], 
            yticklabels=['Forest', 'Mountain'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()
