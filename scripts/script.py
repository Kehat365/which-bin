import os
import sys
import dill
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import FunctionTransformer
import mlflow
import mlflow.sklearn
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image, ImageOps
import cv2
from skimage.feature import hog, local_binary_pattern
from typing import Any

# Paths
DATASET_PATH = Path(__file__).parent.parent / "data" / "raw" / "garbage-dataset"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "garbage_classification_pipeline.pkl"

# Constants
CLASS_SIZE = 70
RANDOM_STATE = 42
TARGET_SIZE = (256, 256)

# Image resizing functions
def resize_with_padding(imag):
    img = ImageOps.contain(imag, TARGET_SIZE, Image.Resampling.LANCZOS)        # Resize while maintaining aspect ratio
    new_img = Image.new("RGB", TARGET_SIZE, (255, 255, 255))        # Create a new image with white background and exact size
    
    #- Calculate positions to center the resized image
    offset_x = (TARGET_SIZE[0] - img.size[0]) // 2
    offset_y = (TARGET_SIZE[1] - img.size[1]) // 2

    new_img.paste(img, (offset_x, offset_y))        # Paste the resized image onto the image with white background
    
    return new_img

def resize_images(df):
    for index, row in df.iterrows():
        image_path = row['path']

        try:
            with Image.open(image_path) as img:
                img = resize_with_padding(img)
                #- Update the new dimensions in the DataFrame
                new_width, new_height = img.size
                df.at[index, 'width'] = new_width
                df.at[index, 'height'] = new_height
                df.at[index, 'mode'] = img.mode
                img.save(image_path)  # Replace existing image

        except Exception as e:
            print(f"Erreur lors du redimensionnement de l'image {image_path}: {e}")
    return df

# Feature extraction functions
def flatten_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    return img.flatten()

def hog_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hog_desc, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_desc

def color_histogram(image_path, bins=32):
    img = cv2.imread(image_path)
    hist_features = list()
    for i in range(3):  # Loop over each color channel (B, G, R)
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist_features.extend(hist.flatten())
    return np.array(hist_features)

def lbp_features(image_path, num_points=24, radius=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp = local_binary_pattern(img, num_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist

def resnet_features(image_path):
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_data = keras_image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = resnet_model.predict(img_data)
    return features.flatten()

def combined_features(image_path):
    features = []
#    features.extend(flatten_image(image_path))      
#    features.extend(hog_features(image_path))    
#    features.extend(color_histogram(image_path)) 
#    features.extend(lbp_features(image_path))    
    features.extend(resnet_features(image_path))  
    return np.array(features)

# Function to build pipeline
def build_pipeline() -> Pipeline:
    feature_extractor = FunctionTransformer(lambda x: np.array([combined_features(path) for path in x]))
    pipeline = Pipeline([
        ('feature_extractor', feature_extractor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ])
    return pipeline

# Function to load and split dataset
def load_and_split_dataset(filepath: Path = DATASET_PATH) -> tuple:
    paths = list()
    classes = list()
    widths = list()
    heights = list()
    modes = list()

    #- Browse folders to retrieve image paths and properties
    for dirname, _, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            file_path = os.path.normpath(os.path.join(dirname, filename))       # Build the file path
            paths.append(file_path)
            image_class = os.path.basename(dirname)
            classes.append(image_class)

            with Image.open(file_path) as img:      # Open the image to retrieve its dimensions and number of channels
                width, height = img.size
                widths.append(width)
                heights.append(height)
                modes.append(img.mode)

    class_names = sorted(set(classes))
    normal_mapping = dict(zip(class_names, range(len(class_names))))

    #- Create the DataFrame to store all the information
    df = pd.DataFrame({
        'path': paths,
        'class': classes,
        'label': [normal_mapping[c] for c in classes],  # Map classes to labels
        'width': widths,
        'height': heights,
        'mode': modes
    })
    df = resize_images(df)  # Resize images before extracting features
    X, y = df['path'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
    return (X_train, X_test, y_train, y_test)

# Function to train and save model
def train_model(output_path: Path = OUTPUT_PATH) -> None:
    X_train, X_test, y_train, y_test = load_and_split_dataset()
    model = build_pipeline()

    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Save metrics to score.txt
    score_path = Path("out") / "score.txt"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        f.write(f"Train Score: {train_score}")
        f.write(f"Test Score: {test_score}")
        f.write(f"Accuracy: {accuracy}")
        f.write(f"Precision: {precision}")
        f.write(f"Recall: {recall}")
        f.write(f"F1 Score: {f1}")

    # Save model to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        dill.dump(model, f)
    print(f"Model saved to file: {output_path.resolve()}")

if __name__ == "__main__":
    train_model()
