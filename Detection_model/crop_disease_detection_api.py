import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ----------------------------
# CONFIGURATION
# ----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_PATH = "PlantVillage/"  # <-- Change this to your dataset folder path

MODEL_PATH = "crop_disease_model.h5"

# ----------------------------
# MODEL TRAINING FUNCTION
# ----------------------------
def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=20)
    model.save(MODEL_PATH)

    print("✅ Model training complete and saved as", MODEL_PATH)

# ----------------------------
# FLASK API FOR PREDICTION
# ----------------------------
app = Flask(__name__)
model = None
class_labels = {}

def load_trained_model():
    global model, class_labels
    model = load_model(MODEL_PATH)

    # Load class labels from dataset
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    temp_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical'
    )
    class_labels = {v: k for k, v in temp_gen.class_indices.items()}

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    file = request.files["file"]
    img = load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    result = class_labels[class_index]
    return jsonify({"prediction": result})

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No trained model found. Starting training...")
        train_model()
    load_trained_model()
    app.run(port=5000, debug=True)
