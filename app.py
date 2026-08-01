from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("video_model.h5")

# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")

# Preprocess uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = image.reshape(1, 1, 64, 64, 1)

    # Repeat single frame to 10 frames
    image = np.repeat(image, 10, axis=1)

    return image

@app.route('/')
def home():
    return render_template("index.html", show_frames=False)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Convert image to array
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    input_data = preprocess_image(image)

    prediction = model.predict(input_data)

    print("Prediction shape:", prediction.shape)

    # Save predicted frames
    for i in range(10):
        frame = prediction[0, i, :, :, 0] * 255
        frame = frame.astype(np.uint8)
        cv2.imwrite(f"static/frame_{i}.png", frame)

    return render_template("index.html", show_frames=True)

if __name__ == "__main__":
    app.run(debug=True)
