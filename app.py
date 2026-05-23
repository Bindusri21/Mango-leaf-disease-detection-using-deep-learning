from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os

# Try to import TensorFlow and Keras helpers. These are optional so the
# app can start for UI work without the heavy TensorFlow install. If
# TensorFlow isn't available, the model remains None and POST requests
# will return a clear error message.
has_tensorflow = True
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
except Exception:
    has_tensorflow = False
    load_model = None
    # Minimal fallbacks so image preprocessing doesn't crash if called
    # accidentally. These do NOT perform the same preprocessing as
    # EfficientNet — they just keep the code paths working for the UI.
    def efficientnet_preprocess_input(x):
        # simple normalization to 0-1
        return x.astype('float32') / 255.0

    def img_to_array(image):
        arr = np.array(image)
        # Ensure channels-last and float32
        return arr.astype('float32')

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "mango_leaf_disease_efficientnetb0.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class names
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the input image for EfficientNetB0:
    - Resize to target size
    - Convert to array
    - Use EfficientNetB0 preprocess_input for normalization
    """
    try:
        # Resize image
        image = image.resize(target_size)

        # Convert image to numpy array
        image_array = img_to_array(image)

        # Preprocess using EfficientNetB0's preprocess_input
        image_array = efficientnet_preprocess_input(image_array)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        try:
            # Open the image file
            image = Image.open(file.stream)

            # Preprocess the image
            input_image = preprocess_image(image)
            if input_image is None:
                return jsonify({"error": "Failed to preprocess image"}), 500

            # Predict with the model
            predictions = model.predict(input_image)

            # Get the class with the highest probability
            predicted_class = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)

            return jsonify({"predicted_class": predicted_class, "confidence": confidence})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
