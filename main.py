import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template_string, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Define the Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define the folder for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Dummy class names for demonstration (replace with actual plant disease names)
class_names = ['Healthy', 'Disease_A', 'Disease_B', 'Disease_C', 'Disease_D']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML template as a string
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Identifier</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container">
        <h1 class="mt-5 text-center">Plant Disease Identifier</h1>
        <p class="text-center">Upload an image of a plant to identify the disease.</p>
        <div class="row justify-content-center">
            <div class="col-md-6">
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Choose an image:</label>
                        <input type="file" class="form-control-file" id="file" name="file">
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Upload and Predict</button>
                </form>
                {% with messages = get_flashed_messages() %}
                    {% if messages %}
                        <div class="alert alert-info mt-4">
                            {{ messages[0] }}
                        </div>
                    {% endif %}
                {% endwith %}
            </div>
        </div>
    </div>
</body>
</html>
'''

# Home route
@app.route('/')
def index():
    return render_template_string(html_template)

# Upload and predict route
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_disease(filepath)
        flash(f'The uploaded plant is likely to have: {prediction}')
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

# Function to predict disease from image
def predict_disease(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the disease
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]

# Start the Flask app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
