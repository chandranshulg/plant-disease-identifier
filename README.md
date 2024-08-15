# Plant Disease Identifier

This is a Flask-based web application that allows users to upload images of plants to identify potential diseases using a pre-trained MobileNetV2 model. The model predicts the likelihood of a plant having a specific disease or being healthy.

## Features

- **Image Upload**: Users can upload an image of a plant in PNG, JPG, or JPEG format.
- **Disease Prediction**: The application uses a deep learning model to predict the disease present in the uploaded plant image.
- **Real-time Feedback**: After uploading an image, the user receives a prediction with the name of the disease or if the plant is healthy.

## Installation

## requirments

Flask==2.3.3

tensorflow==2.13.0

numpy==1.25.1

Pillow==10.0.0

werkzeug==2.3.3

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/chandranshulg/plant-disease-identifier.git
    ```
   
2. **Navigate to the project directory:**
    ```bash
    cd plant-disease-identifier
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    python app.py
    ```

5. **Access the application in your browser:**
    ```
    http://127.0.0.1:5000/
    ```

## Model

The application uses the MobileNetV2 model pre-trained on ImageNet, with additional dense layers to classify plant diseases. The final layer outputs a probability distribution over 5 classes (Healthy and 4 types of diseases).

### Class Names

The current implementation includes the following dummy class names (replace these with actual disease names as needed):

- Healthy
- Disease_A
- Disease_B
- Disease_C
- Disease_D

## File Structure

```plaintext
plant-disease-identifier/
│
├── app.py                     # Main Flask application
├── uploads/                   # Directory for uploaded images
├── static/
│   └── images/                # Directory for any static images
│
└── templates/
    └── index.html             # HTML template for the application
