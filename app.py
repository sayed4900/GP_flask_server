from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_preprocess
from PIL import Image
from tensorflow import keras
import numpy as np
# from joblib import joblib

app = Flask(__name__)
CORS(app)


# Function to preprocess the image
def preprocess_image(image_path, target_width, target_height):
    img = img_preprocess.load_img(image_path, target_size=(target_width, target_height), color_mode='grayscale')
    img_array = img_preprocess.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array



# Load the model
eye_tracking_model = keras.models.load_model('./AI_models/model.hdf5')

# Define the desired input size for the model
desired_width = 94
desired_height = 94

# Define directions
directions = ['close', 'forward', 'left', 'right'] 

# Function to preprocess image and get max direction
def get_max_direction(image_array):
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    prediction = eye_tracking_model.predict(np.expand_dims(image_array, axis=0))
    max_index = np.argmax(prediction[0])
    return directions[max_index]

    

@app.route('/predict-eye', methods=['POST'])
def predict_eye_direction():
    # Get image data from the request
    file = request.files['image']
    image = Image.open(file)
    image = image.resize((desired_width, desired_height))
    image_array = np.array(image) / 255.0
    
    # Get max direction
    max_dir = get_max_direction(image_array)
    
    # Return prediction
    return jsonify({'direction': max_dir})

if __name__ == '__main__':
    app.run(debug=True)






# def number_to_char(num):
#     if num < 0 or num > 25:
#         return "Out of range"
#     else:
#         return chr(ord('A') + num)


# @app.route('/predict-hand-written', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     try:
#         # Save the received image 
#         image_path = './temp_image.png'
#         file.save(image_path)

#         # Define your target width and height as expected by the model
#         target_width, target_height = 28, 28  # Replace with the model's input dimensions

#         # Preprocess the image
#         processed_image = preprocess_image(image_path, target_width, target_height)

#         # Make predictions using the model
#         predictions = model_handwirte.predict(processed_image)

#         # Get the predicted class or classes depending on your model's output
#         predicted_class = np.argmax(predictions, axis=1)
#         predicted_class = number_to_char(predicted_class[0])

#         # Return the predicted class
#         return jsonify(predicted_class)

#     except Exception as e:
#         return jsonify({'error': str(e)})