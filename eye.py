# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image

# # Load the eye tracking model
# eye_tracking_model = keras.models.load_model('./AI_models/model.hdf5')

# # image_path = './imgs/eye_closed(3183).png'
# # image_path = './imgs/right_(1902).png'
# # image_path = './imgs/left_(1044).png'
# image_path = './imgs/forward_look (1927).png'

# # Define the desired input size for the model
# desired_width = 94
# desired_height = 94

# # Load the image
# image = Image.open(image_path)
# # Resize the image to match the input size expected by the model
# image = image.resize((desired_width, desired_height))
# # Convert the image to array
# image_array = np.array(image) / 255.0  # Normalize pixel values

# directions  = ['close_look', 'forward', 'left', 'right'] 

# # Ensure the image has three color channels (for RGB images)
# if len(image_array.shape) == 2:
#     image_array = np.stack((image_array,) * 3, axis=-1)

# # Predict using the model
# prediction = eye_tracking_model.predict(np.expand_dims(image_array, axis=0))
# def max_direction (arr):
#     arr = np.array(arr)
#     max_index = np.argmax(arr)
#     return directions[max_index]


# print(max_direction(prediction[0]))
