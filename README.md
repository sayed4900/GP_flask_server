# Eye Tracking Prediction API

A Flask microservice that predicts eye gaze direction from images using a pre-trained TensorFlow/Keras model.

## Features

- **Eye direction prediction**: Classifies eye gaze into four categories: `close`, `forward`, `left`, `right`.
- **Simple REST API**: Accepts image uploads via POST request and returns the predicted direction.
- *(Planned)* **Handwritten character recognition**: Future support for OCR on handwritten letters (commented-out code ready).

## Tech Stack

- **Framework**: Flask (Python)
- **Machine Learning**: TensorFlow / Keras
- **Image Processing**: Pillow (PIL)
- **Cross-Origin Support**: Flask-CORS


