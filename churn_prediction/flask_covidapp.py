# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import traceback
import base64
from io import BytesIO
import lime
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

app = Flask(__name__)
CORS(app, support_credentials=True)

# Load the  models
try:
    dnn_model = load_model('models/dnn_model.keras')
    cnn_model = load_model('models/cnn_model.keras')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(traceback.format_exc())

def get_lime_explanation(model, image):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=3, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 2 + 0.5, mask)
    return img_boundry1

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image found'}), 400

        image_file = request.files['image']
        image = Image.open(image_file)
        image = image.convert('L')
        image = image.resize((128, 128))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        print("Image pre-processed successfully")

        dnn_pred = dnn_model.predict(image_array)
        cnn_pred = cnn_model.predict(image_array)

        print("Predictions made successfully")

        dnn_lime = get_lime_explanation(dnn_model, image_array)
        cnn_lime = get_lime_explanation(cnn_model, image_array)

        dnn_lime_image = Image.fromarray((dnn_lime * 255).astype(np.uint8))
        cnn_lime_image = Image.fromarray((cnn_lime * 255).astype(np.uint8))

        buffer_dnn = BytesIO()
        buffer_cnn = BytesIO()
        dnn_lime_image.save(buffer_dnn, format="PNG")
        cnn_lime_image.save(buffer_cnn, format="PNG")
        dnn_lime_base64 = base64.b64encode(buffer_dnn.getvalue()).decode('utf-8')
        cnn_lime_base64 = base64.b64encode(buffer_cnn.getvalue()).decode('utf-8')

        print("LIME explanations generated successfully")

        return jsonify({
            'dnn_prediction': dnn_pred.tolist(),
            'cnn_prediction': cnn_pred.tolist(),
            'dnn_lime': dnn_lime_base64,
            'cnn_lime': cnn_lime_base64
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
