from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_model.joblib')

# Define reasonable ranges for validation
VALID_RANGES = {
    'N': {'min': 0, 'max': 200},
    'P': {'min': 0, 'max': 200},
    'K': {'min': 0, 'max': 200},
    'temperature': {'min': 0.0, 'max': 50.0},
    'humidity': {'min': 0.0, 'max': 100.0},
    'ph': {'min': 0.0, 'max': 14.0},
    'rainfall': {'min': 0.0, 'max': 400.0}
}

# Mapping for crop images (assuming lowercase, .png, and in static/images)
# You will need to provide these images in C:\Users\HP\gemnini\crop-recommender\static\images
CROP_IMAGE_MAP = {
    'rice': 'rice.png',
    'maize': 'maize.png',
    'chickpea': 'chickpea.png',
    'kidneybeans': 'kidneybeans.png',
    'pigeonpeas': 'pigeonpeas.png',
    'mothbeans': 'mothbeans.png',
    'mungbean': 'mungbean.png',
    'blackgram': 'blackgram.png',
    'lentil': 'lentil.png',
    'pomegranate': 'pomegranate.png',
    'banana': 'banana.png',
    'mango': 'mango.png',
    'grapes': 'grapes.png',
    'watermelon': 'watermelon.png',
    'muskmelon': 'muskmelon.png',
    'apple': 'apple.png',
    'orange': 'orange.png',
    'papaya': 'papaya.png',
    'coconut': 'coconut.png',
    'cotton': 'cotton.png',
    'jute': 'jute.png',
    'coffee': 'coffee.png',
    # Add more crop mappings as needed
}
DEFAULT_CROP_IMAGE = 'default_crop.png' # You need to provide this image

@app.route('/')
def home():
    return render_template('index.html')

def validate_input(form_data):
    errors = []
    for key, ranges in VALID_RANGES.items():
        try:
            value = float(form_data.get(key))
            if not (ranges['min'] <= value <= ranges['max']):
                errors.append(f"{key.capitalize()} value ({value}) is out of the recommended range ({ranges['min']}-{ranges['max']}).")
        except (ValueError, TypeError):
            errors.append(f"Invalid input for {key.capitalize()}. Please enter a number.")
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Server-side validation
    validation_errors = validate_input(form_data)
    if validation_errors:
        return render_template('index.html', error_message='<br>'.join(validation_errors), form_data=form_data)

    # Convert form data to a list of floats in the correct order
    try:
        input_data = [
            float(form_data['N']),
            float(form_data['P']),
            float(form_data['K']),
            float(form_data['temperature']),
            float(form_data['humidity']),
            float(form_data['ph']),
            float(form_data['rainfall'])
        ]
    except (ValueError, KeyError) as e:
        return render_template('index.html', error_message=f"Error processing input: Missing or invalid data for {e}.", form_data=form_data)

    # Make prediction
    prediction = model.predict([input_data])[0]
    
    # Determine crop image path
    crop_image_filename = CROP_IMAGE_MAP.get(prediction.lower(), DEFAULT_CROP_IMAGE)
    image_path_check = os.path.join(app.static_folder, 'images', crop_image_filename)
    
    # Check if the image actually exists, otherwise use default
    if not os.path.exists(image_path_check):
        crop_image_filename = DEFAULT_CROP_IMAGE

    return render_template('index.html', 
                           recommendation=prediction, 
                           form_data=form_data, 
                           crop_image=crop_image_filename)

if __name__ == '__main__':
    app.run(debug=True)