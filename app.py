from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    form_data = request.form.to_dict()
    
    # Convert form data to a list of floats in the correct order
    try:
        input_data = [float(form_data[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    except (ValueError, KeyError) as e:
        return f"Error processing input: {e}", 400

    # Make prediction
    prediction = model.predict([input_data])
    
    # Render the page again with the recommendation and the original form data
    return render_template('index.html', recommendation=prediction[0], form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
