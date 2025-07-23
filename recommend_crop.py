import joblib
import sys

# Load the trained model
model = joblib.load('crop_model.joblib')

if len(sys.argv) != 8:
    print("Usage: python recommend_crop.py <N> <P> <K> <temperature> <humidity> <ph> <rainfall>")
    sys.exit(1)

# Get user input from command line arguments
try:
    N = float(sys.argv[1])
    P = float(sys.argv[2])
    K = float(sys.argv[3])
    temperature = float(sys.argv[4])
    humidity = float(sys.argv[5])
    ph = float(sys.argv[6])
    rainfall = float(sys.argv[7])
except ValueError:
    print("Error: All inputs must be numbers.")
    sys.exit(1)


# Create a list from the user input
user_input = [[N, P, K, temperature, humidity, ph, rainfall]]

# Make a prediction
prediction = model.predict(user_input)

# Display the result
print("\n-------------------------------------")
print(f"Based on the provided data, the recommended crop is: {prediction[0]}")
print("-------------------------------------")
