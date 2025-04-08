'''Original Server For User Study'''

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import numpy as np

app = Flask(__name__)
CORS(app)

used_offsets = set()  
current_offset = 1  
# model = joblib.load("mlp_level_predictor.pkl")

def get_sensor_data():
    heart_rate = random.uniform(60, 100)  
    skin_conductance = random.uniform(0.1, 1.0)
    eeg_signal = random.uniform(0.1, 0.5)
    stress_level = random.uniform(1, 10)
    return np.array([[heart_rate, skin_conductance, eeg_signal, stress_level]])

def get_sequential_offset():
    """Get an incrementing offset value (incremented one by one) and ensure no duplicates"""
    global current_offset
    if current_offset > 23:
        current_offset = 1 
    offset = current_offset
    current_offset += 1
    return offset

def get_unique_random_offset():
    """Generate a unique random number between 1 and 23"""
    global used_offsets
    available_numbers = set(range(1, 24)) - used_offsets  # Calculate remaining available numbers

    if not available_numbers:
        used_offsets.clear()  # If all numbers are used, clear the set and restart
        available_numbers = set(range(1, 24))

    new_offset = random.choice(list(available_numbers))  # Choose a new number
    used_offsets.add(new_offset)  # Record the used number
    return new_offset

@app.route('/predict_level_offset', methods=['GET'])
def predict_level_offset():
    try:
        mode = request.args.get('mode', 'random')  # Get the mode parameter, default is random mode
        sensor_data = get_sensor_data()

        if mode == 'sequential':
            predicted_offset_rounded = get_sequential_offset()
        elif mode == 'random':
            predicted_offset_rounded = get_unique_random_offset()
        elif mode == 'model':
            # Replace this with real model prediction logic
            predicted_offset_rounded = random.randint(1, 23)  # Assume the model predicts a random number
        else:
            return jsonify({"error": "Invalid mode. Use 'sequential', 'random', or 'model'."})

        print("=== Flask Debug ===")
        print("Mode:", mode)
        print("Sensor Data:", sensor_data)
        print("Predicted Level Offset:", predicted_offset_rounded)
        print("Used Offsets (if random mode):", used_offsets)
        print("===================")

        return jsonify({"next_level_offset": predicted_offset_rounded})
    except Exception as e:
        print("Error in predict_level_offset:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
