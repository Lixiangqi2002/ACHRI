'''Real Time Server for User Study'''

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import numpy as np

app = Flask(__name__)
CORS(app)
all_predictions = [] 
used_levels = set()  
DEFAULT_START_LEVEL = 3
last_current_level = 3


def real_time_level_offset():
    global all_predictions
    global used_levels
    global last_current_level

    if len(all_predictions) == 0:
        return 2  
    
    ratio_above_threshold = sum(1 for p in all_predictions if p > 0.6) / len(all_predictions)
    print("Ratio of predictions > 0.6:", ratio_above_threshold)
    ratio_below_threshold = sum(1 for p in all_predictions if p < 0.4) / len(all_predictions)
    print("Ratio of predictions < 0.4:", ratio_below_threshold)

    if ratio_above_threshold > ratio_below_threshold:
        print("More predictions above 0.6")
        offset_level = -1
    else:
        print("More predictions below 0.4")
        offset_level = 1



    print("=== Flask Debug ===")
    print("Predicted Level Offset:", offset_level)
    print("===================")

    return offset_level


@app.route('/predict_level_offset', methods=['POST'])
def receive_prediction():
    global last_current_level
    try:
        data = request.json
        prediction = data.get("prediction")
        current_level = data.get("current_level") 

        if prediction is not None:
            all_predictions.append(prediction) 
            # used_levels.add(last_current_level)
            return jsonify({"prediction": prediction})
        else:
            return jsonify({"error": "No prediction provided"}), 400

    except Exception as e:
        print("Error receiving prediction:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/predict_level_offset', methods=['GET'])
def predict_level_offset():
    try:
        offset_level = real_time_level_offset()
        latest_prediction = all_predictions[-1] if all_predictions else -1

        return jsonify({"next_level_offset": offset_level,
                        "prediction": latest_prediction})
    except Exception as e:
        print("Error in predict_level_offset:", str(e))
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=False)
