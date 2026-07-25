import random
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/battery/status', methods=['GET'])
def get_battery_status():
    status = {
        "Voltage": round(random.uniform(3.0, 4.2), 2),
        "Current": round(random.uniform(-2.0, 2.0), 2),
        "Temp": round(random.uniform(25.0, 45.0), 2),
        "SOC": round(random.uniform(0.0, 100.0), 2),
        "SOH": round(random.uniform(80.0, 100.0), 2)
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
