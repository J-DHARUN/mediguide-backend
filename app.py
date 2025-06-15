from flask import Flask, request, jsonify
from flask_cors import CORS
from prescription_nlp import doctr_extract, group_prescription_blocks, parse_line

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ MediGuide AI backend is live!"

@app.route("/process", methods=["POST"])
def parse_prescription():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = "temp_upload.png"
    file.save(temp_path)

    try:
        lines = doctr_extract(temp_path)
        blocks = group_prescription_blocks(lines)
        parsed = [parse_line(b) for b in blocks if parse_line(b)]
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
