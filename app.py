from flask import Flask, request, jsonify
from flask_cors import CORS
from prescription_nlp import doctr_extract, group_prescription_blocks, parse_line

app = Flask("MediGuide AI")
CORS(app)

@app.route("/parse", methods=["POST"])
def parse_prescription():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = "temp_upload.png"
    file.save(temp_path)

    try:
        lines = doctr_extract(temp_path)
        blocks = group_prescription_blocks(lines)
        parsed = [parse_line(b) for b in blocks if parse_line(b)]
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
