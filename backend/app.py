from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Allow frontend access
UPLOAD_FOLDER = "data"
MODEL_FOLDER = "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global Variables
dataset_path = None  # Stores uploaded dataset path
selected_model = None  # Stores chosen clustering model
cleaned_df = None  # Stores preprocessed data

# @app.route("/run-clustering", methods=["POST"])
# def run_clustering():
#     """Runs clustering and returns results as JSON."""
#     data = df.copy()

#     # Perform dimensionality reduction (optional, for visualization)
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(data)

#     # Run KMeans clustering
#     kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(reduced_data)

#     # Prepare response
#     result = pd.DataFrame(reduced_data, columns=["x", "y"])
#     result["cluster"] = clusters

#     return jsonify(result.to_dict(orient="records"))  # Convert DataFrame to JSON and return

# Upload Dataset
@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    global dataset_path
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(dataset_path)
    
    return jsonify({"message": "File uploaded successfully", "dataset_path": dataset_path})

@app.route("/read-dataset", methods=["GET"])
def read_dataset():
    # Get the dataset path from the request
    dataset_path = request.args.get("dataset_path")

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"error": "Invalid or missing file path"}), 400
    
    try:
        # Load dataset
        df = pd.read_excel(dataset_path)

        return jsonify({
            "message": "Data read successfully",
            "content": df.head().to_dict(orient="records")  # Convert DataFrame to JSON
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle file read errors

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run on port 5000
