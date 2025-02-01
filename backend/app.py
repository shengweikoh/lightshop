from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import os
import json
import io
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Set upload folders
DATA_FOLDER = os.path.join(os.getcwd(), "data/")
ENCODED_FOLDER = os.path.join(os.getcwd(), "encoded_data/")
MODELS_FOLDER = os.path.join(os.getcwd(), "models/")

app.config["UPLOAD_FOLDER"] = DATA_FOLDER
app.config["ENCODED_FOLDER"] = ENCODED_FOLDER
app.config["MODELS_FOLDER"] = MODELS_FOLDER

# Ensure directories exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ENCODED_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Modify model if needed

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Extract spaCy's default stop words list
spacy_stop_words = nlp.Defaults.stop_words

# Upload Dataset
@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
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
    dataset_path = "".join([DATA_FOLDER, request.args.get("dataset_path")])

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
        return jsonify({"error": str(e)}), 500


# Process and Encode Dataset Route
@app.route("/process-dataset", methods=["POST"])
def process_dataset():
    dataset_path = "".join([DATA_FOLDER, request.args.get("dataset_path")])
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"error": "Invalid or missing file path"}), 400

    # Encode dataset and save embeddings
    return encode_dataset(dataset_path)

@app.route("/find-optimal-clusters", methods=["POST"])
def find_optimal_clusters_hierarchical():
    encoded_filename = os.path.basename(request.args.get("dataset_path")).replace(".xlsx", "_encoded.npy")
    encoded_path = os.path.join(app.config["ENCODED_FOLDER"], encoded_filename)
    if not encoded_path or not os.path.exists(encoded_path):
        return jsonify({"error": "Invalid or missing file path"}), 400
    
    try:
        embeddings = np.load(encoded_path)
        base64_img = find_optimal_clusters_hierarchical(embeddings, 50)

        return jsonify({
            "message": "Optimal clusters found successfully",
            "plot": base64_img
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/make-clusters", methods=["POST"])
def make_clusters():
    dataset_path = "".join([DATA_FOLDER, request.args.get("dataset_path")])
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"error": "Invalid or missing file path"}), 400

    encoded_filename = os.path.basename(request.args.get("dataset_path")).replace(".xlsx", "_encoded.npy")
    encoded_path = os.path.join(app.config["ENCODED_FOLDER"], encoded_filename)
    if not encoded_path or not os.path.exists(encoded_path):
        return jsonify({"error": "Invalid or missing file path"}), 400

    n_clusters = int(request.args.get("n_clusters"))
    if not n_clusters:
        return jsonify({"error": "Invalid or missing cluster number"}), 400

    try:
        df = pd.read_excel(dataset_path)
        embeddings = np.load(encoded_path)
        cluster_labels = hierarchical_clustering(embeddings, n_clusters=n_clusters)

        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)

        # Store clusters in a list
        cluster_list = []

        for i, cluster in clusters.items():
            # Gather texts in a dictionary format: {rowId: text}
            cluster_texts = {idx: df["Text"].iloc[idx] for idx in cluster}
            
            # Extract keywords from all texts in the cluster
            keywords = extract_keywords_from_text(list(cluster_texts.values()), top_n=5)

            # Store cluster details
            cluster_dict = {
                "cluster_keywords": keywords,
                "cluster_texts": cluster_texts  # {rowId: text}
            }
            
            # Append to the list
            cluster_list.append(cluster_dict)

        # Save clusters as a model as a json file
        cluster_path = os.path.join(app.config["MODELS_FOLDER"], request.args.get("dataset_path")).replace(".xlsx", "_cluster.json")
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(cluster_list, f, indent=4, ensure_ascii=False)
        
        return jsonify({
            "message": "Clusters formed successfully",
            "clusters": cluster_list
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper Functions
def encode_dataset(dataset_path):
    try:
        df = pd.read_excel(dataset_path, engine="openpyxl")

        if "Text" not in df.columns:
            return jsonify({"error": "Column 'Text' not found in dataset"}), 400

        # Encode the corpus into dense embeddings
        corpus_embeddings = model.encode(df["Text"].tolist(), convert_to_tensor=False)

        # Save embeddings
        encoded_filename = os.path.basename(dataset_path).replace(".xlsx", "_encoded.npy")
        encoded_path = os.path.join(app.config["ENCODED_FOLDER"], encoded_filename)
        np.save(encoded_path, corpus_embeddings)  # Save as NumPy file

        return jsonify({"message": "Dataset encoded successfully", "encoded_path": encoded_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle file read errors

def preprocess_text_with_spacy(text):
    """
    Tokenize, remove stop words using spaCy, and perform lemmatization.
    Args:
        text: A single string.
    Returns:
        List of processed tokens.
    """
    doc = nlp(text)
    processed_tokens = [
        token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in spacy_stop_words
    ]
    return processed_tokens

def extract_keywords_from_text(texts, top_n=5):
    """
    Extract keywords from a list of texts using TF-IDF with spaCy stop word removal and lemmatization.
    Args:
        texts: List of strings representing the cluster.
        top_n: Number of top keywords to extract.
    Returns:
        List of top keywords.
    """
    # Define the custom TF-IDF Vectorizer with a spaCy-based tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text_with_spacy,
        stop_words=None,  # Custom stop words are handled in the tokenizer
        max_features=1000
    )
    X = vectorizer.fit_transform(texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = X.sum(axis=0).A1.argsort()[::-1]  # Sort features by importance

    top_keywords = [feature_array[i] for i in tfidf_sorting[:top_n]]
    return top_keywords

def hierarchical_clustering(embeddings, n_clusters):
    """
    Perform Hierarchical Clustering on embeddings.
    Args:
        embeddings: Array of embeddings (e.g., dense vectors for texts).
        n_clusters: Number of clusters to form.
    Returns:
        List of cluster labels for each data point.
    """
    # Convert embeddings to NumPy if necessary
    if hasattr(embeddings, "cpu"):  # Check if embeddings are tensors
        embeddings = embeddings.cpu().numpy()

    # Perform Agglomerative Clustering
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering_model.fit_predict(embeddings)

    return cluster_labels

def find_optimal_clusters_hierarchical(embeddings, max_clusters=10):
    if hasattr(embeddings, "cpu"):  # Check if embeddings are tensors
        embeddings = embeddings.cpu().numpy()

    silhouette_scores = []

    # Test different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)


    # Plot the Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Hierarchical Clustering")
    plt.grid()

    # Save plot as a Base64 image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    base64_img = base64.b64encode(img_buf.read()).decode("utf-8")
    plt.close()  # Close plot to free memory
    return base64_img

if __name__ == "__main__":
    app.run(debug=True, port=5000)