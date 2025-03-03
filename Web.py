from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_cancer(img_path):
    """
    Simulate prediction - replace with actual model inference
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize

    # Placeholder prediction (Random probabilities)
    lung_cancer_prob = round(np.random.uniform(0, 1) * 100, 2)
    colon_cancer_prob = round(np.random.uniform(0, 1) * 100, 2)

    return lung_cancer_prob, colon_cancer_prob

def generate_graph(lung_cancer_prob, colon_cancer_prob):
    labels = ["Lung Cancer", "Colon Cancer"]
    probabilities = [lung_cancer_prob, colon_cancer_prob]
    
    plt.figure(figsize=(5,5))
    plt.bar(labels, probabilities, color=["blue", "green"])
    plt.ylim([0, 100])
    plt.ylabel("Likelihood (%)")
    plt.title("Cancer Prediction Results")

    graph_path = os.path.join("static", "prediction_graph.png")
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part"
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Predict cancer likelihood
        lung_cancer_prob, colon_cancer_prob = predict_cancer(file_path)
        
        # Generate graph
        graph_path = generate_graph(lung_cancer_prob, colon_cancer_prob)
        
        return render_template(
            "index.html", 
            image_path=file_path, 
            lung_cancer_prob=lung_cancer_prob, 
            colon_cancer_prob=colon_cancer_prob, 
            graph_path=graph_path
        )

if __name__ == "__main__":
    app.run(debug=True)
