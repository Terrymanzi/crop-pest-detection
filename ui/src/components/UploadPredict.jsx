import React, { useState } from "react";
import axios from "axios";
import API_URL from "../config";

const UploadPredict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    processFile(file);
  };

  const processFile = (file) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setError(null);
      setResult(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError("Please select a valid image file");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);

    const file = e.dataTransfer.files[0];
    processFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail || "Prediction failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="card">
      <h2 style={{ marginBottom: "20px", color: "#333" }}>Image Prediction</h2>

      {!preview ? (
        <div
          className={`upload-area ${dragOver ? "dragover" : ""}`}
          onClick={() => document.getElementById("file-input").click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="upload-input"
          />
          <div style={{ fontSize: "3rem", marginBottom: "16px" }}>📷</div>
          <p style={{ fontSize: "1.1rem", marginBottom: "8px" }}>
            Click to upload or drag and drop
          </p>
          <p style={{ color: "#999", fontSize: "0.9rem" }}>
            PNG, JPG, GIF up to 10MB
          </p>
        </div>
      ) : (
        <div style={{ textAlign: "center" }}>
          <img src={preview} alt="Preview" className="preview-image" />

          <div
            style={{
              marginTop: "20px",
              display: "flex",
              gap: "10px",
              justifyContent: "center",
            }}
          >
            <button
              onClick={handlePredict}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? "Analyzing..." : "Predict Pest"}
            </button>
            <button
              onClick={handleClear}
              disabled={loading}
              className="btn btn-secondary"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing image...</p>
        </div>
      )}

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && result.success && (
        <div className="result">
          <h3>Prediction Results</h3>

          <div className="prediction-main">
            <div
              style={{ fontSize: "0.9rem", opacity: 0.9, marginBottom: "8px" }}
            >
              Predicted Pest:
            </div>
            <div style={{ fontSize: "1.8rem", fontWeight: "bold" }}>
              {result.prediction.class}
            </div>
            <div style={{ fontSize: "1.2rem", marginTop: "8px" }}>
              Confidence: {(result.prediction.confidence * 100).toFixed(2)}%
            </div>
          </div>

          <h4
            style={{ marginTop: "20px", marginBottom: "12px", color: "#333" }}
          >
            Top 3 Predictions:
          </h4>

          <ul className="prediction-list">
            {result.top_3_predictions.map((pred, index) => (
              <li key={index} className="prediction-item">
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: "600", marginBottom: "4px" }}>
                    {index + 1}. {pred.class}
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${pred.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div
                  style={{
                    marginLeft: "16px",
                    fontWeight: "bold",
                    color: "#667eea",
                  }}
                >
                  {(pred.confidence * 100).toFixed(2)}%
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadPredict;
