import React, { useState } from "react";
import axios from "axios";
import API_URL from "../config";

const RetrainUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    processFile(file);
  };

  const processFile = (file) => {
    if (file && file.name.endsWith(".zip")) {
      setSelectedFile(file);
      setError(null);
      setSuccess(null);
    } else {
      setError("Please select a valid ZIP file");
      setSelectedFile(null);
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

  const handleRetrain = async () => {
    if (!selectedFile) {
      setError("Please select a ZIP file first");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_URL}/retrain`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 60000, // 60 second timeout
      });

      setSuccess(response.data.message || "Retraining started successfully!");
      setSelectedFile(null);
    } catch (err) {
      setError(
        err.response?.data?.detail || "Retraining failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setSuccess(null);
    setError(null);
  };

  return (
    <div className="card">
      <h2 style={{ marginBottom: "20px", color: "#333" }}>Model Retraining</h2>

      <div
        style={{
          background: "#fff3cd",
          padding: "16px",
          borderRadius: "8px",
          marginBottom: "20px",
          borderLeft: "4px solid #ffc107",
        }}
      >
        <strong>⚠️ Instructions:</strong>
        <ul style={{ marginTop: "8px", marginLeft: "20px" }}>
          <li>Prepare a ZIP file containing training images</li>
          <li>Images should be organized in class folders</li>
          <li>Example structure: pest_data.zip/class_name/image1.jpg</li>
          <li>Retraining may take several minutes</li>
        </ul>
      </div>

      <div
        className={`upload-area ${dragOver ? "dragover" : ""}`}
        onClick={() => document.getElementById("retrain-input").click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          id="retrain-input"
          type="file"
          accept=".zip"
          onChange={handleFileSelect}
          className="upload-input"
        />
        <div style={{ fontSize: "3rem", marginBottom: "16px" }}>📦</div>
        <p style={{ fontSize: "1.1rem", marginBottom: "8px" }}>
          {selectedFile ? selectedFile.name : "Click to upload ZIP file"}
        </p>
        <p style={{ color: "#999", fontSize: "0.9rem" }}>
          ZIP archives containing training data
        </p>
      </div>

      {selectedFile && (
        <div style={{ marginTop: "20px", textAlign: "center" }}>
          <div
            style={{
              background: "white",
              padding: "16px",
              borderRadius: "8px",
              marginBottom: "16px",
            }}
          >
            <strong>Selected file:</strong> {selectedFile.name}
            <br />
            <span style={{ color: "#666", fontSize: "0.9rem" }}>
              Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>

          <div
            style={{ display: "flex", gap: "10px", justifyContent: "center" }}
          >
            <button
              onClick={handleRetrain}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? "Starting Retraining..." : "Start Retraining"}
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
          <p>Starting retraining process...</p>
          <p style={{ fontSize: "0.9rem", color: "#999", marginTop: "8px" }}>
            This may take several minutes. You can close this page.
          </p>
        </div>
      )}

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {success && (
        <div className="success">
          <strong>Success!</strong> {success}
          <p style={{ marginTop: "8px", fontSize: "0.9rem" }}>
            The model will be retrained in the background. Check the metrics
            page to monitor progress.
          </p>
        </div>
      )}
    </div>
  );
};

export default RetrainUpload;
