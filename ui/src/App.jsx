import React, { useState } from "react";
import UploadPredict from "./components/UploadPredict";
import RetrainUpload from "./components/RetrainUpload";
import HealthMetrics from "./components/HealthMetrics";

function App() {
  const [activeTab, setActiveTab] = useState("predict");

  return (
    <div className="app">
      <div className="header">
        <h1>🌾 Crop Pest Detection</h1>
        <p>AI-Powered Pest Classification System</p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === "predict" ? "active" : ""}`}
          onClick={() => setActiveTab("predict")}
        >
          Predict
        </button>
        <button
          className={`tab ${activeTab === "retrain" ? "active" : ""}`}
          onClick={() => setActiveTab("retrain")}
        >
          Retrain Model
        </button>
        <button
          className={`tab ${activeTab === "metrics" ? "active" : ""}`}
          onClick={() => setActiveTab("metrics")}
        >
          Health & Metrics
        </button>
      </div>

      <div className="tab-content">
        {activeTab === "predict" && <UploadPredict />}
        {activeTab === "retrain" && <RetrainUpload />}
        {activeTab === "metrics" && <HealthMetrics />}
      </div>

      <div
        style={{
          textAlign: "center",
          marginTop: "40px",
          color: "#999",
          fontSize: "0.9rem",
        }}
      >
        <p>Powered by MobileNetV2 Transfer Learning</p>
      </div>
    </div>
  );
}

export default App;
