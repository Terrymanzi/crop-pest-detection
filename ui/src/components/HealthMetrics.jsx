import React, { useState, useEffect } from "react";
import axios from "axios";
import API_URL from "../config";

const HealthMetrics = () => {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [classes, setClasses] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchData = async () => {
    try {
      const [healthRes, metricsRes, classesRes] = await Promise.all([
        axios.get(`${API_URL}/health`),
        axios.get(`${API_URL}/metrics`),
        axios.get(`${API_URL}/classes`).catch(() => ({ data: null })),
      ]);

      setHealth(healthRes.data);
      setMetrics(metricsRes.data);
      setClasses(classesRes.data);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error">
        <strong>Error:</strong> {error}
        <button
          onClick={fetchData}
          className="btn btn-primary"
          style={{ marginTop: "16px" }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <h2 style={{ color: "#333", margin: 0 }}>System Health & Metrics</h2>
        <label
          style={{ display: "flex", alignItems: "center", cursor: "pointer" }}
        >
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
            style={{ marginRight: "8px" }}
          />
          Auto-refresh
        </label>
      </div>

      {/* Health Status */}
      {health && (
        <div
          style={{
            background: "white",
            padding: "20px",
            borderRadius: "8px",
            marginBottom: "20px",
          }}
        >
          <h3 style={{ marginBottom: "16px", color: "#333" }}>Health Status</h3>
          <div
            style={{
              display: "flex",
              gap: "20px",
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            <div>
              <span
                className={`status-badge ${
                  health.status === "healthy"
                    ? "status-healthy"
                    : "status-error"
                }`}
              >
                {health.status === "healthy" ? "✓ Healthy" : "✗ Error"}
              </span>
            </div>
            <div>
              <strong>Model Loaded:</strong>{" "}
              <span
                className={`status-badge ${
                  health.model_loaded ? "status-healthy" : "status-error"
                }`}
              >
                {health.model_loaded ? "✓ Yes" : "✗ No"}
              </span>
            </div>
            <div style={{ color: "#666" }}>
              <strong>Uptime:</strong> {health.uptime_seconds?.toFixed(0)}s
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {metrics && (
        <div>
          <h3 style={{ marginBottom: "16px", color: "#333" }}>
            Performance Metrics
          </h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">{metrics.total_requests}</div>
              <div className="metric-label">Total Requests</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{metrics.total_errors}</div>
              <div className="metric-label">Total Errors</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">
                {(metrics.error_rate * 100).toFixed(2)}%
              </div>
              <div className="metric-label">Error Rate</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{metrics.average_latency_ms}ms</div>
              <div className="metric-label">Avg Latency</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">
                {metrics.recent_average_latency_ms}ms
              </div>
              <div className="metric-label">Recent Latency</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">
                {metrics.requests_per_minute.toFixed(1)}
              </div>
              <div className="metric-label">Requests/Min</div>
            </div>
          </div>

          <div
            style={{
              background: "white",
              padding: "20px",
              borderRadius: "8px",
              marginTop: "20px",
            }}
          >
            <div style={{ color: "#666", fontSize: "0.9rem" }}>
              <strong>Uptime:</strong> {metrics.uptime_formatted}
            </div>
            {metrics.last_retrain_time && (
              <div
                style={{ color: "#666", fontSize: "0.9rem", marginTop: "8px" }}
              >
                <strong>Last Retrain:</strong>{" "}
                {new Date(metrics.last_retrain_time).toLocaleString()}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Top Predictions */}
      {metrics &&
        metrics.top_predictions &&
        Object.keys(metrics.top_predictions).length > 0 && (
          <div
            style={{
              background: "white",
              padding: "20px",
              borderRadius: "8px",
              marginTop: "20px",
            }}
          >
            <h3 style={{ marginBottom: "16px", color: "#333" }}>
              Top Predictions
            </h3>
            <ul className="prediction-list">
              {Object.entries(metrics.top_predictions).map(
                ([pestClass, count], index) => (
                  <li key={index} className="prediction-item">
                    <span style={{ fontWeight: "600" }}>{pestClass}</span>
                    <span style={{ color: "#667eea", fontWeight: "bold" }}>
                      {count}
                    </span>
                  </li>
                )
              )}
            </ul>
          </div>
        )}

      {/* Available Classes */}
      {classes && classes.classes && (
        <div
          style={{
            background: "white",
            padding: "20px",
            borderRadius: "8px",
            marginTop: "20px",
          }}
        >
          <h3 style={{ marginBottom: "16px", color: "#333" }}>
            Available Classes ({classes.num_classes})
          </h3>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {classes.classes.map((className, index) => (
              <span
                key={index}
                style={{
                  background: "#e8eeff",
                  color: "#667eea",
                  padding: "6px 12px",
                  borderRadius: "16px",
                  fontSize: "0.9rem",
                  fontWeight: "500",
                }}
              >
                {className}
              </span>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={fetchData}
        className="btn btn-primary"
        style={{ marginTop: "20px", width: "100%" }}
      >
        Refresh Now
      </button>
    </div>
  );
};

export default HealthMetrics;
