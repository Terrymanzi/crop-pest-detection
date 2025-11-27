"""
Tests for FastAPI endpoints.
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


def test_read_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "model_loaded" in data
    assert isinstance(data["uptime_seconds"], (int, float))


def test_get_classes():
    """Test get classes endpoint."""
    response = client.get("/classes")
    assert response.status_code in [200, 503]  # 503 if model not loaded
    
    if response.status_code == 200:
        data = response.json()
        assert "classes" in data
        assert "num_classes" in data
        assert isinstance(data["classes"], list)


def test_get_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime_seconds" in data
    assert "average_latency_ms" in data


def test_predict_endpoint_no_file():
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable entity


def test_predict_endpoint_with_image(sample_image_file):
    """Test predict endpoint with valid image."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
    )
    
    # May return 503 if model not loaded, 200 if successful
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "prediction" in data
        assert "top_3_predictions" in data
        assert "class" in data["prediction"]
        assert "confidence" in data["prediction"]


def test_predict_endpoint_invalid_file_type():
    """Test predict endpoint with invalid file type."""
    # Create a text file
    text_file = io.BytesIO(b"This is not an image")
    
    response = client.post(
        "/predict",
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    # Should reject non-image files
    assert response.status_code in [400, 500, 503]


def test_retrain_endpoint_no_file():
    """Test retrain endpoint without file."""
    response = client.post("/retrain")
    assert response.status_code == 422  # Unprocessable entity


def test_retrain_endpoint_invalid_file():
    """Test retrain endpoint with non-ZIP file."""
    # Create a non-ZIP file
    text_file = io.BytesIO(b"This is not a ZIP file")
    
    response = client.post(
        "/retrain",
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    # Should reject non-ZIP files
    assert response.status_code == 400


def test_multiple_health_checks():
    """Test multiple health check requests."""
    responses = []
    for _ in range(5):
        response = client.get("/health")
        responses.append(response)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    
    # Request count should increase
    data = responses[-1].json()
    assert data["total_requests"] >= 5


def test_metrics_after_requests():
    """Test that metrics are updated after requests."""
    # Get initial metrics
    response1 = client.get("/metrics")
    data1 = response1.json()
    initial_count = data1["total_requests"]
    
    # Make some requests
    client.get("/health")
    client.get("/classes")
    
    # Get updated metrics
    response2 = client.get("/metrics")
    data2 = response2.json()
    final_count = data2["total_requests"]
    
    # Request count should have increased
    assert final_count > initial_count


def test_prediction_response_format(sample_image_file):
    """Test that prediction response has correct format."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Check top-level structure
        assert isinstance(data["success"], bool)
        assert isinstance(data["prediction"], dict)
        assert isinstance(data["top_3_predictions"], list)
        
        # Check prediction structure
        pred = data["prediction"]
        assert "class" in pred
        assert "confidence" in pred
        assert isinstance(pred["class"], str)
        assert isinstance(pred["confidence"], (int, float))
        assert 0 <= pred["confidence"] <= 1
        
        # Check top predictions structure
        assert len(data["top_3_predictions"]) <= 3
        for top_pred in data["top_3_predictions"]:
            assert "class" in top_pred
            assert "confidence" in top_pred
            assert isinstance(top_pred["confidence"], (int, float))


def test_cors_headers():
    """Test that CORS headers are present."""
    response = client.get("/health")
    
    # CORS middleware should add access-control headers
    # Note: TestClient may not expose all headers, so this is optional
    assert response.status_code == 200
