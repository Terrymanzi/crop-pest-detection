"""
Load testing script for crop pest detection API using Locust.

This script simulates concurrent users uploading images for prediction.
Usage: locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import io
from PIL import Image
import random


class PestDetectionUser(HttpUser):
    """Simulated user for load testing the pest detection API."""
    
    # Wait 1-3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user - create sample images."""
        self.images = self._create_sample_images()
    
    def _create_sample_images(self, num_images=5):
        """
        Create sample images for testing.
        
        Args:
            num_images: Number of sample images to create
            
        Returns:
            List of image bytes
        """
        images = []
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
        ]
        
        for i in range(num_images):
            # Create random colored image
            color = colors[i % len(colors)]
            img = Image.new('RGB', (224, 224), color=color)
            
            # Add some noise for variety
            pixels = img.load()
            for x in range(0, 224, 10):
                for y in range(0, 224, 10):
                    noise = random.randint(-20, 20)
                    new_color = tuple(max(0, min(255, c + noise)) for c in color)
                    pixels[x, y] = new_color
            
            # Convert to bytes
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            images.append(buf.getvalue())
        
        return images
    
    @task(5)
    def predict_image(self):
        """
        Task: Upload an image for prediction.
        
        This task has weight 5, making it 5x more likely than other tasks.
        """
        # Select a random image
        image_data = random.choice(self.images)
        
        # Upload for prediction
        files = {'file': ('test_image.jpg', image_data, 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    response.success()
                else:
                    response.failure("Prediction failed")
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def check_health(self):
        """
        Task: Check API health.
        
        This task has weight 2.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy' or data.get('model_loaded') is not None:
                    response.success()
                else:
                    response.failure("Invalid health response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """
        Task: Get API metrics.
        
        This task has weight 1.
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'total_requests' in data:
                    response.success()
                else:
                    response.failure("Invalid metrics response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_classes(self):
        """
        Task: Get available classes.
        
        This task has weight 1.
        """
        with self.client.get("/classes", catch_response=True) as response:
            if response.status_code in [200, 503]:
                if response.status_code == 200:
                    data = response.json()
                    if 'classes' in data:
                        response.success()
                    else:
                        response.failure("Invalid classes response")
                else:
                    # Model not loaded, but this is expected
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class StressTestUser(HttpUser):
    """
    Stress test user that only does predictions.
    
    Use this for stress testing the prediction endpoint specifically.
    """
    
    wait_time = between(0.1, 0.5)  # Faster requests
    
    def on_start(self):
        """Create a single test image."""
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        self.test_image = buf.getvalue()
    
    @task
    def predict_only(self):
        """Only do predictions - for stress testing."""
        files = {'file': ('stress_test.jpg', self.test_image, 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# Run with:
# locust -f locustfile.py --host=http://localhost:8000
# 
# Then open http://localhost:8089 in your browser to start the load test
#
# For headless mode:
# locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 1m --headless
