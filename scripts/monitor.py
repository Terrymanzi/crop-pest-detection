"""
Simple monitoring utilities for tracking API metrics.

This module provides in-memory storage for request counts, latency, and retraining events.
"""

import json
import time
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


class MetricsMonitor:
    """Simple in-memory metrics monitor for API performance tracking."""
    
    def __init__(self):
        """Initialize metrics storage."""
        self.request_count = 0
        self.total_latency = 0.0
        self.latencies = []
        self.error_count = 0
        self.start_time = datetime.now()
        self.last_retrain_time = None
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_latency': 0.0,
            'errors': 0
        })
        self.prediction_counts = defaultdict(int)
    
    def record_request(self, endpoint: str, latency: float, error: bool = False):
        """
        Record a request with latency.
        
        Args:
            endpoint: API endpoint called
            latency: Request latency in seconds
            error: Whether the request resulted in an error
        """
        self.request_count += 1
        self.total_latency += latency
        self.latencies.append(latency)
        
        # Keep only last 1000 latencies to avoid memory issues
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
        
        # Record endpoint-specific metrics
        self.endpoint_metrics[endpoint]['count'] += 1
        self.endpoint_metrics[endpoint]['total_latency'] += latency
        
        if error:
            self.error_count += 1
            self.endpoint_metrics[endpoint]['errors'] += 1
    
    def record_prediction(self, predicted_class: str):
        """
        Record a prediction class.
        
        Args:
            predicted_class: The predicted pest class
        """
        self.prediction_counts[predicted_class] += 1
    
    def record_retrain(self):
        """Record a retraining event."""
        self.last_retrain_time = datetime.now()
    
    def get_average_latency(self) -> float:
        """
        Get average latency across all requests.
        
        Returns:
            Average latency in seconds
        """
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count
    
    def get_recent_average_latency(self, n: int = 100) -> float:
        """
        Get average latency for last N requests.
        
        Args:
            n: Number of recent requests to consider
            
        Returns:
            Average latency in seconds
        """
        if not self.latencies:
            return 0.0
        recent = self.latencies[-n:]
        return sum(recent) / len(recent)
    
    def get_uptime(self) -> float:
        """
        Get uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_metrics(self) -> Dict:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary containing all metrics
        """
        uptime = self.get_uptime()
        
        metrics = {
            'uptime_seconds': round(uptime, 2),
            'uptime_formatted': self._format_uptime(uptime),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': round(self.error_count / max(self.request_count, 1), 4),
            'average_latency_ms': round(self.get_average_latency() * 1000, 2),
            'recent_average_latency_ms': round(self.get_recent_average_latency() * 1000, 2),
            'requests_per_minute': round(self.request_count / max(uptime / 60, 1), 2),
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'endpoint_metrics': dict(self.endpoint_metrics),
            'top_predictions': dict(sorted(
                self.prediction_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }
        
        return metrics
    
    def get_health_status(self) -> Dict:
        """
        Get health status for the /health endpoint.
        
        Returns:
            Dictionary containing health status
        """
        return {
            'status': 'healthy',
            'uptime_seconds': round(self.get_uptime(), 2),
            'total_requests': self.request_count,
            'average_latency_ms': round(self.get_average_latency() * 1000, 2)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()
    
    def _format_uptime(self, seconds: float) -> str:
        """
        Format uptime in human-readable format.
        
        Args:
            seconds: Uptime in seconds
            
        Returns:
            Formatted uptime string
        """
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    def to_json(self) -> str:
        """
        Convert metrics to JSON string.
        
        Returns:
            JSON string of metrics
        """
        return json.dumps(self.get_metrics(), indent=2)


# Global monitor instance
_monitor = MetricsMonitor()


def get_monitor() -> MetricsMonitor:
    """
    Get the global metrics monitor instance.
    
    Returns:
        Global MetricsMonitor instance
    """
    return _monitor


def record_request(endpoint: str, latency: float, error: bool = False):
    """
    Record a request (convenience function).
    
    Args:
        endpoint: API endpoint called
        latency: Request latency in seconds
        error: Whether the request resulted in an error
    """
    _monitor.record_request(endpoint, latency, error)


def record_prediction(predicted_class: str):
    """
    Record a prediction (convenience function).
    
    Args:
        predicted_class: The predicted pest class
    """
    _monitor.record_prediction(predicted_class)


def record_retrain():
    """Record a retraining event (convenience function)."""
    _monitor.record_retrain()


def get_metrics() -> Dict:
    """
    Get all metrics (convenience function).
    
    Returns:
        Dictionary containing all metrics
    """
    return _monitor.get_metrics()


def get_health_status() -> Dict:
    """
    Get health status (convenience function).
    
    Returns:
        Dictionary containing health status
    """
    return _monitor.get_health_status()
