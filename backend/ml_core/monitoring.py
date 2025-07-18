"""
ML Core Monitoring: Logging, metrics, and checkpointing utilities
"""
import torch
import logging
import os

def log_metrics(metrics: dict, log_file: str):
    with open(log_file, 'a') as f:
        f.write(str(metrics) + '\n')

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] 

# --- Implementation: MLMetricsTracker ---
class MLMetricsTracker:
    def __init__(self):
        self.metrics = []
    def record_inference_metrics(self, metrics):
        self.metrics.append(metrics)
    def get_latest_metrics(self):
        if self.metrics:
            return self.metrics[-1]
        return {} 

# --- STUB: OptimizationMonitor ---
class OptimizationMonitor:
    def __init__(self, *args, **kwargs):
        pass 