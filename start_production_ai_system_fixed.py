import os
import subprocess
import time
import logging
import threading
import requests

# List of all modular ML services to launch (production)
SERVICES = [
    {'name': 'AI Gateway', 'cmd': ['python', 'ai_service_flask/ai_gateway.py'], 'health_url': 'http://localhost:8000/health'},
    {'name': 'Federated Learning', 'cmd': ['python', 'ai_service_flask/federated_learning_service.py'], 'health_url': None},
    {'name': 'GNN Inference', 'cmd': ['python', 'ai_service_flask/gnn_inference_service.py'], 'health_url': 'http://localhost:8001/health'},
    {'name': 'AI Pricing', 'cmd': ['python', 'ai_service_flask/ai_pricing_service_wrapper.py'], 'health_url': 'http://localhost:8002/health'},
    {'name': 'Logistics', 'cmd': ['python', 'ai_service_flask/logistics_service_wrapper.py'], 'health_url': 'http://localhost:8003/health'},
    {'name': 'Feedback Orchestrator', 'cmd': ['python', 'backend/ai_feedback_orchestrator.py'], 'health_url': None},
    {'name': 'Service Integration', 'cmd': ['python', 'backend/ai_service_integration.py'], 'health_url': None},
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

processes = []

def launch_service(service):
    logging.info(f"Launching {service['name']}...")
    proc = subprocess.Popen(service['cmd'])
    return proc

def check_health(url, retries=10, delay=3):
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def monitor_services():
    while True:
        for proc, service in zip(processes, SERVICES):
            if proc.poll() is not None:
                logging.error(f"Service {service['name']} has stopped unexpectedly. Restarting...")
                new_proc = launch_service(service)
                idx = processes.index(proc)
                processes[idx] = new_proc
        time.sleep(10)

def main():
    # Launch all services
    for service in SERVICES:
        proc = launch_service(service)
        processes.append(proc)
        # Health check if available
        if service['health_url']:
            if not check_health(service['health_url']):
                logging.error(f"{service['name']} failed health check. Exiting.")
                exit(1)
            else:
                logging.info(f"{service['name']} is healthy.")
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_services, daemon=True)
    monitor_thread.start()
    logging.info("All modular ML services launched. Production system is running.")
    # Wait for all processes
    for proc in processes:
        proc.wait()

if __name__ == '__main__':
    main() 