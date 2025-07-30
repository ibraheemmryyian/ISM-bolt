#!/usr/bin/env python3
"""
SymbioFlows Demo - Keeps running until explicitly terminated
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
import requests
import signal

# Enhanced logging configuration with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Global variables to track processes
running_processes = []

# Add immediate startup message
print("🚀 SymbioFlows Persistent Demo")
print("📝 Logging to: demo.log")
print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 60)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutdown signal received. Cleaning up processes...")
    cleanup_processes()
    print("✅ All processes terminated. Exiting.")
    sys.exit(0)

def cleanup_processes():
    """Clean up all running processes"""
    global running_processes
    
    for name, process in running_processes:
        try:
            print(f"Terminating {name} (PID: {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"✅ {name} terminated gracefully")
            except subprocess.TimeoutExpired:
                print(f"⚠️ {name} not responding, force killing...")
                process.kill()
                print(f"✅ {name} force killed")
        except Exception as e:
            print(f"❌ Error terminating {name}: {e}")

def start_backend():
    """Start the backend server"""
    print("\n🔧 Starting backend server...")
    
    try:
        process = subprocess.Popen(
            ['node', 'app.js'],
            cwd=Path.cwd() / 'backend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        running_processes.append(('backend', process))
        print(f"✅ Backend started with PID: {process.pid}")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def start_ai_services():
    """Start the AI services"""
    print("\n🤖 Starting AI services...")
    
    try:
        process = subprocess.Popen(
            ['python', 'ai_gateway.py'],
            cwd=Path.cwd() / 'ai_service_flask',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        running_processes.append(('ai_services', process))
        print(f"✅ AI services started with PID: {process.pid}")
        return True
    except Exception as e:
        print(f"❌ Failed to start AI services: {e}")
        return False

def check_health():
    """Check health of services"""
    print("\n🔍 Checking service health...")
    
    health_checks = {
        'backend': 'http://localhost:3000/api/health',
        'ai_services': 'http://localhost:5000/health'
    }
    
    healthy_services = []
    
    for service_name, url in health_checks.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}: Healthy")
                healthy_services.append(service_name)
            else:
                print(f"⚠️ {service_name}: Status {response.status_code}")
        except Exception as e:
            print(f"⚠️ {service_name}: Health check failed - {e}")
    
    return healthy_services

def main():
    """Main function"""
    # Set up signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting SymbioFlows services...")
    
    # Start services
    backend_started = start_backend()
    time.sleep(5)  # Give backend time to start
    
    ai_started = start_ai_services()
    time.sleep(5)  # Give AI services time to start
    
    if not backend_started and not ai_started:
        print("❌ No services could be started. Exiting.")
        return False
    
    # Initial health check
    print("\nWaiting for services to initialize...")
    time.sleep(10)
    healthy_services = check_health()
    
    # Display status
    print("\n" + "="*60)
    print("SYMBIOFLOWS DEMO STATUS")
    print("="*60)
    
    for name, _ in running_processes:
        status = "✅ RUNNING" if name in healthy_services else "⚠️ STARTING"
        print(f"{name:<20} {status}")
    
    print("="*60)
    print("Access URLs:")
    print("Backend API: http://localhost:3000")
    print("AI Services: http://localhost:5000")
    print("="*60)
    
    # Keep running message
    print("\n" + "="*60)
    print("SERVICES ARE NOW RUNNING")
    print("="*60)
    print("✅ Services are running in the background")
    print("✅ Press Ctrl+C to stop all services when done")
    print("✅ DO NOT CLOSE THIS WINDOW or services will terminate")
    print("="*60)
    
    # Keep the script running
    try:
        heartbeat_counter = 0
        while True:
            time.sleep(1)
            heartbeat_counter += 1
            
            # Print a heartbeat every 30 seconds
            if heartbeat_counter % 30 == 0:
                print(".", end="", flush=True)
            
            # Check health every 5 minutes
            if heartbeat_counter % 300 == 0:
                healthy_services = check_health()
                
                # Check if processes are still running
                for name, process in list(running_processes):
                    if process.poll() is not None:
                        print(f"\n⚠️ {name} has terminated (exit code: {process.returncode})")
                        running_processes.remove((name, process))
                        
                        # Try to restart
                        if name == 'backend':
                            start_backend()
                        elif name == 'ai_services':
                            start_ai_services()
    
    except KeyboardInterrupt:
        # This should be caught by the signal handler
        pass
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        cleanup_processes()
        sys.exit(1)