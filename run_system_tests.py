#!/usr/bin/env python3
"""
System Test Runner
Runs either basic startup tests or comprehensive system tests
"""

import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_basic_test():
    """Run basic startup test"""
    logger.info("🚀 Running Basic Startup Test...")
    try:
        result = subprocess.run([sys.executable, "test_system_startup.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Basic startup test PASSED")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Basic startup test FAILED")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Basic startup test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Basic startup test ERROR: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive system test"""
    logger.info("🔍 Running Comprehensive System Test...")
    try:
        result = subprocess.run([sys.executable, "comprehensive_system_test.py"], 
                              capture_output=True, text=True, timeout=300)  # 5 minutes
        
        if result.returncode == 0:
            logger.info("✅ Comprehensive test PASSED")
            print(result.stdout)
            return True
        elif result.returncode == 1:
            logger.warning("⚠️ Comprehensive test has WARNINGS")
            print(result.stdout)
            print(result.stderr)
            return True  # Still consider it a pass
        else:
            logger.error("❌ Comprehensive test FAILED")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Comprehensive test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Comprehensive test ERROR: {e}")
        return False

def run_quick_health_check():
    """Run a quick health check of all services"""
    logger.info("🏥 Running Quick Health Check...")
    
    import requests
    import time
    
    services = [
        ("Backend", "http://localhost:3000/api/health"),
        ("Frontend", "http://localhost:5173"),
        ("AI Pricing Service", "http://localhost:5005/health"),
        ("Logistics Service", "http://localhost:5006/health"),
    ]
    
    all_healthy = True
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name}: HEALTHY")
            else:
                logger.warning(f"⚠️ {name}: Status {response.status_code}")
                all_healthy = False
        except Exception as e:
            logger.error(f"❌ {name}: UNHEALTHY - {e}")
            all_healthy = False
    
    return all_healthy

def main():
    parser = argparse.ArgumentParser(description="ISM AI System Test Runner")
    parser.add_argument("--test-type", choices=["basic", "comprehensive", "health", "all"], 
                       default="all", help="Type of test to run")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick health check only")
    
    args = parser.parse_args()
    
    logger.info("🚀 ISM AI System Test Runner")
    logger.info("=" * 50)
    logger.info(f"Test Type: {args.test_type}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 50)
    
    if args.quick:
        success = run_quick_health_check()
        sys.exit(0 if success else 1)
    
    if args.test_type in ["basic", "all"]:
        logger.info("\n📦 Running Basic Startup Test...")
        basic_success = run_basic_test()
        
        if not basic_success and args.test_type == "basic":
            logger.error("❌ Basic test failed. Stopping.")
            sys.exit(1)
    
    if args.test_type in ["comprehensive", "all"]:
        logger.info("\n🔍 Running Comprehensive System Test...")
        comprehensive_success = run_comprehensive_test()
        
        if not comprehensive_success and args.test_type == "comprehensive":
            logger.error("❌ Comprehensive test failed.")
            sys.exit(1)
    
    if args.test_type == "all":
        logger.info("\n🎯 All tests completed!")
        if basic_success and comprehensive_success:
            logger.info("🎉 All tests PASSED! System is ready.")
            sys.exit(0)
        else:
            logger.warning("⚠️ Some tests had issues. Check the results above.")
            sys.exit(1)
    
    logger.info("✅ Test run completed successfully!")

if __name__ == "__main__":
    main() 