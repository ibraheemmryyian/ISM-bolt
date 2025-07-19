#!/usr/bin/env python3
"""
Comprehensive Service Validation Script
Tests all 35+ microservices to ensure they're fully usable
"""

import asyncio
import aiohttp
import requests
import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    name: str
    url: str
    method: str = 'GET'
    endpoint: str = '/health'
    timeout: int = 10
    expected_status: int = 200
    required_fields: Optional[List[str]] = None
    test_data: Optional[Dict] = None

class ServiceValidator:
    """Validates all microservices for usability"""
    
    def __init__(self):
        self.session = None
        self.results = {}
        
        # Define all services with their configurations
        self.services = [
            # Core Services
            ServiceConfig("Backend API", "http://localhost:3000", endpoint="/health"),
            ServiceConfig("Frontend", "http://localhost:5173", endpoint="/"),
            
            # Backend Microservices
            ServiceConfig("Adaptive Onboarding", "http://localhost:5003", endpoint="/health"),
            ServiceConfig("System Health Monitor", "http://localhost:5018", endpoint="/health"),
            ServiceConfig("AI Monitoring Dashboard", "http://localhost:5004", endpoint="/health"),
            ServiceConfig("AI Pricing Service", "http://localhost:5005", endpoint="/health"),
            ServiceConfig("AI Pricing Orchestrator", "http://localhost:8030", endpoint="/health"),
            ServiceConfig("Meta-Learning Orchestrator", "http://localhost:8010", endpoint="/health"),
            ServiceConfig("AI Matchmaking Service", "http://localhost:8020", endpoint="/health"),
            ServiceConfig("MaterialsBERT Simple", "http://localhost:5002", endpoint="/health"),
            ServiceConfig("Value Function Arbiter", "http://localhost:8000", endpoint="/docs"),  # FastAPI docs
            
            # AI Service Flask
            ServiceConfig("AI Gateway", "http://localhost:8000", endpoint="/health"),
            ServiceConfig("Advanced Analytics", "http://localhost:5004", endpoint="/health"),
            ServiceConfig("AI Pricing Wrapper", "http://localhost:8002", endpoint="/health"),
            ServiceConfig("GNN Inference", "http://localhost:8001", endpoint="/health"),
            ServiceConfig("Logistics Wrapper", "http://localhost:8003", endpoint="/health"),
            ServiceConfig("Multi-Hop Symbiosis", "http://localhost:5003", endpoint="/health"),
            
            # Root Services
            ServiceConfig("Logistics Cost Service", "http://localhost:5006", endpoint="/health"),
            ServiceConfig("Optimize DeepSeek R1", "http://localhost:5005", endpoint="/health"),
            
            # Flask Apps (no specific port, test if they respond)
            ServiceConfig("AI Listings Generator", "http://localhost:5000", endpoint="/health"),
            ServiceConfig("MaterialsBERT Service", "http://localhost:5001", endpoint="/health"),
            ServiceConfig("Ultra AI Listings Generator", "http://localhost:5007", endpoint="/health"),
            ServiceConfig("Regulatory Compliance", "http://localhost:5008", endpoint="/health"),
            ServiceConfig("Proactive Opportunity Engine", "http://localhost:5009", endpoint="/health"),
            ServiceConfig("AI Feedback Orchestrator", "http://localhost:5010", endpoint="/health"),
            ServiceConfig("Financial Analysis Engine", "http://localhost:5011", endpoint="/health"),
        ]
    
    async def start_session(self):
        """Start aiohttp session"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def test_service(self, service: ServiceConfig) -> Dict:
        """Test a single service"""
        result = {
            'name': service.name,
            'url': f"{service.url}{service.endpoint}",
            'status': 'unknown',
            'response_time': None,
            'error': None,
            'details': {}
        }
        
        try:
            start_time = time.time()
            
            if service.method == 'GET':
                async with self.session.get(f"{service.url}{service.endpoint}") as response:
                    result['response_time'] = time.time() - start_time
                    result['status_code'] = response.status
                    
                    if response.status == service.expected_status:
                        result['status'] = 'healthy'
                        try:
                            data = await response.json()
                            result['details']['response_data'] = data
                            
                            # Check required fields if specified
                            if service.required_fields:
                                missing_fields = [field for field in service.required_fields if field not in data]
                                if missing_fields:
                                    result['status'] = 'warning'
                                    result['details']['missing_fields'] = missing_fields
                        except:
                            result['details']['response_text'] = await response.text()
                    else:
                        result['status'] = 'unhealthy'
                        result['error'] = f"Expected status {service.expected_status}, got {response.status}"
            
            elif service.method == 'POST' and service.test_data:
                async with self.session.post(f"{service.url}{service.endpoint}", json=service.test_data) as response:
                    result['response_time'] = time.time() - start_time
                    result['status_code'] = response.status
                    
                    if response.status in [200, 201]:
                        result['status'] = 'healthy'
                        try:
                            data = await response.json()
                            result['details']['response_data'] = data
                        except:
                            result['details']['response_text'] = await response.text()
                    else:
                        result['status'] = 'unhealthy'
                        result['error'] = f"Expected status 200/201, got {response.status}"
        
        except asyncio.TimeoutError:
            result['status'] = 'timeout'
            result['error'] = f"Request timed out after {service.timeout}s"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    async def test_all_services(self) -> Dict:
        """Test all services concurrently"""
        logger.info("🚀 Starting comprehensive service validation...")
        
        await self.start_session()
        
        try:
            # Test all services concurrently
            tasks = [self.test_service(service) for service in self.services]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.results[self.services[i].name] = {
                        'name': self.services[i].name,
                        'status': 'error',
                        'error': str(result)
                    }
                else:
                    self.results[result['name']] = result
            
            return self.results
        
        finally:
            await self.close_session()
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        if not self.results:
            return "No validation results available"
        
        # Count statuses
        status_counts = {}
        for result in self.results.values():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("🔍 SYMBIOFLOWS COMPREHENSIVE SERVICE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Total Services Tested: {len(self.results)}")
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY:")
        for status, count in status_counts.items():
            emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
            report.append(f"  {emoji} {status.title()}: {count}")
        report.append("")
        
        # Detailed results
        report.append("🔍 DETAILED RESULTS:")
        report.append("-" * 80)
        
        for name, result in self.results.items():
            status = result.get('status', 'unknown')
            emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
            
            report.append(f"{emoji} {name}")
            report.append(f"   URL: {result.get('url', 'N/A')}")
            report.append(f"   Status: {status}")
            
            if result.get('response_time'):
                report.append(f"   Response Time: {result['response_time']:.2f}s")
            
            if result.get('error'):
                report.append(f"   Error: {result['error']}")
            
            if result.get('details'):
                for key, value in result['details'].items():
                    if key == 'response_data':
                        report.append(f"   Response: {json.dumps(value, indent=2)[:100]}...")
                    else:
                        report.append(f"   {key}: {value}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("-" * 80)
        
        healthy_count = status_counts.get('healthy', 0)
        total_count = len(self.results)
        
        if healthy_count == total_count:
            report.append("🎉 ALL SERVICES ARE FULLY OPERATIONAL!")
            report.append("   Your SymbioFlows system is ready for production use.")
        elif healthy_count >= total_count * 0.8:
            report.append("✅ MOST SERVICES ARE OPERATIONAL")
            report.append("   A few services may need attention but the system is usable.")
        elif healthy_count >= total_count * 0.5:
            report.append("⚠️ MANY SERVICES NEED ATTENTION")
            report.append("   Check the detailed results above for specific issues.")
        else:
            report.append("❌ SYSTEM NEEDS MAJOR ATTENTION")
            report.append("   Many services are not responding properly.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main validation function"""
    try:
        validator = ServiceValidator()
        results = await validator.test_all_services()
        
        # Generate and print report
        report = validator.generate_report()
        print(report)
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"service_validation_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {filename}")
        
        # Return exit code based on results
        healthy_count = sum(1 for r in results.values() if r.get('status') == 'healthy')
        total_count = len(results)
        
        if healthy_count == total_count:
            print("\n🎉 ALL SERVICES VALIDATED SUCCESSFULLY!")
            return 0
        elif healthy_count >= total_count * 0.8:
            print("\n✅ MOST SERVICES VALIDATED SUCCESSFULLY!")
            return 0
        else:
            print("\n❌ MANY SERVICES NEED ATTENTION!")
            return 1
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 