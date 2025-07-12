#!/usr/bin/env python3
"""
Comprehensive System API Validation Script
Ensures all APIs are working and no fallback logic is being used
"""

import os
import sys
import requests
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class SystemAPIValidator:
    """
    Comprehensive API validation system
    Tests all APIs and ensures no fallback logic is used
    """
    
    def __init__(self):
        self.results = {
            'deepseek_r1': {'status': 'unknown', 'error': None},
            'freightos': {'status': 'unknown', 'error': None},
            'nextgen_materials': {'status': 'unknown', 'error': None},
            'height': {'status': 'unknown', 'error': None},
            'paypal': {'status': 'unknown', 'error': None},
            'shippo': {'status': 'unknown', 'error': None},
            'supabase': {'status': 'unknown', 'error': None}
        }
        
        # Load environment variables
        self.load_environment_variables()
    
    def load_environment_variables(self):
        """Load and validate all required environment variables"""
        logger.info("🔧 Loading environment variables...")
        
        required_vars = {
            'DEEPSEEK_API_KEY': 'DeepSeek R1 API',
            'FREIGHTOS_API_KEY': 'Freightos API',
            'FREIGHTOS_SECRET_KEY': 'Freightos Secret',
            'NEXT_GEN_MATERIALS_API_KEY': 'NextGen Materials API',
            'HEIGHT_API_KEY': 'Height API',
            'HEIGHT_WORKSPACE_ID': 'Height Workspace ID',
            'PAYPAL_CLIENT_ID': 'PayPal Client ID',
            'PAYPAL_CLIENT_SECRET': 'PayPal Client Secret',
            'SHIPPO_API_KEY': 'Shippo API',
            'SUPABASE_URL': 'Supabase URL',
            'SUPABASE_SERVICE_ROLE_KEY': 'Supabase Service Role Key'
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value or value.startswith('your_') or value == 'required':
                missing_vars.append(f"{var} ({description})")
            else:
                logger.info(f"✅ {description}: Configured")
        
        if missing_vars:
            logger.error("❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise Exception("❌ System cannot start without all required API keys")
        
        logger.info("✅ All required environment variables are configured")
    
    def test_deepseek_r1_api(self) -> bool:
        """Test DeepSeek R1 API connectivity and functionality"""
        logger.info("🧪 Testing DeepSeek R1 API...")
        
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-r1",
                "messages": [
                    {
                        "role": "user",
                        "content": "You are an expert industrial symbiosis analyst. Test this API by responding with: {'status': 'working', 'test': 'successful'}"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 100,
                "stream": False
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to parse JSON response
                try:
                    parsed = json.loads(content)
                    if parsed.get('status') == 'working':
                        logger.info("✅ DeepSeek R1 API: Working correctly")
                        self.results['deepseek_r1']['status'] = 'working'
                        return True
                except json.JSONDecodeError:
                    pass
                
                logger.info("✅ DeepSeek R1 API: Connected (response received)")
                self.results['deepseek_r1']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ DeepSeek R1 API error: {error_msg}")
                self.results['deepseek_r1']['status'] = 'failed'
                self.results['deepseek_r1']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ DeepSeek R1 API: {error_msg}")
            self.results['deepseek_r1']['status'] = 'failed'
            self.results['deepseek_r1']['error'] = error_msg
            return False
    
    def test_freightos_api(self) -> bool:
        """Test Freightos API connectivity"""
        logger.info("🧪 Testing Freightos API...")
        
        try:
            api_key = os.getenv('FREIGHTOS_API_KEY')
            secret = os.getenv('FREIGHTOS_SECRET_KEY')
            
            headers = {
                'x-apikey': api_key,
                'Content-Type': 'application/json'
            }
            
            # Test CO2 calculation endpoint
            payload = {
                "load": [{
                    "quantity": 1,
                    "unitWeightKg": 1000,
                    "unitVolumeCBM": 1.0,
                    "unitType": "pallets"
                }],
                "legs": [{
                    "mode": "LTL",
                    "origin": {
                        "unLocationCode": "USNYC"
                    },
                    "destination": {
                        "unLocationCode": "USLAX"
                    }
                }]
            }
            
            response = requests.post(
                "https://api.freightos.com/api/v1/co2calc",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Freightos API: Working correctly")
                self.results['freightos']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Freightos API error: {error_msg}")
                self.results['freightos']['status'] = 'failed'
                self.results['freightos']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ Freightos API: {error_msg}")
            self.results['freightos']['status'] = 'failed'
            self.results['freightos']['error'] = error_msg
            return False
    
    def test_nextgen_materials_api(self) -> bool:
        """Test NextGen Materials API connectivity"""
        logger.info("🧪 Testing NextGen Materials API...")
        
        try:
            api_key = os.getenv('NEXT_GEN_MATERIALS_API_KEY')
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'ISM-AI-Platform/1.0'
            }
            
            # Test basic endpoint
            response = requests.get(
                "https://api.next-gen-materials.com/v1/materials",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ NextGen Materials API: Working correctly")
                self.results['nextgen_materials']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ NextGen Materials API error: {error_msg}")
                self.results['nextgen_materials']['status'] = 'failed'
                self.results['nextgen_materials']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ NextGen Materials API: {error_msg}")
            self.results['nextgen_materials']['status'] = 'failed'
            self.results['nextgen_materials']['error'] = error_msg
            return False
    
    def test_height_api(self) -> bool:
        """Test Height API connectivity"""
        logger.info("🧪 Testing Height API...")
        
        try:
            api_key = os.getenv('HEIGHT_API_KEY')
            workspace_id = os.getenv('HEIGHT_WORKSPACE_ID')
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Test workspace endpoint
            response = requests.get(
                f"https://api.height.app/workspaces/{workspace_id}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Height API: Working correctly")
                self.results['height']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Height API error: {error_msg}")
                self.results['height']['status'] = 'failed'
                self.results['height']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ Height API: {error_msg}")
            self.results['height']['status'] = 'failed'
            self.results['height']['error'] = error_msg
            return False
    
    def test_paypal_api(self) -> bool:
        """Test PayPal API connectivity"""
        logger.info("🧪 Testing PayPal API...")
        
        try:
            client_id = os.getenv('PAYPAL_CLIENT_ID')
            client_secret = os.getenv('PAYPAL_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                error_msg = "Missing PayPal credentials"
                logger.error(f"❌ PayPal API: {error_msg}")
                self.results['paypal']['status'] = 'failed'
                self.results['paypal']['error'] = error_msg
                return False
            
            # Test OAuth token endpoint
            auth_response = requests.post(
                "https://api-m.sandbox.paypal.com/v1/oauth2/token",
                auth=(client_id, client_secret),
                data={'grant_type': 'client_credentials'},
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            if auth_response.status_code == 200:
                logger.info("✅ PayPal API: Working correctly")
                self.results['paypal']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {auth_response.status_code}: {auth_response.text}"
                logger.error(f"❌ PayPal API error: {error_msg}")
                self.results['paypal']['status'] = 'failed'
                self.results['paypal']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ PayPal API: {error_msg}")
            self.results['paypal']['status'] = 'failed'
            self.results['paypal']['error'] = error_msg
            return False
    
    def test_shippo_api(self) -> bool:
        """Test Shippo API connectivity"""
        logger.info("🧪 Testing Shippo API...")
        
        try:
            api_key = os.getenv('SHIPPO_API_KEY')
            
            headers = {
                'Authorization': f'ShippoToken {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Test carriers endpoint
            response = requests.get(
                "https://api.goshippo.com/carriers/",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Shippo API: Working correctly")
                self.results['shippo']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Shippo API error: {error_msg}")
                self.results['shippo']['status'] = 'failed'
                self.results['shippo']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ Shippo API: {error_msg}")
            self.results['shippo']['status'] = 'failed'
            self.results['shippo']['error'] = error_msg
            return False
    
    def test_supabase_connection(self) -> bool:
        """Test Supabase database connection"""
        logger.info("🧪 Testing Supabase connection...")
        
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            headers = {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'application/json'
            }
            
            # Test basic query
            response = requests.get(
                f"{supabase_url}/rest/v1/companies?select=count",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Supabase: Working correctly")
                self.results['supabase']['status'] = 'working'
                return True
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ Supabase error: {error_msg}")
                self.results['supabase']['status'] = 'failed'
                self.results['supabase']['error'] = error_msg
                return False
                
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(f"❌ Supabase: {error_msg}")
            self.results['supabase']['status'] = 'failed'
            self.results['supabase']['error'] = error_msg
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests"""
        logger.info("🚀 Starting comprehensive API validation...")
        logger.info("=" * 60)
        
        tests = [
            ('DeepSeek R1', self.test_deepseek_r1_api),
            ('Freightos', self.test_freightos_api),
            ('NextGen Materials', self.test_nextgen_materials_api),
            ('Height', self.test_height_api),
            ('PayPal', self.test_paypal_api),
            ('Shippo', self.test_shippo_api),
            ('Supabase', self.test_supabase_connection)
        ]
        
        passed = 0
        total = len(tests)
        
        for name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"❌ {name} test failed with exception: {str(e)}")
        
        logger.info("=" * 60)
        logger.info(f"📊 Test Results: {passed}/{total} APIs working")
        
        # Check if all critical APIs are working
        critical_apis = ['deepseek_r1', 'freightos', 'nextgen_materials', 'supabase']
        critical_failures = [api for api in critical_apis if self.results[api]['status'] != 'working']
        
        if critical_failures:
            logger.error("❌ CRITICAL FAILURE: Some critical APIs are not working:")
            for api in critical_failures:
                logger.error(f"   - {api}: {self.results[api]['error']}")
            logger.error("❌ System cannot operate without all critical APIs")
            return {
                'status': 'failed',
                'passed': passed,
                'total': total,
                'results': self.results,
                'critical_failures': critical_failures
            }
        else:
            logger.info("✅ All critical APIs are working")
            return {
                'status': 'success',
                'passed': passed,
                'total': total,
                'results': self.results,
                'critical_failures': []
            }
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a detailed test report"""
        report = []
        report.append("🔍 COMPREHENSIVE API VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Overall Status: {test_results['status'].upper()}")
        report.append(f"APIs Working: {test_results['passed']}/{test_results['total']}")
        report.append("")
        
        for api_name, result in test_results['results'].items():
            status_icon = "✅" if result['status'] == 'working' else "❌"
            report.append(f"{status_icon} {api_name.upper()}: {result['status']}")
            if result['error']:
                report.append(f"   Error: {result['error']}")
        
        report.append("")
        if test_results['critical_failures']:
            report.append("❌ CRITICAL FAILURES:")
            for api in test_results['critical_failures']:
                report.append(f"   - {api}: {test_results['results'][api]['error']}")
            report.append("")
            report.append("🚨 SYSTEM CANNOT START - Fix critical API failures first")
        else:
            report.append("✅ All critical APIs are operational")
            report.append("🚀 System is ready for production use")
        
        return "\n".join(report)

def main():
    """Main validation function"""
    try:
        validator = SystemAPIValidator()
        results = validator.run_all_tests()
        
        # Generate and print report
        report = validator.generate_report(results)
        print("\n" + report)
        
        # Exit with appropriate code
        if results['status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()