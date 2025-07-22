#!/usr/bin/env python3
"""
🚀 RUN ULTRA-ADVANCED AI TESTS AND DEMONSTRATION
Simple script to test and demonstrate all ultra-advanced AI capabilities
"""

import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test and demo modules
try:
    from test_ultra_advanced_ai import UltraAdvancedAITester, TestConfig
    from demo_ultra_advanced_ai import UltraAdvancedAIDemo
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all ultra-advanced AI files are in the backend directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_advanced_ai_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def run_tests():
    """Run comprehensive tests"""
    print("\n🧪 RUNNING COMPREHENSIVE TESTS")
    print("="*60)
    
    try:
        # Initialize tester
        config = TestConfig()
        tester = UltraAdvancedAITester(config)
        
        # Run tests
        await tester.run_comprehensive_tests()
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def run_demo():
    """Run demonstration"""
    print("\n🎬 RUNNING DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize demo
        demo = UltraAdvancedAIDemo()
        
        # Run demo
        await demo.run_complete_demo()
        
        print("\n✅ Demonstration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

async def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\n⚡ RUNNING QUICK TEST")
    print("="*60)
    
    try:
        # Test basic imports
        from ultra_advanced_ai_system import UltraAdvancedAISystem, UltraAdvancedAIConfig
        from integrate_ultra_advanced_ai import UltraAdvancedAIIntegration
        
        print("✅ All imports successful")
        
        # Test system initialization
        config = UltraAdvancedAIConfig()
        ultra_ai = UltraAdvancedAISystem(config)
        integration = UltraAdvancedAIIntegration()
        
        print("✅ Systems initialized successfully")
        
        # Test basic functionality
        test_data = torch.randn(10, 100)
        result = ultra_ai.process_industrial_symbiosis(
            test_data, test_data, test_data
        )
        
        print("✅ Basic processing successful")
        print(f"✅ Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

async def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n📊 RUNNING PERFORMANCE BENCHMARK")
    print("="*60)
    
    try:
        from ultra_advanced_ai_system import UltraAdvancedAISystem, UltraAdvancedAIConfig
        import time
        import numpy as np
        
        # Initialize system
        config = UltraAdvancedAIConfig()
        ultra_ai = UltraAdvancedAISystem(config)
        
        # Generate test data
        material_data = torch.randn(100, 100)
        company_data = torch.randn(100, 50)
        market_data = torch.randn(100, 30)
        
        # Benchmark processing time
        times = []
        for i in range(10):
            start_time = time.time()
            result = ultra_ai.process_industrial_symbiosis(
                material_data, company_data, market_data
            )
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i % 2 == 0:
                print(f"  Run {i+1}: {times[-1]:.4f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n📊 PERFORMANCE RESULTS:")
        print(f"   Average processing time: {avg_time:.4f}s")
        print(f"   Standard deviation: {std_time:.4f}s")
        print(f"   Throughput: {1/avg_time:.1f} requests/second")
        
        # Performance assessment
        if avg_time < 0.1:
            print("   🚀 Performance: Excellent (< 100ms)")
        elif avg_time < 0.5:
            print("   ✅ Performance: Good (< 500ms)")
        else:
            print("   ⚠️  Performance: Needs optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def print_usage():
    """Print usage information"""
    print("""
🚀 ULTRA-ADVANCED AI TESTING AND DEMONSTRATION

Usage:
  python run_ultra_advanced_tests.py [OPTION]

Options:
  --tests          Run comprehensive tests
  --demo           Run demonstration
  --quick          Run quick test
  --benchmark      Run performance benchmark
  --all            Run all tests and demo
  --help           Show this help message

Examples:
  python run_ultra_advanced_tests.py --quick
  python run_ultra_advanced_tests.py --demo
  python run_ultra_advanced_tests.py --all
""")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Ultra-Advanced AI Testing and Demonstration')
    parser.add_argument('--tests', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--all', action='store_true', help='Run all tests and demo')
    
    args = parser.parse_args()
    
    # If no arguments provided, show usage
    if not any([args.tests, args.demo, args.quick, args.benchmark, args.all]):
        print_usage()
        return
    
    print("🚀 ULTRA-ADVANCED AI SYSTEM")
    print("Testing and Demonstration Suite")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    results = {}
    
    # Run requested operations
    if args.quick or args.all:
        results['quick_test'] = await run_quick_test()
    
    if args.benchmark or args.all:
        results['benchmark'] = await run_performance_benchmark()
    
    if args.tests or args.all:
        results['tests'] = await run_tests()
    
    if args.demo or args.all:
        results['demo'] = await run_demo()
    
    # Print summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    for operation, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {operation.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print(f"Your ultra-advanced AI system is working perfectly!")
    else:
        print(f"\n⚠️  SOME OPERATIONS FAILED")
        print(f"Check the logs for details")
    
    print(f"\n📁 Results saved to:")
    print(f"   - ultra_advanced_ai_test.log")
    if args.tests or args.all:
        print(f"   - ultra_advanced_ai_test_report.json")
        print(f"   - ultra_advanced_ai_test_results.png")
    
    print("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}") 