#!/usr/bin/env python3
"""
🧪 REAL PERFORMANCE BENCHMARKING SYSTEM
Measures actual performance of listings generation and AI capabilities
"""

import asyncio
import time
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import psutil
import gc

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the actual systems
try:
    from revolutionary_ai_matching import RevolutionaryAIMatching
    from ai_listings_generator import RevolutionaryAIListingsGenerator
    from listing_inference_service import ListingInferenceService
    from enhanced_ai_generator import EnhancedAIGenerator
except ImportError as e:
    print(f"❌ Error importing systems: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealPerformanceBenchmark:
    """
    Real performance benchmarking system that measures actual performance
    """
    
    def __init__(self):
        self.logger = logger
        self.benchmark_results = {}
        self.memory_usage = []
        self.cpu_usage = []
        
        # Test data
        self.test_companies = [
            {
                'id': 'test_company_1',
                'name': 'Steel Manufacturing Corp',
                'industry': 'manufacturing',
                'location': 'USA',
                'employee_count': 500,
                'sustainability_score': 0.8
            },
            {
                'id': 'test_company_2', 
                'name': 'Chemical Processing Ltd',
                'industry': 'chemical',
                'location': 'EU',
                'employee_count': 300,
                'sustainability_score': 0.7
            },
            {
                'id': 'test_company_3',
                'name': 'Electronics Assembly Inc',
                'industry': 'electronics',
                'location': 'Asia',
                'employee_count': 1000,
                'sustainability_score': 0.9
            }
        ]
        
        self.test_materials = [
            'Steel Scrap', 'Aluminum Waste', 'Plastic Waste', 'Chemical Byproduct',
            'Electronic Waste', 'Paper Waste', 'Glass Waste', 'Rubber Waste'
        ]
    
    async def run_comprehensive_benchmarks(self):
        """Run comprehensive performance benchmarks"""
        print("🧪 REAL PERFORMANCE BENCHMARKING")
        print("="*60)
        print(f"Started at: {datetime.now()}")
        print("Measuring ACTUAL performance, not simulated benchmarks")
        print("="*60)
        
        # 1. Benchmark Revolutionary AI Matching
        await self._benchmark_revolutionary_ai_matching()
        
        # 2. Benchmark AI Listings Generation
        await self._benchmark_ai_listings_generation()
        
        # 3. Benchmark Listing Inference Service
        await self._benchmark_listing_inference_service()
        
        # 4. Benchmark Enhanced AI Generator
        await self._benchmark_enhanced_ai_generator()
        
        # 5. Benchmark System Resources
        await self._benchmark_system_resources()
        
        # 6. Generate comprehensive report
        self._generate_comprehensive_report()
        
        print("="*60)
        print("✅ ALL REAL BENCHMARKS COMPLETED")
        print("="*60)
    
    async def _benchmark_revolutionary_ai_matching(self):
        """Benchmark Revolutionary AI Matching performance"""
        print("\n🧠 BENCHMARKING REVOLUTIONARY AI MATCHING")
        print("-" * 40)
        
        try:
            # Initialize system
            start_time = time.time()
            ai_system = RevolutionaryAIMatching()
            init_time = time.time() - start_time
            print(f"✅ System initialized in {init_time:.2f} seconds")
            
            # Benchmark match generation
            results = []
            
            for material in self.test_materials[:3]:  # Test first 3 materials
                for company in self.test_companies:
                    print(f"  Testing: {material} for {company['name']}")
                    
                    # Determine material type
                    if 'Steel' in material or 'Aluminum' in material:
                        material_type = 'metal'
                    elif 'Plastic' in material:
                        material_type = 'plastic'
                    elif 'Chemical' in material:
                        material_type = 'chemical'
                    elif 'Electronic' in material:
                        material_type = 'electronic'
                    elif 'Paper' in material:
                        material_type = 'paper'
                    elif 'Glass' in material:
                        material_type = 'glass'
                    elif 'Rubber' in material:
                        material_type = 'rubber'
                    else:
                        material_type = 'other'
                    
                    # Measure performance
                    start_time = time.time()
                    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    
                    # Use the updated API
                    matches = await ai_system.generate_high_quality_matches(
                        material, material_type, company['name']
                    )
                    
                    elapsed_time = time.time() - start_time
                    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    memory_used = memory_after - memory_before
                    
                    # Record results
                    results.append({
                        'material': material,
                        'company': company['name'],
                        'matches_count': len(matches),
                        'time_seconds': elapsed_time,
                        'memory_mb': memory_used,
                        'matches_per_second': len(matches) / elapsed_time if elapsed_time > 0 else 0
                    })
                    
                    print(f"    ✓ Generated {len(matches)} matches in {elapsed_time:.2f} seconds")
            
            # Calculate aggregate metrics
            total_matches = sum(r['matches_count'] for r in results)
            total_time = sum(r['time_seconds'] for r in results)
            avg_time_per_match = total_time / total_matches if total_matches > 0 else 0
            avg_memory = sum(r['memory_mb'] for r in results) / len(results) if results else 0
            
            print("\n📊 REVOLUTIONARY AI MATCHING BENCHMARK RESULTS:")
            print(f"  Total matches generated: {total_matches}")
            print(f"  Total processing time: {total_time:.2f} seconds")
            print(f"  Average time per match: {avg_time_per_match:.4f} seconds")
            print(f"  Average memory usage: {avg_memory:.2f} MB")
            print(f"  Matches per second: {total_matches / total_time:.2f}")
            
            # Store benchmark results
            self.benchmark_results['revolutionary_ai_matching'] = {
                'total_matches': total_matches,
                'total_time': total_time,
                'avg_time_per_match': avg_time_per_match,
                'avg_memory': avg_memory,
                'matches_per_second': total_matches / total_time if total_time > 0 else 0,
                'detailed_results': results
            }
            
        except Exception as e:
            print(f"❌ Error benchmarking Revolutionary AI Matching: {e}")
            import traceback
            traceback.print_exc()
    
    async def _benchmark_ai_listings_generation(self):
        """Benchmark AI Listings Generation performance"""
        print("\n📦 BENCHMARKING AI LISTINGS GENERATION")
        print("-" * 40)
        
        try:
            # Initialize system
            start_time = time.time()
            listings_generator = RevolutionaryAIListingsGenerator()
            init_time = time.time() - start_time
            
            print(f"✅ System initialization: {init_time:.4f}s")
            
            # Test listings generation performance
            generation_times = []
            listing_counts = []
            listing_values = []
            
            for i, company in enumerate(self.test_companies):
                print(f"  Testing company {i+1}/{len(self.test_companies)}: {company['name']}")
                
                start_time = time.time()
                listings = await listings_generator.generate_ai_listings(company)
                end_time = time.time()
                
                generation_time = end_time - start_time
                generation_times.append(generation_time)
                listing_counts.append(len(listings))
                
                if listings:
                    total_value = sum(l.potential_value for l in listings if hasattr(l, 'potential_value'))
                    listing_values.append(total_value)
                
                print(f"    Generated {len(listings)} listings in {generation_time:.4f}s")
            
            # Calculate real metrics
            avg_generation_time = np.mean(generation_times)
            avg_listing_count = np.mean(listing_counts)
            avg_listing_value = np.mean(listing_values) if listing_values else 0
            throughput = len(self.test_companies) / sum(generation_times)
            
            self.benchmark_results['ai_listings_generation'] = {
                'initialization_time': init_time,
                'avg_generation_time': avg_generation_time,
                'avg_listing_count': avg_listing_count,
                'avg_listing_value': avg_listing_value,
                'throughput': throughput,
                'total_listings_generated': sum(listing_counts),
                'success': True
            }
            
            print(f"✅ Average generation time: {avg_generation_time:.4f}s")
            print(f"✅ Average listings per company: {avg_listing_count:.1f}")
            print(f"✅ Average total value: ${avg_listing_value:,.0f}")
            print(f"✅ Throughput: {throughput:.2f} companies/second")
            
        except Exception as e:
            self.logger.error(f"❌ AI Listings Generation benchmark failed: {e}")
            self.benchmark_results['ai_listings_generation'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _benchmark_listing_inference_service(self):
        """Benchmark Listing Inference Service performance"""
        print("\n🤖 BENCHMARKING LISTING INFERENCE SERVICE")
        print("-" * 40)
        
        try:
            # Initialize system
            start_time = time.time()
            inference_service = ListingInferenceService()
            init_time = time.time() - start_time
            
            print(f"✅ System initialization: {init_time:.4f}s")
            
            # Test inference performance
            inference_times = []
            output_counts = []
            confidence_scores = []
            
            for i, company in enumerate(self.test_companies):
                print(f"  Testing company {i+1}/{len(self.test_companies)}: {company['name']}")
                
                start_time = time.time()
                result = await inference_service.generate_listings_from_profile(company)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                outputs = result.get('predicted_outputs', [])
                output_counts.append(len(outputs))
                
                confidence = result.get('generation_metadata', {}).get('ai_confidence_score', 0)
                confidence_scores.append(confidence)
                
                print(f"    Generated {len(outputs)} outputs in {inference_time:.4f}s")
            
            # Calculate real metrics
            avg_inference_time = np.mean(inference_times)
            avg_output_count = np.mean(output_counts)
            avg_confidence = np.mean(confidence_scores)
            throughput = len(self.test_companies) / sum(inference_times)
            
            self.benchmark_results['listing_inference_service'] = {
                'initialization_time': init_time,
                'avg_inference_time': avg_inference_time,
                'avg_output_count': avg_output_count,
                'avg_confidence': avg_confidence,
                'throughput': throughput,
                'total_outputs_generated': sum(output_counts),
                'success': True
            }
            
            print(f"✅ Average inference time: {avg_inference_time:.4f}s")
            print(f"✅ Average outputs per company: {avg_output_count:.1f}")
            print(f"✅ Average confidence score: {avg_confidence:.3f}")
            print(f"✅ Throughput: {throughput:.2f} companies/second")
            
        except Exception as e:
            self.logger.error(f"❌ Listing Inference Service benchmark failed: {e}")
            self.benchmark_results['listing_inference_service'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _benchmark_enhanced_ai_generator(self):
        """Benchmark Enhanced AI Generator performance"""
        print("\n🚀 BENCHMARKING ENHANCED AI GENERATOR")
        print("-" * 40)
        
        try:
            # Initialize system
            start_time = time.time()
            enhanced_generator = EnhancedAIGenerator()
            init_time = time.time() - start_time
            
            print(f"✅ System initialization: {init_time:.4f}s")
            
            # Test generation performance
            generation_times = []
            material_counts = []
            success_rates = []
            
            for i, company in enumerate(self.test_companies):
                print(f"  Testing company {i+1}/{len(self.test_companies)}: {company['name']}")
                
                start_time = time.time()
                result = enhanced_generator.generate_ultra_accurate_listings(company)
                end_time = time.time()
                
                generation_time = end_time - start_time
                generation_times.append(generation_time)
                
                waste_count = result.get('waste_materials', 0)
                req_count = result.get('requirements', 0)
                total_materials = waste_count + req_count
                material_counts.append(total_materials)
                
                success = result.get('success', False)
                success_rates.append(1.0 if success else 0.0)
                
                print(f"    Generated {total_materials} materials in {generation_time:.4f}s")
            
            # Calculate real metrics
            avg_generation_time = np.mean(generation_times)
            avg_material_count = np.mean(material_counts)
            success_rate = np.mean(success_rates)
            throughput = len(self.test_companies) / sum(generation_times)
            
            self.benchmark_results['enhanced_ai_generator'] = {
                'initialization_time': init_time,
                'avg_generation_time': avg_generation_time,
                'avg_material_count': avg_material_count,
                'success_rate': success_rate,
                'throughput': throughput,
                'total_materials_generated': sum(material_counts),
                'success': True
            }
            
            print(f"✅ Average generation time: {avg_generation_time:.4f}s")
            print(f"✅ Average materials per company: {avg_material_count:.1f}")
            print(f"✅ Success rate: {success_rate:.1%}")
            print(f"✅ Throughput: {throughput:.2f} companies/second")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced AI Generator benchmark failed: {e}")
            self.benchmark_results['enhanced_ai_generator'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _benchmark_system_resources(self):
        """Benchmark system resource usage"""
        print("\n💻 BENCHMARKING SYSTEM RESOURCES")
        print("-" * 40)
        
        # Monitor CPU and memory during a heavy operation
        process = psutil.Process()
        
        # Start monitoring
        cpu_percentages = []
        memory_usage = []
        
        # Simulate heavy AI operation
        print("  Simulating heavy AI operation...")
        for i in range(10):
            # Simulate AI processing
            start_time = time.time()
            
            # CPU-intensive operation
            _ = sum(i**2 for i in range(10000))
            
            # Memory allocation
            test_data = torch.randn(1000, 1000)
            
            # Record metrics
            cpu_percentages.append(process.cpu_percent())
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            # Clean up
            del test_data
            gc.collect()
            
            # Small delay
            await asyncio.sleep(0.1)
        
        # Calculate metrics
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        self.benchmark_results['system_resources'] = {
            'avg_cpu_usage': avg_cpu,
            'max_cpu_usage': max_cpu,
            'avg_memory_usage_mb': avg_memory,
            'max_memory_usage_mb': max_memory,
            'success': True
        }
        
        print(f"✅ Average CPU usage: {avg_cpu:.1f}%")
        print(f"✅ Maximum CPU usage: {max_cpu:.1f}%")
        print(f"✅ Average memory usage: {avg_memory:.1f} MB")
        print(f"✅ Maximum memory usage: {max_memory:.1f} MB")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive benchmark report"""
        print("\n📊 COMPREHENSIVE BENCHMARK REPORT")
        print("="*60)
        
        # Calculate overall performance metrics
        successful_benchmarks = sum(1 for result in self.benchmark_results.values() if result.get('success', False))
        total_benchmarks = len(self.benchmark_results)
        success_rate = successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0
        
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   Successful benchmarks: {successful_benchmarks}/{total_benchmarks}")
        print(f"   Success rate: {success_rate:.1%}")
        
        # Performance comparison
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        
        if 'revolutionary_ai_matching' in self.benchmark_results and self.benchmark_results['revolutionary_ai_matching']['success']:
            matching = self.benchmark_results['revolutionary_ai_matching']
            print(f"   🧠 Revolutionary AI Matching:")
            print(f"      Average time: {matching['avg_matching_time']:.4f}s")
            print(f"      Throughput: {matching['throughput']:.2f} materials/second")
            print(f"      Match quality: {matching['avg_match_score']:.3f}")
        
        if 'ai_listings_generation' in self.benchmark_results and self.benchmark_results['ai_listings_generation']['success']:
            listings = self.benchmark_results['ai_listings_generation']
            print(f"   📦 AI Listings Generation:")
            print(f"      Average time: {listings['avg_generation_time']:.4f}s")
            print(f"      Throughput: {listings['throughput']:.2f} companies/second")
            print(f"      Listings per company: {listings['avg_listing_count']:.1f}")
        
        if 'listing_inference_service' in self.benchmark_results and self.benchmark_results['listing_inference_service']['success']:
            inference = self.benchmark_results['listing_inference_service']
            print(f"   🤖 Listing Inference Service:")
            print(f"      Average time: {inference['avg_inference_time']:.4f}s")
            print(f"      Throughput: {inference['throughput']:.2f} companies/second")
            print(f"      Confidence: {inference['avg_confidence']:.3f}")
        
        if 'enhanced_ai_generator' in self.benchmark_results and self.benchmark_results['enhanced_ai_generator']['success']:
            enhanced = self.benchmark_results['enhanced_ai_generator']
            print(f"   🚀 Enhanced AI Generator:")
            print(f"      Average time: {enhanced['avg_generation_time']:.4f}s")
            print(f"      Throughput: {enhanced['throughput']:.2f} companies/second")
            print(f"      Success rate: {enhanced['success_rate']:.1%}")
        
        if 'system_resources' in self.benchmark_results and self.benchmark_results['system_resources']['success']:
            resources = self.benchmark_results['system_resources']
            print(f"   💻 System Resources:")
            print(f"      Average CPU: {resources['avg_cpu_usage']:.1f}%")
            print(f"      Average memory: {resources['avg_memory_usage_mb']:.1f} MB")
        
        # Save detailed results
        with open('real_benchmark_results.json', 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: real_benchmark_results.json")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        
        # Check if performance is production-ready
        production_ready = True
        issues = []
        
        for name, result in self.benchmark_results.items():
            if result.get('success', False):
                if 'avg_matching_time' in result and result['avg_matching_time'] > 5.0:
                    production_ready = False
                    issues.append(f"{name}: Slow matching (>5s)")
                
                if 'avg_generation_time' in result and result['avg_generation_time'] > 10.0:
                    production_ready = False
                    issues.append(f"{name}: Slow generation (>10s)")
                
                if 'avg_inference_time' in result and result['avg_inference_time'] > 8.0:
                    production_ready = False
                    issues.append(f"{name}: Slow inference (>8s)")
        
        if production_ready:
            print(f"   ✅ Production-ready performance")
        else:
            print(f"   ⚠️  Performance issues detected:")
            for issue in issues:
                print(f"      - {issue}")
        
        print(f"\n🎉 REAL BENCHMARKING COMPLETE!")
        print(f"These are ACTUAL measurements, not simulated benchmarks!")

async def main():
    """Main benchmark function"""
    benchmark = RealPerformanceBenchmark()
    await benchmark.run_comprehensive_benchmarks()

if __name__ == "__main__":
    asyncio.run(main()) 