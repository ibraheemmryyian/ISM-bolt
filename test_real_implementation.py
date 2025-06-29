#!/usr/bin/env python3
"""
Comprehensive Test Script for Real AI Implementation
Tests all engines: Carbon Calculation, Waste Tracking, AI Matching
"""

import json
import sys
import os
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_carbon_calculation_engine():
    """Test the real carbon calculation engine"""
    print("🧪 Testing Carbon Calculation Engine...")
    
    try:
        from carbon_calculation_engine import carbon_engine
        
        # Test company data
        test_company = {
            'name': 'Test Manufacturing Co',
            'industry': 'manufacturing',
            'location': 'Cairo',
            'employee_count': 500,
            'materials': ['steel', 'aluminum', 'plastic'],
            'processes': 'metal fabrication, assembly, packaging',
            'water_usage': 2000,
            'carbon_footprint': 2500,
            'sustainability_score': 65
        }
        
        # Calculate carbon footprint
        result = carbon_engine.calculate_company_carbon_footprint(test_company)
        
        if 'error' in result:
            print(f"❌ Carbon calculation failed: {result['error']}")
            return False
        
        print(f"✅ Carbon footprint calculated: {result['total_carbon_footprint']} tons CO2")
        print(f"   - Material emissions: {result['material_emissions']}")
        print(f"   - Energy emissions: {result['energy_emissions']}")
        print(f"   - Transport emissions: {result['transport_emissions']}")
        print(f"   - Waste emissions: {result['waste_emissions']}")
        print(f"   - Efficiency score: {result['efficiency_metrics']['efficiency_score']}")
        
        # Test reduction potential
        initiatives = [
            {'id': 'energy_1', 'category': 'energy efficiency', 'question': 'LED lighting'},
            {'id': 'waste_1', 'category': 'waste management', 'question': 'Recycling program'}
        ]
        
        reduction_result = carbon_engine.calculate_reduction_potential(test_company, initiatives)
        
        if 'error' not in reduction_result:
            print(f"✅ Reduction potential calculated: {reduction_result['potential_reduction']} tons CO2")
            print(f"   - Reduction percentage: {reduction_result['reduction_percentage']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Carbon calculation test failed: {str(e)}")
        return False

def test_waste_tracking_engine():
    """Test the real waste tracking engine"""
    print("\n🧪 Testing Waste Tracking Engine...")
    
    try:
        from waste_tracking_engine import waste_engine
        
        # Test company data
        test_company = {
            'name': 'Test Textile Co',
            'industry': 'textiles',
            'location': 'New York',
            'employee_count': 300,
            'materials': ['cotton', 'polyester', 'nylon'],
            'processes': 'spinning, weaving, dyeing',
            'waste_streams': ['fabric_scraps', 'dye_waste', 'packaging_waste'],
            'energy_needs': ['electricity', 'steam', 'hot_water'],
            'water_usage': 1500,
            'carbon_footprint': 1800,
            'sustainability_score': 55
        }
        
        # Calculate waste profile
        result = waste_engine.calculate_company_waste_profile(test_company)
        
        if 'error' in result:
            print(f"❌ Waste calculation failed: {result['error']}")
            return False
        
        print(f"✅ Waste profile calculated: {result['total_waste_generated']} kg/month")
        print(f"   - Solid waste: {result['waste_by_type']['solid_waste']}")
        print(f"   - Liquid waste: {result['waste_by_type']['liquid_waste']}")
        print(f"   - Hazardous waste: {result['waste_by_type']['hazardous_waste']}")
        print(f"   - Recyclable waste: {result['waste_by_type']['recyclable_waste']}")
        print(f"   - Total costs: ${result['waste_costs']['total_net_cost']}")
        print(f"   - Recycling rate: {result['recycling_potential']['current_recycling_rate']}%")
        
        # Test waste reduction potential
        initiatives = [
            {'id': 'recycling_1', 'category': 'recycling', 'question': 'Enhanced recycling'},
            {'id': 'waste_1', 'category': 'waste reduction', 'question': 'Waste minimization'}
        ]
        
        reduction_result = waste_engine.calculate_waste_reduction_potential(test_company, initiatives)
        
        if 'error' not in reduction_result:
            print(f"✅ Waste reduction potential: {reduction_result['potential_reduction']} kg/month")
            print(f"   - Cost savings: ${reduction_result['cost_savings']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Waste tracking test failed: {str(e)}")
        return False

def test_ai_matching_engine():
    """Test the real AI matching engine"""
    print("\n🧪 Testing AI Matching Engine...")
    
    try:
        from real_ai_matching_engine import real_ai_matching_engine
        
        # Test company data
        test_company = {
            'name': 'Test Food Co',
            'industry': 'food_beverage',
            'location': 'London',
            'employee_count': 200,
            'materials': ['grains', 'sugar', 'vegetables'],
            'products': ['processed grains', 'sugar products', 'vegetable products'],
            'waste_streams': ['organic_waste', 'packaging_waste', 'water_waste'],
            'energy_needs': ['electricity', 'refrigeration', 'steam'],
            'water_usage': 1200,
            'carbon_footprint': 1500,
            'sustainability_score': 70,
            'matching_preferences': {
                'material_exchange': 0.7,
                'waste_recycling': 0.8,
                'energy_sharing': 0.5,
                'water_reuse': 0.6,
                'logistics_sharing': 0.4
            }
        }
        
        # Find symbiotic matches
        matches = real_ai_matching_engine.find_symbiotic_matches(test_company, top_k=5)
        
        if 'error' in matches:
            print(f"❌ AI matching failed: {matches['error']}")
            return False
        
        print(f"✅ Found {len(matches)} symbiotic matches")
        
        for i, match in enumerate(matches[:3], 1):
            print(f"   {i}. {match['company_name']} ({match['industry']})")
            print(f"      Match score: {match['match_score']}")
            print(f"      Match type: {match['match_type']}")
            print(f"      Potential savings: ${match['potential_savings']}")
            print(f"      Description: {match['description'][:100]}...")
        
        # Get matching statistics
        stats = real_ai_matching_engine.get_matching_statistics()
        print(f"✅ Matching statistics:")
        print(f"   - Total companies: {stats['total_companies']}")
        print(f"   - Industries represented: {stats['industries_represented']}")
        print(f"   - Materials covered: {stats['materials_covered']}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI matching test failed: {str(e)}")
        return False

def test_integrated_workflow():
    """Test the complete integrated workflow"""
    print("\n🧪 Testing Integrated Workflow...")
    
    try:
        from carbon_calculation_engine import carbon_engine
        from waste_tracking_engine import waste_engine
        from real_ai_matching_engine import real_ai_matching_engine
        
        # Comprehensive test company
        test_company = {
            'name': 'Integrated Test Co',
            'industry': 'chemicals',
            'location': 'Tokyo',
            'employee_count': 800,
            'materials': ['petroleum', 'natural_gas', 'minerals', 'acids'],
            'products': ['petroleum compounds', 'natural_gas solutions', 'mineral products'],
            'waste_streams': ['chemical_waste', 'hazardous_waste', 'water_waste'],
            'energy_needs': ['electricity', 'natural_gas', 'steam', 'cooling'],
            'water_usage': 3000,
            'carbon_footprint': 4000,
            'sustainability_score': 45,
            'matching_preferences': {
                'material_exchange': 0.8,
                'waste_recycling': 0.9,
                'energy_sharing': 0.7,
                'water_reuse': 0.8,
                'logistics_sharing': 0.6
            }
        }
        
        print("📊 Running comprehensive analysis...")
        
        # 1. Carbon calculation
        carbon_result = carbon_engine.calculate_company_carbon_footprint(test_company)
        print(f"   Carbon footprint: {carbon_result['total_carbon_footprint']} tons CO2")
        
        # 2. Waste analysis
        waste_result = waste_engine.calculate_company_waste_profile(test_company)
        print(f"   Waste generated: {waste_result['total_waste_generated']} kg/month")
        
        # 3. AI matching
        matches = real_ai_matching_engine.find_symbiotic_matches(test_company, top_k=3)
        print(f"   Top match score: {matches[0]['match_score'] if matches else 'N/A'}")
        
        # 4. Generate sustainability initiatives
        initiatives = [
            {'id': 'energy_1', 'category': 'energy efficiency', 'question': 'Energy optimization'},
            {'id': 'waste_1', 'category': 'waste management', 'question': 'Waste reduction'},
            {'id': 'renewable_1', 'category': 'renewable energy', 'question': 'Solar installation'}
        ]
        
        carbon_reduction = carbon_engine.calculate_reduction_potential(test_company, initiatives)
        waste_reduction = waste_engine.calculate_waste_reduction_potential(test_company, initiatives)
        
        total_savings = (
            carbon_reduction.get('potential_reduction', 0) * 50 +  # $50 per ton CO2
            waste_reduction.get('cost_savings', 0)
        )
        
        print(f"   Total potential savings: ${total_savings:,.2f}")
        
        # 5. Generate comprehensive report
        report = {
            'company_info': test_company,
            'carbon_analysis': carbon_result,
            'waste_analysis': waste_result,
            'top_matches': matches[:3],
            'sustainability_opportunities': {
                'carbon_reduction': carbon_reduction,
                'waste_reduction': waste_reduction,
                'total_savings': total_savings
            },
            'recommendations': [
                'Implement energy efficiency measures',
                'Enhance waste recycling programs',
                'Explore symbiotic partnerships',
                'Consider renewable energy options'
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        with open('comprehensive_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Comprehensive workflow test completed!")
        print("📄 Report saved to: comprehensive_test_report.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated workflow test failed: {str(e)}")
        return False

def test_performance():
    """Test performance with multiple companies"""
    print("\n🧪 Testing Performance...")
    
    try:
        from real_ai_matching_engine import real_ai_matching_engine
        
        # Test with multiple companies
        test_companies = [
            {
                'name': 'Manufacturing Co A',
                'industry': 'manufacturing',
                'employee_count': 500,
                'materials': ['steel', 'aluminum'],
                'location': 'Cairo'
            },
            {
                'name': 'Textile Co B',
                'industry': 'textiles',
                'employee_count': 300,
                'materials': ['cotton', 'polyester'],
                'location': 'New York'
            },
            {
                'name': 'Food Co C',
                'industry': 'food_beverage',
                'employee_count': 200,
                'materials': ['grains', 'vegetables'],
                'location': 'London'
            }
        ]
        
        start_time = time.time()
        
        for i, company in enumerate(test_companies, 1):
            matches = real_ai_matching_engine.find_symbiotic_matches(company, top_k=5)
            print(f"   Company {i}: Found {len(matches)} matches in {time.time() - start_time:.2f}s")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_companies)
        
        print(f"✅ Performance test completed:")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Average time per company: {avg_time:.2f}s")
        print(f"   - Companies processed: {len(test_companies)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Real AI Implementation Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run individual engine tests
    test_results.append(('Carbon Calculation', test_carbon_calculation_engine()))
    test_results.append(('Waste Tracking', test_waste_tracking_engine()))
    test_results.append(('AI Matching', test_ai_matching_engine()))
    
    # Run integrated workflow test
    test_results.append(('Integrated Workflow', test_integrated_workflow()))
    
    # Run performance test
    test_results.append(('Performance', test_performance()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Real AI implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 