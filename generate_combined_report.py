#!/usr/bin/env python3
"""
Combined Test Report Generator
Combines all test reports into a comprehensive production readiness report
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

def load_test_report(filename: str) -> Dict[str, Any]:
    """Load a test report from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        else:
            return {
                'summary': {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'warning_tests': 0,
                    'success_rate': 0
                },
                'results': {},
                'error': f'Report file {filename} not found'
            }
    except Exception as e:
        return {
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'warning_tests': 0,
                'success_rate': 0
            },
            'results': {},
            'error': f'Error loading {filename}: {str(e)}'
        }

def generate_combined_report():
    """Generate combined test report"""
    
    # Load all test reports
    reports = {
        'production': load_test_report('production_test_report.json'),
        'ai_services': load_test_report('ai_services_test_report.json'),
        'database': load_test_report('database_integration_test_report.json'),
        'frontend': load_test_report('frontend_integration_test_report.json')
    }
    
    # Calculate overall statistics
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    for report_name, report_data in reports.items():
        if 'summary' in report_data:
            summary = report_data['summary']
            total_tests += summary.get('total_tests', 0)
            total_passed += summary.get('passed_tests', 0)
            total_failed += summary.get('failed_tests', 0)
            total_warnings += summary.get('warning_tests', 0)
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Generate combined report
    combined_report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_test_suites': len(reports),
            'overall_status': 'PRODUCTION_READY' if overall_success_rate >= 90 else 
                             'PRODUCTION_READY_WITH_WARNINGS' if overall_success_rate >= 70 else 
                             'NOT_PRODUCTION_READY'
        },
        'overall_summary': {
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'failed_tests': total_failed,
            'warning_tests': total_warnings,
            'success_rate': overall_success_rate
        },
        'test_suites': reports,
        'recommendations': generate_recommendations(reports, overall_success_rate),
        'production_readiness_score': calculate_readiness_score(reports, overall_success_rate)
    }
    
    # Save combined report
    with open('combined_test_report.json', 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("COMBINED PRODUCTION TEST REPORT")
    print("=" * 60)
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    print(f"Warnings: {total_warnings} ⚠️")
    print()
    
    # Print individual suite results
    for suite_name, suite_data in reports.items():
        if 'summary' in suite_data:
            summary = suite_data['summary']
            success_rate = summary.get('success_rate', 0)
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            print(f"{status} {suite_name.replace('_', ' ').title()}: {success_rate:.1f}%")
    
    print()
    print(f"Production Readiness: {combined_report['metadata']['overall_status']}")
    print(f"Readiness Score: {combined_report['production_readiness_score']}/100")
    print()
    
    # Print key recommendations
    print("KEY RECOMMENDATIONS:")
    print("-" * 30)
    for i, rec in enumerate(combined_report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    print()
    print("📄 Combined report saved to: combined_test_report.json")

def generate_recommendations(reports: Dict[str, Any], overall_success_rate: float) -> List[str]:
    """Generate recommendations based on test results"""
    recommendations = []
    
    # Overall recommendations
    if overall_success_rate < 70:
        recommendations.append("CRITICAL: Fix failed tests before production deployment")
    elif overall_success_rate < 90:
        recommendations.append("IMPORTANT: Address warnings and improve test coverage")
    
    # Specific recommendations based on test suites
    for suite_name, suite_data in reports.items():
        if 'summary' in suite_data:
            summary = suite_data['summary']
            success_rate = summary.get('success_rate', 0)
            
            if success_rate < 60:
                if suite_name == 'ai_services':
                    recommendations.append("CRITICAL: Fix AI services - core functionality affected")
                elif suite_name == 'database':
                    recommendations.append("CRITICAL: Fix database integration - data operations affected")
                elif suite_name == 'frontend':
                    recommendations.append("CRITICAL: Fix frontend integration - user experience affected")
                elif suite_name == 'production':
                    recommendations.append("CRITICAL: Fix production environment issues")
            elif success_rate < 80:
                if suite_name == 'ai_services':
                    recommendations.append("IMPORTANT: Improve AI services reliability")
                elif suite_name == 'database':
                    recommendations.append("IMPORTANT: Optimize database performance")
                elif suite_name == 'frontend':
                    recommendations.append("IMPORTANT: Enhance frontend user experience")
                elif suite_name == 'production':
                    recommendations.append("IMPORTANT: Address production environment warnings")
    
    # Add positive recommendations
    if overall_success_rate >= 90:
        recommendations.append("EXCELLENT: System is production-ready with high confidence")
    elif overall_success_rate >= 80:
        recommendations.append("GOOD: System is production-ready with minor improvements needed")
    
    # Add general recommendations
    recommendations.append("MONITOR: Set up production monitoring and alerting")
    recommendations.append("BACKUP: Ensure database backups are configured")
    recommendations.append("SECURITY: Review security configurations before deployment")
    
    return recommendations

def calculate_readiness_score(reports: Dict[str, Any], overall_success_rate: float) -> int:
    """Calculate production readiness score (0-100)"""
    base_score = overall_success_rate
    
    # Bonus points for high-performing components
    bonus_points = 0
    for suite_name, suite_data in reports.items():
        if 'summary' in suite_data:
            summary = suite_data['summary']
            success_rate = summary.get('success_rate', 0)
            
            # Critical components get more weight
            if suite_name in ['ai_services', 'database'] and success_rate >= 90:
                bonus_points += 5
            elif suite_name in ['frontend', 'production'] and success_rate >= 90:
                bonus_points += 3
    
    # Penalty for critical failures
    penalty_points = 0
    for suite_name, suite_data in reports.items():
        if 'summary' in suite_data:
            summary = suite_data['summary']
            success_rate = summary.get('success_rate', 0)
            
            if success_rate < 60:
                if suite_name in ['ai_services', 'database']:
                    penalty_points += 20  # Critical components
                else:
                    penalty_points += 10  # Other components
    
    final_score = max(0, min(100, base_score + bonus_points - penalty_points))
    return int(final_score)

if __name__ == "__main__":
    generate_combined_report() 