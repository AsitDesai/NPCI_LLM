#!/usr/bin/env python3
"""
Comprehensive test runner for RAG system components.
Runs all connectivity tests and provides a detailed summary.
"""

import os
import sys
import subprocess
import time
from typing import Dict, Any, List
from datetime import datetime

def run_test_script(script_path: str, test_name: str) -> Dict[str, Any]:
    """Run a test script and capture its output and exit code."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'name': test_name,
            'script_path': script_path,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'name': test_name,
            'script_path': script_path,
            'exit_code': -1,
            'stdout': '',
            'stderr': 'Test timed out after 60 seconds',
            'duration': 60,
            'success': False
        }
    except Exception as e:
        return {
            'name': test_name,
            'script_path': script_path,
            'exit_code': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': 0,
            'success': False
        }

def analyze_test_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze test results and provide insights."""
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    # Categorize failures
    connectivity_issues = []
    configuration_issues = []
    other_issues = []
    
    for result in results:
        if not result['success']:
            error_text = result['stderr'].lower()
            if any(keyword in error_text for keyword in ['connection', 'timeout', 'refused']):
                connectivity_issues.append(result['name'])
            elif any(keyword in error_text for keyword in ['config', 'api_key', 'endpoint']):
                configuration_issues.append(result['name'])
            else:
                other_issues.append(result['name'])
    
    return {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
        'connectivity_issues': connectivity_issues,
        'configuration_issues': configuration_issues,
        'other_issues': other_issues,
        'overall_status': 'PASS' if successful_tests == total_tests else 'PARTIAL' if successful_tests > 0 else 'FAIL'
    }

def generate_report(results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
    """Generate a comprehensive test report."""
    report = []
    report.append("🔍 RAG SYSTEM CONNECTIVITY TEST REPORT")
    report.append("=" * 60)
    report.append(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"📊 Overall Status: {analysis['overall_status']}")
    report.append(f"✅ Tests Passed: {analysis['successful_tests']}/{analysis['total_tests']}")
    report.append(f"📈 Success Rate: {analysis['success_rate']:.1f}%")
    report.append("")
    
    # Detailed results
    report.append("📋 DETAILED RESULTS")
    report.append("-" * 40)
    
    for result in results:
        status_emoji = "✅" if result['success'] else "❌"
        report.append(f"{status_emoji} {result['name']}")
        report.append(f"   Duration: {result['duration']:.2f}s")
        report.append(f"   Exit Code: {result['exit_code']}")
        
        if not result['success'] and result['stderr']:
            error_summary = result['stderr'][:100] + "..." if len(result['stderr']) > 100 else result['stderr']
            report.append(f"   Error: {error_summary}")
        report.append("")
    
    # Issues summary
    if analysis['connectivity_issues'] or analysis['configuration_issues'] or analysis['other_issues']:
        report.append("⚠️  ISSUES DETECTED")
        report.append("-" * 40)
        
        if analysis['connectivity_issues']:
            report.append("🔌 Connectivity Issues:")
            for issue in analysis['connectivity_issues']:
                report.append(f"   • {issue}")
            report.append("")
        
        if analysis['configuration_issues']:
            report.append("⚙️  Configuration Issues:")
            for issue in analysis['configuration_issues']:
                report.append(f"   • {issue}")
            report.append("")
        
        if analysis['other_issues']:
            report.append("❓ Other Issues:")
            for issue in analysis['other_issues']:
                report.append(f"   • {issue}")
            report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 40)
    
    if analysis['success_rate'] == 100:
        report.append("🎉 All systems are operational!")
        report.append("   Your RAG system is ready for use.")
    elif analysis['success_rate'] >= 66:
        report.append("⚠️  Most systems are operational with minor issues.")
        report.append("   Review the issues above and consider fixing them.")
    elif analysis['success_rate'] >= 33:
        report.append("⚠️  Some systems are operational but significant issues exist.")
        report.append("   Address the connectivity and configuration issues.")
    else:
        report.append("❌ Multiple system failures detected.")
        report.append("   Review your server configuration and network connectivity.")
    
    return "\n".join(report)

def main():
    """Main test runner function."""
    print("🚀 Starting RAG System Connectivity Tests")
    print("=" * 60)
    
    # Define test scripts
    test_scripts = [
        ("scripts/test_server_connection.py", "Server Connectivity Test"),
        ("scripts/test_model_endpoint.py", "Model Endpoint Test"),
        ("scripts/test_qdrant_connection.py", "Qdrant Database Test")
    ]
    
    # Run all tests
    results = []
    for script_path, test_name in test_scripts:
        if os.path.exists(script_path):
            result = run_test_script(script_path, test_name)
            results.append(result)
        else:
            print(f"❌ Test script not found: {script_path}")
    
    # Analyze results
    analysis = analyze_test_results(results)
    
    # Generate and display report
    report = generate_report(results, analysis)
    print("\n" + report)
    
    # Return appropriate exit code
    if analysis['overall_status'] == 'PASS':
        print("\n🎉 All tests passed! RAG system is ready.")
        return 0
    elif analysis['overall_status'] == 'PARTIAL':
        print("\n⚠️  Some tests failed. Review the report above.")
        return 1
    else:
        print("\n❌ Multiple tests failed. Check your configuration.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
