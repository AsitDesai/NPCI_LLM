#!/usr/bin/env python3
"""
System Status Summary for RAG System.
Provides a comprehensive overview of all system components.
"""

import os
import sys
import subprocess
import time
from typing import Dict, Any, List
from datetime import datetime

def run_quick_test(script_path: str, test_name: str) -> Dict[str, Any]:
    """Run a quick test and return basic status."""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            'name': test_name,
            'success': result.returncode == 0,
            'exit_code': result.returncode,
            'error': result.stderr if result.stderr else None
        }
    except Exception as e:
        return {
            'name': test_name,
            'success': False,
            'exit_code': -1,
            'error': str(e)
        }

def check_environment_config() -> Dict[str, Any]:
    """Check environment configuration."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.server')
    
    if not os.path.exists(env_file):
        return {
            'success': False,
            'error': 'env.server file not found'
        }
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check for key configurations
        checks = {
            'API_HOST': 'API_HOST' in content,
            'API_PORT': 'API_PORT' in content,
            'SERVER_MODEL_ENDPOINT': 'SERVER_MODEL_ENDPOINT' in content,
            'QDRANT_HOST': 'QDRANT_HOST' in content,
            'VECTOR_DB_NAME': 'VECTOR_DB_NAME' in content,
            'EMBEDDING_MODEL_NAME': 'EMBEDDING_MODEL_NAME' in content
        }
        
        return {
            'success': True,
            'file_exists': True,
            'checks': checks,
            'total_checks': len(checks),
            'passed_checks': sum(checks.values())
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are installed."""
    required_packages = [
        'qdrant-client',
        'requests',
        'aiohttp',
        'numpy',
        'llama-index',
        'mistralai'
    ]
    
    results = {}
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            results[package] = True
        except ImportError:
            results[package] = False
    
    return {
        'success': all(results.values()),
        'packages': results,
        'total_packages': len(required_packages),
        'installed_packages': sum(results.values())
    }

def check_network_connectivity() -> Dict[str, Any]:
    """Check basic network connectivity."""
    endpoints = [
        ('localhost', 8000),
        ('183.82.7.228', 9519),
        ('localhost', 6334)
    ]
    
    results = []
    for host, port in endpoints:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            results.append({
                'host': host,
                'port': port,
                'success': result == 0
            })
        except Exception as e:
            results.append({
                'host': host,
                'port': port,
                'success': False,
                'error': str(e)
            })
    
    return {
        'success': any(r['success'] for r in results),
        'endpoints': results,
        'total_endpoints': len(results),
        'reachable_endpoints': sum(1 for r in results if r['success'])
    }

def generate_status_report() -> str:
    """Generate a comprehensive status report."""
    report = []
    report.append("🔍 RAG SYSTEM STATUS SUMMARY")
    report.append("=" * 60)
    report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Environment Configuration
    report.append("📋 ENVIRONMENT CONFIGURATION")
    report.append("-" * 40)
    env_check = check_environment_config()
    if env_check['success']:
        report.append(f"✅ Configuration file: env.server")
        report.append(f"   Checks passed: {env_check['passed_checks']}/{env_check['total_checks']}")
        for check, passed in env_check['checks'].items():
            status = "✅" if passed else "❌"
            report.append(f"   {status} {check}")
    else:
        report.append(f"❌ Configuration error: {env_check['error']}")
    report.append("")
    
    # Dependencies
    report.append("📦 DEPENDENCIES")
    report.append("-" * 40)
    deps_check = check_dependencies()
    if deps_check['success']:
        report.append(f"✅ All dependencies installed")
    else:
        report.append(f"⚠️  {deps_check['installed_packages']}/{deps_check['total_packages']} packages installed")
        for package, installed in deps_check['packages'].items():
            status = "✅" if installed else "❌"
            report.append(f"   {status} {package}")
    report.append("")
    
    # Network Connectivity
    report.append("🌐 NETWORK CONNECTIVITY")
    report.append("-" * 40)
    net_check = check_network_connectivity()
    report.append(f"📡 {net_check['reachable_endpoints']}/{net_check['total_endpoints']} endpoints reachable")
    for endpoint in net_check['endpoints']:
        status = "✅" if endpoint['success'] else "❌"
        report.append(f"   {status} {endpoint['host']}:{endpoint['port']}")
    report.append("")
    
    # Component Tests
    report.append("🧪 COMPONENT TESTS")
    report.append("-" * 40)
    
    test_scripts = [
        ("scripts/test_server_connection.py", "Server Connectivity"),
        ("scripts/test_model_endpoint.py", "Model Endpoint"),
        ("scripts/test_qdrant_connection.py", "Qdrant Database")
    ]
    
    test_results = []
    for script_path, test_name in test_scripts:
        if os.path.exists(script_path):
            result = run_quick_test(script_path, test_name)
            test_results.append(result)
            status = "✅" if result['success'] else "❌"
            report.append(f"   {status} {test_name}")
        else:
            report.append(f"   ❓ {test_name} (script not found)")
    
    # Overall Status
    report.append("")
    report.append("📊 OVERALL STATUS")
    report.append("-" * 40)
    
    # Calculate overall health
    env_ok = env_check['success']
    deps_ok = deps_check['success']
    net_ok = net_check['success']
    tests_ok = any(r['success'] for r in test_results)
    
    if env_ok and deps_ok and net_ok and tests_ok:
        overall_status = "🟢 HEALTHY"
        status_description = "All systems operational"
    elif env_ok and deps_ok and (net_ok or tests_ok):
        overall_status = "🟡 PARTIAL"
        status_description = "Most systems operational with minor issues"
    else:
        overall_status = "🔴 DEGRADED"
        status_description = "Multiple system issues detected"
    
    report.append(f"   {overall_status}")
    report.append(f"   {status_description}")
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 40)
    
    if not env_ok:
        report.append("🔧 Fix environment configuration")
    if not deps_ok:
        report.append("🔧 Install missing dependencies")
    if not net_ok:
        report.append("🔧 Check network connectivity")
    if not tests_ok:
        report.append("🔧 Review component-specific issues")
    
    if env_ok and deps_ok and net_ok and tests_ok:
        report.append("🎉 System is ready for RAG operations!")
    elif env_ok and deps_ok:
        report.append("⚠️  Address connectivity issues before proceeding")
    else:
        report.append("❌ Fix configuration and dependency issues first")
    
    return "\n".join(report)

def main():
    """Main function."""
    print("🚀 Generating System Status Summary")
    print("=" * 60)
    
    report = generate_status_report()
    print(report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"system_status_{timestamp}.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Status report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
