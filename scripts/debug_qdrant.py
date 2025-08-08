#!/usr/bin/env python3
"""
Qdrant debugging script to diagnose connection issues.
Provides detailed diagnostics and potential solutions.
"""

import os
import sys
import subprocess
import socket
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

def load_environment() -> Dict[str, Any]:
    """Load environment variables from env.server file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.server')
    load_dotenv(env_file)
    
    return {
        'qdrant_host': os.getenv('QDRANT_HOST', '0.0.0.0:6333'),
        'qdrant_port': int(os.getenv('QDRANT_PORT', 6333)),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
    }

def check_port_availability(host: str, port: int) -> Dict[str, Any]:
    """Check if a port is open and listening."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        return {
            'success': result == 0,
            'host': host,
            'port': port,
            'status': 'open' if result == 0 else 'closed'
        }
    except Exception as e:
        return {
            'success': False,
            'host': host,
            'port': port,
            'error': str(e)
        }

def check_qdrant_process() -> Dict[str, Any]:
    """Check if Qdrant process is running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        
        qdrant_processes = []
        for line in result.stdout.split('\n'):
            if 'qdrant' in line.lower() and not 'grep' in line:
                qdrant_processes.append(line.strip())
        
        return {
            'success': len(qdrant_processes) > 0,
            'processes': qdrant_processes,
            'count': len(qdrant_processes)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def test_http_endpoints() -> List[Dict[str, Any]]:
    """Test various HTTP endpoints for Qdrant."""
    endpoints = [
        'http://localhost:6333',
        'http://127.0.0.1:6333',
        'http://0.0.0.0:6333',
        'http://localhost:6333/collections',
        'http://127.0.0.1:6333/collections',
        'http://localhost:6333/health',
        'http://127.0.0.1:6333/health'
    ]
    
    results = []
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            results.append({
                'endpoint': endpoint,
                'success': True,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'content_length': len(response.content)
            })
        except requests.exceptions.ConnectionError:
            results.append({
                'endpoint': endpoint,
                'success': False,
                'error': 'Connection refused'
            })
        except requests.exceptions.Timeout:
            results.append({
                'endpoint': endpoint,
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            results.append({
                'endpoint': endpoint,
                'success': False,
                'error': str(e)
            })
    
    return results

def check_network_interfaces() -> List[Dict[str, Any]]:
    """Check network interfaces and their status."""
    try:
        result = subprocess.run(
            ['ip', 'addr', 'show'], 
            capture_output=True, 
            text=True
        )
        
        interfaces = []
        current_interface = None
        
        for line in result.stdout.split('\n'):
            if line.strip().startswith('inet '):
                if current_interface:
                    current_interface['addresses'].append(line.strip())
            elif line.strip().startswith('inet6 '):
                if current_interface:
                    current_interface['addresses'].append(line.strip())
            elif ':' in line and not line.startswith(' '):
                if current_interface:
                    interfaces.append(current_interface)
                current_interface = {
                    'name': line.split(':')[0].strip(),
                    'addresses': []
                }
        
        if current_interface:
            interfaces.append(current_interface)
        
        return interfaces
    except Exception as e:
        return [{'error': str(e)}]

def suggest_solutions(issues: List[str]) -> List[str]:
    """Suggest solutions based on detected issues."""
    solutions = []
    
    if 'port_closed' in issues:
        solutions.append("🔧 Start Qdrant server: docker run -p 6333:6333 qdrant/qdrant")
        solutions.append("🔧 Or install and start Qdrant service")
    
    if 'connection_timeout' in issues:
        solutions.append("🔧 Check firewall settings")
        solutions.append("🔧 Verify Qdrant is listening on correct interface")
        solutions.append("🔧 Try restarting Qdrant service")
    
    if 'process_not_running' in issues:
        solutions.append("🔧 Start Qdrant process manually")
        solutions.append("🔧 Check systemd service: sudo systemctl status qdrant")
        solutions.append("🔧 Restart service: sudo systemctl restart qdrant")
    
    if 'ssl_issues' in issues:
        solutions.append("🔧 Use HTTP instead of HTTPS for local connections")
        solutions.append("🔧 Check SSL certificate configuration")
    
    if not solutions:
        solutions.append("🔧 Check Qdrant logs for more details")
        solutions.append("🔧 Verify Qdrant configuration file")
    
    return solutions

def main():
    """Main debugging function."""
    print("🔍 Qdrant Connection Diagnostics")
    print("=" * 50)
    
    # Load configuration
    config = load_environment()
    print(f"📋 Configuration:")
    print(f"   Host: {config['qdrant_host']}")
    print(f"   Port: {config['qdrant_port']}")
    print()
    
    # Check 1: Process status
    print("1️⃣ Checking Qdrant Process")
    print("-" * 30)
    process_result = check_qdrant_process()
    if process_result['success']:
        print(f"✅ Qdrant process is running")
        print(f"   Found {process_result['count']} process(es)")
        for proc in process_result['processes'][:2]:  # Show first 2
            print(f"   • {proc[:80]}...")
    else:
        print(f"❌ Qdrant process not found")
        if 'error' in process_result:
            print(f"   Error: {process_result['error']}")
    print()
    
    # Check 2: Port availability
    print("2️⃣ Checking Port Availability")
    print("-" * 30)
    hosts_to_check = ['localhost', '127.0.0.1', '0.0.0.0']
    port_issues = []
    
    for host in hosts_to_check:
        port_result = check_port_availability(host, config['qdrant_port'])
        if port_result['success']:
            print(f"✅ Port {config['qdrant_port']} is open on {host}")
        else:
            print(f"❌ Port {config['qdrant_port']} is closed on {host}")
            port_issues.append('port_closed')
    
    if port_issues:
        print("   ⚠️  Port connectivity issues detected")
    print()
    
    # Check 3: HTTP endpoints
    print("3️⃣ Testing HTTP Endpoints")
    print("-" * 30)
    http_results = test_http_endpoints()
    http_issues = []
    
    for result in http_results:
        if result['success']:
            print(f"✅ {result['endpoint']} - Status: {result['status_code']}")
        else:
            print(f"❌ {result['endpoint']} - {result['error']}")
            if 'timeout' in result['error'].lower():
                http_issues.append('connection_timeout')
            elif 'refused' in result['error'].lower():
                http_issues.append('connection_refused')
    
    if http_issues:
        print("   ⚠️  HTTP connectivity issues detected")
    print()
    
    # Check 4: Network interfaces
    print("4️⃣ Network Interface Status")
    print("-" * 30)
    interfaces = check_network_interfaces()
    for interface in interfaces[:3]:  # Show first 3 interfaces
        if 'error' not in interface:
            print(f"📡 {interface['name']}")
            for addr in interface['addresses'][:2]:  # Show first 2 addresses
                print(f"   {addr}")
    print()
    
    # Summary and recommendations
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 50)
    
    all_issues = []
    if not process_result['success']:
        all_issues.append('process_not_running')
    if port_issues:
        all_issues.extend(port_issues)
    if http_issues:
        all_issues.extend(http_issues)
    
    if not all_issues:
        print("✅ All basic checks passed")
        print("   Qdrant appears to be running correctly")
    else:
        print(f"❌ Issues detected: {len(all_issues)}")
        for issue in set(all_issues):
            print(f"   • {issue}")
    
    # Solutions
    if all_issues:
        print("\n💡 RECOMMENDED SOLUTIONS")
        print("-" * 30)
        solutions = suggest_solutions(all_issues)
        for solution in solutions:
            print(solution)
    
    print("\n🔧 NEXT STEPS")
    print("-" * 30)
    if all_issues:
        print("1. Try the recommended solutions above")
        print("2. Check Qdrant logs for detailed error messages")
        print("3. Verify Qdrant configuration")
        print("4. Restart Qdrant service if needed")
    else:
        print("1. Qdrant appears to be working correctly")
        print("2. The issue might be with the Python client configuration")
        print("3. Try updating the qdrant-client package")
        print("4. Check if there are any SSL/TLS issues")
    
    return 0 if not all_issues else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
