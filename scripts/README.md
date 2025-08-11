# RAG System Test Scripts

This directory contains comprehensive test scripts for validating the RAG system connectivity and functionality.

## Test Scripts Overview

### 1. `test_server_connection.py`
**Purpose**: Tests basic server connectivity and network reachability
- ✅ Basic TCP connectivity
- ✅ HTTP endpoint accessibility  
- ✅ Async connectivity
- ✅ Model endpoint reachability

**Usage**:
```bash
python3 scripts/test_server_connection.py
```

### 2. `test_model_endpoint.py`
**Purpose**: Tests the model endpoint functionality and response quality
- ✅ Endpoint connectivity
- ✅ Model inference with multiple payload formats
- ✅ Async inference
- ✅ Response quality analysis
- ✅ Mistral fallback testing

**Usage**:
```bash
python3 scripts/test_model_endpoint.py
```

### 3. `test_qdrant_connection.py`
**Purpose**: Tests Qdrant vector database connectivity and operations
- ✅ Basic Qdrant connection
- ✅ Collection operations (create, list, delete)
- ✅ Vector operations (insert, search, count)
- ✅ Existing collection validation

**Usage**:
```bash
python3 scripts/test_qdrant_connection.py
```

### 4. `debug_qdrant.py`
**Purpose**: Detailed Qdrant diagnostics and troubleshooting
- ✅ Process status checking
- ✅ Port availability testing
- ✅ HTTP endpoint testing
- ✅ Network interface analysis
- ✅ Issue categorization and solution suggestions

**Usage**:
```bash
python3 scripts/debug_qdrant.py
```

### 5. `run_all_tests.py`
**Purpose**: Comprehensive test runner that executes all tests and generates a detailed report
- ✅ Runs all test scripts
- ✅ Analyzes results and categorizes issues
- ✅ Generates comprehensive report
- ✅ Saves results to timestamped file

**Usage**:
```bash
python3 scripts/run_all_tests.py
```

### 6. `system_status_summary.py`
**Purpose**: High-level system status overview
- ✅ Environment configuration validation
- ✅ Dependency checking
- ✅ Network connectivity testing
- ✅ Component test execution
- ✅ Overall health assessment

**Usage**:
```bash
python3 scripts/system_status_summary.py
```

## Test Results Summary

Based on the latest test run:

### ✅ Working Components
- **Server Connectivity**: All basic connectivity tests passed
- **Model Endpoint**: Successfully connecting to `http://183.82.7.228:9519`
- **Environment Configuration**: All required settings are properly configured
- **Dependencies**: All required packages are installed
- **Network**: All endpoints are reachable

### ⚠️ Issues Identified
- **Qdrant Database**: Connection issues detected
  - Process is running but HTTP endpoints are not responding
  - Port 6334 is open but HTTP requests are refused
  - May require Qdrant service restart or configuration adjustment

## Quick Start

1. **Run all tests**:
   ```bash
   python3 scripts/run_all_tests.py
   ```

2. **Check system status**:
   ```bash
   python3 scripts/system_status_summary.py
   ```

3. **Debug specific issues**:
   ```bash
   python3 scripts/debug_qdrant.py
   ```

## Configuration

All tests use the configuration from `env.server` file. Key settings:

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `SERVER_MODEL_ENDPOINT`: Model service URL
- `QDRANT_HOST`: Qdrant host and port
- `VECTOR_DB_NAME`: Vector database name
- `EMBEDDING_MODEL_NAME`: Embedding model configuration

## Troubleshooting

### Qdrant Connection Issues
If Qdrant tests fail:
1. Run `python3 scripts/debug_qdrant.py` for detailed diagnostics
2. Check if Qdrant process is running: `ps aux | grep qdrant`
3. Verify port is open: `netstat -tlnp | grep 6334`
4. Try restarting Qdrant service

### Model Endpoint Issues
If model tests fail:
1. Verify the endpoint URL is correct in `env.server`
2. Check if the model service is running
3. Test with curl: `curl -X POST http://183.82.7.228:9519/v1/chat/completions`

### Network Connectivity Issues
If connectivity tests fail:
1. Check firewall settings
2. Verify network configuration
3. Test with ping and telnet

## Report Files

**Note**: File saving functionality has been removed. Test results are now displayed only in the console output.

## Exit Codes

- `0`: All tests passed or system is healthy
- `1`: Some tests failed but system is partially operational
- `2`: Multiple failures detected

## Dependencies

Required Python packages:
- `requests`: HTTP client
- `aiohttp`: Async HTTP client
- `qdrant-client`: Qdrant vector database client
- `numpy`: Numerical operations
- `python-dotenv`: Environment variable loading
- `mistralai`: Mistral AI client (optional)

All dependencies are listed in `requirements.txt`.
